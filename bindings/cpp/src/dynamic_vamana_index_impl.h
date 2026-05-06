/*
 * Copyright 2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "svs/runtime/vamana_index.h"

#include "svs_runtime_utils.h"

#include <svs/core/data.h>
#include <svs/core/distance.h>
#include <svs/core/graph.h>
#include <svs/core/query_result.h>
#include <svs/extensions/vamana/scalar.h>
#include <svs/lib/file.h>
#include <svs/lib/float16.h>
#include <svs/orchestrators/dynamic_vamana.h>
#include <svs/quantization/scalar/scalar.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <variant>
#include <vector>

namespace svs {
namespace runtime {

// Dynamic Vamana index implementation
class DynamicVamanaIndexImpl {
    using allocator_type = svs::data::Blocked<svs::lib::Allocator<float>>;

  public:
    DynamicVamanaIndexImpl(
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        const VamanaIndex::BuildParams& params,
        const VamanaIndex::SearchParams& default_search_params,
        const VamanaIndex::DynamicIndexParams& dynamic_index_params
    )
        : dim_{dim}
        , metric_type_{metric}
        , storage_kind_{storage_kind}
        , build_params_{params}
        , default_search_params_{default_search_params}
        , dynamic_index_params_{dynamic_index_params} {
        if (!storage::is_supported_storage_kind(storage_kind)) {
            throw StatusException{
                ErrorCode::INVALID_ARGUMENT,
                "The specified storage kind is not compatible with the "
                "DynamicVamanaIndex"};
        }
        // Validate the deferred-compression configuration up front so misconfiguration
        // surfaces at construction time rather than only on the first add.
        if (dynamic_index_params_.deferred_compression_threshold > 0) {
            const auto initial = dynamic_index_params_.initial_storage_kind;
            if (initial != StorageKind::FP32 && initial != StorageKind::FP16) {
                throw StatusException{
                    ErrorCode::INVALID_ARGUMENT,
                    "Deferred compression initial_storage_kind must be FP32 or FP16"};
            }
            if (!storage::is_supported_storage_kind(initial)) {
                throw StatusException{
                    ErrorCode::INVALID_ARGUMENT,
                    "Deferred compression initial_storage_kind is not supported on "
                    "this CPU"};
            }
        }
        current_storage_kind_ = effective_initial_storage_kind();
    }

    size_t size() const { return impl_ ? impl_->size() : 0; }

    size_t blocksize_bytes() const { return 1u << dynamic_index_params_.blocksize_exp; }

    size_t dimensions() const { return dim_; }

    MetricType metric_type() const { return metric_type_; }

    StorageKind get_storage_kind() const { return storage_kind_; }

    /// @brief Storage kind currently backing the index.
    ///
    /// Equal to `get_storage_kind()` unless deferred compression is enabled and the
    /// threshold has not yet been crossed, in which case this returns the *initial*
    /// (uncompressed) storage kind. Public so that tests and Python bindings can
    /// observe the deferred-compression transition.
    StorageKind get_current_storage_kind() const { return current_storage_kind_; }

    /// @brief Whether deferred compression is configured (threshold > 0 and target
    /// requires training).
    bool deferred_compression_enabled() const {
        return dynamic_index_params_.deferred_compression_threshold > 0 &&
               storage_kind_ != effective_initial_storage_kind();
    }

    void add(data::ConstSimpleDataView<float> data, std::span<const size_t> labels) {
        if (!impl_) {
            auto blocksize_bytes = lib::PowerOfTwo(dynamic_index_params_.blocksize_exp);
            init_impl(data, labels, blocksize_bytes);
        } else {
            impl_->add_points(data, labels);
        }
        // Deferred compression: once the live count reaches the threshold and we are
        // still on the initial (uncompressed) storage, train the target compression
        // and swap the dataset in-place while reusing the existing graph.
        maybe_swap_to_target_storage();
    }

    void search(
        svs::QueryResultView<size_t> result,
        svs::data::ConstSimpleDataView<float> queries,
        const VamanaIndex::SearchParams* params = nullptr,
        IDFilter* filter = nullptr
    ) const {
        if (!impl_) {
            auto& dists = result.distances();
            std::fill(dists.begin(), dists.end(), Unspecify<float>());
            auto& inds = result.indices();
            std::fill(inds.begin(), inds.end(), Unspecify<size_t>());
            throw StatusException{ErrorCode::NOT_INITIALIZED, "Index not initialized"};
        }

        if (queries.size() == 0) {
            return;
        }

        const size_t k = result.n_neighbors();
        if (k == 0) {
            throw StatusException{ErrorCode::INVALID_ARGUMENT, "k must be greater than 0"};
        }

        auto sp = make_search_parameters(params);

        // Simple search
        if (filter == nullptr) {
            impl_->search(result, queries, sp);
            return;
        }

        // Selective search with IDSelector
        auto old_sp = impl_->get_search_parameters();
        impl_->set_search_parameters(sp);

        auto search_closure = [&](const auto& range, uint64_t SVS_UNUSED(tid)) {
            for (auto i : range) {
                // For every query
                auto query = queries.get_datum(i);
                auto iterator = impl_->batch_iterator(query);
                size_t found = 0;
                do {
                    iterator.next(k);
                    for (auto& neighbor : iterator.results()) {
                        if (filter->is_member(neighbor.id())) {
                            result.set(neighbor, i, found);
                            found++;
                            if (found == k) {
                                break;
                            }
                        }
                    }
                } while (found < k && !iterator.done());

                // Pad results if not enough neighbors found
                if (found < k) {
                    for (size_t j = found; j < k; ++j) {
                        result.set(Neighbor{Unspecify<size_t>(), Unspecify<float>()}, i, j);
                    }
                }
            }
        };

        auto threadpool = default_threadpool();

        svs::threads::parallel_for(
            threadpool, svs::threads::StaticPartition{queries.size()}, search_closure
        );

        impl_->set_search_parameters(old_sp);
    }

    void range_search(
        svs::data::ConstSimpleDataView<float> queries,
        float radius,
        const ResultsAllocator& results,
        const VamanaIndex::SearchParams* params = nullptr,
        IDFilter* filter = nullptr
    ) const {
        if (!impl_) {
            throw StatusException{ErrorCode::NOT_INITIALIZED, "Index not initialized"};
        }
        if (radius <= 0) {
            throw StatusException{
                ErrorCode::INVALID_ARGUMENT, "radius must be greater than 0"};
        }

        const size_t n = queries.size();
        if (n == 0) {
            return;
        }

        auto sp = make_search_parameters(params);
        auto old_sp = impl_->get_search_parameters();
        impl_->set_search_parameters(sp);

        // Using ResultHandler makes no sense due to it's complexity, overhead and
        // missed features; e.g. add_result() does not indicate whether result added
        // or not - we have to manually manage threshold comparison and id
        // selection.

        // Prepare output buffers
        std::vector<std::vector<svs::Neighbor<size_t>>> all_results(n);
        // Reserve space for allocation to avoid multiple reallocations
        // Use search_buffer_capacity as a heuristic
        const auto result_capacity = sp.buffer_config_.get_total_capacity();
        for (auto& res : all_results) {
            res.reserve(result_capacity);
        }

        svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric_type_));

        std::function<bool(float, float)> compare = distance_dispatcher([](auto&& dist) {
            return std::function<bool(float, float)>{svs::distance::comparator(dist)};
        });

        std::function<bool(size_t)> select = [](size_t) { return true; };
        if (filter != nullptr) {
            select = [&](size_t id) { return filter->is_member(id); };
        }

        // Set iterator batch size to search window size
        auto batch_size = sp.buffer_config_.get_search_window_size();

        auto range_search_closure = [&](const auto& range, uint64_t SVS_UNUSED(tid)) {
            for (auto i : range) {
                // For every query
                auto query = queries.get_datum(i);

                auto iterator = impl_->batch_iterator(query);
                bool in_range = true;

                do {
                    iterator.next(batch_size);
                    for (auto& neighbor : iterator.results()) {
                        // SVS comparator functor returns true if the first distance
                        // is 'closer' than the second one
                        in_range = compare(neighbor.distance(), radius);
                        if (in_range) {
                            // Selective search with IDSelector
                            if (select(neighbor.id())) {
                                all_results[i].push_back(neighbor);
                            }
                        } else {
                            // Since iterator.results() are ordered by distance, we
                            // can stop processing
                            break;
                        }
                    }
                } while (in_range && !iterator.done());
            }
        };

        auto threadpool = default_threadpool();

        svs::threads::parallel_for(
            threadpool, svs::threads::StaticPartition{n}, range_search_closure
        );

        // Allocate output
        std::vector<size_t> result_counts(n);
        std::transform(
            all_results.begin(),
            all_results.end(),
            result_counts.begin(),
            [](const auto& res) { return res.size(); }
        );
        auto results_storage = results(result_counts);

        // Fill in results
        for (size_t q = 0, ofs = 0; q < n; ++q) {
            for (const auto& [id, distance] : all_results[q]) {
                results_storage.labels[ofs] = id;
                results_storage.distances[ofs] = distance;
                ofs++;
            }
        }

        impl_->set_search_parameters(old_sp);
    }

    size_t remove(std::span<const size_t> labels) {
        if (!impl_) {
            throw StatusException{ErrorCode::NOT_INITIALIZED, "Index not initialized"};
        }

        // SVS deletion is a soft deletion, meaning the corresponding vectors are
        // marked as deleted but still present in both the dataset and the graph,
        // and will be navigated through during search.
        // Actual cleanup happens once a large enough number of soft deleted vectors
        // are collected.
        impl_->delete_points(labels);
        ntotal_soft_deleted += labels.size();

        auto ntotal = impl_->size();
        const float cleanup_threshold = .5f;
        if (ntotal == 0 || (float)ntotal_soft_deleted / ntotal > cleanup_threshold) {
            impl_->consolidate();
            impl_->compact();
            ntotal_soft_deleted = 0;
        }
        return labels.size();
    }

    size_t remove_selected(const IDFilter& selector) {
        if (!impl_) {
            throw StatusException{ErrorCode::NOT_INITIALIZED, "Index not initialized"};
        }

        auto ids = impl_->all_ids();
        std::vector<size_t> ids_to_delete;
        std::copy_if(
            ids.begin(),
            ids.end(),
            std::back_inserter(ids_to_delete),
            [&](size_t id) { return selector(id); }
        );

        return remove(ids_to_delete);
    }

    void reset() {
        impl_.reset();
        ntotal_soft_deleted = 0;
    }

    void save(std::ostream& out) const {
        if (!impl_) {
            throw StatusException{
                ErrorCode::NOT_INITIALIZED, "Cannot serialize: SVS index not initialized."};
        }

        impl_->save(out);
    }

  protected:
    // Utility functions
    svs::index::vamana::VamanaBuildParameters vamana_build_parameters() const {
        svs::index::vamana::VamanaBuildParameters result;
        set_if_specified(result.alpha, build_params_.alpha);
        set_if_specified(result.graph_max_degree, build_params_.graph_max_degree);
        set_if_specified(result.window_size, build_params_.construction_window_size);
        set_if_specified(
            result.max_candidate_pool_size, build_params_.max_candidate_pool_size
        );
        set_if_specified(result.prune_to, build_params_.prune_to);
        if (is_specified(build_params_.use_full_search_history)) {
            result.use_full_search_history =
                build_params_.use_full_search_history.is_enabled();
        }
        return result;
    }

    svs::index::vamana::VamanaSearchParameters
    make_search_parameters(const VamanaIndex::SearchParams* params) const {
        if (!impl_) {
            throw StatusException{ErrorCode::NOT_INITIALIZED, "Index not initialized"};
        }

        // Copy default search parameters
        auto search_params = default_search_params_;
        // Update with user-specified parameters
        if (params) {
            set_if_specified(search_params.search_window_size, params->search_window_size);
            set_if_specified(
                search_params.search_buffer_capacity, params->search_buffer_capacity
            );
            set_if_specified(search_params.prefetch_lookahead, params->prefetch_lookahead);
            set_if_specified(search_params.prefetch_step, params->prefetch_step);
        }

        // Get current search parameters from the index
        auto result = impl_->get_search_parameters();
        // Update with specified parameters
        if (is_specified(search_params.search_window_size)) {
            if (is_specified(search_params.search_buffer_capacity)) {
                result.buffer_config(
                    {search_params.search_window_size, search_params.search_buffer_capacity}
                );
            } else {
                result.buffer_config(search_params.search_window_size);
            }
        } else if (is_specified(search_params.search_buffer_capacity)) {
            result.buffer_config(search_params.search_buffer_capacity);
        }

        set_if_specified(result.prefetch_lookahead_, search_params.prefetch_lookahead);
        set_if_specified(result.prefetch_step_, search_params.prefetch_step);

        return result;
    }

    template <typename Tag, typename... StorageArgs>
    static svs::DynamicVamana* build_impl(
        Tag&& tag,
        MetricType metric,
        const index::vamana::VamanaBuildParameters& parameters,
        const svs::data::ConstSimpleDataView<float>& data,
        std::span<const size_t> labels,
        svs::lib::PowerOfTwo blocksize_bytes,
        StorageArgs&&... storage_args
    ) {
        auto threadpool = default_threadpool();
        using storage_alloc_t = typename Tag::allocator_type;
        auto allocator = storage::make_allocator<storage_alloc_t>(blocksize_bytes);

        auto storage = make_storage(
            std::forward<Tag>(tag),
            data,
            threadpool,
            allocator,
            std::forward<StorageArgs>(storage_args)...
        );

        svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric));
        return distance_dispatcher([&](auto&& distance) {
            return new svs::DynamicVamana(svs::DynamicVamana::build<float>(
                parameters,
                std::move(storage),
                std::move(labels),
                std::forward<decltype(distance)>(distance),
                std::move(threadpool)
            ));
        });
    }

    virtual void init_impl(
        data::ConstSimpleDataView<float> data,
        std::span<const size_t> labels,
        lib::PowerOfTwo blocksize_bytes
    ) {
        // Deferred compression fast-path: if the very first add already meets the
        // threshold there is no benefit to building an uncompressed graph and
        // immediately retraining it. Skip the deferred path entirely and build the
        // target compressed backend directly.
        if (deferred_compression_enabled() &&
            data.size() >= dynamic_index_params_.deferred_compression_threshold) {
            current_storage_kind_ = storage_kind_;
        }
        // When deferred compression is enabled (and the fast-path above didn't fire),
        // build the initial backend with the uncompressed `initial_storage_kind`.
        // Otherwise (eager path) build directly with the target kind. The swap
        // closure (if any) is installed by `setup_deferred_compression_swap` after
        // the build.
        const auto build_kind = current_storage_kind_;
        impl_.reset(storage::dispatch_storage_kind<allocator_type>(
            build_kind,
            [this](
                auto&& tag,
                data::ConstSimpleDataView<float> data,
                std::span<const size_t> labels,
                lib::PowerOfTwo blocksize_bytes
            ) {
                using Tag = std::decay_t<decltype(tag)>;
                return build_impl(
                    Tag{},
                    this->metric_type_,
                    this->vamana_build_parameters(),
                    data,
                    labels,
                    blocksize_bytes
                );
            },
            data,
            labels,
            blocksize_bytes
        ));
        if (deferred_compression_enabled() &&
            build_kind != storage_kind_) {
            setup_deferred_compression_swap(build_kind, blocksize_bytes);
        }
    }

    /// @brief Install the closure that performs the eventual deferred-compression
    /// swap. Called after `init_impl` builds the initial uncompressed backend.
    ///
    /// The base implementation handles SQ / LVQ / LeanVec (with PCA-trained matrices
    /// and default leanvec_dims). Subclasses (e.g. `DynamicVamanaIndexLeanVecImpl`)
    /// override this to inject pre-trained matrices or a user-specified
    /// ``leanvec_dims``.
    virtual void setup_deferred_compression_swap(
        StorageKind initial_kind, lib::PowerOfTwo blocksize_bytes
    ) {
        storage::dispatch_storage_kind<allocator_type>(
            initial_kind,
            [&](auto&& tag) {
                using Tag = std::decay_t<decltype(tag)>;
                this->install_swap_closure<Tag>(blocksize_bytes);
            }
        );
    }

    // Constructor used during loading
    DynamicVamanaIndexImpl(
        std::unique_ptr<svs::DynamicVamana>&& impl,
        MetricType metric,
        StorageKind storage_kind
    )
        : impl_{std::move(impl)} {
        dim_ = impl_->dimensions();
        const auto& buffer_config = impl_->get_search_parameters().buffer_config_;
        default_search_params_ = {
            buffer_config.get_search_window_size(), buffer_config.get_total_capacity()};
        metric_type_ = metric;
        storage_kind_ = storage_kind;
        current_storage_kind_ = storage_kind;
        build_params_ = VamanaIndex::BuildParams{
            impl_->get_graph_max_degree(),
            impl_->get_prune_to(),
            impl_->get_alpha(),
            impl_->get_construction_window_size(),
            impl_->get_max_candidates(),
            impl_->get_full_search_history()};
    }

    template <StorageKind Kind, typename Alloc>
    static svs::DynamicVamana* load_impl_t(
        storage::StorageType<Kind, Alloc>&& SVS_UNUSED(tag),
        std::istream& stream,
        MetricType metric
    ) {
        if constexpr (!storage::is_supported_storage_kind_v<Kind>) {
            throw StatusException(
                ErrorCode::NOT_IMPLEMENTED, "Requested storage kind is not supported"
            );
        } else {
            using storage_type = storage::StorageType_t<Kind, Alloc>;
            auto threadpool = default_threadpool();

            svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric));

            return distance_dispatcher([&](auto&& distance) {
                return new svs::DynamicVamana(
                    svs::DynamicVamana::assemble<float, storage_type>(
                        stream,
                        std::forward<decltype(distance)>(distance),
                        std::move(threadpool)
                    )
                );
            });
        }
    }

  public:
    static DynamicVamanaIndexImpl*
    load(std::istream& stream, MetricType metric, StorageKind storage_kind) {
        return storage::dispatch_storage_kind<allocator_type>(
            storage_kind,
            [&](auto&& tag, std::istream& stream, MetricType metric) {
                using Tag = std::decay_t<decltype(tag)>;
                std::unique_ptr<svs::DynamicVamana> impl{
                    load_impl_t(std::forward<Tag>(tag), stream, metric)};

                return new DynamicVamanaIndexImpl(std::move(impl), metric, storage_kind);
            },
            stream,
            metric
        );
    }

    // Data members
  protected:
    size_t dim_;
    MetricType metric_type_;
    StorageKind storage_kind_;
    VamanaIndex::BuildParams build_params_;
    VamanaIndex::SearchParams default_search_params_;
    VamanaIndex::DynamicIndexParams dynamic_index_params_;
    std::unique_ptr<svs::DynamicVamana> impl_;
    size_t ntotal_soft_deleted{0};

    /// Storage kind currently backing `impl_`. Differs from `storage_kind_` only when
    /// deferred compression is enabled and the threshold has not yet been crossed.
    StorageKind current_storage_kind_{StorageKind::FP32};

    /// Closure that, when invoked, trains the target compressed dataset from the
    /// current uncompressed `impl_` and reseats `impl_` with a new
    /// `MutableVamanaIndex<...>` whose `Data` template is the trained type. Reuses the
    /// existing graph + ID translation. Set during `init_impl` only when delayed
    /// compression is enabled. Reset to empty after a successful swap.
    std::function<void()> swap_to_target_fn_;

    /// @brief Choose the storage kind used while accumulating below the threshold.
    StorageKind effective_initial_storage_kind() const {
        if (dynamic_index_params_.deferred_compression_threshold == 0) {
            // Eager path: build directly with the requested storage kind.
            return storage_kind_;
        }
        // If the user requested an untrained target there's nothing to delay.
        if (storage_kind_ == StorageKind::FP32 ||
            storage_kind_ == StorageKind::FP16) {
            return storage_kind_;
        }
        return dynamic_index_params_.initial_storage_kind;
    }

    /// @brief If deferred compression is configured and the threshold is reached, run
    /// the swap. Called after every successful `add()`.
    void maybe_swap_to_target_storage() {
        if (!swap_to_target_fn_ || !impl_) {
            return;
        }
        if (impl_->size() < dynamic_index_params_.deferred_compression_threshold) {
            return;
        }
        try {
            swap_to_target_fn_();
            // Successful swap: clear the closure so we don't try again.
            swap_to_target_fn_ = nullptr;
            current_storage_kind_ = storage_kind_;
        } catch (const std::exception&) {
            // Leave the existing uncompressed index intact; retry on next add().
            // (See plan section "Rollback on swap failure".)
        }
    }

    /// @brief Set up the swap closure. Captures the current source storage tag (so we
    /// know the concrete `MutableVamanaIndex<...>` type to downcast to inside the
    /// type-erased orchestrator) and the target storage kind.
    ///
    /// Instantiated for every storage tag from `init_impl`'s dispatch but only
    /// effective for FP32/FP16 sources — the constructor validates
    /// `initial_storage_kind` is one of those.
    template <typename SourceTag>
    void install_swap_closure(lib::PowerOfTwo blocksize_bytes) {
        install_swap_closure_with_trainer<SourceTag>(
            blocksize_bytes, DefaultTrainer{}
        );
    }

    /// @brief Install a swap closure that uses a caller-supplied trainer callable
    /// instead of the default one.
    ///
    /// The trainer is invoked as ``trainer(target_tag, source_data, threadpool,
    /// allocator)`` and must return the trained compressed dataset of type
    /// ``typename decltype(target_tag)::type``. Used by `DynamicVamanaIndexLeanVecImpl`
    /// to inject pre-trained LeanVec matrices and a user-specified ``leanvec_dims``.
    template <typename SourceTag, typename Trainer>
    void install_swap_closure_with_trainer(
        lib::PowerOfTwo blocksize_bytes, Trainer trainer
    ) {
        if constexpr (is_uncompressed_source_tag_v<SourceTag>) {
            const auto target_kind = storage_kind_;
            swap_to_target_fn_ =
                [this, target_kind, blocksize_bytes, trainer = std::move(trainer)]() {
                    this->do_swap_to_target_storage<SourceTag>(
                        target_kind, blocksize_bytes, trainer
                    );
                };
        } else {
            // Compressed source kinds are rejected at construction time. Avoid
            // instantiating the swap path for them.
            (void)blocksize_bytes;
            (void)trainer;
        }
    }

    /// @brief Trait: is the source storage tag one of the uncompressed (trainable
    /// from) backends FP32 / FP16?
    template <typename Tag> struct is_uncompressed_source_tag : std::false_type {};
    template <typename Alloc>
    struct is_uncompressed_source_tag<storage::StorageType<StorageKind::FP32, Alloc>>
        : std::true_type {};
    template <typename Alloc>
    struct is_uncompressed_source_tag<storage::StorageType<StorageKind::FP16, Alloc>>
        : std::true_type {};
    template <typename Tag>
    static constexpr bool is_uncompressed_source_tag_v =
        is_uncompressed_source_tag<Tag>::value;

    /// @brief Trait: is the target storage tag a *trainable* compressed type that the
    /// deferred-compression swap can produce from an uncompressed source?
    ///
    /// Used to avoid instantiating the swap body for FP32/FP16 target tags (which
    /// would require returning a default-constructed dataset, which not all dataset
    /// types support).
    template <typename Tag> struct is_trainable_target_tag : std::false_type {};
    template <typename Alloc>
    struct is_trainable_target_tag<storage::StorageType<StorageKind::SQI8, Alloc>>
        : std::true_type {};
#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
    template <typename Alloc>
    struct is_trainable_target_tag<storage::StorageType<StorageKind::LVQ4x0, Alloc>>
        : std::true_type {};
    template <typename Alloc>
    struct is_trainable_target_tag<storage::StorageType<StorageKind::LVQ8x0, Alloc>>
        : std::true_type {};
    template <typename Alloc>
    struct is_trainable_target_tag<storage::StorageType<StorageKind::LVQ4x4, Alloc>>
        : std::true_type {};
    template <typename Alloc>
    struct is_trainable_target_tag<storage::StorageType<StorageKind::LVQ4x8, Alloc>>
        : std::true_type {};
    template <typename Alloc>
    struct is_trainable_target_tag<storage::StorageType<StorageKind::LeanVec4x4, Alloc>>
        : std::true_type {};
    template <typename Alloc>
    struct is_trainable_target_tag<storage::StorageType<StorageKind::LeanVec4x8, Alloc>>
        : std::true_type {};
    template <typename Alloc>
    struct is_trainable_target_tag<storage::StorageType<StorageKind::LeanVec8x8, Alloc>>
        : std::true_type {};
#endif
    template <typename Tag>
    static constexpr bool is_trainable_target_tag_v = is_trainable_target_tag<Tag>::value;

    /// @brief Perform the actual swap: train the compressed dataset from the
    /// accumulated source dataset, then transplant onto a fresh `MutableVamanaIndex`
    /// with the trained `Data` type, reusing graph + translator + status + entry-point.
    template <typename SourceTag, typename Trainer>
    void do_swap_to_target_storage(
        StorageKind target_kind,
        lib::PowerOfTwo blocksize_bytes,
        const Trainer& trainer
    ) {
        using SourceData = typename SourceTag::type;
        using Graph = svs::graphs::SimpleBlockedGraph<uint32_t>;

        // The DynamicVamanaIndexImpl always builds with `lib::Types<float>` as the
        // query type set (see build_impl below).
        using QueryTypes = svs::lib::Types<float>;

        svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric_type_));
        distance_dispatcher([&](auto distance_function) {
            using Dist = std::decay_t<decltype(distance_function)>;
            using SourceIndex =
                svs::index::vamana::MutableVamanaIndex<Graph, SourceData, Dist>;

            auto* concrete =
                impl_->template get_typed_impl<QueryTypes, SourceIndex>();
            if (concrete == nullptr) {
                throw StatusException{
                    ErrorCode::RUNTIME_ERROR,
                    "Deferred compression swap: unable to recover concrete index type"};
            }

            // Train the new compressed dataset from the source. Uses a freshly owned
            // threadpool because `concrete`'s pool is about to be released.
            auto train_pool = default_threadpool();

            // Dispatch to the target storage tag so we have the static target type.
            // Then build the compressed dataset and transplant onto a new
            // MutableVamanaIndex via the friendly transplant ctor (which moves out
            // of `*concrete` internally).
            storage::dispatch_storage_kind<allocator_type>(
                target_kind,
                [&](auto&& target_tag) {
                    using TargetTag = std::decay_t<decltype(target_tag)>;
                    if constexpr (!is_trainable_target_tag_v<TargetTag> ||
                                  !Trainer::template supports<TargetTag>) {
                        // Either the target storage isn't a trainable kind (FP32 /
                        // FP16) or the active trainer doesn't support this target
                        // (e.g. the LeanVec subclass's trainer doesn't handle SQ /
                        // LVQ targets). Both are configuration errors caught at
                        // construction time, so this branch is unreachable in
                        // practice; we simply avoid instantiating the body.
                        throw StatusException{
                            ErrorCode::INVALID_ARGUMENT,
                            "Deferred compression: trainer does not support the "
                            "configured target storage kind"};
                    } else {
                        using TargetData = typename TargetTag::type;
                        using TargetAlloc = typename TargetTag::allocator_type;

                        auto allocator =
                            storage::make_allocator<TargetAlloc>(blocksize_bytes);

                        // Train the compressed dataset directly from the source
                        // dataset using the caller-supplied trainer. The default
                        // trainer dispatches to each backend's native factory; the
                        // LeanVec subclass uses a trainer that injects pre-trained
                        // matrices and ``leanvec_dims``.
                        TargetData new_data = trainer(
                            TargetTag{},
                            concrete->view_data(),
                            train_pool,
                            allocator
                        );

                        // Construct the new MutableVamanaIndex via the transplant
                        // constructor. The ctor moves out of `*concrete` internally
                        // (graph / status / entry-point / translator / distance /
                        // build & search params), so this single call replaces the
                        // explicit release_*() shuffle.
                        using TargetIndex = svs::index::vamana::
                            MutableVamanaIndex<Graph, TargetData, Dist>;
                        auto new_index = TargetIndex{
                            typename TargetIndex::TransplantTag{},
                            std::move(*concrete),
                            std::move(new_data),
                            default_threadpool()
                        };

                        // Reseat `impl_` with a freshly type-erased DynamicVamana
                        // around the new compressed-backend index. Destroying the
                        // old `impl_` also destroys the (now moved-from) source
                        // MutableVamanaIndex.
                        impl_ = std::make_unique<svs::DynamicVamana>(
                            svs::DynamicVamana::AssembleTag{},
                            QueryTypes{},
                            std::move(new_index)
                        );
                    } // end of `if constexpr (is_trainable_target_tag_v<TargetTag>)`
                }
            );
        });
    }

    /// @brief Default trainer used by the base class. Dispatches to each backend's
    /// native training factory (SQ::compress / LVQ::compress / LeanDataset::reduce
    /// with auto-PCA matrices and default leanvec_dims).
    struct DefaultTrainer {
        // Supports every trainable target tag.
        template <typename TargetTag>
        static constexpr bool supports = is_trainable_target_tag_v<TargetTag>;

        template <typename TargetTag, typename Source, typename Pool, typename Alloc>
        auto operator()(
            TargetTag, const Source& source, Pool& pool, const Alloc& allocator
        ) const {
            using TargetData = typename TargetTag::type;
            if constexpr (svs::quantization::scalar::IsSQData<TargetData>) {
                return TargetData::compress(source, pool, allocator);
            }
#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
            else if constexpr (svs::quantization::lvq::IsLVQDataset<TargetData>) {
                return TargetData::compress(source, pool, 0, allocator);
            } else if constexpr (svs::leanvec::IsLeanDataset<TargetData>) {
                const size_t leanvec_d = (source.dimensions() + 1) / 2;
                return TargetData::reduce(
                    source,
                    std::nullopt,
                    pool,
                    0,
                    svs::lib::MaybeStatic{leanvec_d},
                    allocator
                );
            }
#endif
            else {
                static_assert(
                    !sizeof(TargetData*),
                    "DefaultTrainer instantiated for an unsupported target type"
                );
            }
        }
    };
};

} // namespace runtime
} // namespace svs
