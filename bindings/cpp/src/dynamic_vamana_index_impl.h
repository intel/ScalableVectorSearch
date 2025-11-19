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

#include "svs_runtime_utils.h"

#include <svs/runtime/dynamic_vamana_index.h>

#include <algorithm>
#include <memory>
#include <variant>
#include <vector>

#include <svs/core/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/extensions/vamana/scalar.h>
#include <svs/lib/float16.h>
#include <svs/orchestrators/dynamic_vamana.h>
#include <svs/quantization/scalar/scalar.h>

namespace svs {
namespace runtime {

// Vamana index implementation
class DynamicVamanaIndexImpl {
  public:
    DynamicVamanaIndexImpl(
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        const VamanaIndex::BuildParams& params,
        const VamanaIndex::SearchParams& default_search_params
    )
        : dim_{dim}
        , metric_type_{metric}
        , storage_kind_{storage_kind}
        , build_params_{params}
        , default_search_params_{default_search_params} {
        if (build_params_.prune_to == 0) {
            build_params_.prune_to = build_params_.graph_max_degree < 4
                                         ? build_params_.graph_max_degree
                                         : build_params_.graph_max_degree - 4;
        }
        if (build_params_.alpha == 0) {
            build_params_.alpha = metric == MetricType::L2 ? 1.2f : 0.95f;
        }
    }

    size_t size() const { return impl_ ? impl_->size() : 0; }

    size_t dimensions() const { return dim_; }

    MetricType metric_type() const { return metric_type_; }

    StorageKind get_storage_kind() const { return storage_kind_; }

    void add(data::ConstSimpleDataView<float> data, std::span<const size_t> labels) {
        if (!impl_) {
            return init_impl(data, labels);
        }

        impl_->add_points(data, labels);
    }

    void search(
        svs::QueryResultView<size_t> result,
        svs::data::ConstSimpleDataView<float> queries,
        const VamanaIndex::SearchParams* params = nullptr,
        IDFilter* filter = nullptr
    ) const {
        if (!impl_) {
            auto& dists = result.distances();
            std::fill(dists.begin(), dists.end(), std::numeric_limits<float>::infinity());
            auto& inds = result.indices();
            std::fill(inds.begin(), inds.end(), static_cast<size_t>(-1));
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
                    auto& dists = result.distances();
                    std::fill(
                        dists.begin() + found,
                        dists.end(),
                        std::numeric_limits<float>::infinity()
                    );
                    auto& inds = result.indices();
                    std::fill(inds.begin() + found, inds.end(), static_cast<size_t>(-1));
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
        return svs::index::vamana::VamanaBuildParameters{
            build_params_.alpha,
            build_params_.graph_max_degree,
            build_params_.construction_window_size,
            build_params_.max_candidate_pool_size,
            build_params_.prune_to,
            build_params_.use_full_search_history};
    }

    svs::index::vamana::VamanaSearchParameters
    make_search_parameters(const VamanaIndex::SearchParams* params) const {
        if (!impl_) {
            throw StatusException{ErrorCode::NOT_INITIALIZED, "Index not initialized"};
        }

        auto sp = impl_->get_search_parameters();

        auto search_window_size = default_search_params_.search_window_size;
        auto search_buffer_capacity = default_search_params_.search_buffer_capacity;
        if (default_search_params_.prefetch_lookahead > 0) {
            sp = sp.prefetch_lookahead(default_search_params_.prefetch_lookahead);
        }
        if (default_search_params_.prefetch_step > 0) {
            sp = sp.prefetch_step(default_search_params_.prefetch_step);
        }

        if (params != nullptr) {
            if (params->search_window_size > 0)
                search_window_size = params->search_window_size;
            if (params->search_buffer_capacity > 0)
                search_buffer_capacity = params->search_buffer_capacity;
            if (params->prefetch_lookahead > 0) {
                sp = sp.prefetch_lookahead(params->prefetch_lookahead);
            }
            if (params->prefetch_step > 0) {
                sp = sp.prefetch_step(params->prefetch_step);
            }
        }

        return impl_->get_search_parameters().buffer_config(
            {search_window_size, search_buffer_capacity}
        );
    }

    template <typename Tag, typename... StorageArgs>
    static svs::DynamicVamana* build_impl(
        Tag&& tag,
        MetricType metric,
        const index::vamana::VamanaBuildParameters& parameters,
        const svs::data::ConstSimpleDataView<float>& data,
        std::span<const size_t> labels,
        StorageArgs&&... storage_args
    ) {
        auto threadpool = default_threadpool();

        auto storage = make_storage(
            std::forward<Tag>(tag),
            data,
            threadpool,
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

    virtual void
    init_impl(data::ConstSimpleDataView<float> data, std::span<const size_t> labels) {
        impl_.reset(storage::dispatch_storage_kind(
            get_storage_kind(),
            [this](
                auto&& tag,
                data::ConstSimpleDataView<float> data,
                std::span<const size_t> labels
            ) {
                using Tag = std::decay_t<decltype(tag)>;
                return build_impl(
                    std::forward<Tag>(tag),
                    this->metric_type_,
                    this->vamana_build_parameters(),
                    data,
                    labels
                );
            },
            data,
            labels
        ));
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
        build_params_ = {
            impl_->get_graph_max_degree(),
            impl_->get_prune_to(),
            impl_->get_alpha(),
            impl_->get_construction_window_size(),
            impl_->get_max_candidates(),
            impl_->get_full_search_history()};
    }

    template <storage::StorageTag Tag>
    static svs::DynamicVamana*
    load_impl_t(Tag&& SVS_UNUSED(tag), std::istream& stream, MetricType metric) {
        auto threadpool = default_threadpool();

        svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric));

        // Create allocator with custom block size for data loading
        auto parameters =
            svs::data::BlockingParameters{.blocksize_bytes = svs::lib::PowerOfTwo(26)};
        auto allocator = svs::data::Blocked<svs::lib::Allocator<std::byte>>{parameters};

        return distance_dispatcher([&](auto&& distance) {
            return new svs::DynamicVamana(
                svs::DynamicVamana::assemble<float, storage::StorageType_t<Tag>>(
                    stream,
                    std::forward<decltype(distance)>(distance),
                    std::move(threadpool),
                    allocator
                )
            );
        });
    }

  public:
    static DynamicVamanaIndexImpl*
    load(std::istream& stream, MetricType metric, StorageKind storage_kind) {
        return storage::dispatch_storage_kind(
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
    std::unique_ptr<svs::DynamicVamana> impl_;
    size_t ntotal_soft_deleted{0};
};

#if SVS_RUNTIME_ENABLE_LVQ_LEANVEC
struct DynamicVamanaIndexLeanVecImpl : public DynamicVamanaIndexImpl {
    DynamicVamanaIndexLeanVecImpl(
        std::unique_ptr<svs::DynamicVamana>&& impl,
        MetricType metric,
        StorageKind storage_kind
    )
        : DynamicVamanaIndexImpl{std::move(impl), metric, storage_kind}
        , leanvec_dims_{0}
        , leanvec_matrices_{std::nullopt} {
        check_storage_kind(storage_kind);
    }

    DynamicVamanaIndexLeanVecImpl(
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        const LeanVecTrainingDataImpl& training_data,
        const VamanaIndex::BuildParams& params,
        const VamanaIndex::SearchParams& default_search_params = {10, 10}
    )
        : DynamicVamanaIndexImpl{dim, metric, storage_kind, params, default_search_params}
        , leanvec_dims_{training_data.get_leanvec_dims()}
        , leanvec_matrices_{training_data.get_leanvec_matrices()} {
        check_storage_kind(storage_kind);
    }

    DynamicVamanaIndexLeanVecImpl(
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        size_t leanvec_dims,
        const VamanaIndex::BuildParams& params,
        const VamanaIndex::SearchParams& default_search_params = {10, 10}
    )
        : DynamicVamanaIndexImpl{dim, metric, storage_kind, params, default_search_params}
        , leanvec_dims_{leanvec_dims}
        , leanvec_matrices_{std::nullopt} {
        check_storage_kind(storage_kind);
    }

    template <typename F, typename... Args>
    static auto dispatch_leanvec_storage_kind(StorageKind kind, F&& f, Args&&... args) {
        switch (kind) {
            case StorageKind::LeanVec4x4:
                return f(storage::LeanVec4x4Tag{}, std::forward<Args>(args)...);
            case StorageKind::LeanVec4x8:
                return f(storage::LeanVec4x8Tag{}, std::forward<Args>(args)...);
            case StorageKind::LeanVec8x8:
                return f(storage::LeanVec8x8Tag{}, std::forward<Args>(args)...);
            default:
                throw StatusException{
                    ErrorCode::INVALID_ARGUMENT, "SVS LeanVec storage kind required"};
        }
    }

    void init_impl(data::ConstSimpleDataView<float> data, std::span<const size_t> labels)
        override {
        assert(storage::is_leanvec_storage(this->storage_kind_));
        impl_.reset(dispatch_leanvec_storage_kind(
            this->storage_kind_,
            [this](
                auto&& tag,
                data::ConstSimpleDataView<float> data,
                std::span<const size_t> labels
            ) {
                using Tag = std::decay_t<decltype(tag)>;
                return DynamicVamanaIndexImpl::build_impl(
                    std::forward<Tag>(tag),
                    this->metric_type_,
                    this->vamana_build_parameters(),
                    data,
                    labels,
                    this->leanvec_dims_,
                    this->leanvec_matrices_
                );
            },
            data,
            labels
        ));
    }

  protected:
    size_t leanvec_dims_;
    std::optional<LeanVecMatricesType> leanvec_matrices_;

    StorageKind check_storage_kind(StorageKind kind) {
        if (!storage::is_leanvec_storage(kind)) {
            throw StatusException(
                ErrorCode::INVALID_ARGUMENT, "SVS LeanVec storage kind required"
            );
        }
        if (!svs::detail::lvq_leanvec_enabled()) {
            throw StatusException(
                ErrorCode::NOT_IMPLEMENTED,
                "LeanVec storage kind requested but not supported by CPU"
            );
        }
        return kind;
    }
};
#endif

} // namespace runtime
} // namespace svs
