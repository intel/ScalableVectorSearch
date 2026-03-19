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
#include "training_impl.h"

#include <svs/core/data.h>
#include <svs/core/distance.h>
#include <svs/core/graph.h>
#include <svs/core/query_result.h>
#include <svs/extensions/vamana/scalar.h>
#include <svs/lib/file.h>
#include <svs/lib/float16.h>
#include <svs/lib/memory.h>
#include <svs/orchestrators/vamana.h>
#include <svs/quantization/scalar/scalar.h>

#include <algorithm>
#include <memory>
#include <variant>
#include <vector>

namespace svs {
namespace runtime {

// Vamana index implementation
class VamanaIndexImpl {
    using allocator_type = svs::lib::Allocator<float>;

  public:
    VamanaIndexImpl(
        const svs::data::ConstSimpleDataView<float>& data,
        MetricType metric,
        StorageKind storage_kind,
        const VamanaIndex::BuildParams& build_params,
        const VamanaIndex::SearchParams& default_search_params
    )
        : VamanaIndexImpl(nullptr, metric, storage_kind) {
        VamanaIndexImpl::init_impl(data, build_params, default_search_params);
    }

    size_t size() const { return get_impl()->size(); }

    size_t dimensions() const { return get_impl()->dimensions(); }

    MetricType metric_type() const { return metric_type_; }

    StorageKind get_storage_kind() const { return storage_kind_; }

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
            get_impl()->search(result, queries, sp);
            return;
        }

        // Selective search with IDSelector
        auto old_sp = get_impl()->get_search_parameters();
        get_impl()->set_search_parameters(sp);

        auto search_closure = [&](const auto& range, uint64_t SVS_UNUSED(tid)) {
            for (auto i : range) {
                // For every query
                auto query = queries.get_datum(i);
                auto iterator = get_impl()->batch_iterator(query);
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
                    std::fill(dists.begin() + found, dists.end(), Unspecify<float>());
                    auto& inds = result.indices();
                    std::fill(inds.begin() + found, inds.end(), Unspecify<size_t>());
                }
            }
        };

        auto threadpool = default_threadpool();

        svs::threads::parallel_for(
            threadpool, svs::threads::StaticPartition{queries.size()}, search_closure
        );

        get_impl()->set_search_parameters(old_sp);
    }

    void range_search(
        svs::data::ConstSimpleDataView<float> queries,
        float radius,
        const ResultsAllocator& results,
        const VamanaIndex::SearchParams* params = nullptr,
        IDFilter* filter = nullptr
    ) const {
        if (radius <= 0) {
            throw StatusException{
                ErrorCode::INVALID_ARGUMENT, "radius must be greater than 0"};
        }

        const size_t n = queries.size();
        if (n == 0) {
            return;
        }

        auto sp = make_search_parameters(params);
        auto old_sp = get_impl()->get_search_parameters();
        get_impl()->set_search_parameters(sp);

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

                auto iterator = get_impl()->batch_iterator(query);
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

        get_impl()->set_search_parameters(old_sp);
    }

    void save(std::ostream& out) const {
        lib::UniqueTempDirectory tempdir{"svs_vamana_save"};
        const auto config_dir = tempdir.get() / "config";
        const auto graph_dir = tempdir.get() / "graph";
        const auto data_dir = tempdir.get() / "data";
        std::filesystem::create_directories(config_dir);
        std::filesystem::create_directories(graph_dir);
        std::filesystem::create_directories(data_dir);
        get_impl()->save(config_dir, graph_dir, data_dir);
        lib::DirectoryArchiver::pack(tempdir, out);
    }

  protected:
    // Utility functions
    svs::Vamana* get_impl() const {
        if (!impl_) {
            throw StatusException{ErrorCode::NOT_INITIALIZED, "Index not initialized"};
        }
        return impl_.get();
    }

    static svs::index::vamana::VamanaBuildParameters
    make_build_parameters(const VamanaIndex::BuildParams& build_params) {
        svs::index::vamana::VamanaBuildParameters result;
        set_if_specified(result.alpha, build_params.alpha);
        set_if_specified(result.graph_max_degree, build_params.graph_max_degree);
        set_if_specified(result.window_size, build_params.construction_window_size);
        set_if_specified(
            result.max_candidate_pool_size, build_params.max_candidate_pool_size
        );
        set_if_specified(result.prune_to, build_params.prune_to);
        if (is_specified(build_params.use_full_search_history)) {
            result.use_full_search_history =
                build_params.use_full_search_history.is_enabled();
        }
        return result;
    }

    svs::index::vamana::VamanaSearchParameters
    make_search_parameters(const VamanaIndex::SearchParams* params) const {
        // Get current search parameters from the index
        auto result = get_impl()->get_search_parameters();
        if (!params) {
            return result;
        }
        // else: update with user-specified parameters
        if (is_specified(params->search_window_size)) {
            if (is_specified(params->search_buffer_capacity)) {
                result.buffer_config(
                    {params->search_window_size, params->search_buffer_capacity}
                );
            } else {
                result.buffer_config(params->search_window_size);
            }
        } else if (is_specified(params->search_buffer_capacity)) {
            result.buffer_config(params->search_buffer_capacity);
        }

        set_if_specified(result.prefetch_lookahead_, params->prefetch_lookahead);
        set_if_specified(result.prefetch_step_, params->prefetch_step);

        return result;
    }

    template <typename Tag, typename... StorageArgs>
    static svs::Vamana* build_impl(
        Tag&& tag,
        MetricType metric,
        const index::vamana::VamanaBuildParameters& parameters,
        const svs::data::ConstSimpleDataView<float>& data,
        StorageArgs&&... storage_args
    ) {
        auto threadpool = default_threadpool();
        using storage_alloc_t = typename Tag::allocator_type;
        auto allocator = storage::make_allocator<storage_alloc_t>(svs::lib::PowerOfTwo{0});

        auto storage = make_storage(
            std::forward<Tag>(tag),
            data,
            threadpool,
            allocator,
            std::forward<StorageArgs>(storage_args)...
        );

        svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric));
        return distance_dispatcher([&](auto&& distance) {
            return new svs::Vamana(svs::Vamana::build<float>(
                parameters,
                std::move(storage),
                std::forward<decltype(distance)>(distance),
                std::move(threadpool)
            ));
        });
    }

    void init_impl(
        const data::ConstSimpleDataView<float>& data,
        const VamanaIndex::BuildParams& build_params,
        const VamanaIndex::SearchParams& default_search_params
    ) {
        impl_.reset(storage::dispatch_storage_kind<allocator_type>(
            get_storage_kind(),
            [&](auto&& tag, const data::ConstSimpleDataView<float>& data) {
                using Tag = std::decay_t<decltype(tag)>;
                return build_impl(
                    std::forward<Tag>(tag),
                    this->metric_type_,
                    make_build_parameters(build_params),
                    data
                );
            },
            data
        ));
        get_impl()->set_search_parameters(make_search_parameters(&default_search_params));
    }

    // Constructor used during loading
    VamanaIndexImpl(
        std::unique_ptr<svs::Vamana>&& impl, MetricType metric, StorageKind storage_kind
    )
        : metric_type_{metric}
        , storage_kind_{storage_kind}
        , impl_{std::move(impl)} {
        if (!storage::is_supported_storage_kind(storage_kind)) {
            throw StatusException{
                ErrorCode::INVALID_ARGUMENT,
                "The specified storage kind is not compatible with the "
                "DynamicVamanaIndex"};
        }
    }

    template <typename Tag>
    static svs::Vamana* load_impl_t(Tag&& tag, std::istream& stream, MetricType metric) {
        namespace fs = std::filesystem;
        lib::UniqueTempDirectory tempdir{"svs_vamana_load"};
        lib::DirectoryArchiver::unpack(stream, tempdir);

        const auto config_path = tempdir.get() / "config";
        if (!fs::is_directory(config_path)) {
            throw StatusException{
                ErrorCode::RUNTIME_ERROR,
                "Invalid Vamana index archive: missing config directory!"};
        }

        const auto graph_path = tempdir.get() / "graph";
        if (!fs::is_directory(graph_path)) {
            throw StatusException{
                ErrorCode::RUNTIME_ERROR,
                "Invalid Vamana index archive: missing graph directory!"};
        }

        const auto data_path = tempdir.get() / "data";
        if (!fs::is_directory(data_path)) {
            throw StatusException{
                ErrorCode::RUNTIME_ERROR,
                "Invalid Vamana index archive: missing data directory!"};
        }

        auto storage = storage::load_storage(std::forward<Tag>(tag), data_path);
        auto threadpool = default_threadpool();

        svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric));

        return distance_dispatcher([&](auto&& distance) {
            return new svs::Vamana(svs::Vamana::assemble<float>(
                config_path,
                svs::GraphLoader{graph_path},
                std::move(storage),
                std::forward<decltype(distance)>(distance),
                std::move(threadpool)
            ));
        });
    }

  public:
    static VamanaIndexImpl*
    load(std::istream& stream, MetricType metric, StorageKind storage_kind) {
        return storage::dispatch_storage_kind<allocator_type>(
            storage_kind,
            [&](auto&& tag, std::istream& stream, MetricType metric) {
                using Tag = std::decay_t<decltype(tag)>;
                std::unique_ptr<svs::Vamana> impl{
                    load_impl_t(std::forward<Tag>(tag), stream, metric)};

                return new VamanaIndexImpl(std::move(impl), metric, storage_kind);
            },
            stream,
            metric
        );
    }

    // Data members
  protected:
    MetricType metric_type_;
    StorageKind storage_kind_;
    std::unique_ptr<svs::Vamana> impl_;
};

struct VamanaIndexLeanVecImpl : public VamanaIndexImpl {
    using LeanVecMatricesType = LeanVecTrainingDataImpl::LeanVecMatricesType;
    using allocator_type = svs::lib::Allocator<std::byte>;

    VamanaIndexLeanVecImpl(
        std::unique_ptr<svs::Vamana>&& impl, MetricType metric, StorageKind storage_kind
    )
        : VamanaIndexImpl{std::move(impl), metric, storage_kind}
        , leanvec_dims_{0}
        , leanvec_matrices_{std::nullopt} {
        check_storage_kind(storage_kind);
    }

    VamanaIndexLeanVecImpl(
        const data::ConstSimpleDataView<float>& data,
        MetricType metric,
        StorageKind storage_kind,
        const LeanVecTrainingDataImpl& training_data,
        const VamanaIndex::BuildParams& params,
        const VamanaIndex::SearchParams& default_search_params
    )
        : VamanaIndexImpl{nullptr, metric, storage_kind}
        , leanvec_dims_{training_data.get_leanvec_dims()}
        , leanvec_matrices_{training_data.get_leanvec_matrices()} {
        check_storage_kind(storage_kind);
        init_impl(data, params, default_search_params, leanvec_dims_, leanvec_matrices_);
    }

    VamanaIndexLeanVecImpl(
        const data::ConstSimpleDataView<float>& data,
        MetricType metric,
        StorageKind storage_kind,
        size_t leanvec_dims,
        const VamanaIndex::BuildParams& params,
        const VamanaIndex::SearchParams& default_search_params
    )
        : VamanaIndexImpl{nullptr, metric, storage_kind}
        , leanvec_dims_{leanvec_dims}
        , leanvec_matrices_{std::nullopt} {
        check_storage_kind(storage_kind);
        init_impl(data, params, default_search_params, leanvec_dims_, leanvec_matrices_);
    }

    template <typename F, typename... Args>
    static auto dispatch_leanvec_storage_kind(StorageKind kind, F&& f, Args&&... args) {
        switch (kind) {
            case StorageKind::LeanVec4x4:
                return f(
                    storage::StorageType<StorageKind::LeanVec4x4, allocator_type>{},
                    std::forward<Args>(args)...
                );
            case StorageKind::LeanVec4x8:
                return f(
                    storage::StorageType<StorageKind::LeanVec4x8, allocator_type>{},
                    std::forward<Args>(args)...
                );
            case StorageKind::LeanVec8x8:
                return f(
                    storage::StorageType<StorageKind::LeanVec8x8, allocator_type>{},
                    std::forward<Args>(args)...
                );
            default:
                throw StatusException{
                    ErrorCode::INVALID_ARGUMENT, "SVS LeanVec storage kind required"};
        }
    }

    void init_impl(
        const data::ConstSimpleDataView<float>& data,
        const VamanaIndex::BuildParams& build_params,
        const VamanaIndex::SearchParams& default_search_params,
        size_t leanvec_dims,
        const std::optional<LeanVecMatricesType>& leanvec_matrices
    ) {
        assert(storage::is_leanvec_storage(this->storage_kind_));
        impl_.reset(dispatch_leanvec_storage_kind(
            this->storage_kind_,
            [&](auto&& tag, const data::ConstSimpleDataView<float>& data) {
                using Tag = std::decay_t<decltype(tag)>;
                return VamanaIndexImpl::build_impl(
                    std::forward<Tag>(tag),
                    this->metric_type_,
                    make_build_parameters(build_params),
                    data,
                    leanvec_dims,
                    leanvec_matrices
                );
            },
            data
        ));
        impl_->set_search_parameters(make_search_parameters(&default_search_params));
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

} // namespace runtime
} // namespace svs
