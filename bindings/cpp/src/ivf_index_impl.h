/*
 * Copyright 2026 Intel Corporation
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

#include "svs/runtime/ivf_index.h"
#include "svs_runtime_utils.h"

#include <svs/core/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/index/ivf/common.h>
#include <svs/lib/file.h>
#include <svs/orchestrators/dynamic_ivf.h>
#include <svs/orchestrators/ivf.h>

#include <algorithm>
#include <memory>
#include <variant>
#include <vector>

namespace svs {
namespace runtime {

// IVF storage kind support - IVF supports a subset of storage kinds
namespace ivf_storage {

// IVF supports FP32 and FP16 storage kinds
inline bool is_supported_storage_kind(StorageKind kind) {
    switch (kind) {
        case StorageKind::FP32:
        case StorageKind::FP16:
            return true;
        default:
            return false;
    }
}

// IVF data type for static index (uses lib::Allocator)
template <typename T>
using IVFDataType = svs::data::SimpleData<T, svs::Dynamic, svs::lib::Allocator<T>>;

// IVF data type for dynamic index (uses Blocked allocator)
template <typename T>
using IVFBlockedDataType =
    svs::data::SimpleData<T, svs::Dynamic, svs::data::Blocked<svs::lib::Allocator<T>>>;

// Dispatch on storage kind for IVF operations
template <typename F, typename... Args>
auto dispatch_ivf_storage_kind(StorageKind kind, F&& f, Args&&... args) {
    switch (kind) {
        case StorageKind::FP32:
            return f(svs::lib::Type<IVFDataType<float>>{}, std::forward<Args>(args)...);
        case StorageKind::FP16:
            return f(
                svs::lib::Type<IVFDataType<svs::Float16>>{}, std::forward<Args>(args)...
            );
        default:
            throw StatusException{
                ErrorCode::NOT_IMPLEMENTED,
                "Requested storage kind is not supported for IVF index"};
    }
}

// Dispatch on storage kind for Dynamic IVF operations (uses blocked allocator)
template <typename F, typename... Args>
auto dispatch_ivf_blocked_storage_kind(StorageKind kind, F&& f, Args&&... args) {
    switch (kind) {
        case StorageKind::FP32:
            return f(
                svs::lib::Type<IVFBlockedDataType<float>>{}, std::forward<Args>(args)...
            );
        case StorageKind::FP16:
            return f(
                svs::lib::Type<IVFBlockedDataType<svs::Float16>>{},
                std::forward<Args>(args)...
            );
        default:
            throw StatusException{
                ErrorCode::NOT_IMPLEMENTED,
                "Requested storage kind is not supported for Dynamic IVF index"};
    }
}

} // namespace ivf_storage

// Static IVF index implementation
class StaticIVFIndexImpl {
  public:
    StaticIVFIndexImpl(
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        const IVFIndex::BuildParams& params,
        const IVFIndex::SearchParams& default_search_params,
        size_t num_threads,
        size_t intra_query_threads
    )
        : dim_{dim}
        , metric_type_{metric}
        , storage_kind_{storage_kind}
        , build_params_{params}
        , default_search_params_{default_search_params}
        , num_threads_{num_threads == 0 ? static_cast<size_t>(omp_get_max_threads()) : num_threads}
        , intra_query_threads_{intra_query_threads} {
        if (!ivf_storage::is_supported_storage_kind(storage_kind)) {
            throw StatusException{
                ErrorCode::INVALID_ARGUMENT,
                "The specified storage kind is not compatible with StaticIVFIndex"};
        }
    }

    size_t size() const { return impl_ ? impl_->size() : 0; }

    size_t dimensions() const { return dim_; }

    MetricType metric_type() const { return metric_type_; }

    StorageKind get_storage_kind() const { return storage_kind_; }

    void build(data::ConstSimpleDataView<float> data) {
        if (impl_) {
            throw StatusException{ErrorCode::RUNTIME_ERROR, "Index already initialized"};
        }
        init_impl(data);
    }

    void search(
        svs::QueryResultView<size_t> result,
        svs::data::ConstSimpleDataView<float> queries,
        const IVFIndex::SearchParams* params = nullptr
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
        impl_->set_search_parameters(sp);
        impl_->search(result, queries, {});
    }

    void save(std::ostream& out) const {
        if (!impl_) {
            throw StatusException{
                ErrorCode::NOT_INITIALIZED, "Cannot serialize: IVF index not initialized."};
        }
        impl_->save(out);
    }

    static StaticIVFIndexImpl* load(
        std::istream& in,
        MetricType metric,
        StorageKind storage_kind,
        size_t num_threads,
        size_t intra_query_threads
    ) {
        if (num_threads == 0) {
            num_threads = static_cast<size_t>(omp_get_max_threads());
        }

        // Dispatch on storage kind to load with correct data type
        return ivf_storage::dispatch_ivf_storage_kind(
            storage_kind,
            [&]<typename DataType>(svs::lib::Type<DataType>) {
                svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric));
                return distance_dispatcher([&](auto&& distance) {
                    auto impl = std::make_unique<svs::IVF>(
                        svs::IVF::assemble<float, svs::BFloat16, DataType>(
                            in,
                            std::forward<decltype(distance)>(distance),
                            num_threads,
                            intra_query_threads
                        )
                    );
                    return new StaticIVFIndexImpl(
                        std::move(impl),
                        metric,
                        storage_kind,
                        num_threads,
                        intra_query_threads
                    );
                });
            }
        );
    }

  protected:
    // Constructor used during loading
    StaticIVFIndexImpl(
        std::unique_ptr<svs::IVF>&& impl,
        MetricType metric,
        StorageKind storage_kind,
        size_t num_threads,
        size_t intra_query_threads
    )
        : dim_{impl->dimensions()}
        , metric_type_{metric}
        , storage_kind_{storage_kind}
        , num_threads_{num_threads}
        , intra_query_threads_{intra_query_threads}
        , impl_{std::move(impl)} {
        // Extract default search params from loaded index
        auto loaded_params = impl_->get_search_parameters();
        default_search_params_ = {loaded_params.n_probes_, loaded_params.k_reorder_};
    }

    svs::index::ivf::IVFBuildParameters ivf_build_parameters() const {
        svs::index::ivf::IVFBuildParameters result;
        set_if_specified(result.num_centroids_, build_params_.num_centroids);
        set_if_specified(result.minibatch_size_, build_params_.minibatch_size);
        set_if_specified(result.num_iterations_, build_params_.num_iterations);
        if (is_specified(build_params_.is_hierarchical)) {
            result.is_hierarchical_ = build_params_.is_hierarchical.is_enabled();
        }
        set_if_specified(result.training_fraction_, build_params_.training_fraction);
        set_if_specified(
            result.hierarchical_level1_clusters_, build_params_.hierarchical_level1_clusters
        );
        set_if_specified(result.seed_, build_params_.seed);
        return result;
    }

    svs::index::ivf::IVFSearchParameters
    make_search_parameters(const IVFIndex::SearchParams* params) const {
        // Start with default parameters
        svs::index::ivf::IVFSearchParameters result;
        if (is_specified(default_search_params_.n_probes)) {
            result.n_probes_ = default_search_params_.n_probes;
        }
        if (is_specified(default_search_params_.k_reorder)) {
            result.k_reorder_ = default_search_params_.k_reorder;
        }

        // Override with user-specified parameters
        if (params) {
            set_if_specified(result.n_probes_, params->n_probes);
            set_if_specified(result.k_reorder_, params->k_reorder);
        }

        return result;
    }

    void init_impl(data::ConstSimpleDataView<float> data) {
        auto build_params = ivf_build_parameters();

        // Single copy of data - required because IVF assembly deduces internal types from
        // data type, and ConstSimpleDataView<float> has const element type which breaks
        // internal type deduction. This copy is also passed directly to assemble which
        // partitions it into clusters (no additional copy for FP32 storage).
        auto owned_data = svs::data::SimpleData<float>(data.size(), data.dimensions());
        svs::data::copy(data, owned_data);

        svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric_type_));

        impl_.reset(distance_dispatcher([&](auto&& distance) {
            // Build clustering using BFloat16 for efficiency (AMX support)
            // Note: build_clustering takes const ref, doesn't consume data
            auto clustering = svs::IVF::build_clustering<svs::BFloat16>(
                build_params,
                owned_data,
                std::forward<decltype(distance)>(distance),
                num_threads_
            );

            // Dispatch on storage kind to assemble with correct data type
            return ivf_storage::dispatch_ivf_storage_kind(
                storage_kind_,
                [&]<typename DataType>(svs::lib::Type<DataType>) {
                    using TargetElement = typename DataType::element_type;

                    // For FP32: pass owned_data directly (moved into clusters)
                    // For FP16: convert from owned_data
                    if constexpr (std::is_same_v<TargetElement, float>) {
                        return new svs::IVF(svs::IVF::assemble_from_clustering<float>(
                            std::move(clustering),
                            owned_data,
                            std::forward<decltype(distance)>(distance),
                            num_threads_,
                            intra_query_threads_
                        ));
                    } else {
                        // Convert to target type (e.g., FP16)
                        DataType converted_data(owned_data.size(), owned_data.dimensions());
                        svs::data::copy(owned_data, converted_data);
                        return new svs::IVF(svs::IVF::assemble_from_clustering<float>(
                            std::move(clustering),
                            std::move(converted_data),
                            std::forward<decltype(distance)>(distance),
                            num_threads_,
                            intra_query_threads_
                        ));
                    }
                }
            );
        }));
    }

    // Data members
    size_t dim_;
    MetricType metric_type_;
    StorageKind storage_kind_;
    IVFIndex::BuildParams build_params_;
    IVFIndex::SearchParams default_search_params_;
    size_t num_threads_;
    size_t intra_query_threads_;
    std::unique_ptr<svs::IVF> impl_;
};

// Dynamic IVF index implementation
class DynamicIVFIndexImpl {
  public:
    DynamicIVFIndexImpl(
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        const IVFIndex::BuildParams& params,
        const IVFIndex::SearchParams& default_search_params,
        size_t num_threads,
        size_t intra_query_threads
    )
        : dim_{dim}
        , metric_type_{metric}
        , storage_kind_{storage_kind}
        , build_params_{params}
        , default_search_params_{default_search_params}
        , num_threads_{num_threads == 0 ? static_cast<size_t>(omp_get_max_threads()) : num_threads}
        , intra_query_threads_{intra_query_threads} {
        if (!ivf_storage::is_supported_storage_kind(storage_kind)) {
            throw StatusException{
                ErrorCode::INVALID_ARGUMENT,
                "The specified storage kind is not compatible with DynamicIVFIndex"};
        }
    }

    size_t size() const { return impl_ ? impl_->size() : 0; }

    size_t dimensions() const { return dim_; }

    MetricType metric_type() const { return metric_type_; }

    StorageKind get_storage_kind() const { return storage_kind_; }

    void build(data::ConstSimpleDataView<float> data, std::span<const size_t> ids) {
        if (impl_) {
            throw StatusException{ErrorCode::RUNTIME_ERROR, "Index already initialized"};
        }
        init_impl(data, ids);
    }

    void
    add(data::ConstSimpleDataView<float> data,
        std::span<const size_t> ids,
        bool reuse_empty = false) {
        if (!impl_) {
            // First add initializes the index
            init_impl(data, ids);
            return;
        }
        impl_->add_points(data, ids, reuse_empty);
    }

    size_t remove(std::span<const size_t> ids) {
        if (!impl_) {
            throw StatusException{ErrorCode::NOT_INITIALIZED, "Index not initialized"};
        }
        return impl_->delete_points(ids);
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

        return impl_->delete_points(ids_to_delete);
    }

    bool has_id(size_t id) const {
        if (!impl_) {
            return false;
        }
        return impl_->has_id(id);
    }

    void consolidate() {
        if (!impl_) {
            throw StatusException{ErrorCode::NOT_INITIALIZED, "Index not initialized"};
        }
        impl_->consolidate();
    }

    void compact(size_t batchsize = 1'000'000) {
        if (!impl_) {
            throw StatusException{ErrorCode::NOT_INITIALIZED, "Index not initialized"};
        }
        impl_->compact(batchsize);
    }

    void search(
        svs::QueryResultView<size_t> result,
        svs::data::ConstSimpleDataView<float> queries,
        const IVFIndex::SearchParams* params = nullptr
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
        impl_->set_search_parameters(sp);
        impl_->search(result, queries, {});
    }

    void save(std::ostream& out) const {
        if (!impl_) {
            throw StatusException{
                ErrorCode::NOT_INITIALIZED,
                "Cannot serialize: DynamicIVF index not initialized."};
        }
        impl_->save(out);
    }

    static DynamicIVFIndexImpl* load(
        std::istream& in,
        MetricType metric,
        StorageKind storage_kind,
        size_t num_threads,
        size_t intra_query_threads
    ) {
        if (num_threads == 0) {
            num_threads = static_cast<size_t>(omp_get_max_threads());
        }

        // Dispatch on storage kind to load with correct data type
        return ivf_storage::dispatch_ivf_storage_kind(
            storage_kind,
            [&]<typename DataType>(svs::lib::Type<DataType>) {
                svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric));
                return distance_dispatcher([&](auto&& distance) {
                    auto impl = std::make_unique<svs::DynamicIVF>(
                        svs::DynamicIVF::assemble<float, svs::BFloat16, DataType>(
                            in,
                            std::forward<decltype(distance)>(distance),
                            num_threads,
                            intra_query_threads
                        )
                    );
                    return new DynamicIVFIndexImpl(
                        std::move(impl),
                        metric,
                        storage_kind,
                        num_threads,
                        intra_query_threads
                    );
                });
            }
        );
    }

  protected:
    // Constructor used during loading
    DynamicIVFIndexImpl(
        std::unique_ptr<svs::DynamicIVF>&& impl,
        MetricType metric,
        StorageKind storage_kind,
        size_t num_threads,
        size_t intra_query_threads
    )
        : dim_{impl->dimensions()}
        , metric_type_{metric}
        , storage_kind_{storage_kind}
        , num_threads_{num_threads}
        , intra_query_threads_{intra_query_threads}
        , impl_{std::move(impl)} {
        // Extract default search params from loaded index
        auto loaded_params = impl_->get_search_parameters();
        default_search_params_ = {loaded_params.n_probes_, loaded_params.k_reorder_};
    }

    svs::index::ivf::IVFBuildParameters ivf_build_parameters() const {
        svs::index::ivf::IVFBuildParameters result;
        set_if_specified(result.num_centroids_, build_params_.num_centroids);
        set_if_specified(result.minibatch_size_, build_params_.minibatch_size);
        set_if_specified(result.num_iterations_, build_params_.num_iterations);
        if (is_specified(build_params_.is_hierarchical)) {
            result.is_hierarchical_ = build_params_.is_hierarchical.is_enabled();
        }
        set_if_specified(result.training_fraction_, build_params_.training_fraction);
        set_if_specified(
            result.hierarchical_level1_clusters_, build_params_.hierarchical_level1_clusters
        );
        set_if_specified(result.seed_, build_params_.seed);
        return result;
    }

    svs::index::ivf::IVFSearchParameters
    make_search_parameters(const IVFIndex::SearchParams* params) const {
        // Start with default parameters
        svs::index::ivf::IVFSearchParameters result;
        if (is_specified(default_search_params_.n_probes)) {
            result.n_probes_ = default_search_params_.n_probes;
        }
        if (is_specified(default_search_params_.k_reorder)) {
            result.k_reorder_ = default_search_params_.k_reorder;
        }

        // Override with user-specified parameters
        if (params) {
            set_if_specified(result.n_probes_, params->n_probes);
            set_if_specified(result.k_reorder_, params->k_reorder);
        }

        return result;
    }

    void init_impl(data::ConstSimpleDataView<float> data, std::span<const size_t> ids) {
        auto build_params = ivf_build_parameters();

        // Single copy of data - required because IVF assembly deduces internal types from
        // data type, and ConstSimpleDataView<float> has const element type which breaks
        // internal type deduction. This copy is also passed directly to assemble which
        // partitions it into clusters (no additional copy for FP32 storage).
        auto owned_data = svs::data::SimpleData<float>(data.size(), data.dimensions());
        svs::data::copy(data, owned_data);

        svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric_type_));

        impl_.reset(distance_dispatcher([&](auto&& distance) {
            // Build clustering using BFloat16 for efficiency (AMX support)
            // Note: build_clustering takes const ref, doesn't consume data
            auto clustering = svs::IVF::build_clustering<svs::BFloat16>(
                build_params,
                owned_data,
                std::forward<decltype(distance)>(distance),
                num_threads_
            );

            // Dispatch on storage kind to assemble with correct data type
            return ivf_storage::dispatch_ivf_storage_kind(
                storage_kind_,
                [&]<typename DataType>(svs::lib::Type<DataType>) {
                    using TargetElement = typename DataType::element_type;

                    // For FP32: pass owned_data directly (moved into clusters)
                    // For FP16: convert from owned_data
                    if constexpr (std::is_same_v<TargetElement, float>) {
                        return new svs::DynamicIVF(
                            svs::DynamicIVF::assemble_from_clustering<float>(
                                std::move(clustering),
                                owned_data,
                                ids,
                                std::forward<decltype(distance)>(distance),
                                num_threads_,
                                intra_query_threads_
                            )
                        );
                    } else {
                        // Convert to target type (e.g., FP16)
                        DataType converted_data(owned_data.size(), owned_data.dimensions());
                        svs::data::copy(owned_data, converted_data);
                        return new svs::DynamicIVF(
                            svs::DynamicIVF::assemble_from_clustering<float>(
                                std::move(clustering),
                                std::move(converted_data),
                                ids,
                                std::forward<decltype(distance)>(distance),
                                num_threads_,
                                intra_query_threads_
                            )
                        );
                    }
                }
            );
        }));
    }

    // Data members
    size_t dim_;
    MetricType metric_type_;
    StorageKind storage_kind_;
    IVFIndex::BuildParams build_params_;
    IVFIndex::SearchParams default_search_params_;
    size_t num_threads_;
    size_t intra_query_threads_;
    std::unique_ptr<svs::DynamicIVF> impl_;
};

} // namespace runtime
} // namespace svs
