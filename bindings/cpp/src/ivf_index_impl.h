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

// Include scalar quantization support
#include <svs/extensions/ivf/scalar.h>
#include <svs/quantization/scalar/scalar.h>

#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
#include "training_impl.h"
#endif

#include <algorithm>
#include <memory>
#include <variant>
#include <vector>

namespace svs {
namespace runtime {

// IVF storage kind support - following the Vamana storage pattern
namespace ivf_storage {

// Check if storage kind is LeanVec (requires training data)
inline bool is_leanvec_storage_kind(StorageKind kind) {
#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
    return kind == StorageKind::LeanVec4x4 || kind == StorageKind::LeanVec4x8 ||
           kind == StorageKind::LeanVec8x8;
#else
    (void)kind;
    return false;
#endif
}

// Check if storage kind is supported for IVF (non-LeanVec)
inline bool is_supported_non_leanvec_storage_kind(StorageKind kind) {
    switch (kind) {
        case StorageKind::FP32:
        case StorageKind::FP16:
        case StorageKind::SQI8:
            return true;
#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
        case StorageKind::LVQ4x0:
        case StorageKind::LVQ8x0:
        case StorageKind::LVQ4x4:
        case StorageKind::LVQ4x8:
            return true;
#endif
        default:
            return false;
    }
}

// Check if any storage kind is supported for IVF (including LeanVec)
inline bool is_supported_storage_kind(StorageKind kind) {
    return is_supported_non_leanvec_storage_kind(kind) || is_leanvec_storage_kind(kind);
}

///// Storage Factory /////

template <typename DataType> struct IVFStorageFactory;

// Unsupported storage factory
template <typename Alloc> struct IVFStorageFactory<storage::UnsupportedStorageType<Alloc>> {
    using DataType = storage::
        SimpleDatasetType<float, storage::rebind_extracted_allocator_t<float, Alloc>>;

    template <svs::threads::ThreadPool Pool>
    static DataType
    compress(const svs::data::ConstSimpleDataView<float>&, Pool&, size_t = 0) {
        throw StatusException(
            ErrorCode::NOT_IMPLEMENTED, "Requested storage kind is not supported for IVF"
        );
    }
};

// Simple data factory (FP32, FP16)
template <typename T, size_t Extent, typename Alloc>
struct IVFStorageFactory<svs::data::SimpleData<T, Extent, Alloc>> {
    using DataType = svs::data::SimpleData<T, Extent, Alloc>;

    template <svs::threads::ThreadPool Pool>
    static DataType
    compress(const svs::data::ConstSimpleDataView<float>& data, Pool& pool, size_t = 0) {
        DataType result(data.size(), data.dimensions());
        svs::threads::parallel_for(
            pool,
            svs::threads::StaticPartition(result.size()),
            [&](auto is, auto) {
                for (auto i : is) {
                    result.set_datum(i, data.get_datum(i));
                }
            }
        );
        return result;
    }
};

// Scalar Quantization factory
template <typename T, size_t Extent, typename Alloc>
struct IVFStorageFactory<svs::quantization::scalar::SQDataset<T, Extent, Alloc>> {
    using DataType = svs::quantization::scalar::SQDataset<T, Extent, Alloc>;

    template <svs::threads::ThreadPool Pool>
    static DataType
    compress(const svs::data::ConstSimpleDataView<float>& data, Pool& pool, size_t = 0) {
        return DataType::compress(data, pool);
    }
};

#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
// LVQ factory
template <size_t Primary, size_t Residual, size_t Extent, typename Strategy, typename Alloc>
struct IVFStorageFactory<
    svs::quantization::lvq::LVQDataset<Primary, Residual, Extent, Strategy, Alloc>> {
    using DataType =
        svs::quantization::lvq::LVQDataset<Primary, Residual, Extent, Strategy, Alloc>;

    template <svs::threads::ThreadPool Pool>
    static DataType
    compress(const svs::data::ConstSimpleDataView<float>& data, Pool& pool, size_t = 0) {
        return DataType::compress(data, pool, 0);
    }
};

// LeanVec factory - requires optional matrices for proper training
template <typename Primary, typename Secondary, size_t E1, size_t E2, typename Alloc>
struct IVFStorageFactory<svs::leanvec::LeanDataset<Primary, Secondary, E1, E2, Alloc>> {
    using DataType = svs::leanvec::LeanDataset<Primary, Secondary, E1, E2, Alloc>;
    using LeanVecMatricesType = svs::leanvec::LeanVecMatrices<svs::Dynamic>;

    template <svs::threads::ThreadPool Pool>
    static DataType compress(
        const svs::data::ConstSimpleDataView<float>& data,
        Pool& pool,
        size_t leanvec_d = 0,
        std::optional<LeanVecMatricesType> matrices = std::nullopt
    ) {
        if (leanvec_d == 0) {
            leanvec_d = (data.dimensions() + 1) / 2;
        }
        return DataType::reduce(
            data, std::move(matrices), pool, 0, svs::lib::MaybeStatic{leanvec_d}
        );
    }
};
#endif // SVS_RUNTIME_HAVE_LVQ_LEANVEC

// Helper to make compressed data (non-LeanVec)
template <StorageKind Kind, typename Alloc, svs::threads::ThreadPool Pool>
auto make_ivf_storage(
    storage::StorageType<Kind, Alloc> SVS_UNUSED(tag),
    const svs::data::ConstSimpleDataView<float>& data,
    Pool& pool,
    size_t arg = 0
) {
    static_assert(
        !svs::data::is_blocked_v<Alloc>, "Allocator must not be blocked for IVF storage"
    );
    return IVFStorageFactory<storage::StorageType_t<Kind, Alloc>>::compress(
        data, pool, arg
    );
}

template <StorageKind Kind, typename Alloc, svs::threads::ThreadPool Pool>
auto make_ivf_blocked_storage(
    storage::StorageType<Kind, Alloc> SVS_UNUSED(tag),
    const svs::data::ConstSimpleDataView<float>& data,
    Pool& pool,
    size_t arg = 0
) {
    static_assert(
        svs::data::is_blocked_v<Alloc>, "Allocator must be blocked for IVF storage"
    );
    return IVFStorageFactory<storage::StorageType_t<Kind, Alloc>>::compress(
        data, pool, arg
    );
}

#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
// LeanVec-specific make functions with matrices parameter
template <StorageKind Kind, typename Alloc, svs::threads::ThreadPool Pool>
auto make_ivf_leanvec_storage(
    storage::StorageType<Kind, Alloc> SVS_UNUSED(tag),
    const svs::data::ConstSimpleDataView<float>& data,
    Pool& pool,
    size_t leanvec_d,
    std::optional<svs::leanvec::LeanVecMatrices<svs::Dynamic>> matrices
) {
    static_assert(
        !svs::data::is_blocked_v<Alloc>, "Allocator must not be blocked for IVF storage"
    );
    return IVFStorageFactory<storage::StorageType_t<Kind, Alloc>>::compress(
        data, pool, leanvec_d, std::move(matrices)
    );
}

template <StorageKind Kind, typename Alloc, svs::threads::ThreadPool Pool>
auto make_ivf_blocked_leanvec_storage(
    storage::StorageType<Kind, Alloc> SVS_UNUSED(tag),
    const svs::data::ConstSimpleDataView<float>& data,
    Pool& pool,
    size_t leanvec_d,
    std::optional<svs::leanvec::LeanVecMatrices<svs::Dynamic>> matrices
) {
    static_assert(
        svs::data::is_blocked_v<Alloc>, "Allocator must be blocked for IVF storage"
    );
    return IVFStorageFactory<storage::StorageType_t<Kind, Alloc>>::compress(
        data, pool, leanvec_d, std::move(matrices)
    );
}
#endif // SVS_RUNTIME_HAVE_LVQ_LEANVEC

///// Dispatch Functions /////

// Dispatch on storage kind for IVF operations (excludes LeanVec - handled separately)
template <typename Alloc, typename F, typename... Args>
auto dispatch_ivf_storage_kind(StorageKind kind, F&& f, Args&&... args) {
    switch (kind) {
        case StorageKind::FP32:
            return f(
                storage::StorageType<StorageKind::FP32, Alloc>{},
                std::forward<Args>(args)...
            );
        case StorageKind::FP16:
            return f(
                storage::StorageType<StorageKind::FP16, Alloc>{},
                std::forward<Args>(args)...
            );
        case StorageKind::SQI8:
            return f(
                storage::StorageType<StorageKind::SQI8, Alloc>{},
                std::forward<Args>(args)...
            );
#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
        case StorageKind::LVQ4x0:
            return f(
                storage::StorageType<StorageKind::LVQ4x0, Alloc>{},
                std::forward<Args>(args)...
            );
        case StorageKind::LVQ8x0:
            return f(
                storage::StorageType<StorageKind::LVQ8x0, Alloc>{},
                std::forward<Args>(args)...
            );
        case StorageKind::LVQ4x4:
            return f(
                storage::StorageType<StorageKind::LVQ4x4, Alloc>{},
                std::forward<Args>(args)...
            );
        case StorageKind::LVQ4x8:
            return f(
                storage::StorageType<StorageKind::LVQ4x8, Alloc>{},
                std::forward<Args>(args)...
            );
#endif
        default:
            throw StatusException{
                ErrorCode::NOT_IMPLEMENTED,
                "Requested storage kind is not supported for IVF index"};
    }
}

#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
// Dispatch on LeanVec storage kinds only
template <typename Alloc, typename F, typename... Args>
auto dispatch_ivf_leanvec_storage_kind(StorageKind kind, F&& f, Args&&... args) {
    switch (kind) {
        case StorageKind::LeanVec4x4:
            return f(
                storage::StorageType<StorageKind::LeanVec4x4, Alloc>{},
                std::forward<Args>(args)...
            );
        case StorageKind::LeanVec4x8:
            return f(
                storage::StorageType<StorageKind::LeanVec4x8, Alloc>{},
                std::forward<Args>(args)...
            );
        case StorageKind::LeanVec8x8:
            return f(
                storage::StorageType<StorageKind::LeanVec8x8, Alloc>{},
                std::forward<Args>(args)...
            );
        default:
            throw StatusException{
                ErrorCode::INVALID_ARGUMENT, "LeanVec storage kind required"};
    }
}
#endif // SVS_RUNTIME_HAVE_LVQ_LEANVEC

} // namespace ivf_storage

// Static IVF index implementation (non-LeanVec storage kinds)
class IVFIndexImpl {
    using allocator_type = svs::lib::Allocator<float>;

  public:
    IVFIndexImpl(
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
        if (!ivf_storage::is_supported_non_leanvec_storage_kind(storage_kind)) {
            throw StatusException{
                ErrorCode::INVALID_ARGUMENT,
                "The specified storage kind is not compatible with IVFIndex. "
                "Use IVFIndexLeanVecImpl for LeanVec storage kinds."};
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
        impl_->search(result, queries, sp);
    }

    void save(std::ostream& out) const {
        if (!impl_) {
            throw StatusException{
                ErrorCode::NOT_INITIALIZED, "Cannot serialize: IVF index not initialized."};
        }
        impl_->save(out);
    }

    void set_num_threads(size_t num_threads) {
        if (num_threads == 0) {
            num_threads = static_cast<size_t>(omp_get_max_threads());
        }
        num_threads_ = num_threads;
        if (impl_) {
            impl_->set_threadpool(svs::threads::DefaultThreadPool(num_threads));
        }
    }

    size_t get_num_threads() const {
        if (impl_) {
            return impl_->get_num_threads();
        }
        return num_threads_;
    }

    static IVFIndexImpl* load(
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
        return ivf_storage::dispatch_ivf_storage_kind<allocator_type>(
            storage_kind,
            [&](auto tag) {
                using Tag = decltype(tag);
                using DataType = typename Tag::type;

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
                    return new IVFIndexImpl(
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
    IVFIndexImpl(
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
        default_search_params_.n_probes = loaded_params.n_probes_;
        default_search_params_.k_reorder = loaded_params.k_reorder_;
    }

    // Constructor used by subclasses (LeanVec) that handle their own validation
    IVFIndexImpl(
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        const IVFIndex::BuildParams& params,
        const IVFIndex::SearchParams& default_search_params,
        size_t num_threads,
        size_t intra_query_threads,
        bool /*skip_validation*/
    )
        : dim_{dim}
        , metric_type_{metric}
        , storage_kind_{storage_kind}
        , build_params_{params}
        , default_search_params_{default_search_params}
        , num_threads_{num_threads == 0 ? static_cast<size_t>(omp_get_max_threads()) : num_threads}
        , intra_query_threads_{intra_query_threads} {
        // Subclasses handle their own storage kind validation
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
        auto threadpool =
            svs::threads::ThreadPoolHandle(svs::threads::DefaultThreadPool(num_threads_));

        svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric_type_));

        impl_.reset(distance_dispatcher([&](auto&& distance) {
            // Build clustering using BFloat16 for efficiency (AMX support)
            auto clustering = svs::IVF::build_clustering<svs::BFloat16>(
                build_params, data, std::forward<decltype(distance)>(distance), num_threads_
            );

            // Dispatch on storage kind to compress and assemble
            return ivf_storage::dispatch_ivf_storage_kind<allocator_type>(
                storage_kind_,
                [&](auto tag) {
                    // Compress data to target storage type using the factory
                    auto compressed_data =
                        ivf_storage::make_ivf_storage(tag, data, threadpool);

                    return new svs::IVF(svs::IVF::assemble_from_clustering<float>(
                        std::move(clustering),
                        std::move(compressed_data),
                        std::forward<decltype(distance)>(distance),
                        num_threads_,
                        intra_query_threads_
                    ));
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

#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
// Static IVF index implementation for LeanVec storage kinds
class IVFIndexLeanVecImpl : public IVFIndexImpl {
    using allocator_type = svs::lib::Allocator<std::byte>;

  public:
    using LeanVecMatricesType = LeanVecTrainingDataImpl::LeanVecMatricesType;

    // Constructor for building with training data
    IVFIndexLeanVecImpl(
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        const LeanVecTrainingDataImpl& training_data,
        const IVFIndex::BuildParams& params,
        const IVFIndex::SearchParams& default_search_params,
        size_t num_threads,
        size_t intra_query_threads
    )
        : IVFIndexImpl{dim, metric, storage_kind, params, default_search_params, num_threads, intra_query_threads, /*skip_validation=*/true}
        , leanvec_dims_{training_data.get_leanvec_dims()}
        , leanvec_matrices_{training_data.get_leanvec_matrices()} {
        check_leanvec_storage_kind(storage_kind);
    }

    // Constructor for building without pre-computed matrices (will compute during build)
    IVFIndexLeanVecImpl(
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        size_t leanvec_dims,
        const IVFIndex::BuildParams& params,
        const IVFIndex::SearchParams& default_search_params,
        size_t num_threads,
        size_t intra_query_threads
    )
        : IVFIndexImpl{dim, metric, storage_kind, params, default_search_params, num_threads, intra_query_threads, /*skip_validation=*/true}
        , leanvec_dims_{leanvec_dims}
        , leanvec_matrices_{std::nullopt} {
        check_leanvec_storage_kind(storage_kind);
    }

    // Constructor for loading
    IVFIndexLeanVecImpl(
        std::unique_ptr<svs::IVF>&& impl,
        MetricType metric,
        StorageKind storage_kind,
        size_t num_threads,
        size_t intra_query_threads
    )
        : IVFIndexImpl{std::move(impl), metric, storage_kind, num_threads, intra_query_threads}
        , leanvec_dims_{0}
        , leanvec_matrices_{std::nullopt} {
        check_leanvec_storage_kind(storage_kind);
    }

    void build(data::ConstSimpleDataView<float> data) {
        if (impl_) {
            throw StatusException{ErrorCode::RUNTIME_ERROR, "Index already initialized"};
        }
        init_leanvec_impl(data);
    }

    static IVFIndexLeanVecImpl* load(
        std::istream& in,
        MetricType metric,
        StorageKind storage_kind,
        size_t num_threads,
        size_t intra_query_threads
    ) {
        if (num_threads == 0) {
            num_threads = static_cast<size_t>(omp_get_max_threads());
        }

        return ivf_storage::dispatch_ivf_leanvec_storage_kind<allocator_type>(
            storage_kind,
            [&](auto tag) {
                using Tag = decltype(tag);
                using DataType = typename Tag::type;

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
                    return new IVFIndexLeanVecImpl(
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
    size_t leanvec_dims_;
    std::optional<LeanVecMatricesType> leanvec_matrices_;

    void check_leanvec_storage_kind(StorageKind kind) {
        if (!ivf_storage::is_leanvec_storage_kind(kind)) {
            throw StatusException(
                ErrorCode::INVALID_ARGUMENT, "LeanVec storage kind required"
            );
        }
        if (!svs::detail::lvq_leanvec_enabled()) {
            throw StatusException(
                ErrorCode::NOT_IMPLEMENTED,
                "LeanVec storage kind requested but not supported by CPU"
            );
        }
    }

    void init_leanvec_impl(data::ConstSimpleDataView<float> data) {
        auto build_params = ivf_build_parameters();
        auto threadpool =
            svs::threads::ThreadPoolHandle(svs::threads::DefaultThreadPool(num_threads_));

        svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric_type_));

        impl_.reset(distance_dispatcher([&](auto&& distance) {
            // Build clustering using BFloat16 for efficiency (AMX support)
            auto clustering = svs::IVF::build_clustering<svs::BFloat16>(
                build_params, data, std::forward<decltype(distance)>(distance), num_threads_
            );

            // Dispatch on LeanVec storage kind to compress and assemble
            return ivf_storage::dispatch_ivf_leanvec_storage_kind<allocator_type>(
                storage_kind_,
                [&](auto tag) {
                    // Compress data to LeanVec storage type using the factory with matrices
                    auto compressed_data = ivf_storage::make_ivf_leanvec_storage(
                        tag, data, threadpool, leanvec_dims_, leanvec_matrices_
                    );

                    return new svs::IVF(svs::IVF::assemble_from_clustering<float>(
                        std::move(clustering),
                        std::move(compressed_data),
                        std::forward<decltype(distance)>(distance),
                        num_threads_,
                        intra_query_threads_
                    ));
                }
            );
        }));
    }
};
#endif // SVS_RUNTIME_HAVE_LVQ_LEANVEC

} // namespace runtime
} // namespace svs
