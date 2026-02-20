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

// Conditionally include LVQ/LeanVec headers
#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
#include "training_impl.h"
#include <svs/extensions/ivf/leanvec.h>
#include <svs/extensions/ivf/lvq.h>
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

///// IVF Data Types /////

// Simple uncompressed data types
template <typename T>
using IVFSimpleDataType = svs::data::SimpleData<T, svs::Dynamic, svs::lib::Allocator<T>>;

template <typename T>
using IVFBlockedSimpleDataType =
    svs::data::SimpleData<T, svs::Dynamic, svs::data::Blocked<svs::lib::Allocator<T>>>;

// Scalar Quantization data types
template <typename T>
using IVFSQDataType =
    svs::quantization::scalar::SQDataset<T, svs::Dynamic, svs::lib::Allocator<T>>;

template <typename T>
using IVFBlockedSQDataType = svs::quantization::scalar::
    SQDataset<T, svs::Dynamic, svs::data::Blocked<svs::lib::Allocator<T>>>;

#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
// LVQ data types
template <size_t Primary, size_t Residual, typename Strategy>
using IVFLVQDataType = svs::quantization::lvq::
    LVQDataset<Primary, Residual, svs::Dynamic, Strategy, svs::lib::Allocator<std::byte>>;

template <size_t Primary, size_t Residual, typename Strategy>
using IVFBlockedLVQDataType = svs::quantization::lvq::LVQDataset<
    Primary,
    Residual,
    svs::Dynamic,
    Strategy,
    svs::data::Blocked<svs::lib::Allocator<std::byte>>>;

using Sequential = svs::quantization::lvq::Sequential;
using Turbo16x8 = svs::quantization::lvq::Turbo<16, 8>;

// LeanVec data types
template <size_t I1, size_t I2>
using IVFLeanVecDataType = svs::leanvec::LeanDataset<
    svs::leanvec::UsingLVQ<I1>,
    svs::leanvec::UsingLVQ<I2>,
    svs::Dynamic,
    svs::Dynamic,
    svs::lib::Allocator<std::byte>>;

template <size_t I1, size_t I2>
using IVFBlockedLeanVecDataType = svs::leanvec::LeanDataset<
    svs::leanvec::UsingLVQ<I1>,
    svs::leanvec::UsingLVQ<I2>,
    svs::Dynamic,
    svs::Dynamic,
    svs::data::Blocked<svs::lib::Allocator<std::byte>>>;
#endif // SVS_RUNTIME_HAVE_LVQ_LEANVEC

///// Storage Type Mapping /////

// Map StorageKind to data type using storage tags
template <storage::StorageTag Tag> struct IVFStorageType {
    using type = storage::UnsupportedStorageType;
};

template <storage::StorageTag Tag> struct IVFBlockedStorageType {
    using type = storage::UnsupportedStorageType;
};

template <storage::StorageTag Tag>
using IVFStorageType_t = typename IVFStorageType<Tag>::type;

template <storage::StorageTag Tag>
using IVFBlockedStorageType_t = typename IVFBlockedStorageType<Tag>::type;

// clang-format off
template <> struct IVFStorageType<storage::FP32Tag> { using type = IVFSimpleDataType<float>; };
template <> struct IVFStorageType<storage::FP16Tag> { using type = IVFSimpleDataType<svs::Float16>; };
template <> struct IVFStorageType<storage::SQI8Tag> { using type = IVFSQDataType<std::int8_t>; };

template <> struct IVFBlockedStorageType<storage::FP32Tag> { using type = IVFBlockedSimpleDataType<float>; };
template <> struct IVFBlockedStorageType<storage::FP16Tag> { using type = IVFBlockedSimpleDataType<svs::Float16>; };
template <> struct IVFBlockedStorageType<storage::SQI8Tag> { using type = IVFBlockedSQDataType<std::int8_t>; };
// clang-format on

#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
// clang-format off
template <> struct IVFStorageType<storage::LVQ4x0Tag> { using type = IVFLVQDataType<4, 0, Turbo16x8>; };
template <> struct IVFStorageType<storage::LVQ8x0Tag> { using type = IVFLVQDataType<8, 0, Sequential>; };
template <> struct IVFStorageType<storage::LVQ4x4Tag> { using type = IVFLVQDataType<4, 4, Turbo16x8>; };
template <> struct IVFStorageType<storage::LVQ4x8Tag> { using type = IVFLVQDataType<4, 8, Turbo16x8>; };

template <> struct IVFBlockedStorageType<storage::LVQ4x0Tag> { using type = IVFBlockedLVQDataType<4, 0, Turbo16x8>; };
template <> struct IVFBlockedStorageType<storage::LVQ8x0Tag> { using type = IVFBlockedLVQDataType<8, 0, Sequential>; };
template <> struct IVFBlockedStorageType<storage::LVQ4x4Tag> { using type = IVFBlockedLVQDataType<4, 4, Turbo16x8>; };
template <> struct IVFBlockedStorageType<storage::LVQ4x8Tag> { using type = IVFBlockedLVQDataType<4, 8, Turbo16x8>; };
// clang-format on

// clang-format off
template <> struct IVFStorageType<storage::LeanVec4x4Tag> { using type = IVFLeanVecDataType<4, 4>; };
template <> struct IVFStorageType<storage::LeanVec4x8Tag> { using type = IVFLeanVecDataType<4, 8>; };
template <> struct IVFStorageType<storage::LeanVec8x8Tag> { using type = IVFLeanVecDataType<8, 8>; };

template <> struct IVFBlockedStorageType<storage::LeanVec4x4Tag> { using type = IVFBlockedLeanVecDataType<4, 4>; };
template <> struct IVFBlockedStorageType<storage::LeanVec4x8Tag> { using type = IVFBlockedLeanVecDataType<4, 8>; };
template <> struct IVFBlockedStorageType<storage::LeanVec8x8Tag> { using type = IVFBlockedLeanVecDataType<8, 8>; };
// clang-format on
#endif // SVS_RUNTIME_HAVE_LVQ_LEANVEC

///// Storage Factory /////

template <typename DataType> struct IVFStorageFactory;

// Unsupported storage factory
template <> struct IVFStorageFactory<storage::UnsupportedStorageType> {
    using DataType = IVFSimpleDataType<float>;

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
template <typename Tag, svs::threads::ThreadPool Pool>
    requires storage::StorageTag<std::decay_t<Tag>>
auto make_ivf_storage(
    Tag&&, const svs::data::ConstSimpleDataView<float>& data, Pool& pool, size_t arg = 0
) {
    using TagDecay = std::decay_t<Tag>;
    return IVFStorageFactory<IVFStorageType_t<TagDecay>>::compress(data, pool, arg);
}

template <typename Tag, svs::threads::ThreadPool Pool>
    requires storage::StorageTag<std::decay_t<Tag>>
auto make_ivf_blocked_storage(
    Tag&&, const svs::data::ConstSimpleDataView<float>& data, Pool& pool, size_t arg = 0
) {
    using TagDecay = std::decay_t<Tag>;
    return IVFStorageFactory<IVFBlockedStorageType_t<TagDecay>>::compress(data, pool, arg);
}

#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
// LeanVec-specific make functions with matrices parameter
template <typename Tag, svs::threads::ThreadPool Pool>
    requires storage::StorageTag<std::decay_t<Tag>>
auto make_ivf_leanvec_storage(
    Tag&&,
    const svs::data::ConstSimpleDataView<float>& data,
    Pool& pool,
    size_t leanvec_d,
    std::optional<svs::leanvec::LeanVecMatrices<svs::Dynamic>> matrices
) {
    using TagDecay = std::decay_t<Tag>;
    return IVFStorageFactory<IVFStorageType_t<TagDecay>>::compress(
        data, pool, leanvec_d, std::move(matrices)
    );
}

template <typename Tag, svs::threads::ThreadPool Pool>
    requires storage::StorageTag<std::decay_t<Tag>>
auto make_ivf_blocked_leanvec_storage(
    Tag&&,
    const svs::data::ConstSimpleDataView<float>& data,
    Pool& pool,
    size_t leanvec_d,
    std::optional<svs::leanvec::LeanVecMatrices<svs::Dynamic>> matrices
) {
    using TagDecay = std::decay_t<Tag>;
    return IVFStorageFactory<IVFBlockedStorageType_t<TagDecay>>::compress(
        data, pool, leanvec_d, std::move(matrices)
    );
}
#endif // SVS_RUNTIME_HAVE_LVQ_LEANVEC

///// Dispatch Functions /////

// Dispatch on storage kind for IVF operations (excludes LeanVec - handled separately)
template <typename F, typename... Args>
auto dispatch_ivf_storage_kind(StorageKind kind, F&& f, Args&&... args) {
    switch (kind) {
        case StorageKind::FP32:
            return f(storage::FP32Tag{}, std::forward<Args>(args)...);
        case StorageKind::FP16:
            return f(storage::FP16Tag{}, std::forward<Args>(args)...);
        case StorageKind::SQI8:
            return f(storage::SQI8Tag{}, std::forward<Args>(args)...);
#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
        case StorageKind::LVQ4x0:
            return f(storage::LVQ4x0Tag{}, std::forward<Args>(args)...);
        case StorageKind::LVQ8x0:
            return f(storage::LVQ8x0Tag{}, std::forward<Args>(args)...);
        case StorageKind::LVQ4x4:
            return f(storage::LVQ4x4Tag{}, std::forward<Args>(args)...);
        case StorageKind::LVQ4x8:
            return f(storage::LVQ4x8Tag{}, std::forward<Args>(args)...);
#endif
        default:
            throw StatusException{
                ErrorCode::NOT_IMPLEMENTED,
                "Requested storage kind is not supported for IVF index"};
    }
}

#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
// Dispatch on LeanVec storage kinds only
template <typename F, typename... Args>
auto dispatch_ivf_leanvec_storage_kind(StorageKind kind, F&& f, Args&&... args) {
    switch (kind) {
        case StorageKind::LeanVec4x4:
            return f(storage::LeanVec4x4Tag{}, std::forward<Args>(args)...);
        case StorageKind::LeanVec4x8:
            return f(storage::LeanVec4x8Tag{}, std::forward<Args>(args)...);
        case StorageKind::LeanVec8x8:
            return f(storage::LeanVec8x8Tag{}, std::forward<Args>(args)...);
        default:
            throw StatusException{
                ErrorCode::INVALID_ARGUMENT, "LeanVec storage kind required"};
    }
}
#endif // SVS_RUNTIME_HAVE_LVQ_LEANVEC

} // namespace ivf_storage

// Static IVF index implementation (non-LeanVec storage kinds)
class IVFIndexImpl {
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
        return ivf_storage::dispatch_ivf_storage_kind(storage_kind, [&](auto tag) {
            using Tag = decltype(tag);
            using DataType = ivf_storage::IVFStorageType_t<Tag>;

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
                    std::move(impl), metric, storage_kind, num_threads, intra_query_threads
                );
            });
        });
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
            return ivf_storage::dispatch_ivf_storage_kind(storage_kind_, [&](auto tag) {
                // Compress data to target storage type using the factory
                auto compressed_data = ivf_storage::make_ivf_storage(tag, data, threadpool);

                return new svs::IVF(svs::IVF::assemble_from_clustering<float>(
                    std::move(clustering),
                    std::move(compressed_data),
                    std::forward<decltype(distance)>(distance),
                    num_threads_,
                    intra_query_threads_
                ));
            });
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

        return ivf_storage::dispatch_ivf_leanvec_storage_kind(storage_kind, [&](auto tag) {
            using Tag = decltype(tag);
            using DataType = ivf_storage::IVFStorageType_t<Tag>;

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
                    std::move(impl), metric, storage_kind, num_threads, intra_query_threads
                );
            });
        });
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
            return ivf_storage::dispatch_ivf_leanvec_storage_kind(
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
