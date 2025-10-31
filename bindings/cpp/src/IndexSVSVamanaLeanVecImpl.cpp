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

#include "IndexSVSVamanaLeanVecImpl.h"
#include "IndexSVSImplUtils.h"

#include <variant>

#include <svs/core/medioid.h>
#include <svs/cpuid.h>
#include <svs/extensions/vamana/scalar.h>
#include <svs/orchestrators/dynamic_vamana.h>
#include <svs/quantization/scalar/scalar.h>

#include SVS_LEANVEC_HEADER

namespace svs::runtime {

namespace {
using blocked_alloc_type = svs::data::Blocked<svs::lib::Allocator<std::byte>>;
using blocked_alloc_type_sq = svs::data::Blocked<svs::lib::Allocator<std::int8_t>>;

using storage_type_4x4 = svs::leanvec::LeanDataset<
    svs::leanvec::UsingLVQ<4>,
    svs::leanvec::UsingLVQ<4>,
    svs::Dynamic,
    svs::Dynamic,
    blocked_alloc_type>;
using storage_type_4x8 = svs::leanvec::LeanDataset<
    svs::leanvec::UsingLVQ<4>,
    svs::leanvec::UsingLVQ<8>,
    svs::Dynamic,
    svs::Dynamic,
    blocked_alloc_type>;
using storage_type_8x8 = svs::leanvec::LeanDataset<
    svs::leanvec::UsingLVQ<8>,
    svs::leanvec::UsingLVQ<8>,
    svs::Dynamic,
    svs::Dynamic,
    blocked_alloc_type>;
using storage_type_sq =
    svs::quantization::scalar::SQDataset<std::int8_t, svs::Dynamic, blocked_alloc_type_sq>;

svs::index::vamana::VamanaBuildParameters
get_build_parameters(const IndexSVSVamanaImpl::BuildParams& params) {
    return svs::index::vamana::VamanaBuildParameters{
        params.alpha,
        params.graph_max_degree,
        params.construction_window_size,
        params.max_candidate_pool_size,
        params.prune_to,
        params.use_full_search_history
    };
}

template <typename StorageType, typename ThreadPoolProto>
svs::DynamicVamana* init_impl_t(
    IndexSVSVamanaImpl* index,
    StorageType&& storage,
    MetricType metric,
    ThreadPoolProto&& threadpool
) {
    std::vector<size_t> labels(storage.size());
    std::iota(labels.begin(), labels.end(), 0);

    svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric));
    return distance_dispatcher([&](auto&& distance) {
        return new svs::DynamicVamana(
            svs::DynamicVamana::build<float>(
                get_build_parameters(index->build_params),
                std::forward<StorageType>(storage),
                std::move(labels),
                std::forward<decltype(distance)>(distance),
                std::forward<ThreadPoolProto>(threadpool)
            )
        );
    });
}

template <typename StorageType>
svs::DynamicVamana* deserialize_impl_t(std::istream& stream, MetricType metric) {
    auto threadpool =
        svs::threads::ThreadPoolHandle(svs::threads::OMPThreadPool(omp_get_max_threads()));

    svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric));
    return distance_dispatcher([&](auto&& distance) {
        return new svs::DynamicVamana(
            svs::DynamicVamana::assemble<float, StorageType>(
                stream, std::forward<decltype(distance)>(distance), std::move(threadpool)
            )
        );
    });
}

} // namespace

IndexSVSVamanaLeanVecImpl* IndexSVSVamanaLeanVecImpl::build(
    size_t dim,
    MetricType metric,
    const BuildParams& params,
    size_t leanvec_dims,
    LeanVecLevel leanvec_level
) noexcept {
    try {
        auto index = new IndexSVSVamanaLeanVecImpl(
            dim, params.graph_max_degree, metric, leanvec_dims, leanvec_level
        );
        index->build_params = params;
        return index;
    } catch (...) { return nullptr; }
}

IndexSVSVamanaLeanVecImpl::IndexSVSVamanaLeanVecImpl() = default;

IndexSVSVamanaLeanVecImpl::IndexSVSVamanaLeanVecImpl(
    size_t d,
    size_t degree,
    MetricType metric,
    size_t leanvec_dims,
    LeanVecLevel leanvec_level
)
    : IndexSVSVamanaImpl(d, degree, metric)
    , leanvec_d{leanvec_dims == 0 ? d / 2 : leanvec_dims}
    , leanvec_level{leanvec_level} {}

IndexSVSVamanaLeanVecImpl::~IndexSVSVamanaLeanVecImpl() = default;

void IndexSVSVamanaLeanVecImpl::reset() noexcept {
    IndexSVSVamanaImpl::reset();
    leanvec_matrix.reset();
}

Status IndexSVSVamanaLeanVecImpl::train(size_t n, const float* x) noexcept {
    const auto data = svs::data::SimpleDataView<float>(const_cast<float*>(x), n, dim_);
    auto threadpool =
        svs::threads::ThreadPoolHandle(svs::threads::OMPThreadPool(omp_get_max_threads()));
    auto means = svs::utils::compute_medioid(data, threadpool);
    auto matrix = svs::leanvec::compute_leanvec_matrix<svs::Dynamic, svs::Dynamic>(
        data, means, threadpool, svs::lib::MaybeStatic<svs::Dynamic>{leanvec_d}
    );
    leanvec_matrix =
        std::make_unique<svs::leanvec::LeanVecMatrices<svs::Dynamic>>(matrix, matrix);
    trained = true;
    return Status_Ok;
}

Status IndexSVSVamanaLeanVecImpl::init_impl(size_t n, const float* x) noexcept {
    if (impl) {
        return Status{ErrorCode::UNKNOWN_ERROR, "Index already initialized"};
    }

    if (!is_trained()) {
        return {
            ErrorCode::NOT_INITIALIZED,
            "Cannot initialize SVS LeanVec index without training first."
        };
    }

    // TODO: support ConstSimpleDataView in SVS shared/static lib
    const auto data = svs::data::SimpleDataView<float>(const_cast<float*>(x), n, dim_);
    std::vector<size_t> labels(n);
    auto threadpool =
        svs::threads::ThreadPoolHandle(svs::threads::OMPThreadPool(omp_get_max_threads()));

    std::variant<
        std::monostate,
        storage_type_4x4,
        storage_type_4x8,
        storage_type_8x8,
        storage_type_sq>
        compressed_data;

    if (svs::detail::intel_enabled()) {
        switch (leanvec_level) {
            case LeanVecLevel::LeanVec4x4:
                compressed_data = storage_type_4x4::reduce(
                    data,
                    *leanvec_matrix,
                    threadpool,
                    0,
                    svs::lib::MaybeStatic<svs::Dynamic>(leanvec_d),
                    blocked_alloc_type{}
                );
                break;
            case LeanVecLevel::LeanVec4x8:
                compressed_data = storage_type_4x8::reduce(
                    data,
                    *leanvec_matrix,
                    threadpool,
                    0,
                    svs::lib::MaybeStatic<svs::Dynamic>(leanvec_d),
                    blocked_alloc_type{}
                );
                break;
            case LeanVecLevel::LeanVec8x8:
                compressed_data = storage_type_8x8::reduce(
                    data,
                    *leanvec_matrix,
                    threadpool,
                    0,
                    svs::lib::MaybeStatic<svs::Dynamic>(leanvec_d),
                    blocked_alloc_type{}
                );
                break;
            default:
                return Status{
                    ErrorCode::NOT_IMPLEMENTED, "not supported SVS LeanVec level"
                };
        }
    } else {
        compressed_data =
            storage_type_sq::compress(data, threadpool, blocked_alloc_type_sq{});
    }

    return std::visit(
        [&](auto&& storage) {
            if constexpr (std::is_same_v<std::decay_t<decltype(storage)>, std::monostate>) {
                return Status{
                    ErrorCode::NOT_INITIALIZED, "SVS LeanVec data is not initialized."
                };
            } else {
                impl.reset(init_impl_t(
                    this,
                    std::forward<decltype(storage)>(storage),
                    metric_type_,
                    std::move(threadpool)
                ));
                return Status_Ok;
            }
        },
        compressed_data
    );
}

Status IndexSVSVamanaLeanVecImpl::deserialize_impl(std::istream& in) noexcept {
    if (impl) {
        return Status{
            ErrorCode::INVALID_ARGUMENT,
            "Cannot deserialize: SVS index already initialized."
        };
    }

    if (svs::detail::intel_enabled()) {
        switch (leanvec_level) {
            case LeanVecLevel::LeanVec4x4:
                impl.reset(deserialize_impl_t<storage_type_4x4>(in, metric_type_));
                break;
            case LeanVecLevel::LeanVec4x8:
                impl.reset(deserialize_impl_t<storage_type_4x8>(in, metric_type_));
                break;
            case LeanVecLevel::LeanVec8x8:
                impl.reset(deserialize_impl_t<storage_type_8x8>(in, metric_type_));
                break;
            default:
                return Status(
                    ErrorCode::NOT_IMPLEMENTED, "not supported SVS LeanVec level"
                );
        }
    } else {
        impl.reset(deserialize_impl_t<storage_type_sq>(in, metric_type_));
    }

    trained = true;
    return Status_Ok;
}

} // namespace svs::runtime
