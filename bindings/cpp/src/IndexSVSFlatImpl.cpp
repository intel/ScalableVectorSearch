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

#include "IndexSVSFlatImpl.h"
#include "IndexSVSImplUtils.h"

#include <svs/core/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>

#include <svs/orchestrators/exhaustive.h>

namespace svs {

namespace runtime {

IndexSVSFlatImpl* IndexSVSFlatImpl::build(size_t dim, MetricType metric) noexcept {
    return new IndexSVSFlatImpl(dim, metric);
}

void IndexSVSFlatImpl::destroy(IndexSVSFlatImpl* impl) noexcept { delete impl; }

Status IndexSVSFlatImpl::init_impl(size_t n, const float* x) noexcept {
    auto data = svs::data::SimpleData<float>(n, dim_);
    auto threadpool =
        svs::threads::ThreadPoolHandle(svs::threads::OMPThreadPool(omp_get_max_threads()));

    svs::threads::parallel_for(
        threadpool,
        svs::threads::StaticPartition(n),
        [&](auto is, auto SVS_UNUSED(tid)) {
            for (auto i : is) {
                data.set_datum(i, std::span<const float>(x + i * dim_, dim_));
            }
        }
    );

    switch (metric_type_) {
        case MetricType::INNER_PRODUCT:
            impl.reset(new svs::Flat(svs::Flat::assemble<float>(
                std::move(data), svs::DistanceIP(), std::move(threadpool)
            )));
            break;
        case MetricType::L2:
            impl.reset(new svs::Flat(svs::Flat::assemble<float>(
                std::move(data), svs::DistanceL2(), std::move(threadpool)
            )));
            break;
        default:
            impl = nullptr;
            return {ErrorCode::INVALID_ARGUMENT, "not supported SVS distance"};
    }
    return Status_Ok;
}

Status IndexSVSFlatImpl::add(size_t n, const float* x) noexcept {
    if (!impl) {
        return init_impl(n, x);
    }

    return {
        ErrorCode::NOT_IMPLEMENTED,
        "IndexSVSFlat does not support adding points after initialization"};
}

void IndexSVSFlatImpl::reset() noexcept { impl.reset(); }

Status IndexSVSFlatImpl::search(
    size_t n, const float* x, size_t k, float* distances, size_t* labels
) const noexcept {
    if (!impl) {
        return {ErrorCode::UNKNOWN_ERROR, "SVS index not initialized"};
    }
    if (k == 0) {
        return {ErrorCode::INVALID_ARGUMENT, "k must be greater than 0"};
    }

    auto queries = svs::data::ConstSimpleDataView<float>(x, n, dim_);

    auto results = svs::QueryResult<size_t>{queries.size(), static_cast<size_t>(k)};
    impl->search(results.view(), queries, {});

    svs::threads::parallel_for(
        impl->get_threadpool_handle(),
        svs::threads::StaticPartition(n),
        [&](auto is, auto SVS_UNUSED(tid)) {
            for (auto i : is) {
                for (size_t j = 0; j < k; ++j) {
                    labels[j + i * k] = results.index(i, j);
                    distances[j + i * k] = results.distance(i, j);
                }
            }
        }
    );
    return Status_Ok;
}

Status IndexSVSFlatImpl::serialize(std::ostream& out) const noexcept {
    if (!impl) {
        return {ErrorCode::UNKNOWN_ERROR, "Cannot serialize: SVS index not initialized."};
    }

    impl->save(out);
    return Status_Ok;
}

Status IndexSVSFlatImpl::deserialize(std::istream& in) noexcept {
    if (impl) {
        return {
            ErrorCode::UNKNOWN_ERROR, "Cannot deserialize: SVS index already initialized."};
    }

    auto threadpool =
        svs::threads::ThreadPoolHandle(svs::threads::OMPThreadPool(omp_get_max_threads()));
    using storage_type = typename svs::VectorDataLoader<float>::return_type;

    svs::DistanceDispatcher dispatcher(to_svs_distance(metric_type_));
    dispatcher([&](auto&& distance) {
        impl.reset(new svs::Flat(svs::Flat::assemble<float, storage_type>(
            in, std::move(distance), std::move(threadpool)
        )));
    });

    return Status_Ok;
}

IndexSVSFlatImpl::IndexSVSFlatImpl(size_t dim, MetricType metric)
    : metric_type_(metric)
    , dim_(dim) {}
IndexSVSFlatImpl::~IndexSVSFlatImpl() = default;

} // namespace runtime
} // namespace svs
