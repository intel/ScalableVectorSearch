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

#include "svs/runtime/flat_index.h"

#include "svs_runtime_utils.h"

#include <svs/core/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/orchestrators/exhaustive.h>

#include <algorithm>
#include <memory>
#include <variant>
#include <vector>

namespace svs {
namespace runtime {

// Vamana index implementation
class FlatIndexImpl {
  public:
    FlatIndexImpl(size_t dim, MetricType metric)
        : dim_{dim}
        , metric_type_{metric} {}

    size_t size() const { return impl_ ? impl_->size() : 0; }

    size_t dimensions() const { return dim_; }

    MetricType metric_type() const { return metric_type_; }

    void add(data::ConstSimpleDataView<float> data) {
        if (!impl_) {
            return init_impl(data);
        }

        throw StatusException{
            ErrorCode::NOT_IMPLEMENTED,
            "Flat index does not support adding points after initialization"};
    }

    void search(
        svs::QueryResultView<size_t> result,
        svs::data::ConstSimpleDataView<float> queries,
        IDFilter* filter = nullptr
    ) const {
        if (!impl_) {
            auto& dists = result.distances();
            std::fill(dists.begin(), dists.end(), std::numeric_limits<float>::infinity());
            auto& inds = result.indices();
            std::fill(inds.begin(), inds.end(), static_cast<size_t>(-1));
            throw StatusException{ErrorCode::NOT_INITIALIZED, "Index not initialized"};
        }

        const size_t k = result.n_neighbors();
        if (k == 0) {
            throw StatusException{ErrorCode::INVALID_ARGUMENT, "k must be greater than 0"};
        }

        // Simple search
        if (filter == nullptr) {
            impl_->search(result, queries, {});
        } else {
            throw StatusException{
                ErrorCode::NOT_IMPLEMENTED, "Filtered search not implemented yet"};
        }
    }

    void reset() { impl_.reset(); }

    void save(std::ostream& out) const {
        if (!impl_) {
            throw StatusException{
                ErrorCode::NOT_INITIALIZED, "Cannot serialize: SVS index not initialized."};
        }

        impl_->save(out);
    }

    static FlatIndexImpl* load(std::istream& in, MetricType metric) {
        auto threadpool = default_threadpool();
        using storage_type = svs::runtime::storage::StorageType_t<storage::FP32Tag>;

        svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric));
        return distance_dispatcher([&](auto&& distance) {
            auto impl = new svs::Flat{svs::Flat::assemble<float, storage_type>(
                in, std::forward<decltype(distance)>(distance), std::move(threadpool)
            )};

            return new FlatIndexImpl(std::unique_ptr<svs::Flat>{impl}, metric);
        });
    }

  protected:
    // Constructor used during loading
    FlatIndexImpl(std::unique_ptr<svs::Flat>&& impl, MetricType metric)
        : dim_{impl->dimensions()}
        , metric_type_{metric}
        , impl_{std::move(impl)} {}

    void init_impl(data::ConstSimpleDataView<float> data) {
        auto threadpool = default_threadpool();

        auto storage = svs::runtime::storage::make_storage(
            svs::runtime::storage::FP32Tag{}, data, threadpool
        );

        svs::DistanceDispatcher distance_dispatcher(to_svs_distance(metric_type_));
        impl_.reset(distance_dispatcher([&](auto&& distance) {
            return new svs::Flat(svs::Flat::assemble<float>(
                std::move(storage),
                std::forward<decltype(distance)>(distance),
                std::move(threadpool)
            ));
        }));
    }

    // Data members
    size_t dim_;
    MetricType metric_type_;
    std::unique_ptr<svs::Flat> impl_;
};
} // namespace runtime
} // namespace svs
