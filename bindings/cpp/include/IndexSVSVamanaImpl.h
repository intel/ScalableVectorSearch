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
#include "IndexSVSImplDefs.h"

#include <cstddef>
#include <istream>
#include <memory>
#include <numeric>
#include <ostream>
#include <vector>

namespace svs {
class DynamicVamana;

namespace runtime {

struct SVS_RUNTIME_API IndexSVSVamanaImpl {
    struct SearchParams {
        size_t search_window_size = 0;
        size_t search_buffer_capacity = 0;
    };

    enum StorageKind { FP32, FP16, SQI8 } storage_kind = StorageKind::FP32;

    struct BuildParams {
        StorageKind storage_kind;
        size_t graph_max_degree;
        size_t prune_to;
        float alpha = 1.2;
        size_t construction_window_size = 40;
        size_t max_candidate_pool_size = 200;
        bool use_full_search_history = true;
    };

    static IndexSVSVamanaImpl*
    build(size_t dim, MetricType metric, const BuildParams& params) noexcept;
    static void destroy(IndexSVSVamanaImpl* impl) noexcept;

    Status add(size_t n, const float* x) noexcept;

    Status search(
        size_t n,
        const float* x,
        size_t k,
        float* distances,
        size_t* labels,
        const SearchParams* params = nullptr,
        IDFilter* filter = nullptr
    ) const noexcept;

    Status range_search(
        size_t n,
        const float* x,
        float radius,
        const ResultsAllocator& results,
        const SearchParams* params = nullptr,
        IDFilter* filter = nullptr
    ) const noexcept;

    size_t remove_ids(const IDFilter& selector) noexcept;

    virtual void reset() noexcept;

    /* Serialization and deserialization helpers */
    Status serialize_impl(std::ostream& out) const noexcept;
    virtual Status deserialize_impl(std::istream& in) noexcept;

    MetricType metric_type_;
    size_t dim_;
    SearchParams default_search_params{10, 10};
    BuildParams build_params{};

  protected:
    IndexSVSVamanaImpl();

    IndexSVSVamanaImpl(
        size_t d,
        size_t degree,
        MetricType metric = MetricType::L2,
        StorageKind storage = StorageKind::FP32
    );

    virtual ~IndexSVSVamanaImpl();

    /* Initializes the implementation, using the provided data */
    virtual Status init_impl(size_t n, const float* x) noexcept;

    /* The actual SVS implementation */
    std::unique_ptr<svs::DynamicVamana> impl{nullptr};
    size_t ntotal_soft_deleted{0};
};

} // namespace runtime
} // namespace svs
