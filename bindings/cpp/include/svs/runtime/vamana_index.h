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
#include <svs/runtime/api_defs.h>
#include <svs/runtime/training.h>

#include <cstddef>
#include <iosfwd>

namespace svs {
namespace runtime {
inline namespace v0 {

namespace detail {
struct VamanaBuildParameters {
    size_t graph_max_degree = Unspecify<size_t>();
    size_t prune_to = Unspecify<size_t>();
    float alpha = Unspecify<float>();
    size_t construction_window_size = Unspecify<size_t>();
    size_t max_candidate_pool_size = Unspecify<size_t>();
    OptionalBool use_full_search_history = Unspecify<bool>();
};

struct VamanaSearchParameters {
    size_t search_window_size = Unspecify<size_t>();
    size_t search_buffer_capacity = Unspecify<size_t>();
    size_t prefetch_lookahead = Unspecify<size_t>();
    size_t prefetch_step = Unspecify<size_t>();
    // Minimum filter hit rate to continue filtered search.
    // If the hit rate after the first round falls below this threshold,
    // stop and return empty results (caller can fall back to exact search).
    // Default unspecified means never give up (treated as 0).
    float filter_stop = Unspecify<float>();
    // Enable pre-search filter sampling to estimate hit rate before graph traversal.
    // Uses a random sample of IDs to set initial batch size and trigger early exit.
    OptionalBool filter_estimate_batch = Unspecify<bool>();
};
} // namespace detail

// Abstract interface for Vamana-based indices.
struct SVS_RUNTIME_API VamanaIndex {
    virtual ~VamanaIndex();

    using BuildParams = detail::VamanaBuildParameters;
    using SearchParams = detail::VamanaSearchParameters;

    struct DynamicIndexParams {
        size_t blocksize_exp = 30;
    };

    virtual Status add(size_t n, const float* x) noexcept = 0;
    virtual Status reset() noexcept = 0;

    virtual Status search(
        size_t n,
        const float* x,
        size_t k,
        float* distances,
        size_t* labels,
        const SearchParams* params = nullptr,
        IDFilter* filter = nullptr
    ) const noexcept = 0;

    virtual Status range_search(
        size_t n,
        const float* x,
        float radius,
        const ResultsAllocator& results,
        const SearchParams* params = nullptr,
        IDFilter* filter = nullptr
    ) const noexcept = 0;

    // Utility function to check storage kind support
    static Status check_storage_kind(StorageKind storage_kind) noexcept;

    // Static constructors and destructors
    static Status build(
        VamanaIndex** index,
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        const VamanaIndex::BuildParams& params = VamanaIndex::BuildParams{},
        const VamanaIndex::SearchParams& default_search_params = VamanaIndex::SearchParams{}
    ) noexcept;

    static Status destroy(VamanaIndex* index) noexcept;

    virtual Status save(std::ostream& out) const noexcept = 0;
    static Status load(
        VamanaIndex** index, std::istream& in, MetricType metric, StorageKind storage_kind
    ) noexcept;
};

struct SVS_RUNTIME_API VamanaIndexLeanVec : public VamanaIndex {
    // Specialization to build LeanVec-based Vamana index with specified leanvec dims
    static Status build(
        VamanaIndex** index,
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        size_t leanvec_dims,
        const VamanaIndex::BuildParams& params = {},
        const VamanaIndex::SearchParams& default_search_params = {}
    ) noexcept;

    // Specialization to build LeanVec-based Vamana index with provided training data
    static Status build(
        VamanaIndex** index,
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        const LeanVecTrainingData* training_data,
        const VamanaIndex::BuildParams& params = {},
        const VamanaIndex::SearchParams& default_search_params = {}
    ) noexcept;
};

} // namespace v0
} // namespace runtime
} // namespace svs
