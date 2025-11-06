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

namespace svs {
namespace runtime {

// Abstract interface for Vamana-based indices.
// NOTE VamanaIndex is not implemented directly, only DynamicVamanaIndex is implemented.
struct SVS_RUNTIME_API VamanaIndex {
    virtual ~VamanaIndex();

    struct BuildParams {
        size_t graph_max_degree;
        size_t prune_to = 0;
        float alpha = 0;
        size_t construction_window_size = 40;
        size_t max_candidate_pool_size = 200;
        bool use_full_search_history = true;
    };

    struct SearchParams {
        size_t search_window_size = 10;
        size_t search_buffer_capacity = 10;
        size_t prefetch_lookahead = 0;
        size_t prefetch_step = 0;
    };

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
};
} // namespace runtime
} // namespace svs
