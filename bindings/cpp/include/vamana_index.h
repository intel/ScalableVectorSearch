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
namespace runtime {

// Abstract interface for Vamana-based indexes.
struct SVS_RUNTIME_API VamanaIndex {
    enum StorageKind { 
        FP32, FP16, SQI8,
        LVQ4x0, LVQ4x4, LVQ4x8,
        LeanVec4x4, LeanVec4x8, LeanVec8x8,
    };

    // TODO:
    // 1. Should StorageKind, metric, be a part of BuildParams?
    // 2. Does it make sense to have "Common" BuildParams{dim, metric} struct for other index algos (Flat, IVF)?
    //    Or dim, metric, storage kind should be passed separately to the build() method?
    //    What about storage kind in Flat, IVF?
    struct BuildParams {
        size_t dim;
        size_t graph_max_degree;
        size_t prune_to = 0;
        float alpha = 0;
        size_t construction_window_size = 40;
        size_t max_candidate_pool_size = 200;
        bool use_full_search_history = true;
    };

    struct SearchParams {
        size_t search_window_size = 0;
        size_t search_buffer_capacity = 0;
    };

    // Unused for now:
    virtual size_t size() const noexcept = 0;
    virtual size_t dimensions() const noexcept = 0;
    virtual MetricType metric_type() const noexcept = 0;
    virtual StorageKind get_storage_kind() const noexcept = 0;

    virtual Status add(size_t n, const size_t* labels, const float* x) noexcept = 0;
    virtual Status remove_selected(size_t* num_removed, const IDFilter& selector) noexcept = 0;
    // Further method for deletion can be added later:
    // virtual Status remove(size_t n, const size_t* labels) noexcept = 0;

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

    virtual Status reset() noexcept = 0;
    // TODO: Does it make sense to rename it to "save()"?
    virtual Status serialize(std::ostream& out) const noexcept = 0;

    // Static constructors and destructors
    static Status build(
        VamanaIndex** index,
        MetricType metric,
        StorageKind storage_kind,
        const VamanaIndex::BuildParams& params,
        const VamanaIndex::SearchParams& default_search_params = {10,10}
    ) noexcept;

    static Status destroy(VamanaIndex* index) noexcept;
    // TODO: Does it make sense to rename it to "load()"?
    // TODO: is it possible to get metric and storage kind from the stream instead of passing them explicitly?
    static Status deserialize(VamanaIndex** index, std::istream& in, MetricType metric, VamanaIndex::StorageKind storage_kind) noexcept;
};

struct SVS_RUNTIME_API VamanaIndexLeanVecFactory {
    static Status train(
        VamanaIndexLeanVecFactory** factory,
        size_t d,
        size_t n,
        const float* x,
        size_t leanvec_dims
    ) noexcept;

    static Status destroy(VamanaIndexLeanVecFactory* factory) noexcept;

    virtual Status serialize(std::ostream& out) const noexcept;
    static Status deserialize(VamanaIndexLeanVecFactory** factory, std::istream& in) noexcept;

    virtual Status buildIndex(
        VamanaIndex** index,
        size_t dim,
        MetricType metric,
        const VamanaIndex::BuildParams& params,
        const VamanaIndex::SearchParams& default_search_params = {}
    ) noexcept;
};

} // namespace runtime
} // namespace svs
