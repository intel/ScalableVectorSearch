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
#include <svs/runtime/vamana_index.h>

#include <cstddef>
#include <istream>
#include <ostream>

namespace svs {
namespace runtime {
namespace v0 {

// Abstract interface for Dynamic Vamana-based indexes.
struct SVS_RUNTIME_API DynamicVamanaIndex : public VamanaIndex {
    virtual Status add(size_t n, const size_t* labels, const float* x) noexcept = 0;
    virtual Status
    remove_selected(size_t* num_removed, const IDFilter& selector) noexcept = 0;
    virtual Status remove(size_t n, const size_t* labels) noexcept = 0;

    virtual Status reset() noexcept = 0;

    // Utility function to check storage kind support
    static Status check_storage_kind(StorageKind storage_kind) noexcept;

    static Status check_params(const VamanaIndex::DynamicIndexParams& dynamic_index_params
    ) noexcept;

    // Static constructors and destructors
    // ABI backward compatibility
    static Status build(
        DynamicVamanaIndex** index,
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        const VamanaIndex::BuildParams& params = {},
        const VamanaIndex::SearchParams& default_search_params = {}
    ) noexcept;

    static Status build(
        DynamicVamanaIndex** index,
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        const VamanaIndex::BuildParams& params,
        const VamanaIndex::SearchParams& default_search_params,
        const VamanaIndex::DynamicIndexParams& dynamic_index_params
    ) noexcept;

    static Status destroy(DynamicVamanaIndex* index) noexcept;

    virtual Status save(std::ostream& out) const noexcept = 0;
    static Status load(
        DynamicVamanaIndex** index,
        std::istream& in,
        MetricType metric,
        StorageKind storage_kind
    ) noexcept;

    virtual size_t blocksize_bytes() const noexcept = 0;
};

struct SVS_RUNTIME_API DynamicVamanaIndexLeanVec : public DynamicVamanaIndex {
    // Specialization to build LeanVec-based Vamana index with specified leanvec dims
    // ABI backward compatibility
    static Status build(
        DynamicVamanaIndex** index,
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        size_t leanvec_dims,
        const VamanaIndex::BuildParams& params = {},
        const VamanaIndex::SearchParams& default_search_params = {}
    ) noexcept;

    static Status build(
        DynamicVamanaIndex** index,
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        size_t leanvec_dims,
        const VamanaIndex::BuildParams& params,
        const VamanaIndex::SearchParams& default_search_params,
        const VamanaIndex::DynamicIndexParams& dynamic_index_params
    ) noexcept;

    // Specialization to build LeanVec-based Vamana index with provided training data
    // ABI backward compatibility
    static Status build(
        DynamicVamanaIndex** index,
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        const LeanVecTrainingData* training_data,
        const VamanaIndex::BuildParams& params = {},
        const VamanaIndex::SearchParams& default_search_params = {}
    ) noexcept;

    static Status build(
        DynamicVamanaIndex** index,
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        const LeanVecTrainingData* training_data,
        const VamanaIndex::BuildParams& params,
        const VamanaIndex::SearchParams& default_search_params,
        const VamanaIndex::DynamicIndexParams& dynamic_index_params
    ) noexcept;
};

} // namespace v0
} // namespace runtime
} // namespace svs
