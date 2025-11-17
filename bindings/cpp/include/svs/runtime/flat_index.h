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

#include <cstddef>
#include <istream>
#include <ostream>

namespace svs {
namespace runtime {
namespace v0 {

// Abstract interface for Flat indices.
struct SVS_RUNTIME_API FlatIndex {
    // Static constructors and destructors
    static Status check_storage_kind(StorageKind storage_kind) noexcept;

    static Status build(FlatIndex** index, size_t dim, MetricType metric) noexcept;
    static Status destroy(FlatIndex* index) noexcept;
    virtual ~FlatIndex();

    virtual Status search(
        size_t n, const float* x, size_t k, float* distances, size_t* labels
    ) const noexcept = 0;

    virtual Status add(size_t n, const float* x) noexcept = 0;
    virtual Status reset() noexcept = 0;

    virtual Status save(std::ostream& out) const noexcept = 0;
    static Status load(FlatIndex** index, std::istream& in, MetricType metric) noexcept;
};

} // namespace v0
} // namespace runtime
} // namespace svs
