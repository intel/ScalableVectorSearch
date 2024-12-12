/*
 * Copyright 2024 Intel Corporation
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

// svs-benchmark
#include "svs-benchmark/benchmark.h"

// third-party
#include "tsl/robin_set.h"

// stl
#include <span>
#include <vector>

namespace svsbenchmark::inverted {

// Entry-point for registering inverted index-related executables.
void register_executables(svsbenchmark::ExecutableDispatcher& dispatcher);

///
/// @brief Validate externally supplied centroids for correctness.
///
/// Checks:
///
/// (1) IDs are sorted.
/// (2) All IDs are in-bounds.
/// (3) No ID is repeated.
///
template <std::integral I>
void validate_external_centroids(std::span<const I> ids, size_t max_valid_id) {
    if (ids.empty()) {
        throw ANNEXCEPTION("Centroid list is empty!");
    }

    auto itr = ids.begin();
    auto p = *itr;
    ++itr;
    const auto end = ids.end();
    for (; itr != end; ++itr) {
        auto n = *itr;
        if (n == p) {
            throw ANNEXCEPTION("Centroids have duplicate ids: {}", n);
        }
        if (n < p) {
            throw ANNEXCEPTION("Centroids are not sorted in increasing order!");
        }
        if (n > max_valid_id) {
            throw ANNEXCEPTION(
                "Centroid ID {} is out of bounds. Maximum allowed is {}.", n, max_valid_id
            );
        }
        p = n;
    }
}

} // namespace svsbenchmark::inverted
