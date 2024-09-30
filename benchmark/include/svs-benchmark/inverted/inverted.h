/*
 * Copyright (C) 2024 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
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
