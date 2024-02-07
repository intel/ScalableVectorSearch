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
