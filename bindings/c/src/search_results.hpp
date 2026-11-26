/*
 * Copyright 2026 Intel Corporation
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

#include "svs/c/svs_c.h"

#include "error.hpp"

#include <svs/core/query_result.h>
#include <svs/lib/neighbor.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

namespace svs::c_runtime {

// Non-owning view over a caller-provided svs_search_results_t; buffers follow the C API
// ownership contract (owns_buffers).
class SearchResults {
  public:
    explicit SearchResults(svs_search_results_t& results) noexcept
        : r_{results} {}

    void validate() const {
        INVALID_ARGUMENT_IF(
            r_.version > svs_get_version(), "Incompatible svs_search_results_t version"
        );
        INVALID_ARGUMENT_IF(
            r_.struct_size > sizeof(svs_search_results_t),
            "Incompatible svs_search_results_t struct_size"
        );
    }

    // Grows library-owned buffers to hold counts[q] results per query; caller-owned
    // buffers that are too small are an error.
    void reserve(std::span<const size_t> counts) {
        const size_t offsets_capacity = counts.size() + 1;
        const size_t results_capacity =
            std::accumulate(counts.begin(), counts.end(), size_t{0});
        if (!r_.owns_buffers && r_.offsets == nullptr && r_.indices == nullptr &&
            r_.distances == nullptr) {
            // Zero-initialized descriptor: the library takes ownership.
            r_.offsets_capacity = 0;
            r_.results_capacity = 0;
            r_.owns_buffers = true;
        }
        if (!r_.owns_buffers) {
            INVALID_ARGUMENT_IF(
                r_.offsets_capacity < offsets_capacity ||
                    r_.results_capacity < results_capacity,
                "Caller-provided svs_search_results buffers are too small"
            );
            return;
        }

        std::unique_ptr<size_t[]> new_offsets;
        std::unique_ptr<size_t[]> new_indices;
        std::unique_ptr<float[]> new_distances;
        if (r_.offsets_capacity < offsets_capacity) {
            new_offsets = std::make_unique_for_overwrite<size_t[]>(offsets_capacity);
        }
        if (r_.results_capacity < results_capacity) {
            new_indices = std::make_unique_for_overwrite<size_t[]>(results_capacity);
            new_distances = std::make_unique_for_overwrite<float[]>(results_capacity);
        }

        // Commit only after every required allocation has succeeded, preserving
        // the existing CSR results if any allocation above throws.
        if (new_offsets) {
            delete[] r_.offsets;
            r_.offsets = new_offsets.release();
            r_.offsets_capacity = offsets_capacity;
        }
        if (new_indices) {
            delete[] r_.indices;
            r_.indices = new_indices.release();
            delete[] r_.distances;
            r_.distances = new_distances.release();
            r_.results_capacity = results_capacity;
        }
    }

    void reserve(size_t num_queries, size_t k) {
        const std::vector<size_t> counts(num_queries, k);
        reserve(counts);
    }

    // Writes the first counts[q] neighbors of each query in CSR layout.
    void assign(const svs::QueryResult<size_t>& results, std::span<const size_t> counts) {
        assert(counts.size() == results.n_queries());
        const size_t num_queries = counts.size();
        reserve(counts);

        size_t pos = 0;
        for (size_t q = 0; q < num_queries; ++q) {
            r_.offsets[q] = pos;
            for (size_t j = 0; j < counts[q]; ++j, ++pos) {
                r_.indices[pos] = results.index(q, j);
                r_.distances[pos] = results.distance(q, j);
            }
        }
        r_.offsets[num_queries] = pos;
        r_.num_queries = num_queries;
        r_.total_results = pos;
    }

    // Writes one neighbor list per query in CSR layout.
    void assign(std::span<const std::vector<svs::Neighbor<size_t>>> neighbors) {
        const size_t num_queries = neighbors.size();
        std::vector<size_t> counts(num_queries);
        for (size_t q = 0; q < num_queries; ++q) {
            counts[q] = neighbors[q].size();
        }
        reserve(counts);

        size_t pos = 0;
        for (size_t q = 0; q < num_queries; ++q) {
            r_.offsets[q] = pos;
            for (const auto& neighbor : neighbors[q]) {
                r_.indices[pos] = neighbor.id();
                r_.distances[pos] = neighbor.distance();
                ++pos;
            }
        }
        r_.offsets[num_queries] = pos;
        r_.num_queries = num_queries;
        r_.total_results = pos;
    }

    // Releases library-owned buffers; caller-owned pointers and capacities are kept.
    void reset() noexcept {
        r_.num_queries = 0;
        r_.total_results = 0;
        if (!r_.owns_buffers) {
            return;
        }
        delete[] r_.offsets;
        delete[] r_.indices;
        delete[] r_.distances;
        r_.offsets = nullptr;
        r_.indices = nullptr;
        r_.distances = nullptr;
        r_.offsets_capacity = 0;
        r_.results_capacity = 0;
        r_.owns_buffers = false;
    }

  private:
    svs_search_results_t& r_;
};

} // namespace svs::c_runtime
