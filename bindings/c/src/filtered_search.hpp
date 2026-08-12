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

#include "types_support.hpp"

#include <svs/concepts/data.h>
#include <svs/core/query_result.h>
#include <svs/lib/misc.h>
#include <svs/lib/threads.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>

namespace svs::c_runtime {

/// @brief Estimate the batch size for filtered search based on the number of total
/// candidates, hits, goal, hint, and limit.
/// @param total The total number of candidates.
/// @param hits The number of filter hits.
/// @param goal The target number of hits to achieve.
/// @param hint A hint for the batch size - usually based on prior knowledge.
/// @param limit The maximum allowed batch size. E.g. index size.
/// @return The estimated batch size.
inline size_t
estimate_batch_size(size_t total, size_t hits, size_t goal, size_t hint, size_t limit) {
    assert(total >= hits);
    assert(goal > 0);
    if (total == 0 || hits == 0 || hits >= goal) {
        return std::min(hint, limit);
    }
    const auto hit_rate_inv = static_cast<float>(total) / static_cast<float>(hits);
    size_t estimated = static_cast<size_t>(static_cast<float>(goal - hits) * hit_rate_inv);
    estimated = std::max(estimated, size_t{1});
    return std::min(estimated, limit);
}

/// @brief Check if the actual hit rate is sufficient based on the minimum required filter
/// rate.
/// @param total The total number of candidates.
/// @param hits The number of filter hits.
/// @param filter_rate The minimum required filter rate.
/// @return True if the hit rate is sufficient, false otherwise.
inline bool hit_rate_sufficient(size_t total, size_t hits, float filter_rate) {
    // by default, assume that the hit rate is sufficient
    if (filter_rate <= 0.0f || total == 0) {
        return true;
    }
    const auto hit_rate = static_cast<float>(hits) / static_cast<float>(total);
    return hit_rate >= filter_rate;
}

/// @brief Estimate the initial batch size for filtered search based on the actual filter
/// rate by generating sample IDs and filtering them through the ID filter.
/// @param id_filter The ID filter interface.
/// @param sample_generator A function that generates sample IDs.
/// @param min_sample_size The minimum sample size to consider.
/// @param goal The target number of hits to achieve - usually is K (from TopK).
/// @param hint A hint for the batch size - usually based on prior knowledge.
/// @param limit The maximum allowed batch size. E.g. index size.
/// @return The estimated initial batch size, or 0 if the hit rate is insufficient.
inline size_t estimate_initial_batch_size(
    const IDFilterInterface* id_filter,
    std::function<size_t()> sample_generator,
    size_t min_sample_size,
    size_t goal,
    size_t hint,
    size_t limit
) {
    assert(id_filter != nullptr);
    const auto filter_rate = id_filter->filter_rate();
    if (filter_rate <= 0.0f) {
        // If filter rate is 0.0 or negative, return the `hint` as the initial batch size -
        // clamped to `limit` to avoid oversizing.
        return std::min(hint, limit);
    }

    auto sample_size =
        std::max({goal, min_sample_size, static_cast<size_t>(1.f / filter_rate)});
    assert(sample_size > 0);
    size_t hits = 0;
    for (size_t i = 0; i < sample_size; ++i) {
        size_t id = sample_generator();
        // Stop if the sample generator returns an invalid ID
        if (id == static_cast<size_t>(-1)) {
            sample_size = i; // Adjust sample size to the number of valid samples
            break;
        }
        if (id_filter->is_member(id)) {
            hits++;
        }
    }
    // if hit rate is less than filter_rate, return 0 - which means we should not even start
    // the search
    if (!hit_rate_sufficient(sample_size, hits, filter_rate)) {
        return 0;
    }
    return estimate_batch_size(sample_size, hits, goal, hint, limit);
}

/// @brief Pad the result with empty neighbors starting from a specific index.
/// @param result The query result to pad.
/// @param query_index The index of the query within the result.
/// @param neighbor_start The starting index of neighbors to pad.
inline void
pad_result(svs::QueryResult<size_t>& result, size_t query_index, size_t neighbor_start) {
    assert(query_index < result.n_queries());
    assert(neighbor_start <= result.n_neighbors());

    static constexpr svs::Neighbor<size_t> empty_neighbor{
        static_cast<size_t>(-1), std::numeric_limits<float>::infinity()};

    for (size_t i = neighbor_start; i < result.n_neighbors(); ++i) {
        result.set(empty_neighbor, query_index, i);
    }
}

/// @brief Set the query result to an empty state, with all distances set to infinity and
/// all indices set to -1.
/// @param result The query result to set as empty.
inline void set_empty_result(svs::QueryResult<size_t>& result) {
    std::fill(
        result.distances().begin(),
        result.distances().end(),
        std::numeric_limits<float>::infinity()
    );
    std::fill(result.indices().begin(), result.indices().end(), static_cast<size_t>(-1));
}

// Perform a filtered nearest-neighbor search by iterating over batches of candidates and
// keeping only those that pass the filter. The batch size is estimated adaptively based on
// the observed hit rate. Results are written into `results`.
template <typename IndexType>
void filtered_topk_search(
    IndexType& index,
    svs::QueryResult<size_t>& results,
    svs::data::ConstSimpleDataView<float> queries,
    size_t initial_batch_hint,
    const IDFilterInterface* id_filter,
    const std::function<size_t()>& sample_generator
) {
    // Minimum number of samples to estimate the filter hit rate. This is a trade-off
    // between accuracy and performance. A larger sample size gives a more accurate
    // estimate of the filter hit rate, but takes longer to compute.
    const size_t MIN_SAMPLE_SIZE = 200;

    // Filtered search: we need to estimate the batch size based on the filter rate and
    // the number of hits
    const auto num_neighbors = results.n_neighbors();
    const auto index_size = index.size();

    auto initial_batch_size = estimate_initial_batch_size(
        id_filter,
        sample_generator,
        MIN_SAMPLE_SIZE,
        num_neighbors,
        initial_batch_hint,
        index_size
    );
    if (initial_batch_size == 0) {
        // If the batch size is 0, it means that the filter rate is too low than
        // expected and we should not even start the search
        set_empty_result(results);
        return;
    }

    const auto filter_rate = id_filter->filter_rate();

    auto search_closure = [&](const auto& range, uint64_t SVS_UNUSED(tid)) {
        for (auto i : range) {
            auto query = queries.get_datum(i);
            auto iterator = index.batch_iterator(query);
            size_t found = 0;
            size_t total_checked = 0;
            auto batch_size = initial_batch_size;
            do {
                batch_size = estimate_batch_size(
                    total_checked, found, num_neighbors, batch_size, index_size
                );
                iterator.next(batch_size);
                total_checked += iterator.size();
                for (auto& neighbor : iterator.results()) {
                    if (id_filter->is_member(neighbor.id())) {
                        results.set(neighbor, i, found);
                        found++;
                        if (found == num_neighbors) {
                            break;
                        }
                    }
                }
                // TODO: clarify the contract here - should we return partial or no
                // result if the hit rate is too low
                if (found < num_neighbors &&
                    !hit_rate_sufficient(total_checked, found, filter_rate)) {
                    found = 0;
                    break;
                }
            } while (found < num_neighbors && !iterator.done());

            // Pad results if not enough neighbors found
            pad_result(results, i, found);
        }
    };

    svs::threads::parallel_for(
        index.get_threadpool_handle(),
        svs::threads::StaticPartition{queries.size()},
        search_closure
    );
}

} // namespace svs::c_runtime
