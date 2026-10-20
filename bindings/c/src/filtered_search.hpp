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
#include <svs/index/vamana/search_buffer.h>
#include <svs/lib/misc.h>
#include <svs/lib/preprocessor.h>
#include <svs/lib/threads.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

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
/// @param measured_pass_rate If not null, receives the sampled pass rate (hits / sampled
/// ids; 0 if nothing was sampled). The sample is taken even when `filter_rate()` is 0.
/// @return The estimated initial batch size, or 0 if the hit rate is insufficient.
inline size_t estimate_initial_batch_size(
    const IDFilterInterface* id_filter,
    std::function<size_t()> sample_generator,
    size_t min_sample_size,
    size_t goal,
    size_t hint,
    size_t limit,
    float* measured_pass_rate = nullptr
) {
    assert(id_filter != nullptr);
    const auto filter_rate = id_filter->filter_rate();
    if (measured_pass_rate != nullptr) {
        *measured_pass_rate = 0.0f;
    }

    auto sample_size = std::max(goal, min_sample_size);
    if (filter_rate > 0.0f) {
        sample_size = std::max(sample_size, static_cast<size_t>(1.f / filter_rate));
    }
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
    if (measured_pass_rate != nullptr && sample_size > 0) {
        *measured_pass_rate = static_cast<float>(hits) / static_cast<float>(sample_size);
    }
    if (filter_rate <= 0.0f) {
        // No filter rate estimate from the caller: use the `hint` as the initial batch
        // size - clamped to `limit` to avoid oversizing.
        return std::min(hint, limit);
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

// Perform a filtered nearest-neighbor search. The pass rate of the filter is measured on a
// sample of ids, then:
//  * pass rate >= GREEDY_MIN_PASS_RATE: one plain greedy search per query for
//    max(hint, GREEDY_OVERSAMPLE * K / pass rate) neighbors, filtered in distance order.
//    Queries that end up with fewer than K hits are redone with the batch loop below.
//  * otherwise: iterate over batches of candidates and keep only those that pass the
//    filter. The batch size is estimated adaptively based on the observed hit rate, and the
//    iterator gets extra search buffer capacity that grows as the pass rate falls (60 to
//    200 slots).
// Results are written into `results`.
// returns a vector containing the number of neighbors found for each query.
template <typename IndexType, typename SearchParams>
std::vector<size_t> filtered_topk_search(
    IndexType& index,
    const SearchParams& params,
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
    // Sampled pass rate from which the greedy search is tried first.
    const float GREEDY_MIN_PASS_RATE = 0.5f;
    // The greedy search asks for this many times the expected number of candidates.
    const float GREEDY_OVERSAMPLE = 1.2f;
    // Extra search buffer capacity of the batch iterator (library default is 100):
    // ceil(BATCH_EXTRA_PER_K * K / pass rate), clamped to [MIN, MAX]; MAX if nothing
    // passed. With MIN < 55, a 10k-vector index at window 100 and 10 % pass rate lost
    // recall.
    const float BATCH_EXTRA_PER_K = 0.2f;
    const size_t BATCH_EXTRA_MIN = 60;
    const size_t BATCH_EXTRA_MAX = 200;
    // Upper bound on neighbor slots of one greedy search call (limits memory).
    const size_t GREEDY_MAX_SLOTS = size_t{1} << 22;

    assert(results.n_queries() == queries.size());

    // Filtered search: we need to estimate the batch size based on the filter rate and
    // the number of hits
    const auto num_neighbors = results.n_neighbors();
    const auto index_size = index.size();
    std::vector<size_t> found_neighbors_per_query(queries.size(), 0);

    float pass_rate = 0.0f;
    auto initial_batch_size = estimate_initial_batch_size(
        id_filter,
        sample_generator,
        MIN_SAMPLE_SIZE,
        num_neighbors,
        initial_batch_hint,
        index_size,
        &pass_rate
    );
    if (initial_batch_size == 0) {
        // If the batch size is 0, it means that the filter rate is too low than
        // expected and we should not even start the search
        set_empty_result(results);
        return found_neighbors_per_query;
    }

    const auto filter_rate = id_filter->filter_rate();
    const bool greedy_first = pass_rate >= GREEDY_MIN_PASS_RATE;
    const size_t batch_extra_slots =
        pass_rate > 0.0f
            ? std::clamp(
                  static_cast<size_t>(std::ceil(
                      BATCH_EXTRA_PER_K * static_cast<float>(num_neighbors) / pass_rate
                  )),
                  BATCH_EXTRA_MIN,
                  BATCH_EXTRA_MAX
              )
            : BATCH_EXTRA_MAX;

    // Batch iterator + filter loop for query `i`; returns the number of neighbors found.
    auto batch_search = [&](size_t i, size_t extra_slots) {
        auto query = queries.get_datum(i);
        auto iterator = index.batch_iterator(query, extra_slots);
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
        return found;
    };

    if (!greedy_first) {
        auto search_closure = [&](const auto& range, uint64_t SVS_UNUSED(tid)) {
            for (auto i : range) {
                found_neighbors_per_query[i] = batch_search(i, batch_extra_slots);
            }
        };
        svs::threads::parallel_for(
            index.get_threadpool_handle(),
            svs::threads::StaticPartition{queries.size()},
            search_closure
        );
        return found_neighbors_per_query;
    }

    // Window and capacity both equal the number of neighbors asked for.
    const auto greedy_n = std::min(
        std::max(
            initial_batch_hint,
            static_cast<size_t>(
                std::ceil(GREEDY_OVERSAMPLE * static_cast<float>(num_neighbors) / pass_rate)
            )
        ),
        index_size
    );
    auto greedy_params = params;
    greedy_params.buffer_config_ = svs::index::vamana::SearchBufferConfig{greedy_n};
    const size_t chunk =
        std::max(GREEDY_MAX_SLOTS / std::max(greedy_n, size_t{1}), size_t{1});
    const size_t dim = queries.dimensions();

    for (size_t first = 0; first < queries.size(); first += chunk) {
        const size_t count = std::min(chunk, queries.size() - first);
        auto greedy = svs::QueryResult<size_t>(count, greedy_n);
        // Parallelized by the index's own thread pool.
        index.search(
            greedy.view(),
            svs::data::ConstSimpleDataView<float>(queries.data() + first * dim, count, dim),
            greedy_params
        );

        auto filter_closure = [&](const auto& range, uint64_t SVS_UNUSED(tid)) {
            for (auto c : range) {
                const auto i = first + c;
                size_t found = 0;
                for (size_t j = 0; j < greedy_n && found < num_neighbors; ++j) {
                    const auto id = greedy.index(c, j);
                    if (id != static_cast<size_t>(-1) && id_filter->is_member(id)) {
                        results.set(
                            svs::Neighbor<size_t>{id, greedy.distance(c, j)}, i, found
                        );
                        found++;
                    }
                }
                if (found < num_neighbors) {
                    // Not enough hits: redo this query with the batch loop.
                    found = batch_search(i, svs::UNSIGNED_INTEGER_PLACEHOLDER);
                } else {
                    pad_result(results, i, found);
                }
                found_neighbors_per_query[i] = found;
            }
        };
        svs::threads::parallel_for(
            index.get_threadpool_handle(),
            svs::threads::StaticPartition{count},
            filter_closure
        );
    }
    return found_neighbors_per_query;
}

} // namespace svs::c_runtime
