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

// Single-pass filtered top-k search for the dynamic Vamana index, following hnswlib's
// filtered `searchBaseLayerST`.
//
// - `candidates`: min-heap of nodes still to expand.
// - `results`: max-heap of nodes that pass the filter and are not deleted, capped at
//   `ef`. Nodes that fail are expanded but never enter `results`.
// - The search stops when the best candidate is worse than the worst result and
//   `results` is full.
// - The visited set is exact: one tag per slot, reset per query by a generation counter.
#pragma once

#include "svs/concepts/data.h"
#include "svs/concepts/distance.h"
#include "svs/core/data/simple.h"
#include "svs/core/query_result.h"
#include "svs/index/vamana/extensions.h"
#include "svs/index/vamana/greedy_search.h"
#include "svs/lib/neighbor.h"
#include "svs/lib/prefetch.h"
#include "svs/lib/threads.h"
#include "svs/quantization/scalar/scalar.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace svs::index::vamana {
namespace detail {

// Per-thread scratch reused across queries.
template <typename Idx> struct TwoHeapScratch {
    // `tags[slot] == generation` means the slot was visited by the current query.
    std::vector<uint32_t> tags;
    uint32_t generation = 0;
    std::vector<std::pair<float, Idx>> candidates;
    std::vector<std::pair<float, Idx>> results;
};

template <typename Idx> TwoHeapScratch<Idx>& two_heap_scratch() {
    thread_local TwoHeapScratch<Idx> scratch;
    return scratch;
}

// Searches one query. Writes up to `output.n_neighbors()` neighbors (external ids) into
// row `query_index` of `output` and returns how many were written.
template <
    typename Index,
    typename Graph,
    typename Data,
    typename Accessor,
    typename DistFn,
    typename Query,
    typename Pred>
size_t two_heap_search_one(
    const Index& index,
    const Graph& graph,
    const Data& data,
    Accessor& accessor,
    DistFn& distance,
    std::span<const typename Graph::index_type> entry_points,
    const Query& query,
    size_t ef,
    const Pred& pred,
    GreedySearchPrefetchParameters prefetch_parameters,
    QueryResultView<size_t>& output,
    size_t query_index
) {
    using Idx = typename Graph::index_type;
    using Entry = std::pair<float, Idx>;

    auto& scratch = two_heap_scratch<Idx>();
    auto& candidates = scratch.candidates;
    auto& results = scratch.results;
    candidates.clear();
    results.clear();

    // Start a new generation; on wrap-around, clear all tags.
    if (scratch.tags.size() < data.size()) {
        scratch.tags.resize(data.size(), 0);
    }
    if (++scratch.generation == 0) {
        std::fill(scratch.tags.begin(), scratch.tags.end(), 0);
        scratch.generation = 1;
    }
    uint32_t* tags = scratch.tags.data();
    const uint32_t generation = scratch.generation;

    distance::maybe_fix_argument(distance, query);
    // `cmp(a, b)` is true if distance `a` is strictly better than `b`.
    // std heaps keep the "largest" element on top: best candidate, worst result.
    const auto cmp = distance::comparator(distance);
    const auto candidate_less = [&](const Entry& a, const Entry& b) {
        return cmp(b.first, a.first);
    };
    const auto result_less = [&](const Entry& a, const Entry& b) {
        return cmp(a.first, b.first);
    };

    // Check deletion first: deleted slots may have no external id.
    const auto passes = [&](Idx i) {
        return !index.is_deleted(i) && pred(index.translate_internal_id(i));
    };
    const auto compute = [&](Idx i) {
        return static_cast<float>(distance::compute(distance, query, accessor(data, i)));
    };
    // Keep `results` at most `ef` long by dropping the worst.
    const auto add_result = [&](float dist, Idx i) {
        results.emplace_back(dist, i);
        std::push_heap(results.begin(), results.end(), result_less);
        if (results.size() > ef) {
            std::pop_heap(results.begin(), results.end(), result_less);
            results.pop_back();
        }
    };

    // Entry points are always expanded; they are results only if they pass.
    for (Idx ep : entry_points) {
        tags[ep] = generation;
        float dist = compute(ep);
        candidates.emplace_back(dist, ep);
        std::push_heap(candidates.begin(), candidates.end(), candidate_less);
        if (passes(ep)) {
            add_result(dist, ep);
        }
    }

    while (!candidates.empty()) {
        const auto [candidate_dist, candidate_id] = candidates.front();
        // Best candidate is worse than the worst of a full result set.
        if (results.size() == ef && cmp(results.front().first, candidate_dist)) {
            break;
        }
        std::pop_heap(candidates.begin(), candidates.end(), candidate_less);
        candidates.pop_back();

        auto neighbors = graph.get_node(candidate_id);
        const size_t num_neighbors = neighbors.size();
        // Same prefetching scheme as `greedy_search`.
        auto prefetcher = lib::make_prefetcher(
            lib::PrefetchParameters{
                prefetch_parameters.lookahead, prefetch_parameters.step},
            num_neighbors,
            [&](size_t i) { accessor.prefetch(data, neighbors[i]); },
            [&](size_t i) {
                if (i + 1 < num_neighbors) {
                    lib::prefetch_l0(tags + neighbors[i + 1]);
                }
                return tags[neighbors[i]] != generation;
            }
        );

        prefetcher();
        for (auto id : neighbors) {
            if (tags[id] == generation) {
                continue;
            }
            tags[id] = generation;
            prefetcher();

            float dist = compute(id);
            // Only nodes that could improve `results` are queued, and only those
            // evaluate the filter.
            if (results.size() < ef || cmp(dist, results.front().first)) {
                candidates.emplace_back(dist, id);
                std::push_heap(candidates.begin(), candidates.end(), candidate_less);
                if (passes(id)) {
                    add_result(dist, id);
                }
            }
        }
    }

    // Best result first.
    std::sort_heap(results.begin(), results.end(), result_less);
    const size_t found = std::min(results.size(), output.n_neighbors());
    for (size_t j = 0; j < found; ++j) {
        output.set(
            Neighbor<size_t>{
                index.translate_internal_id(results[j].second), results[j].first},
            query_index,
            j
        );
    }
    return found;
}

} // namespace detail

/// Data types for which `two_heap_filtered_search` is valid: the default search path
/// (identity accessor, no reranking) must apply.
template <typename Data> inline constexpr bool two_heap_supported = false;
template <typename T, size_t Extent, typename Alloc>
inline constexpr bool two_heap_supported<data::SimpleData<T, Extent, Alloc>> = true;
template <typename T, size_t Extent, typename Alloc>
inline constexpr bool
    two_heap_supported<quantization::scalar::SQDataset<T, Extent, Alloc>> = true;

///
/// @brief Filtered top-k search over a dynamic Vamana index.
///
/// @param index The `MutableVamanaIndex` to search.
/// @param output Row `i` receives the neighbors (external ids) of query `i`. Rows with
///     fewer than `n_neighbors` results are not padded.
/// @param queries One datum per query.
/// @param ef Result-set capacity (the search window). Raised to `n_neighbors` if smaller.
/// @param pred Callable `bool(size_t external_id)`. Must be thread safe.
/// @param found Resized to `queries.size()`; receives the number of results per query.
/// @param prefetch_parameters Prefetch lookahead and step for neighbor vectors.
///
/// Requires `two_heap_supported<Data>`.
///
template <typename Index, typename Queries, typename Pred>
void two_heap_filtered_search(
    Index& index,
    QueryResultView<size_t> output,
    const Queries& queries,
    size_t ef,
    const Pred& pred,
    std::vector<size_t>& found,
    GreedySearchPrefetchParameters prefetch_parameters = {}
) {
    using Idx = typename Index::internal_id_type;
    static_assert(
        two_heap_supported<typename Index::data_type>,
        "Data type needs a custom search path (e.g. reranking)."
    );
    ef = std::max(ef, output.n_neighbors());
    found.assign(queries.size(), 0);
    auto& pool = index.get_threadpool_handle();

    index.experimental_escape_hatch([&](const auto& graph,
                                        const auto& data,
                                        const auto& distance,
                                        std::span<const Idx> entry_points) {
        threads::parallel_for(
            pool,
            threads::StaticPartition{queries.size()},
            [&](const auto is, uint64_t SVS_UNUSED(tid)) {
                auto dist_fn = extensions::per_thread_batch_search_setup(data, distance);
                auto accessor = data::GetDatumAccessor();
                for (auto i : is) {
                    found[i] = detail::two_heap_search_one(
                        index,
                        graph,
                        data,
                        accessor,
                        dist_fn,
                        entry_points,
                        queries.get_datum(i),
                        ef,
                        pred,
                        prefetch_parameters,
                        output,
                        i
                    );
                }
            }
        );
    });
}

} // namespace svs::index::vamana
