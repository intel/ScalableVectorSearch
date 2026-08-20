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

#include "svs/concurrent/graph.h"

#include "svs/concepts/data.h"
#include "svs/concepts/distance.h"

// ``distance::compute`` is called with a *qualified* name from both this header and
// upstream's ``greedy_search.h``, so the concrete overloads must be declared before either
// definition is parsed -- qualified lookup in a template happens at the point of
// definition, not instantiation. Upstream's ``dynamic_index.h`` gets this for free because
// its alphabetically-sorted include list puts ``core/distance.h`` above
// ``index/vamana/greedy_search.h``; being explicit here removes the ordering hazard.
#include "svs/core/distance.h"

#include "svs/index/vamana/greedy_search.h"
#include "svs/lib/spinlock.h" // svs::detail::pause

#include <algorithm>
#include <optional>

namespace svs::concurrent {

// Reuse upstream's trackers, prefetch parameters, initializer and neighbor builder
// verbatim -- only the node-expansion loop needs to change.
using svs::index::vamana::EntryPointInitializer;
using svs::index::vamana::GreedySearchPrefetchParameters;
using svs::index::vamana::GreedySearchTracker;
using svs::index::vamana::NeighborBuilder;
using svs::index::vamana::NullTracker;

///
/// @brief Greedy graph search that tolerates a concurrent writer.
///
/// Behaviourally identical to ``svs::index::vamana::greedy_search``; the only difference
/// is that each node expansion is wrapped in a sequence-lock read section:
///
/// 1. ``read_begin`` -- bail out and spin if a write to this node is in flight.
/// 2. Read the adjacency list through an ``AtomicSpan`` (relaxed atomic loads, which
///    compile to plain ``MOV`` on x86) and expand its neighbors.
/// 3. ``read_validate`` -- if the writer touched this node meanwhile, retry the
///    expansion.
///
/// Retrying is safe rather than merely tolerable: a rejected read can only have inserted
/// neighbors with *valid* IDs and *correctly computed* distances into the search buffer
/// (the graph never publishes a degree covering an unwritten slot), and
/// ``search_buffer.insert`` deduplicates by ID. So a retry can add redundant work but
/// cannot corrupt the result.
///
/// This lives in ``svs::concurrent`` and is a separate routine from upstream's, so
/// ``svs/index/vamana/greedy_search.h`` -- shared with the *static* Vamana index -- is
/// left completely untouched and pays nothing for this feature.
///
template <
    std::unsigned_integral Idx,
    svs::data::ImmutableMemoryDataset Dataset,
    svs::data::AccessorFor<Dataset> Accessor,
    typename QueryType,
    svs::distance::Distance<QueryType, typename Dataset::const_value_type> Dist,
    typename Buffer,
    typename Initializer,
    typename Builder,
    GreedySearchTracker<Idx> Tracker>
void seqlock_greedy_search(
    const SeqLockGraph<Idx>& graph,
    const Dataset& dataset,
    Accessor& accessor,
    const QueryType& query,
    Dist& distance_function,
    Buffer& search_buffer,
    const Initializer& initializer,
    const Builder& builder,
    Tracker& search_tracker,
    GreedySearchPrefetchParameters prefetch_parameters = {},
    const svs::lib::DefaultPredicate& cancel = svs::lib::Returns(svs::lib::Const<false>())
) {
    using I = Idx;

    // Fix the query if needed by the distance function.
    svs::distance::maybe_fix_argument(distance_function, query);

    // Initialize the search buffer.
    {
        auto computer = [&](std::integral auto id) {
            return svs::distance::compute(distance_function, query, accessor(dataset, id));
        };
        initializer(search_buffer, computer, graph, builder, search_tracker);
    }

    // Main search routine.
    while (!search_buffer.done()) {
        // Check if request to cancel the search
        if (cancel()) {
            return;
        }
        // Get the next unvisited vertex.
        //
        // Copy it out by value rather than holding the reference: unlike upstream, the
        // expansion below can run more than once, and the ``search_buffer.insert`` calls
        // it performs may reorder the buffer and invalidate a reference into it.
        const auto tracked = svs::Neighbor<I>{search_buffer.next()};
        const auto node_id = tracked.id();

        const auto& node_seqlock = graph.seqlock(node_id);

        for (;;) { // Sequence-lock read section.
            auto maybe_seq = node_seqlock.read_begin();
            if (!maybe_seq) {
                // A write to this node is in flight; wait for it to land.
                svs::detail::pause();
                continue;
            }

            // Get the adjacency list for this vertex and prepare prefetching logic.
            auto neighbors = graph.get_node_atomic(node_id);
            const size_t num_neighbors = neighbors.size();
            search_tracker.visited(tracked, num_neighbors);

            auto prefetcher = svs::lib::make_prefetcher(
                svs::lib::PrefetchParameters{
                    prefetch_parameters.lookahead, prefetch_parameters.step},
                num_neighbors,
                [&](size_t i) { accessor.prefetch(dataset, neighbors[i]); },
                [&](size_t i) {
                    // Perform the visited set enabled check just once.
                    if (search_buffer.visited_set_enabled()) {
                        // Prefetch next bucket so it's (hopefully) in the cache when we
                        // next consult the visited filter.
                        if (i + 1 < num_neighbors) {
                            search_buffer.unsafe_prefetch_visited(neighbors[i + 1]);
                        }
                        return !search_buffer.unsafe_is_visited(neighbors[i]);
                    }

                    // Otherwise, always prefetch the next data item.
                    return true;
                }
            );

            ///// Neighbor expansion.
            prefetcher();
            for (auto id : neighbors) {
                if (search_buffer.emplace_visited(id)) {
                    continue;
                }

                // Run the prefetcher.
                prefetcher();

                // Compute distance and update search buffer.
                auto dist =
                    svs::distance::compute(distance_function, query, accessor(dataset, id));
                search_buffer.insert(builder(id, dist));
            }

            if (node_seqlock.read_validate(*maybe_seq)) {
                break; // Consistent read -- move on to the next node.
            }
            // The writer modified this node mid-expansion. Anything we inserted is valid
            // but possibly stale, so re-expand to pick up the current adjacency list.
            svs::detail::pause();
        }
    }
}

/// @brief Overload supplying a default (null) search tracker.
template <
    std::unsigned_integral Idx,
    svs::data::ImmutableMemoryDataset Dataset,
    svs::data::AccessorFor<Dataset> Accessor,
    typename QueryType,
    svs::distance::Distance<QueryType, typename Dataset::const_value_type> Dist,
    typename Buffer,
    typename Initializer,
    typename Builder = NeighborBuilder>
void seqlock_greedy_search(
    const SeqLockGraph<Idx>& graph,
    const Dataset& dataset,
    Accessor& accessor,
    QueryType query,
    Dist& distance_function,
    Buffer& search_buffer,
    const Initializer& initializer,
    const Builder& builder = NeighborBuilder(),
    GreedySearchPrefetchParameters prefetch_parameters = {},
    const svs::lib::DefaultPredicate& cancel = svs::lib::Returns(svs::lib::Const<false>())
) {
    auto null_tracker = NullTracker{};
    seqlock_greedy_search(
        graph,
        dataset,
        accessor,
        query,
        distance_function,
        search_buffer,
        initializer,
        builder,
        null_tracker,
        prefetch_parameters,
        cancel
    );
}

} // namespace svs::concurrent
