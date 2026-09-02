/*
 * Copyright 2023 Intel Corporation
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

#include "svs/concepts/data.h"
#include "svs/concepts/distance.h"
#include "svs/concurrent/graph_concepts.h"
#include "svs/concurrent/spinlock.h"
// For `svs::index::vamana::GreedySearchPrefetchParameters`, which this stack reuses
// unchanged -- see the using-declaration below.
#include "svs/index/vamana/greedy_search.h"
#include "svs/index/vamana/search_buffer.h"
#include "svs/lib/concurrency/seqlock.h"

#include <algorithm>
#include <memory>

namespace svs::index::vamana::concurrent {

// The greedy-search *scaffolding* -- the tracker API, the tracker concept, the entry-point
// initializer, the default neighbor builder, and the prefetch parameters -- is unchanged by
// this stack; only `greedy_search` itself gains the SeqLock retry loop. Alias the
// pre-existing entities rather than redeclaring them: a redeclaration inside `concurrent`
// would shadow the enclosing namespace's with a distinct look-alike, and values arriving
// from facilities reused verbatim (e.g. `SearchScratchspace::prefetch_parameters`, or
// `RestartInitializer`'s `NullTracker` parameter) would then fail to convert.
using svs::index::vamana::GreedySearchPrefetchParameters;
using svs::index::vamana::GreedySearchTracker;
using svs::index::vamana::NullTracker;

/////
///// Initialization Customization.
/////

using svs::index::vamana::EntryPointInitializer;

/////
///// Greedy Search
/////

using svs::index::vamana::NeighborBuilder;

template <
    graphs::ImmutableMemoryGraph Graph,
    data::ImmutableMemoryDataset Dataset,
    data::AccessorFor<Dataset> Accessor,
    typename QueryType,
    distance::Distance<QueryType, typename Dataset::const_value_type> Dist,
    typename Buffer,
    typename Initializer,
    typename Builder,
    GreedySearchTracker<typename Graph::index_type> Tracker>
void greedy_search(
    const Graph& graph,
    const Dataset& dataset,
    Accessor& accessor,
    const QueryType& query,
    Dist& distance_function,
    Buffer& search_buffer,
    const Initializer& initializer,
    const Builder& builder,
    Tracker& search_tracker,
    GreedySearchPrefetchParameters prefetch_parameters = {},
    const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
) {
    using I = typename Graph::index_type;

    // Fix the query if needed by the distance function.
    distance::maybe_fix_argument(distance_function, query);

    // Initialize the search buffer.
    {
        // A lambda that wraps the distance computation to avoid propagating everything
        // into the initializer.
        auto computer = [&](std::integral auto id) {
            return distance::compute(distance_function, query, accessor(dataset, id));
        };
        initializer(search_buffer, computer, graph, builder, search_tracker);
    }

    // Main search routine.
    while (!search_buffer.done()) {
        // Check if request to cancel the search
        if (cancel()) {
            return;
        }
        // Get the next unvisited vertex. Copy rather than bind a reference: the
        // SeqLock retry loop below can re-enter after `search_buffer.insert()` has
        // reallocated the candidates buffer, which would dangle a reference here.
        const auto node = search_buffer.next();
        auto node_id = node.id();

        for (;;) { // SeqLock retry loop
            auto maybe_seq = graph.seq_counters()[node_id].read_begin();
            if (!maybe_seq) {
                svs::detail::pause();
                continue;
            }

            // Get the adjacency list for this vertex and prepare prefetching logic.
            auto neighbors = graph.get_node(node_id);
            const size_t num_neighbors = neighbors.size();
            search_tracker.visited(Neighbor<I>{node}, num_neighbors);

            auto prefetcher = lib::make_prefetcher(
                lib::PrefetchParameters{
                    prefetch_parameters.lookahead, prefetch_parameters.step},
                num_neighbors,
                [&](size_t i) { accessor.prefetch(dataset, neighbors[i]); },
                [&](size_t i) {
                    // Perform the visited set enabled check just once.
                    if (search_buffer.visited_set_enabled()) {
                        // Prefetch next bucket so it's (hopefully) in the cache when
                        // we next consult the visited filter.
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
                    distance::compute(distance_function, query, accessor(dataset, id));
                search_buffer.insert(builder(id, dist));
            }

            // Validate that no concurrent write occurred during the read.
            if (graph.seq_counters()[node_id].read_validate(*maybe_seq)) {
                break; // Consistent read — proceed to the next node.
            }
            svs::detail::pause();
            // Retry: stale entries from the invalid read remain in the search buffer.
            // They have valid IDs and distances, and insert() deduplicates by ID.
        }
    }
}

// Overload to provide a default search tracker because search trackers are taken by
// lvalue reference.
template <
    graphs::ImmutableMemoryGraph Graph,
    data::ImmutableMemoryDataset Dataset,
    data::AccessorFor<Dataset> Accessor,
    typename QueryType,
    distance::Distance<QueryType, typename Dataset::const_value_type> Dist,
    typename Buffer,
    typename Initializer,
    typename Builder = NeighborBuilder>
void greedy_search(
    const Graph& graph,
    const Dataset& dataset,
    Accessor& accessor,
    QueryType query,
    Dist& distance_function,
    Buffer& search_buffer,
    const Initializer& initializer,
    const Builder& builder = NeighborBuilder(),
    GreedySearchPrefetchParameters prefetch_parameters = {},
    const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
) {
    auto null_tracker = NullTracker{};
    concurrent::greedy_search(
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
} // namespace svs::index::vamana::concurrent
