/*
 * Copyright (C) 2023 Intel Corporation
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

#include "svs/concepts/data.h"
#include "svs/concepts/distance.h"
#include "svs/concepts/graph.h"
#include "svs/index/vamana/search_buffer.h"

#include <algorithm>
#include <memory>

namespace svs::index::vamana {

// Include a stat - tracker API.
class NullTracker {
  public:
    NullTracker() = default;

    template <typename I>
    void visited(Neighbor<I> /*neighbor*/, size_t /*num_distance_computations*/) {}
};

template <typename T, typename I>
concept GreedySearchTracker =
    requires(T tracker, Neighbor<I> neighbor, size_t num_distance_computations) {
        tracker.visited(neighbor, num_distance_computations);
    };

struct GreedySearchPrefetchParameters {
  public:
    /// The number of iterations to prefetch ahead.
    size_t lookahead{4};
    /// The number of neighbors to prefetch at a time until `lookahead` is reached.
    size_t step{2};
};

/////
///// Greedy Search
/////

// Default builder for generating neighbor elements.
// Alternative builders that return some builder-like object are supported to enable
// alternative search buffers to be used.
struct NeighborBuilder {
    template <typename I>
    constexpr SearchNeighbor<I> operator()(I i, float distance) const {
        return SearchNeighbor<I>{i, distance};
    }
};

template <
    graphs::ImmutableMemoryGraph Graph,
    data::ImmutableMemoryDataset Dataset,
    data::AccessorFor<Dataset> Accessor,
    typename QueryType,
    distance::Distance<QueryType, typename Dataset::const_value_type> Dist,
    typename Buffer,
    typename Ep,
    typename Builder,
    GreedySearchTracker<typename Graph::index_type> Tracker>
void greedy_search(
    const Graph& graph,
    const Dataset& dataset,
    Accessor& accessor,
    const QueryType& query,
    Dist& distance_function,
    Buffer& search_buffer,
    const Ep& entry_points,
    const Builder& builder,
    Tracker& search_tracker,
    GreedySearchPrefetchParameters prefetch_parameters = {}
) {
    using I = typename Graph::index_type;

    // Fix the query if needed by the distance function.
    distance::maybe_fix_argument(distance_function, query);

    // Initialize entry points.
    for (const auto& id : entry_points) {
        accessor.prefetch(dataset, id);
    }

    // Populate initial points.
    search_buffer.clear();
    for (const auto& id : entry_points) {
        auto dist = distance::compute(distance_function, query, accessor(dataset, id));
        search_buffer.push_back(builder(id, dist));
        graph.prefetch_node(id);
        search_tracker.visited(Neighbor<I>{id, dist}, 1);
    }

    // Main search routine.
    search_buffer.sort();
    while (!search_buffer.done()) {
        // Get the next unvisited vertex.
        const auto& node = search_buffer.next();
        auto node_id = node.id();

        // Get the adjacency list for this vertex and prepare prefetching logic.
        auto neighbors = graph.get_node(node_id);
        const size_t num_neighbors = neighbors.size();
        search_tracker.visited(Neighbor<I>{node}, neighbors.size());

        auto prefetcher = lib::make_prefetcher(
            lib::PrefetchParameters{
                prefetch_parameters.lookahead, prefetch_parameters.step},
            num_neighbors,
            [&](size_t i) { accessor.prefetch(dataset, neighbors[i]); },
            [&](size_t i) {
                // Perform the visited set enabled check just once.
                if (search_buffer.visited_set_enabled()) {
                    // Prefetch next bucket so it's (hopefully) in the cache when we next
                    // consult the visited filter.
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
            auto dist = distance::compute(distance_function, query, accessor(dataset, id));
            search_buffer.insert(builder(id, dist));
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
    typename Ep,
    typename Builder = NeighborBuilder>
void greedy_search(
    const Graph& graph,
    const Dataset& dataset,
    Accessor& accessor,
    QueryType query,
    Dist& distance_function,
    Buffer& search_buffer,
    const Ep& entry_points,
    const Builder& builder = NeighborBuilder(),
    GreedySearchPrefetchParameters prefetch_parameters = {}
) {
    auto null_tracker = NullTracker{};
    greedy_search(
        graph,
        dataset,
        accessor,
        query,
        distance_function,
        search_buffer,
        entry_points,
        builder,
        null_tracker,
        prefetch_parameters
    );
}
} // namespace svs::index::vamana
