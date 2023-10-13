/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
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
    // How far from the start of the neighbor list to begin prefetching.
    size_t offset{0};
    // The number of neighbors to prefetch at a time.
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
    const Accessor& accessor,
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
    const size_t prefetch_step = prefetch_parameters.step;
    while (!search_buffer.done()) {
        // Get the next unvisited vertex.
        const auto& node = search_buffer.next();
        auto node_id = node.id();

        // Get the adjacency list for this vertex and prepare prefetching logic.
        auto neighbors = graph.get_node(node_id);
        auto prefetch_start = prefetch_parameters.offset;
        search_tracker.visited(Neighbor<I>{node}, neighbors.size());
        for (auto id : neighbors) {
            if (search_buffer.visited(id)) {
                continue;
            }

            // Prefetch loop.
            if (prefetch_start < neighbors.size()) {
                size_t upper = std::min(neighbors.size(), prefetch_start + prefetch_step);
                for (size_t i = prefetch_start; i < upper; ++i) {
                    accessor.prefetch(dataset, neighbors[i]);
                }
                prefetch_start += prefetch_step;
            }

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
    const Accessor& accessor,
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
