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

// Repairs graph connectivity a build can break by skipping most nodes (e.g. GEM starts).
#include "svs/concepts/data.h"
#include "svs/concepts/graph.h"
#include "svs/index/vamana/build_params.h"
#include "svs/index/vamana/extensions.h"
#include "svs/index/vamana/greedy_search.h"
#include "svs/index/vamana/prune.h"
#include "svs/index/vamana/search_buffer.h"
#include "svs/lib/misc.h"
#include "svs/lib/narrow.h"
#include "svs/lib/neighbor.h"
#include "svs/lib/threads.h"

// stl
#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

namespace svs::index::vamana {

/// Breadth-first search over out-edges from `roots`; returns the node ids `roots` cannot
/// reach, ascending.
template <graphs::ImmutableMemoryGraph Graph>
std::vector<typename Graph::index_type>
unreachable_nodes(const Graph& graph, std::span<const typename Graph::index_type> roots) {
    using Idx = typename Graph::index_type;
    size_t n = graph.n_nodes();
    std::vector<bool> visited(n, false);

    std::vector<Idx> queue;
    queue.reserve(n);
    for (Idx root : roots) {
        if (!visited[root]) {
            visited[root] = true;
            queue.push_back(root);
        }
    }
    for (size_t head = 0; head < queue.size(); ++head) {
        for (Idx v : graph.get_node(queue[head])) {
            if (!visited[v]) {
                visited[v] = true;
                queue.push_back(v);
            }
        }
    }

    std::vector<Idx> result;
    for (size_t i = 0; i < n; ++i) {
        if (!visited[i]) {
            result.push_back(lib::narrow_cast<Idx>(i));
        }
    }
    return result;
}

/// Outcome of `repair_connectivity`. edges_forced_unsafe, unlike the other edge counts,
/// has no safety guarantee and can orphan the node it displaced. `stalled` means a round
/// made no progress, which stops the repair before `max_rounds`.
struct ConnectivityReport {
    size_t rounds = 0;
    size_t unreachable_before = 0;
    size_t unreachable_after = 0;
    size_t edges_appended = 0;
    size_t edges_accepted_through_prune = 0;
    size_t edges_forced = 0;
    size_t edges_forced_unsafe = 0;
    bool stalled = false;
};

/// Reconnects every node `unreachable_nodes` reports, round by round, until none remain
/// or `max_rounds` is spent. Candidates are searched in parallel per round, but applied
/// serially in ascending node-id order since accepting v for one node changes deg(v).
template <
    graphs::MemoryGraph Graph,
    data::ImmutableMemoryDataset Data,
    typename Dist,
    threads::ThreadPool Pool>
ConnectivityReport repair_connectivity(
    Graph& graph,
    const Data& data,
    const Dist& distance,
    const VamanaBuildParameters& params,
    std::span<const typename Graph::index_type> roots,
    Pool& threadpool,
    size_t max_rounds = 4
) {
    using Idx = typename Graph::index_type;

    std::vector<bool> is_root(graph.n_nodes(), false);
    for (Idx root : roots) {
        is_root[root] = true;
    }

    ConnectivityReport report{};
    auto unreachable = unreachable_nodes(graph, roots);
    report.unreachable_before = unreachable.size();

    for (size_t round = 0; round < max_rounds && !unreachable.empty(); ++round) {
        ++report.rounds;

        // Reachability snapshot at round start: proposals below are filtered against it.
        std::vector<bool> reachable(graph.n_nodes(), true);
        for (Idx u : unreachable) {
            reachable[u] = false;
        }

        // Kept current as edges change, so the forced step can detect a stranding drop.
        std::vector<size_t> reachable_in_degree(graph.n_nodes(), 0);
        for (size_t x = 0; x < graph.n_nodes(); ++x) {
            if (reachable[x]) {
                for (Idx y : graph.get_node(lib::narrow_cast<Idx>(x))) {
                    ++reachable_in_degree[y];
                }
            }
        }

        // Parallel search only; Phase B is serial since deg(v) changes within a round.
        std::vector<std::vector<Idx>> proposals(unreachable.size());
        threads::parallel_for(
            threadpool,
            threads::StaticPartition{unreachable.size()},
            [&](const auto& is, uint64_t /*tid*/) {
                auto build_adaptor = extensions::build_adaptor(data, distance);
                auto accessor = build_adaptor.graph_search_accessor();
                auto&& search_distance = build_adaptor.graph_search_distance();
                auto search_buffer =
                    SearchBuffer<Idx, distance::compare_t<Dist>>{params.window_size};
                auto tracker = NullTracker{};

                for (size_t k : is) {
                    Idx u = unreachable[k];
                    search_buffer.clear();
                    const auto& query =
                        build_adaptor.access_query_for_graph_search(data, u);
                    greedy_search(
                        graph,
                        data,
                        accessor,
                        query,
                        search_distance,
                        search_buffer,
                        EntryPointInitializer<Idx>{roots},
                        NeighborBuilder(),
                        tracker
                    );
                    search_buffer.sort();

                    auto& candidates = proposals[k];
                    for (size_t i = 0, imax = search_buffer.size(); i < imax; ++i) {
                        Idx id = search_buffer[i].id();
                        if (reachable[id]) {
                            candidates.push_back(id);
                        }
                    }
                }
            }
        );

        // Phase B (serial, ascending u).
        auto build_adaptor = extensions::build_adaptor(data, distance);
        auto accessor = build_adaptor.general_accessor();
        auto&& general_distance = build_adaptor.general_distance();
        auto cmp = distance::comparator(general_distance);

        std::vector<size_t> order(unreachable.size());
        for (size_t k = 0; k < order.size(); ++k) {
            order[k] = k;
        }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return unreachable[a] < unreachable[b];
        });

        // Safe to drop: a root (always reachable) or has another reachable in-edge.
        auto is_safe_to_drop = [&](Idx n) {
            return is_root[n] || reachable_in_degree[n] >= 2;
        };

        std::vector<Neighbor<Idx>> pool;
        std::vector<Idx> pruned;
        std::vector<Idx> old_neighbors;
        for (size_t k : order) {
            Idx u = unreachable[k];
            const auto& candidates = proposals[k];
            bool accepted = false;

            // Pass 1: append if room, else re-prune and accept only if all drops are safe.
            for (Idx v : candidates) {
                if (graph.get_node_degree(v) < params.graph_max_degree) {
                    graph.add_edge(v, u);
                    ++reachable_in_degree[u];
                    ++report.edges_appended;
                    accepted = true;
                    break;
                }

                const auto& v_data = accessor(data, v);
                distance::maybe_fix_argument(general_distance, v_data);
                pool.clear();
                old_neighbors.clear();
                for (Idx n : graph.get_node(v)) {
                    old_neighbors.push_back(n);
                    pool.emplace_back(
                        n, distance::compute(general_distance, v_data, accessor(data, n))
                    );
                }
                pool.emplace_back(
                    u, distance::compute(general_distance, v_data, accessor(data, u))
                );
                std::sort(pool.begin(), pool.end(), TotalOrder(cmp));

                heuristic_prune_neighbors(
                    prune_strategy(distance),
                    params.graph_max_degree,
                    params.alpha,
                    data,
                    accessor,
                    general_distance,
                    v,
                    lib::as_const_span(pool),
                    pruned
                );

                if (std::find(pruned.begin(), pruned.end(), u) == pruned.end()) {
                    continue;
                }

                bool all_drops_safe = true;
                for (Idx old_id : old_neighbors) {
                    bool dropped =
                        std::find(pruned.begin(), pruned.end(), old_id) == pruned.end();
                    if (dropped && !is_safe_to_drop(old_id)) {
                        all_drops_safe = false;
                        break;
                    }
                }
                if (!all_drops_safe) {
                    continue;
                }

                graph.replace_node(v, pruned);
                for (Idx old_id : old_neighbors) {
                    if (std::find(pruned.begin(), pruned.end(), old_id) == pruned.end()) {
                        --reachable_in_degree[old_id];
                    }
                }
                ++reachable_in_degree[u];
                ++report.edges_accepted_through_prune;
                accepted = true;
                break;
            }

            if (accepted) {
                continue;
            }

            // Pass 2: forced replacement of a safe neighbor; candidates is never empty.
            bool forced_safe = false;
            Idx safe_v{};
            Idx safe_w{};
            for (Idx v : candidates) {
                const auto& v_data = accessor(data, v);
                distance::maybe_fix_argument(general_distance, v_data);

                std::vector<Neighbor<Idx>> safe_candidates;
                for (Idx n : graph.get_node(v)) {
                    if (is_safe_to_drop(n)) {
                        safe_candidates.emplace_back(
                            n,
                            distance::compute(general_distance, v_data, accessor(data, n))
                        );
                    }
                }
                if (!safe_candidates.empty()) {
                    auto farthest = std::max_element(
                        safe_candidates.begin(), safe_candidates.end(), TotalOrder(cmp)
                    );
                    safe_v = v;
                    safe_w = farthest->id();
                    forced_safe = true;
                    break;
                }
            }

            // Pass 3: forces the nearest candidate regardless of safety, orphaning `w`.
            Idx v{};
            Idx w{};
            if (forced_safe) {
                v = safe_v;
                w = safe_w;
            } else {
                v = candidates.front();
                const auto& v_data = accessor(data, v);
                distance::maybe_fix_argument(general_distance, v_data);

                std::vector<Neighbor<Idx>> ranked;
                for (Idx n : graph.get_node(v)) {
                    ranked.emplace_back(
                        n, distance::compute(general_distance, v_data, accessor(data, n))
                    );
                }
                auto farthest =
                    std::max_element(ranked.begin(), ranked.end(), TotalOrder(cmp));
                w = farthest->id();
            }

            std::vector<Idx> new_neighbors;
            for (Idx n : graph.get_node(v)) {
                if (n != w) {
                    new_neighbors.push_back(n);
                }
            }
            new_neighbors.push_back(u);
            graph.replace_node(v, new_neighbors);

            --reachable_in_degree[w];
            ++reachable_in_degree[u];

            if (forced_safe) {
                ++report.edges_forced;
            } else {
                ++report.edges_forced_unsafe;
            }
        }

        size_t unreachable_before_round = unreachable.size();
        unreachable = unreachable_nodes(graph, roots);
        if (unreachable.size() >= unreachable_before_round) {
            report.stalled = true;
            break;
        }
    }
    report.unreachable_after = unreachable.size();
    return report;
}

} // namespace svs::index::vamana
