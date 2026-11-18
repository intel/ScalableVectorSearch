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

// Primal network simplex for the dense balanced transportation problem, scalar and
// unconditionally built: the AVX_AVAILABILITY::NONE path and the test oracle that
// TransportSimplex's SIMD-pricing variants are checked against. Algorithm: Ahuja,
// Magnanti & Orlin, "Network Flows" (1993), ch. 11 (artificial-root initialization and
// strongly feasible spanning trees); Cunningham, "A network simplex method",
// Mathematical Programming 11 (1976), for the leaving-arc rule that keeps a tree
// strongly feasible under degenerate pivots. Problem shape (quantized ground distance
// over merged centroid weights) follows Bonneel et al., "Displacement Interpolation
// Using Lagrangian Mass Transport", SIGGRAPH Asia 2011.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace svs::distance::transport_reference_detail {

// Copies supply/demand into the solver's own buffers, checks total balance to within
// 1e-6*max(total_supply, 1), and folds a smaller residual into the last demand entry;
// throws std::runtime_error on a larger mismatch or a fold that leaves that entry
// non-positive. Shared by TransportSimplexReference and TransportSimplex (called before
// either orients rows/columns, so the fold always lands on the caller's own last demand
// entry, never its supply) so the input contract cannot drift between them. `Caller`
// (TransportSimplexReference, or TransportSimplex<Avx>::Impl<MaxSide>) is a tag only, so
// each -march TU instantiates its own copy instead of sharing one weak symbol.
template <typename Caller>
inline double prepare_marginals(
    size_t ua,
    const double* supply,
    double* supply_out,
    size_t ub,
    const double* demand,
    double* demand_out
) {
    double total_s = 0.0, total_d = 0.0;
    for (size_t i = 0; i < ua; ++i) {
        supply_out[i] = supply[i];
        total_s += supply[i];
    }
    for (size_t j = 0; j < ub; ++j) {
        demand_out[j] = demand[j];
        total_d += demand[j];
    }
    double tol = 1e-6 * std::max(total_s, 1.0);
    double diff = total_s - total_d;
    if (diff > tol || diff < -tol) {
        throw std::runtime_error(
            "TransportSimplex::solve: supply (" + std::to_string(total_s) +
            ") and demand (" + std::to_string(total_d) +
            ") are not balanced for ua=" + std::to_string(ua) + ", ub=" + std::to_string(ub)
        );
    }
    double last_demand = demand_out[ub - 1];
    demand_out[ub - 1] += diff;
    if (demand_out[ub - 1] <= 0.0) {
        throw std::runtime_error(
            "TransportSimplex::solve: the last demand entry (" +
            std::to_string(last_demand) +
            ") is too small to absorb the balance residual (" + std::to_string(diff) +
            ") for ua=" + std::to_string(ua) + ", ub=" + std::to_string(ub)
        );
    }
    return total_s;
}

// Finds the most negative reduced cost in a cost row and its column, for the block-search
// entering rule in TransportSimplexReference::solve.
inline void price_row(
    const double* cost_row,
    const double* v,
    size_t ub,
    double u_i,
    double& best_val,
    size_t& best_col
) {
    for (size_t j = 0; j < ub; ++j) {
        double r = cost_row[j] - u_i - v[j];
        if (r < best_val) {
            best_val = r;
            best_col = j;
        }
    }
}

// Scratch for one thread's in-flight solve, reused across calls and grow-only: a bigger
// (ua, ub) resizes these up, a smaller one just uses a prefix, so steady-state solves
// (same CentroidTable, same decoded token counts) allocate nothing. Sized for ua+ub+1
// nodes: sources [0, ua), sinks [ua, ua+ub), and one artificial root node at ua+ub.
struct Workspace {
    std::vector<double> cost; // ua * ub, row-major
    std::vector<double> supply;
    std::vector<double> demand;
    std::vector<int64_t> parent; // size ua+ub+1; -1 at the root, -2 = unvisited (scratch)
    std::vector<size_t> depth;   // size ua+ub+1
    std::vector<double> flow_to_parent; // size ua+ub+1; flow on the arc to parent[v]
    std::vector<double> u;              // source potentials, size ua
    std::vector<double> v;              // sink potentials, size ub
    std::vector<size_t> adj_head; // CSR adjacency over ua+ub+1 nodes, rebuilt every pivot
    std::vector<size_t> adj_next;
    std::vector<int64_t> adj_to;
    std::vector<double> adj_cost; // the physical arc's cost (M for an artificial arc)
    std::vector<double> adj_flow;
    std::vector<int64_t> bfs_order;
    std::vector<int64_t> path_p;
    std::vector<int64_t> path_q;

    void reserve(size_t ua, size_t ub) {
        size_t n = ua + ub + 1;
        if (cost.size() < ua * ub) {
            cost.resize(ua * ub);
        }
        if (supply.size() < ua) {
            supply.resize(ua);
        }
        if (demand.size() < ub) {
            demand.resize(ub);
        }
        if (parent.size() < n) {
            parent.resize(n);
        }
        if (depth.size() < n) {
            depth.resize(n);
        }
        if (flow_to_parent.size() < n) {
            flow_to_parent.resize(n);
        }
        if (u.size() < ua) {
            u.resize(ua);
        }
        if (v.size() < ub) {
            v.resize(ub);
        }
        if (adj_head.size() < n) {
            adj_head.resize(n);
        }
        size_t edges = (n > 0) ? n - 1 : 0;
        if (adj_next.size() < 2 * edges) {
            adj_next.resize(2 * edges);
            adj_to.resize(2 * edges);
            adj_cost.resize(2 * edges);
            adj_flow.resize(2 * edges);
        }
        if (bfs_order.size() < n) {
            bfs_order.resize(n);
        }
        if (path_p.size() < n) {
            path_p.resize(n);
        }
        if (path_q.size() < n) {
            path_q.resize(n);
        }
    }
};

// One directed adjacency slot: node `to`, reached from `from` over an arc with physical
// cost `cost` (the real cost(i, j) for a transportation arc, or M for an artificial one)
// and current flow `flow`.
inline void add_adj(
    Workspace& ws, size_t edge_slot, int64_t from, int64_t to, double cost, double flow
) {
    ws.adj_to[edge_slot] = to;
    ws.adj_cost[edge_slot] = cost;
    ws.adj_flow[edge_slot] = flow;
    ws.adj_next[edge_slot] = ws.adj_head[static_cast<size_t>(from)];
    ws.adj_head[static_cast<size_t>(from)] = edge_slot;
}

// Builds the artificial-root star: one extra node R = ua+ub, an arc (source i, R) with
// flow supply[i] for every source and (R, sink j) with flow demand[j] for every sink, all
// at cost M. Every arc starts at strictly positive flow (ws.supply/ws.demand are qEMD's
// merged weights, never zero), so this tree is strongly feasible by construction: the
// strongly-feasible condition only constrains zero-flow tree arcs, and there are none
// (Ahuja-Magnanti-Orlin ch. 11.2). Northwest-corner is not used here because it is not
// automatically strongly feasible.
inline void artificial_root_adjacency(Workspace& ws, size_t ua, size_t ub, double big_m) {
    size_t root = ua + ub;
    std::fill(
        ws.adj_head.begin(),
        ws.adj_head.begin() + static_cast<ptrdiff_t>(root + 1),
        SIZE_MAX
    );
    size_t slot = 0;
    for (size_t i = 0; i < ua; ++i) {
        add_adj(
            ws,
            slot,
            static_cast<int64_t>(i),
            static_cast<int64_t>(root),
            big_m,
            ws.supply[i]
        );
        add_adj(
            ws,
            slot + 1,
            static_cast<int64_t>(root),
            static_cast<int64_t>(i),
            big_m,
            ws.supply[i]
        );
        slot += 2;
    }
    for (size_t j = 0; j < ub; ++j) {
        add_adj(
            ws,
            slot,
            static_cast<int64_t>(root),
            static_cast<int64_t>(ua + j),
            big_m,
            ws.demand[j]
        );
        add_adj(
            ws,
            slot + 1,
            static_cast<int64_t>(ua + j),
            static_cast<int64_t>(root),
            big_m,
            ws.demand[j]
        );
        slot += 2;
    }
}

// Breadth-first search from the artificial root R = ua+ub (always the root: R is
// permanently part of the spanning tree, see solve()'s doc comment) over the adjacency
// already built in ws.adj_*, filling parent/depth/flow_to_parent and the dual potentials
// u/v. A source is always the tail of its arc to its tree parent, whether that arc is a
// real one (to a sink) or artificial (to R); a sink is always the head. That fixes the
// sign in the potential update below without per-arc direction bookkeeping. Every tree
// arc has zero reduced cost by construction, which is what lets solve()'s pricing scan
// treat "negative reduced cost" as "not yet basic" with no separate basis-membership
// check.
inline void rebuild_tree(Workspace& ws, size_t ua, size_t ub) {
    size_t root = ua + ub;
    size_t n = root + 1;
    std::fill(ws.parent.begin(), ws.parent.begin() + static_cast<ptrdiff_t>(n), -2);
    ws.parent[root] = -1;
    ws.depth[root] = 0;
    ws.bfs_order[0] = static_cast<int64_t>(root);
    size_t order_end = 1;
    for (size_t qi = 0; qi < order_end; ++qi) {
        int64_t node = ws.bfs_order[qi];
        double node_y =
            (node == static_cast<int64_t>(root))
                ? 0.0
                : (node < static_cast<int64_t>(ua) ? ws.u[static_cast<size_t>(node)]
                                                   : -ws.v[static_cast<size_t>(node) - ua]);
        for (size_t e = ws.adj_head[static_cast<size_t>(node)]; e != SIZE_MAX;
             e = ws.adj_next[e]) {
            int64_t to = ws.adj_to[e];
            if (ws.parent[static_cast<size_t>(to)] != -2) {
                continue;
            }
            ws.parent[static_cast<size_t>(to)] = node;
            ws.depth[static_cast<size_t>(to)] = ws.depth[static_cast<size_t>(node)] + 1;
            ws.flow_to_parent[static_cast<size_t>(to)] = ws.adj_flow[e];
            double cost = ws.adj_cost[e];
            if (to < static_cast<int64_t>(ua)) {
                ws.u[static_cast<size_t>(to)] =
                    node_y + cost; // source: tail, y = y(parent) + cost
            } else {
                ws.v[static_cast<size_t>(to) - ua] = cost - node_y; // sink: head
            }
            ws.bfs_order[order_end++] = to;
        }
    }
}

} // namespace svs::distance::transport_reference_detail

namespace svs::distance {

/// Reference implementation of the dense balanced transportation problem solver: `ua`
/// sources with supply `supply[i]`, `ub` sinks with demand `demand[j]`, a complete
/// bipartite arc set, and a per-arc cost from `cost(i, j)`. Minimizes
/// `sum_ij flow_ij * cost(i, j)` subject to `sum_j flow_ij == supply[i]`,
/// `sum_i flow_ij == demand[j]`, `flow_ij >= 0`.
///
/// Requires `ua >= 1`, `ub >= 1`, every supply/demand entry strictly positive (true for
/// qEMD's merged weights), and `sum(supply) == sum(demand)` to within `1e-6 *
/// max(sum(supply), 1.0)`; any smaller residual (the kind two independent floating-point
/// sums that are mathematically equal can differ by) is folded into the last demand entry
/// so the solver never over- or under-subscribes a side. A larger mismatch, or a residual
/// that leaves the last demand entry non-positive, throws std::runtime_error, since both
/// signal mismatched inputs rather than rounding.
///
/// Initializes with an artificial root node and big-M artificial arcs to every source and
/// sink (Ahuja-Magnanti-Orlin ch. 11.2), which is strongly feasible from the start since
/// every artificial arc carries strictly positive flow. Each pivot then picks the leaving
/// arc as the *last* blocking (flow-decreasing) arc encountered while walking the cycle
/// from its apex (the two endpoints' lowest common tree ancestor) in the entering arc's
/// own direction (Cunningham, 1976); maintaining a strongly feasible tree this way
/// guarantees termination regardless of which negative-reduced-cost arc enters next, so
/// the entering rule is free to optimize for speed: a block-cyclic search over the dense
/// cost matrix, picking the most negative reduced cost in the first block (of
/// `~sqrt(ua*ub)` arcs, rounded to whole rows) that has one, resuming from where the
/// previous search left off. M is `(ua+ub) * max(cost) + 1`, the largest an optimal basic
/// solution's total real cost could be plus 1, so no finite combination of real arcs is
/// ever more expensive than keeping one unit of flow on an artificial arc.
///
/// This type runs the scalar pricing loop unconditionally: it is the `AVX_AVAILABILITY::
/// NONE` path, and it is the oracle TransportSimplex's results (not its algorithm) are
/// checked against.
struct TransportSimplexReference {
    template <typename CostFn>
    static double
    solve(size_t ua, const double* supply, size_t ub, const double* demand, CostFn&& cost) {
        using namespace transport_reference_detail;
        thread_local Workspace ws;
        ws.reserve(ua, ub);
        size_t root = ua + ub;

        prepare_marginals<TransportSimplexReference>(
            ua, supply, ws.supply.data(), ub, demand, ws.demand.data()
        );

        double max_cost = 0.0;
        for (size_t i = 0; i < ua; ++i) {
            double* row = &ws.cost[i * ub];
            for (size_t j = 0; j < ub; ++j) {
                double c = static_cast<double>(cost(i, j));
                row[j] = c;
                max_cost = std::max(max_cost, c);
            }
        }
        // Bigger than any optimal basic solution's real cost could ever be, so the
        // simplex never prefers routing flow through the artificial root once a real
        // spanning tree is available; see the class doc comment.
        double big_m = static_cast<double>(ua + ub) * max_cost + 1.0;

        artificial_root_adjacency(ws, ua, ub, big_m);
        rebuild_tree(ws, ua, ub);

        // Block size: ~sqrt(ua*ub) arcs, rounded to a whole number of rows so each block
        // scan stays an integer number of rows.
        size_t block_rows = static_cast<size_t>(std::round(
            std::sqrt(static_cast<double>(ua) * static_cast<double>(ub)) /
            static_cast<double>(ub)
        ));
        block_rows = std::clamp<size_t>(block_rows, 1, ua);
        size_t cursor_row = 0;

        // No polynomial bound exists for primal network simplex iteration count (even
        // with a strongly feasible tree, only finiteness is guaranteed, not a bound), so
        // this is a generous safety net against a latent bug, not an expected limit.
        size_t max_iters = 1000 * (ua + ub) + 10000;
        for (size_t iter = 0; iter < max_iters; ++iter) {
            size_t best_i = ua, best_j = 0, rows_scanned = 0;
            size_t row = cursor_row;
            while (rows_scanned < ua) {
                double block_best_val = 0.0;
                size_t block_best_i = ua, block_best_j = 0;
                for (size_t b = 0; b < block_rows && rows_scanned < ua;
                     ++b, ++rows_scanned) {
                    double val = 0.0;
                    size_t col = 0;
                    price_row(&ws.cost[row * ub], ws.v.data(), ub, ws.u[row], val, col);
                    if (val < block_best_val) {
                        block_best_val = val;
                        block_best_i = row;
                        block_best_j = col;
                    }
                    row = (row + 1) % ua;
                }
                if (block_best_i != ua) {
                    best_i = block_best_i;
                    best_j = block_best_j;
                    cursor_row = row;
                    break;
                }
            }
            if (best_i == ua) {
                double total_cost = 0.0;
                for (size_t node = 0; node < root; ++node) {
                    if (static_cast<size_t>(ws.parent[node]) == root) {
                        continue; // artificial arc: not part of the real transport cost
                    }
                    size_t ai, aj;
                    if (node < ua) {
                        ai = node;
                        aj = static_cast<size_t>(ws.parent[node]) - ua;
                    } else {
                        ai = static_cast<size_t>(ws.parent[node]);
                        aj = node - ua;
                    }
                    total_cost += ws.flow_to_parent[node] * ws.cost[ai * ub + aj];
                }
                return total_cost;
            }

            int64_t p_node = static_cast<int64_t>(best_i);
            int64_t q_node = static_cast<int64_t>(ua + best_j);
            ws.path_p.clear();
            ws.path_p.push_back(p_node);
            ws.path_q.clear();
            ws.path_q.push_back(q_node);
            int64_t pu = p_node, qu = q_node;
            while (ws.depth[static_cast<size_t>(pu)] > ws.depth[static_cast<size_t>(qu)]) {
                pu = ws.parent[static_cast<size_t>(pu)];
                ws.path_p.push_back(pu);
            }
            while (ws.depth[static_cast<size_t>(qu)] > ws.depth[static_cast<size_t>(pu)]) {
                qu = ws.parent[static_cast<size_t>(qu)];
                ws.path_q.push_back(qu);
            }
            while (pu != qu) {
                pu = ws.parent[static_cast<size_t>(pu)];
                ws.path_p.push_back(pu);
                qu = ws.parent[static_cast<size_t>(qu)];
                ws.path_q.push_back(qu);
            }
            size_t lp = ws.path_p.size();
            size_t lq = ws.path_q.size();

            // A source is always the tail of its tree arc (to a sink or to the root), a
            // sink always the head (see rebuild_tree's doc comment), so walking
            // parent->child is "forward" (flow increases) exactly when the child is a
            // sink, and walking child->parent is forward exactly when the child is a
            // source.
            auto is_forward_down = [&](int64_t child) {
                return child >= static_cast<int64_t>(ua);
            };
            auto is_forward_up = [&](int64_t child) {
                return child < static_cast<int64_t>(ua);
            };

            // Leaving arc: walk the cycle apex (= LCA = the end of either path array)
            // down to p, then from q back up to the apex -- the entering arc's own
            // orientation -- and take the *last* blocking (backward) arc seen, using <=
            // so a later tie overwrites an earlier one (Cunningham's rule; this is what
            // keeps the tree strongly feasible, not the choice of entering arc).
            double theta = std::numeric_limits<double>::infinity();
            int64_t leave_node = -1;
            for (size_t t = lp; t-- > 1;) {
                int64_t child = ws.path_p[t - 1];
                if (!is_forward_down(child)) {
                    double flow = ws.flow_to_parent[static_cast<size_t>(child)];
                    if (flow <= theta) {
                        theta = flow;
                        leave_node = child;
                    }
                }
            }
            for (size_t t = 0; t + 1 < lq; ++t) {
                int64_t child = ws.path_q[t];
                if (!is_forward_up(child)) {
                    double flow = ws.flow_to_parent[static_cast<size_t>(child)];
                    if (flow <= theta) {
                        theta = flow;
                        leave_node = child;
                    }
                }
            }

            // Apply the flow delta to every tree edge on the cycle (the entering arc is
            // not one of these; it is added fresh below with flow == theta). Order does
            // not matter here, unlike the leaving-arc search above.
            for (size_t t = 0; t + 1 < lp; ++t) {
                int64_t child = ws.path_p[t];
                double delta = is_forward_down(child) ? theta : -theta;
                ws.flow_to_parent[static_cast<size_t>(child)] += delta;
            }
            for (size_t t = 0; t + 1 < lq; ++t) {
                int64_t child = ws.path_q[t];
                double delta = is_forward_up(child) ? theta : -theta;
                ws.flow_to_parent[static_cast<size_t>(child)] += delta;
            }

            // Rebuild adjacency from the old tree minus the leaving edge, plus the
            // entering arc (p_node, q_node, theta); rebuilding by BFS instead of
            // patching parent/depth in place avoids re-rooting a subtree incrementally,
            // which costs O(ua+ub) here against O(ua*ub) for pricing, so it is not the
            // bottleneck.
            std::fill(
                ws.adj_head.begin(),
                ws.adj_head.begin() + static_cast<ptrdiff_t>(root + 1),
                SIZE_MAX
            );
            size_t slot = 0;
            for (int64_t node = 0; node < static_cast<int64_t>(root); ++node) {
                if (node == leave_node) {
                    continue;
                }
                int64_t parent_node = ws.parent[static_cast<size_t>(node)];
                double edge_cost;
                if (static_cast<size_t>(parent_node) == root) {
                    edge_cost = big_m;
                } else if (node < static_cast<int64_t>(ua)) {
                    edge_cost = ws.cost
                                    [static_cast<size_t>(node) * ub +
                                     (static_cast<size_t>(parent_node) - ua)];
                } else {
                    edge_cost = ws.cost
                                    [static_cast<size_t>(parent_node) * ub +
                                     (static_cast<size_t>(node) - ua)];
                }
                double flow = ws.flow_to_parent[static_cast<size_t>(node)];
                add_adj(ws, slot, node, parent_node, edge_cost, flow);
                add_adj(ws, slot + 1, parent_node, node, edge_cost, flow);
                slot += 2;
            }
            double entering_cost = ws.cost[best_i * ub + best_j];
            add_adj(ws, slot, p_node, q_node, entering_cost, theta);
            add_adj(ws, slot + 1, q_node, p_node, entering_cost, theta);

            rebuild_tree(ws, ua, ub);
        }
        throw std::runtime_error(
            "TransportSimplexReference::solve: exceeded the iteration limit (" +
            std::to_string(max_iters) + ") for ua=" + std::to_string(ua) +
            ", ub=" + std::to_string(ub)
        );
    }
};

} // namespace svs::distance
