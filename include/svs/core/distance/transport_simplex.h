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

// Primal network simplex for the dense balanced transportation problem, specialized to a
// bipartite tree with no artificial root and no capacities. Northwest-corner
// initialization and the tree-array representation (parent/thread/rev_thread/succ_num/
// last_succ) and incremental pivot update (findJoinNode/findLeavingArc/changeFlow/
// updateTreeStructure/updatePotential) are ported from LEMON's network_simplex, by way of
// Nicolas Bonneel's single-file adaptation vendored into POT; see the notices below. The
// float-proposes/double-confirms pricing split and the strongly-feasible northwest-corner
// start (replacing LEMON/POT's artificial-root big-M start) are this file's own. Problem
// shape (quantized ground distance over merged centroid weights) follows Bonneel et al.,
// "Displacement Interpolation Using Lagrangian Mass Transport", SIGGRAPH Asia 2011.
//
// Portions of this file are derived from:
//
// POT (Python Optimal Transport), `ot/lp/`, tag 0.9.5.
// https://github.com/PythonOT/POT
// Copyright (c) 2016-2023 POT contributors.
// Licensed under the MIT License.
//
// which itself adapted, as `network_simplex_simple.h` (Nicolas Bonneel, 2013), from
// LEMON's `network_simplex.h`:
//
// Copyright (C) 2003-2010
// Egervary Jeno Kombinatorikus Optimalizalasi Kutatocsoport
// (Egervary Research Group on Combinatorial Optimization, EGRES).
//
// Permission to use, modify and distribute this software is granted
// provided that this copyright notice appears in all copies. For
// precise terms see the accompanying LICENSE file.
//
// This software is provided "AS IS" with no warranty of any kind,
// express or implied, and with no claim as to its suitability for any
// purpose.

#include "svs/core/distance/distance_core.h"
#include "svs/core/distance/transport_simplex_pricing.h"
#include "svs/core/distance/transport_simplex_reference.h"
#include "svs/lib/meta.h"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace svs::distance::transport_detail {

// uint16_t node ids halve the tree arrays' footprint for the common small-problem bucket;
// Dynamic has no such bound so ids must cover an arbitrary ua+ub. A pure type trait (no
// function bodies), so it carries none of the cross-TU weak-symbol risk the rest of this
// file's per-ISA machinery is keyed on TransportSimplex<Avx> to avoid.
template <size_t MaxSide> struct TransportIndexTraits {
    using node_type = uint32_t;
    using arc_type = uint32_t;
};
template <> struct TransportIndexTraits<256> {
    using node_type = uint16_t;
    using arc_type = uint32_t;
};

} // namespace svs::distance::transport_detail

namespace svs::distance {

/// Solves the dense balanced transportation problem exactly to within
/// `error_bound(ua, ub, max_cost, total_mass)`. Same contract as
/// `TransportSimplexReference::solve`, plus: `CostFn` must return `float` (pricing stores
/// costs as float, exact for callers whose costs already are, e.g. `CentroidTable`).
///
/// Orients to put the smaller side in rows, then starts from a northwest-corner walk over
/// columns grouped by their cheapest row (ties broken by ascending column id), trading a
/// one-pass cost scan for fewer pivots than a cost-blind NW start needs.
/// Strong feasibility means every zero-flow tree arc has a *source*
/// child (points toward the root, LEMON's convention); the only zero-flow arc the start
/// ever adds is a value tie's degenerate arc, and that arc's child is always a fresh,
/// untouched row, never a sink. Every arc whose child is a sink instead carries that
/// sink's own fresh demand (or a fresh row's full supply), so it is always strictly
/// positive. Each pivot picks the leaving arc by Cunningham's rule (the *last* blocking
/// arc on the cycle in the entering arc's own direction), which preserves this invariant
/// under degenerate pivots and so guarantees termination regardless of which
/// negative-reduced-cost arc enters next.
/// Pricing scans the dense cost matrix in a block-cyclic resumable order in float (`Avx`
/// selects the row scan's SIMD width); a float candidate is only pivoted once its reduced
/// cost is confirmed negative in double (exactly recoverable, since the cost matrix is
/// itself stored as float), and pricing falls back to double (and a from-scratch
/// potential recompute) once a full float pass finds nothing, or once a float candidate
/// fails double confirmation. Below `narrow_index_max_side`, node ids are `uint16_t`
/// instead of `uint32_t`, halving the tree arrays' footprint; callers never see this.
///
/// Every helper this type needs (potentials, tolerances, the dense workspace, ...) is a
/// member keyed on `Avx` (and, where it needs a node-id width, on `Impl`'s `MaxSide`
/// too), rather than a free function: an inline free function not keyed on `Avx` becomes
/// one weak symbol shared across the differently-`-march`-compiled translation units that
/// explicitly instantiate this type per ISA (see `gem.h`, near `merge_sorted_duplicates`).
template <AVX_AVAILABILITY Avx> struct TransportSimplex {
    /// Upper bound on `|solve() - exact optimum|`.
    static double error_bound(size_t ua, size_t ub, double max_cost, double total_mass) {
        return optimality_gap_bound(ua, ub, max_cost, total_mass);
    }

#if SVS_ENABLE_STATS
    /// Pivots performed by this thread's solves since the caller last reset it. Stats
    /// builds only (`SVS_EXPERIMENTAL_ENABLE_STATS`). Per ISA instantiation: read it
    /// through the same `Avx` the dispatched solver used, or it reads 0.
    static size_t& thread_pivot_count() {
        thread_local size_t count = 0;
        return count;
    }
#endif

    template <typename CostFn>
    static double
    solve(size_t ua, const double* supply, size_t ub, const double* demand, CostFn&& cost)
        requires std::same_as<std::invoke_result_t<CostFn&, size_t, size_t>, float>
    {
        if (std::max(ua, ub) <= narrow_index_max_side) {
            return Impl<narrow_index_max_side>::solve_impl(
                ua, supply, ub, demand, std::forward<CostFn>(cost)
            );
        }
        return Impl<svs::Dynamic>::solve_impl(
            ua, supply, ub, demand, std::forward<CostFn>(cost)
        );
    }

  private:
    static constexpr size_t narrow_index_max_side = 256;
    static_assert(
        2 * narrow_index_max_side <=
        std::numeric_limits<typename transport_detail::TransportIndexTraits<
            narrow_index_max_side>::node_type>::max()
    );

    // node is a source iff node < m; fixes both the entering arc's direction (row -> col)
    // and every tree arc's direction (LEMON's generic `_forward` collapses to this one
    // comparison in a bipartite tree with no artificial root).
    static bool is_source(size_t node, size_t m) { return node < m; }

    // Potential magnitude bound: a tree path sum of at most n-1 signed costs.
    static double potential_bound(size_t n, double max_cost) {
        return static_cast<double>(n > 0 ? n - 1 : 0) * max_cost;
    }

    // Tier-2/fallback threshold for an exact (walk_potential-based) reduced cost; see the
    // commit body for the derivation. A candidate beating -walk_tol is truly negative
    // independent of any potential drift.
    static double walk_tol(size_t n, double max_cost) {
        double eps_mach_d = std::numeric_limits<double>::epsilon();
        return 2.0 * static_cast<double>(n) * eps_mach_d *
               (max_cost + potential_bound(n, max_cost));
    }

    // Tier-1 fast-accept threshold for a *drifted* double reduced cost (current u_d/v_d,
    // stale by up to `recompute_period` pivots since the last fresh recompute):
    // drift_tol = 2*(n-1)*R*9*eps*P + walk_tol, the accumulated per-pivot drift plus the
    // same fresh-recompute-and-subtraction rounding walk_tol already bounds.
    static double drift_tol(size_t n, double max_cost, size_t recompute_period) {
        constexpr double c = 9.0;
        double eps_mach_d = std::numeric_limits<double>::epsilon();
        double per_pivot_per_arc = c * eps_mach_d * potential_bound(n, max_cost);
        double accumulated_drift = 2.0 * static_cast<double>(n > 0 ? n - 1 : 0) *
                                   static_cast<double>(recompute_period) *
                                   per_pivot_per_arc;
        return accumulated_drift + walk_tol(n, max_cost);
    }

    // Upper bound on |solve() - exact optimum|: solve() stops once every fresh r >=
    // -walk_tol, and a fresh r differs from the exact reduced cost by at most walk_tol
    // (the same bound, applied once more), so the exact value is >= -2*walk_tol; plus
    // flow_rounding (the final flow recompute's rounding) and final_sum_rounding (the
    // double rounding of summing and multiplying at most n-1 flow*cost terms).
    static double
    optimality_gap_bound(size_t ua, size_t ub, double max_cost, double total_mass) {
        size_t n = ua + ub;
        double eps_mach_d = std::numeric_limits<double>::epsilon();
        double n_d = static_cast<double>(n);
        double final_sum_rounding =
            2.0 * static_cast<double>(n > 0 ? n - 1 : 0) * eps_mach_d * max_cost;
        double flow_rounding = n_d * n_d * eps_mach_d * max_cost;
        return (2.0 * walk_tol(n, max_cost) + flow_rounding + final_sum_rounding) *
               total_mass;
    }

    template <size_t MaxSide> struct Impl {
        using NodeT = typename transport_detail::TransportIndexTraits<MaxSide>::node_type;

      private:
        // Scratch for one `solve<CostFn>` instantiation, grow-only so a steady-state
        // solve (same shape or smaller) allocates nothing. Node ids: sources [0, m),
        // sinks [m, m+k), m <= k.
        struct SolverWorkspace {
            std::vector<float> cost_f; // m * padded_k, +inf in [k, padded_k) of every row
            std::vector<float>
                price_f; // same shape; tree-arc cells hold +inf so pricing skips them
            std::vector<double> supply; // m; destructively consumed by the NW-corner build
            std::vector<double> demand; // k; destructively consumed by the NW-corner build
            std::vector<double> supply_marginal,
                demand_marginal; // folded marginals, untouched
            // Fold-before-orient scratch, each sized to k (>= both ua and ub): holds the
            // caller's own supply/demand, pre-orientation, until m/k are known.
            std::vector<double> supply_folded, demand_folded;
            // Cost-aware column order: col_min/col_argmin (per-column cheapest cost/row)
            // feed col_order, a stable counting sort by col_argmin (scratch col_count).
            std::vector<float> col_min;
            std::vector<NodeT> col_argmin;
            std::vector<NodeT> col_order;
            std::vector<NodeT> col_count;
            std::vector<double> child_sum;      // final flow recompute scratch, size n
            std::vector<NodeT> parent;          // n = m+k; parent[0] (root) unused
            std::vector<NodeT> thread;          // preorder successor, circular over all n
            std::vector<NodeT> rev_thread;      // inverse of thread
            std::vector<NodeT> succ_num;        // subtree size, including self
            std::vector<NodeT> last_succ;       // last node of subtree, in preorder
            std::vector<double> flow_to_parent; // flow on the arc to parent[node]
            std::vector<double> u_d, v_d;       // source/sink potentials, double master
            std::vector<float> u_f, v_f;        // float mirrors, updated with every u_d/v_d
            std::vector<uint8_t> in_tree;       // NW-corner bootstrap scratch
            std::vector<NodeT> first_child, next_sibling, dfs_stack,
                preorder;                     // bootstrap scratch
            std::vector<size_t> preorder_pos; // bootstrap scratch
            std::vector<NodeT>
                dirty_revs; // updateTreeStructure scratch, cleared (not freed) per pivot
            std::vector<NodeT> walk_path; // walk_potential scratch, cleared per call

            void reserve(size_t m, size_t k, size_t padded_k) {
                size_t n = m + k;
                if (cost_f.size() < m * padded_k) {
                    cost_f.resize(m * padded_k);
                    price_f.resize(m * padded_k);
                }
                if (supply.size() < m) {
                    supply.resize(m);
                    supply_marginal.resize(m);
                }
                if (col_count.size() < m + 1) {
                    col_count.resize(m + 1);
                }
                if (demand.size() < k) {
                    demand.resize(k);
                    demand_marginal.resize(k);
                    supply_folded.resize(k);
                    demand_folded.resize(k);
                    col_min.resize(k);
                    col_argmin.resize(k);
                    col_order.resize(k);
                }
                if (parent.size() < n) {
                    parent.resize(n);
                    thread.resize(n);
                    rev_thread.resize(n);
                    succ_num.resize(n);
                    last_succ.resize(n);
                    flow_to_parent.resize(n);
                    in_tree.resize(n);
                    first_child.resize(n);
                    next_sibling.resize(n);
                    dfs_stack.resize(n);
                    preorder.resize(n);
                    preorder_pos.resize(n);
                    dirty_revs.reserve(n);
                    walk_path.reserve(n);
                    child_sum.resize(n);
                }
                if (u_d.size() < m) {
                    u_d.resize(m);
                    u_f.resize(m);
                }
                if (v_d.size() < padded_k) {
                    v_d.resize(padded_k);
                    v_f.resize(padded_k);
                }
            }
        };

        // The dense-matrix cell for the arc between a node and its tree parent: the row
        // is whichever of the two is a source, the column the other's offset by m.
        static size_t arc_cell(size_t m, size_t padded_k, NodeT node, NodeT parent) {
            size_t row = is_source(node, m) ? static_cast<size_t>(node)
                                            : static_cast<size_t>(parent);
            size_t col = is_source(node, m) ? static_cast<size_t>(parent) - m
                                            : static_cast<size_t>(node) - m;
            return row * padded_k + col;
        }

        // A node's pred-arc cost. Recomputed on demand instead of cached per node, since
        // the tree-arc identity (node, parent[node]) already pins down the cell.
        static float pred_cost_f(
            const std::vector<float>& cost_f, size_t padded_k, size_t m, NodeT node, NodeT p
        ) {
            return cost_f[arc_cell(m, padded_k, node, p)];
        }

        // Restores a leaving arc's price cell to its real cost once it leaves the tree.
        static void unmark_tree_arc(
            std::vector<float>& price_f,
            const std::vector<float>& cost_f,
            size_t padded_k,
            size_t m,
            NodeT node,
            NodeT p
        ) {
            size_t cell = arc_cell(m, padded_k, node, p);
            price_f[cell] = cost_f[cell];
        }

        // Exact potential for `node`: walk parent[] up to the root, then re-derive it
        // root-down with recompute_potentials' same single-subtraction-per-step formula
        // (so a walked value equals a freshly recomputed one bit for bit).
        static double
        walk_potential(SolverWorkspace& ws, size_t m, size_t padded_k, NodeT node) {
            ws.walk_path.clear();
            for (NodeT cur = node; cur != 0; cur = ws.parent[cur]) {
                ws.walk_path.push_back(cur);
            }
            double val = 0.0;
            for (size_t idx = ws.walk_path.size(); idx-- > 0;) {
                NodeT nd = ws.walk_path[idx];
                NodeT p = ws.parent[nd];
                double c = static_cast<double>(pred_cost_f(ws.cost_f, padded_k, m, nd, p));
                val = c - val;
            }
            return val;
        }

        // Builds thread/rev_thread/succ_num/last_succ from parent[] alone (the
        // northwest-corner build only produces parent[]): a preorder walk, then subtree
        // sizes via that preorder in reverse (a node's children, at higher preorder
        // index, are already totalled by then).
        static void build_tree_arrays(SolverWorkspace& ws, size_t n) {
            constexpr NodeT no_node = std::numeric_limits<NodeT>::max();
            for (size_t v = 0; v < n; ++v) {
                ws.first_child[v] = no_node;
            }
            for (size_t v = 1; v < n; ++v) {
                NodeT p = ws.parent[v];
                ws.next_sibling[v] = ws.first_child[p];
                ws.first_child[p] = static_cast<NodeT>(v);
            }
            size_t top = 0;
            ws.dfs_stack[top++] = 0;
            size_t pre_count = 0;
            while (top > 0) {
                NodeT v = ws.dfs_stack[--top];
                ws.preorder[pre_count] = v;
                ws.preorder_pos[v] = pre_count;
                ++pre_count;
                for (NodeT c = ws.first_child[v]; c != no_node; c = ws.next_sibling[c]) {
                    ws.dfs_stack[top++] = c;
                }
            }
            for (size_t v = 0; v < n; ++v) {
                ws.succ_num[v] = 1;
            }
            for (size_t idx = n; idx-- > 1;) {
                NodeT v = ws.preorder[idx];
                ws.succ_num[ws.parent[v]] =
                    static_cast<NodeT>(ws.succ_num[ws.parent[v]] + ws.succ_num[v]);
            }
            for (size_t v = 0; v < n; ++v) {
                size_t last_idx =
                    ws.preorder_pos[v] + static_cast<size_t>(ws.succ_num[v]) - 1;
                ws.last_succ[v] = ws.preorder[last_idx];
            }
            for (size_t idx = 0; idx + 1 < n; ++idx) {
                ws.thread[ws.preorder[idx]] = ws.preorder[idx + 1];
                ws.rev_thread[ws.preorder[idx + 1]] = ws.preorder[idx];
            }
            ws.thread[ws.preorder[n - 1]] = 0;
            ws.rev_thread[0] = ws.preorder[n - 1];
        }

        // Fills u_d/v_d/u_f/v_f from scratch via a thread (preorder) walk: a parent's
        // potential is always known before its children's, since thread visits a node
        // before its subtree.
        static void
        recompute_potentials(SolverWorkspace& ws, size_t m, size_t k, size_t padded_k) {
            // Hoisted so the compiler doesn't reload each vector's base pointer every
            // iteration (a store through one could otherwise alias another's data()).
            const NodeT* const __restrict thread = ws.thread.data();
            const NodeT* const __restrict parent = ws.parent.data();
            double* const __restrict u_d = ws.u_d.data();
            float* const __restrict u_f = ws.u_f.data();
            double* const __restrict v_d = ws.v_d.data();
            float* const __restrict v_f = ws.v_f.data();
            u_d[0] = 0.0;
            u_f[0] = 0.0F;
            NodeT node = thread[0];
            size_t n = m + k;
            for (size_t count = 1; count < n; ++count, node = thread[node]) {
                NodeT p = parent[node];
                double c =
                    static_cast<double>(pred_cost_f(ws.cost_f, padded_k, m, node, p));
                double val = (static_cast<size_t>(node) < m)
                                 ? c - v_d[static_cast<size_t>(p) - m]
                                 : c - u_d[p];
                if (static_cast<size_t>(node) < m) {
                    u_d[node] = val;
                    u_f[node] = static_cast<float>(val);
                } else {
                    v_d[static_cast<size_t>(node) - m] = val;
                    v_f[static_cast<size_t>(node) - m] = static_cast<float>(val);
                }
            }
        }

      public:
        template <typename CostFn>
        static double solve_impl(
            size_t ua, const double* supply, size_t ub, const double* demand, CostFn&& cost
        ) {
            if constexpr (MaxSide != svs::Dynamic) {
                size_t largest = std::max(ua, ub);
                if (largest > MaxSide) {
                    throw std::runtime_error(
                        "TransportSimplex<" + std::to_string(MaxSide) +
                        ">::solve: max(ua, ub) = " + std::to_string(largest) +
                        " exceeds the bound " + std::to_string(MaxSide)
                    );
                }
            }

            bool transposed = ua > ub;
            size_t m = transposed ? ub : ua;
            size_t k = transposed ? ua : ub;

            thread_local SolverWorkspace ws;
            size_t padded_k = transport_detail::pad_to_lanes(k);
            ws.reserve(m, k, padded_k);
            size_t n = m + k;

            // Folds before orienting, on the caller's own supply/demand: folding after
            // swapping rows/columns for ua > ub would land the residual on the caller's
            // last supply entry instead of its last demand entry.
            transport_reference_detail::prepare_marginals<Impl>(
                ua, supply, ws.supply_folded.data(), ub, demand, ws.demand_folded.data()
            );
            const double* supply_src =
                transposed ? ws.demand_folded.data() : ws.supply_folded.data();
            const double* demand_src =
                transposed ? ws.supply_folded.data() : ws.demand_folded.data();
            std::copy_n(supply_src, m, ws.supply.begin());
            std::copy_n(demand_src, k, ws.demand.begin());
            // Kept aside, untouched by the NW-corner walk below, so final_cost can
            // recompute every tree flow fresh from these instead of the pivot loop's
            // incrementally-accumulated flow_to_parent.
            std::copy_n(ws.supply.begin(), m, ws.supply_marginal.begin());
            std::copy_n(ws.demand.begin(), k, ws.demand_marginal.begin());

            double max_cost = 0.0;
            if (transposed) {
                fill_cost_matrix<true>(ws, m, k, padded_k, cost, max_cost);
            } else {
                fill_cost_matrix<false>(ws, m, k, padded_k, cost, max_cost);
            }
            build_col_order(ws, m, k);
            for (size_t j = k; j < padded_k; ++j) {
                ws.v_d[j] = 0.0;
                ws.v_f[j] = 0.0F;
            }
            std::copy_n(ws.cost_f.begin(), m * padded_k, ws.price_f.begin());

            // Every zero-flow tree arc must have a source child (point toward the
            // root); only the value-tie arc below is ever zero, and its child is a
            // fresh, untouched (so later strictly positive) row.
            std::fill(
                ws.in_tree.begin(), ws.in_tree.begin() + static_cast<ptrdiff_t>(n), 0
            );
            ws.in_tree[0] = 1;
            auto add_tree_arc = [&](size_t i, size_t j, double flow) {
                size_t sink_node = m + j;
                if (!ws.in_tree[sink_node]) {
                    ws.parent[sink_node] = static_cast<NodeT>(i);
                    ws.flow_to_parent[sink_node] = flow;
                    ws.in_tree[sink_node] = 1;
                } else {
                    ws.parent[i] = static_cast<NodeT>(sink_node);
                    ws.flow_to_parent[i] = flow;
                    ws.in_tree[i] = 1;
                }
                ws.price_f[i * padded_k + j] = std::numeric_limits<float>::infinity();
            };
            {
                size_t i = 0;
                // p is the walk's position among columns; j = col_order[p] is the real
                // column id. The walk's correctness depends only on visiting order.
                size_t p = 0;
                size_t j = static_cast<size_t>(ws.col_order[p]);
                // True when row i was the side most recently entered fresh (so, at the
                // final cell, is still untouched); false when column j was.
                bool fresh_row = false;
                while (true) {
                    bool last_row = (i == m - 1);
                    bool last_col = (p == k - 1);
                    if (last_row && last_col) {
                        // The fresh side's value is untouched and strictly positive; the
                        // other side's residual lives in supply/demand, not a tree arc.
                        double x = fresh_row ? ws.supply[i] : ws.demand[j];
                        add_tree_arc(i, j, x);
                        ws.supply[i] -= x;
                        ws.demand[j] -= x;
                        break;
                    }
                    if (last_col) {
                        double x = ws.supply[i];
                        add_tree_arc(i, j, x);
                        ws.supply[i] -= x;
                        ws.demand[j] -= x;
                        i += 1;
                        fresh_row = true;
                    } else if (last_row) {
                        double x = ws.demand[j];
                        add_tree_arc(i, j, x);
                        ws.supply[i] -= x;
                        ws.demand[j] -= x;
                        p += 1;
                        j = static_cast<size_t>(ws.col_order[p]);
                        fresh_row = false;
                    } else {
                        double x = std::min(ws.supply[i], ws.demand[j]);
                        add_tree_arc(i, j, x);
                        ws.supply[i] -= x;
                        ws.demand[j] -= x;
                        bool s_zero = ws.supply[i] <= 0.0;
                        bool d_zero = ws.demand[j] <= 0.0;
                        if (s_zero && d_zero) {
                            // Degenerate tie: the zero-flow arc gets a source child
                            // (row i+1, fresh), not a sink child, per the class doc.
                            add_tree_arc(i + 1, j, 0.0);
                            i += 1;
                            p += 1;
                            j = static_cast<size_t>(ws.col_order[p]);
                            fresh_row = true;
                        } else if (s_zero) {
                            i += 1;
                            fresh_row = true;
                        } else {
                            p += 1;
                            j = static_cast<size_t>(ws.col_order[p]);
                            fresh_row = false;
                        }
                    }
                }
            }
            // parent[root] is never a real node; updateTreeStructure's last_succ walk
            // relies on this exact sentinel to stop a chain that may pass through root.
            ws.parent[0] = std::numeric_limits<NodeT>::max();
            build_tree_arrays(ws, n);
            recompute_potentials(ws, m, k, padded_k);

            size_t recompute_period = n;
            double drift_threshold = drift_tol(n, max_cost, recompute_period);
            double walk_threshold = walk_tol(n, max_cost);

            size_t block_rows = static_cast<size_t>(std::round(
                std::sqrt(static_cast<double>(m) * static_cast<double>(k)) /
                static_cast<double>(k)
            ));
            block_rows = std::clamp<size_t>(block_rows, 1, m);
            size_t cursor_row = 0;
            bool double_mode = false;
            size_t pivots_since_recompute = 0;

            auto do_pivot = [&](size_t row, size_t col) {
#if SVS_ENABLE_STATS
                ++thread_pivot_count();
#endif
                pivot(ws, m, padded_k, row, col);
                ++pivots_since_recompute;
                if (pivots_since_recompute >= recompute_period) {
                    recompute_potentials(ws, m, k, padded_k);
                    pivots_since_recompute = 0;
                }
            };

            // No polynomial bound exists for primal network simplex iteration count, so
            // this is a generous safety net against a latent bug, not an expected limit.
            size_t max_iters = 1000 * n + 10000;
            for (size_t iter = 0; iter < max_iters; ++iter) {
                size_t row = 0, col = 0;
                bool found = !double_mode ? find_entering<float>(
                                                ws.price_f,
                                                ws.u_f,
                                                ws.v_f,
                                                m,
                                                padded_k,
                                                block_rows,
                                                cursor_row,
                                                0.0F,
                                                row,
                                                col
                                            )
                                          : find_entering<double>(
                                                ws.price_f,
                                                ws.u_d,
                                                ws.v_d,
                                                m,
                                                padded_k,
                                                block_rows,
                                                cursor_row,
                                                0.0,
                                                row,
                                                col
                                            );
                if (found) {
                    double r_d = static_cast<double>(ws.cost_f[row * padded_k + col]) -
                                 ws.u_d[row] - ws.v_d[col];
                    // Tier 1: drift can't have made this one look falsely negative.
                    if (r_d < -drift_threshold) {
                        do_pivot(row, col);
                        continue;
                    }
                    // Tier 2: borderline; an exact root-path walk settles it either way.
                    if (r_d < 0.0) {
                        NodeT i_node = static_cast<NodeT>(row);
                        NodeT j_node = static_cast<NodeT>(m + col);
                        double r_walk =
                            static_cast<double>(ws.cost_f[row * padded_k + col]) -
                            walk_potential(ws, m, padded_k, i_node) -
                            walk_potential(ws, m, padded_k, j_node);
                        if (r_walk < -walk_threshold) {
                            do_pivot(row, col);
                            continue;
                        }
                    }
                }

                // Tier 3: neither tier above gave a trustworthy candidate (or none was
                // found); the only authoritative check left is a full double pass on
                // freshly recomputed potentials, which needs no further walk to trust.
                recompute_potentials(ws, m, k, padded_k);
                pivots_since_recompute = 0;
                size_t fresh_cursor = 0;
                bool found3 = find_entering<double>(
                    ws.price_f,
                    ws.u_d,
                    ws.v_d,
                    m,
                    padded_k,
                    block_rows,
                    fresh_cursor,
                    -walk_threshold,
                    row,
                    col
                );
                if (!found3) {
                    return final_cost(ws, m, padded_k, n);
                }
                double_mode = true;
                do_pivot(row, col);
            }
            throw std::runtime_error(
                "TransportSimplex::solve: exceeded the iteration limit (" +
                std::to_string(max_iters) + ") for ua=" + std::to_string(ua) +
                ", ub=" + std::to_string(ub)
            );
        }

      private:
        // Fills cost_f/padding; Transposed avoids a runtime orientation check. Also folds
        // each column's cheapest cost/row into col_min/col_argmin while filling.
        template <bool Transposed, typename CostFn>
        static void fill_cost_matrix(
            SolverWorkspace& ws,
            size_t m,
            size_t k,
            size_t padded_k,
            CostFn& cost,
            double& max_cost
        ) {
            for (size_t i = 0; i < m; ++i) {
                float* row = &ws.cost_f[i * padded_k];
                for (size_t j = 0; j < k; ++j) {
                    double c = Transposed ? static_cast<double>(cost(j, i))
                                          : static_cast<double>(cost(i, j));
                    row[j] = static_cast<float>(c);
                    max_cost = std::max(max_cost, c);
                    // Row 0 seeds col_min by assignment rather than a separate +inf
                    // pass; strict `<` below keeps the smallest row on a tie.
                    float cf = static_cast<float>(c);
                    if (i == 0 || cf < ws.col_min[j]) {
                        ws.col_min[j] = cf;
                        ws.col_argmin[j] = static_cast<NodeT>(i);
                    }
                }
                for (size_t j = k; j < padded_k; ++j) {
                    row[j] = std::numeric_limits<float>::infinity();
                }
            }
        }

        // Stable counting sort of columns by col_argmin, ties kept in ascending column
        // id: col_count's prefix sums give each row's bucket start before the fill pass.
        static void build_col_order(SolverWorkspace& ws, size_t m, size_t k) {
            std::fill(
                ws.col_count.begin(),
                ws.col_count.begin() + static_cast<ptrdiff_t>(m + 1),
                NodeT{0}
            );
            for (size_t j = 0; j < k; ++j) {
                ++ws.col_count[static_cast<size_t>(ws.col_argmin[j]) + 1];
            }
            for (size_t r = 0; r < m; ++r) {
                ws.col_count[r + 1] =
                    static_cast<NodeT>(ws.col_count[r + 1] + ws.col_count[r]);
            }
            for (size_t j = 0; j < k; ++j) {
                size_t row = ws.col_argmin[j];
                ws.col_order[ws.col_count[row]] = static_cast<NodeT>(j);
                ws.col_count[row] = static_cast<NodeT>(ws.col_count[row] + 1);
            }
        }

        // Block-cyclic resumable row scan: returns the first block (of `block_rows`
        // rows, cycling from `cursor`) with an entry below `threshold`, or false once
        // every row has been scanned once with none found.
        template <typename PotT>
        static bool find_entering(
            const std::vector<float>& price_f,
            const std::vector<PotT>& u_pot,
            const std::vector<PotT>& v_pot,
            size_t m,
            size_t padded_k,
            size_t block_rows,
            size_t& cursor,
            PotT threshold,
            size_t& out_row,
            size_t& out_col
        ) {
            size_t row = cursor;
            size_t rows_scanned = 0;
            while (rows_scanned < m) {
                PotT block_best_val = threshold;
                size_t block_best_row = m;
                uint32_t block_best_col = 0;
                for (size_t b = 0; b < block_rows && rows_scanned < m;
                     ++b, ++rows_scanned) {
                    PotT val = threshold;
                    uint32_t col = 0;
                    transport_detail::RowMin<Avx>::min_in_row(
                        &price_f[row * padded_k],
                        v_pot.data(),
                        padded_k,
                        u_pot[row],
                        val,
                        col
                    );
                    if (val < block_best_val) {
                        block_best_val = val;
                        block_best_row = row;
                        block_best_col = col;
                    }
                    row = (row + 1) % m;
                }
                if (block_best_row != m) {
                    out_row = block_best_row;
                    out_col = block_best_col;
                    cursor = row;
                    return true;
                }
            }
            return false;
        }

        // Recomputes every tree flow from the folded marginals in post-order instead of
        // the pivot loop's incrementally-drifted flow_to_parent; see the commit body.
        static double final_cost(SolverWorkspace& ws, size_t m, size_t padded_k, size_t n) {
            NodeT node = 0;
            for (size_t idx = 0; idx < n; ++idx, node = ws.thread[node]) {
                ws.preorder[idx] = node;
            }
            std::fill(
                ws.child_sum.begin(), ws.child_sum.begin() + static_cast<ptrdiff_t>(n), 0.0
            );
            for (size_t idx = n; idx-- > 1;) {
                NodeT nd = ws.preorder[idx];
                NodeT p = ws.parent[nd];
                double marginal = is_source(nd, m)
                                      ? ws.supply_marginal[nd]
                                      : ws.demand_marginal[static_cast<size_t>(nd) - m];
                double flow = marginal - ws.child_sum[nd];
                ws.flow_to_parent[nd] = flow;
                ws.child_sum[p] += flow;
            }

            double total = 0.0;
            for (size_t idx = 1; idx < n; ++idx) {
                NodeT nd = static_cast<NodeT>(idx);
                NodeT p = ws.parent[nd];
                double c = static_cast<double>(pred_cost_f(ws.cost_f, padded_k, m, nd, p));
                total += ws.flow_to_parent[nd] * c;
            }
            return total;
        }

        static void
        pivot(SolverWorkspace& ws, size_t m, size_t padded_k, size_t row, size_t col) {
            NodeT i_node = static_cast<NodeT>(row);
            NodeT j_node = static_cast<NodeT>(m + col);

            NodeT join = find_join_node(ws, i_node, j_node);

            NodeT u_in, v_in, u_out = 0;
            double theta = find_leaving_arc(ws, i_node, j_node, join, m, u_in, v_in, u_out);

            // Capture the leaving cell before updateTreeStructure overwrites
            // parent[u_out].
            NodeT leave_parent = ws.parent[u_out];
            unmark_tree_arc(ws.price_f, ws.cost_f, padded_k, m, u_out, leave_parent);

            change_flow(ws, i_node, j_node, join, m, theta);
            update_tree_structure(ws, join, u_in, v_in, u_out, theta);
            update_potential(ws, i_node, j_node, u_in, v_in, m, padded_k);

            ws.price_f[row * padded_k + col] = std::numeric_limits<float>::infinity();
        }

        static NodeT find_join_node(SolverWorkspace& ws, NodeT i_node, NodeT j_node) {
            const NodeT* const __restrict succ_num = ws.succ_num.data();
            const NodeT* const __restrict parent = ws.parent.data();
            size_t u = i_node, v = j_node;
            while (u != v) {
                if (succ_num[u] < succ_num[v]) {
                    u = parent[u];
                } else {
                    v = parent[v];
                }
            }
            return static_cast<NodeT>(u);
        }

        // Cunningham's rule: the *last* blocking arc on each side's path to the join,
        // walked in the entering arc's own direction, second path's tie using <= so a
        // later tie overrides an earlier one. This keeps the tree strongly feasible.
        static double find_leaving_arc(
            SolverWorkspace& ws,
            NodeT i_node,
            NodeT j_node,
            NodeT join,
            size_t m,
            NodeT& u_in,
            NodeT& v_in,
            NodeT& u_out
        ) {
            const NodeT* const __restrict parent = ws.parent.data();
            const double* const __restrict flow_to_parent = ws.flow_to_parent.data();
            double delta = std::numeric_limits<double>::infinity();
            bool leaving_on_j_side = false;
            for (size_t u = i_node; u != join; u = parent[u]) {
                double d = is_source(u, m) ? flow_to_parent[u]
                                           : std::numeric_limits<double>::infinity();
                if (d < delta) {
                    delta = d;
                    u_out = static_cast<NodeT>(u);
                }
            }
            for (size_t u = j_node; u != join; u = parent[u]) {
                double d = is_source(u, m) ? std::numeric_limits<double>::infinity()
                                           : flow_to_parent[u];
                if (d <= delta) {
                    delta = d;
                    u_out = static_cast<NodeT>(u);
                    leaving_on_j_side = true;
                }
            }
            if (leaving_on_j_side) {
                u_in = j_node;
                v_in = i_node;
            } else {
                u_in = i_node;
                v_in = j_node;
            }
            return delta;
        }

        static void change_flow(
            SolverWorkspace& ws,
            NodeT i_node,
            NodeT j_node,
            NodeT join,
            size_t m,
            double theta
        ) {
            if (theta <= 0.0) {
                return;
            }
            const NodeT* const __restrict parent = ws.parent.data();
            double* const __restrict flow_to_parent = ws.flow_to_parent.data();
            for (size_t u = i_node; u != join; u = parent[u]) {
                flow_to_parent[u] += is_source(u, m) ? -theta : theta;
            }
            for (size_t u = j_node; u != join; u = parent[u]) {
                flow_to_parent[u] += is_source(u, m) ? theta : -theta;
            }
        }

        // Ports LEMON's updateTreeStructure: reverses the parent chain from u_in up to
        // (excluding) u_out and splices thread/rev_thread accordingly. A tree arc's flow
        // lives at whichever endpoint is the child, so reversing a link moves it too.
        static void update_tree_structure(
            SolverWorkspace& ws,
            NodeT join,
            NodeT u_in,
            NodeT v_in,
            NodeT u_out,
            double theta
        ) {
            // Hoisted: none of these vectors alias each other, but without __restrict a
            // store through one forces a reload of every other's data() each iteration.
            NodeT* const __restrict parent = ws.parent.data();
            NodeT* const __restrict thread = ws.thread.data();
            NodeT* const __restrict rev_thread = ws.rev_thread.data();
            NodeT* const __restrict succ_num = ws.succ_num.data();
            NodeT* const __restrict last_succ = ws.last_succ.data();
            double* const __restrict flow_to_parent = ws.flow_to_parent.data();

            size_t old_rev_thread = rev_thread[u_out];
            NodeT old_succ_num = succ_num[u_out];
            size_t old_last_succ = last_succ[u_out];
            NodeT v_out = parent[u_out];

            size_t u = last_succ[u_in];
            size_t right = thread[u];
            size_t last = (old_rev_thread == static_cast<size_t>(v_in))
                              ? thread[last_succ[u_out]]
                              : thread[v_in];

            thread[v_in] = u_in;
            size_t stem = u_in;
            ws.dirty_revs.clear();
            ws.dirty_revs.push_back(v_in);
            size_t par_stem = v_in;
            while (stem != u_out) {
                size_t new_stem = parent[stem];
                thread[u] = static_cast<NodeT>(new_stem);
                ws.dirty_revs.push_back(static_cast<NodeT>(u));

                size_t w = rev_thread[stem];
                thread[w] = static_cast<NodeT>(right);
                rev_thread[right] = static_cast<NodeT>(w);

                parent[stem] = static_cast<NodeT>(par_stem);
                par_stem = stem;
                stem = new_stem;

                u = (last_succ[stem] == last_succ[par_stem]) ? rev_thread[par_stem]
                                                             : last_succ[stem];
                right = thread[u];
            }
            parent[u_out] = static_cast<NodeT>(par_stem);
            thread[u] = static_cast<NodeT>(last);
            rev_thread[last] = static_cast<NodeT>(u);
            last_succ[u_out] = static_cast<NodeT>(u);

            if (old_rev_thread != static_cast<size_t>(v_in)) {
                thread[old_rev_thread] = static_cast<NodeT>(right);
                rev_thread[right] = static_cast<NodeT>(old_rev_thread);
            }
            for (NodeT dirty : ws.dirty_revs) {
                rev_thread[thread[dirty]] = dirty;
            }

            NodeT tmp_sc = 0;
            size_t tmp_ls = last_succ[u_out];
            u = u_out;
            while (u != static_cast<size_t>(u_in)) {
                size_t w = parent[u];
                flow_to_parent[u] = flow_to_parent[w];
                tmp_sc = static_cast<NodeT>(tmp_sc + succ_num[u] - succ_num[w]);
                succ_num[u] = tmp_sc;
                last_succ[w] = static_cast<NodeT>(tmp_ls);
                u = w;
            }
            flow_to_parent[u_in] = theta;
            succ_num[u_in] = old_succ_num;

            // parent[root]'s sentinel (see solve_impl) matches this "disabled" value, so
            // a chain left unbounded below is safe to walk off the end of the tree.
            constexpr size_t no_node = std::numeric_limits<NodeT>::max();
            size_t up_limit_in = no_node, up_limit_out = no_node;
            if (last_succ[join] == v_in) {
                up_limit_out = join;
            } else {
                up_limit_in = join;
            }
            for (u = v_in; u != up_limit_in && last_succ[u] == v_in; u = parent[u]) {
                last_succ[u] = last_succ[u_out];
            }
            if (join != old_rev_thread && static_cast<size_t>(v_in) != old_rev_thread) {
                for (u = v_out; u != up_limit_out && last_succ[u] == old_last_succ;
                     u = parent[u]) {
                    last_succ[u] = static_cast<NodeT>(old_rev_thread);
                }
            } else {
                for (u = v_out; u != up_limit_out && last_succ[u] == old_last_succ;
                     u = parent[u]) {
                    last_succ[u] = last_succ[u_out];
                }
            }
            for (u = v_in; u != static_cast<size_t>(join); u = parent[u]) {
                succ_num[u] = static_cast<NodeT>(succ_num[u] + old_succ_num);
            }
            for (u = v_out; u != static_cast<size_t>(join); u = parent[u]) {
                succ_num[u] = static_cast<NodeT>(succ_num[u] - old_succ_num);
            }
        }

        // sigma shifts every node in u_in's moved subtree so the entering arc's
        // invariant (u_i + v_j == cost) holds exactly; sources/sinks shift by
        // -sigma/+sigma (not the same sign) since the invariant is a sum, not a diff.
        static void update_potential(
            SolverWorkspace& ws,
            NodeT i_node,
            NodeT j_node,
            NodeT u_in,
            NodeT v_in,
            size_t m,
            size_t padded_k
        ) {
            auto pi_of = [&](NodeT node) -> double {
                return is_source(node, m) ? -ws.u_d[node]
                                          : ws.v_d[static_cast<size_t>(node) - m];
            };
            double entering_cost =
                static_cast<double>(ws.cost_f
                                        [static_cast<size_t>(i_node) * padded_k +
                                         (static_cast<size_t>(j_node) - m)]);
            double sigma = is_source(u_in, m) ? pi_of(v_in) - pi_of(u_in) - entering_cost
                                              : pi_of(v_in) - pi_of(u_in) + entering_cost;

            // Hoisted: these never alias each other, but without __restrict every store
            // through one forces a reload of the others' data() each iteration.
            const NodeT* const __restrict thread = ws.thread.data();
            double* const __restrict u_d = ws.u_d.data();
            float* const __restrict u_f = ws.u_f.data();
            double* const __restrict v_d = ws.v_d.data();
            float* const __restrict v_f = ws.v_f.data();
            size_t end = thread[ws.last_succ[u_in]];
            for (size_t node = u_in; node != end; node = thread[node]) {
                if (is_source(node, m)) {
                    u_d[node] -= sigma;
                    u_f[node] -= static_cast<float>(sigma);
                } else {
                    size_t j = node - m;
                    v_d[j] += sigma;
                    v_f[j] += static_cast<float>(sigma);
                }
            }
        }
    };
};

// NONE has no RowMin specialization of its own; it just runs the plain scalar algorithm
// every ISA is cross-checked against.
template <> struct TransportSimplex<AVX_AVAILABILITY::NONE> {
    // NONE is the scalar baseline (no -march variation to key against), so this repeats
    // TransportSimplex<Avx>'s own gap-bound formula rather than sharing its private one.
    static double error_bound(size_t ua, size_t ub, double max_cost, double total_mass) {
        size_t n = ua + ub;
        double eps_mach_d = std::numeric_limits<double>::epsilon();
        double n_d = static_cast<double>(n);
        double p = static_cast<double>(n > 0 ? n - 1 : 0) * max_cost;
        double walk_tol = 2.0 * n_d * eps_mach_d * (max_cost + p);
        double flow_rounding = n_d * n_d * eps_mach_d * max_cost;
        double final_sum_rounding =
            2.0 * static_cast<double>(n > 0 ? n - 1 : 0) * eps_mach_d * max_cost;
        return (2.0 * walk_tol + flow_rounding + final_sum_rounding) * total_mass;
    }

    template <typename CostFn>
    static double
    solve(size_t ua, const double* supply, size_t ub, const double* demand, CostFn&& cost) {
        return TransportSimplexReference::solve(
            ua, supply, ub, demand, std::forward<CostFn>(cost)
        );
    }
};

} // namespace svs::distance
