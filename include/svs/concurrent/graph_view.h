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

#include "svs/concepts/graph.h"
#include "svs/lib/spinlock.h" // svs::detail::pause

#include <algorithm>
#include <span>
#include <vector>

namespace svs::concurrent {

///
/// @brief A read-only view of a ``SeqLockGraph`` that makes *unmodified* upstream
/// graph-traversal code safe against a concurrent writer.
///
/// The trick is where the sequence lock lives. ``seqlock_greedy_search`` puts the whole
/// node expansion -- distance computations included -- inside the read section, which is
/// zero-copy but means every traversal routine has to be rewritten around the retry loop.
/// This class instead pushes the read section *down into ``get_node``*: it copies the
/// adjacency list into a scratch buffer, retrying until the sequence counter certifies the
/// copy, and hands back a span over that stable buffer. The caller sees an ordinary,
/// immutable adjacency list and needs to know nothing about concurrency.
///
/// So this type satisfies ``svs::graphs::ImmutableMemoryGraph`` and can be handed to
/// upstream's ``greedy_search``, ``BatchIterator``, and anything else that only *reads* a
/// graph, with no changes to any of them. That is how the prototype supports range/batch
/// queries without duplicating upstream's 340-line ``iterator.h``.
///
/// Cost: one copy of up to ``max_degree`` indices per node visited (128 bytes at degree 32
/// with ``uint32_t`` IDs), against the ~32 full distance computations that visit provokes.
/// Correctness cost: none -- the copy is certified by the same sequence counter.
///
/// **Not thread-safe.** One instance per searching thread; the scratch buffer is per-view.
/// This is the same contract as a search buffer, and it is why views are created inside the
/// search call rather than stored on the index.
///
template <std::unsigned_integral Idx> class SeqLockGraphView {
  public:
    using graph_type = SeqLockGraph<Idx>;
    using index_type = Idx;
    // Both alias the same const span: this view is read-only, so there is no mutable
    // reference to hand out. `reference` exists only because the concept requires the name.
    using reference = std::span<const Idx>;
    using const_reference = std::span<const Idx>;

    explicit SeqLockGraphView(const graph_type& graph)
        : graph_{&graph}
        , scratch_(graph.max_degree()) {}

    size_t max_degree() const { return graph_->max_degree(); }
    size_t n_nodes() const { return graph_->n_nodes(); }
    void prefetch_node(Idx i) const { graph_->prefetch_node(i); }

    /// @brief A certified-consistent snapshot of node ``i``'s adjacency list.
    ///
    /// The returned span is valid until the next call to ``get_node`` on this view.
    const_reference get_node(Idx i) const {
        const auto& seqlock = graph_->seqlock(i);
        for (;;) {
            auto maybe_seq = seqlock.read_begin();
            if (!maybe_seq) {
                svs::detail::pause();
                continue;
            }
            auto neighbors = graph_->get_node_atomic(i);
            // The degree is a single aligned store of a value never exceeding max_degree,
            // so it cannot tear into something larger -- but clamping is free and keeps a
            // future writer-side bug from turning into a buffer overrun here.
            const size_t degree = std::min(neighbors.size(), scratch_.size());
            for (size_t j = 0; j < degree; ++j) {
                scratch_[j] = neighbors[j];
            }
            if (seqlock.read_validate(*maybe_seq)) {
                return const_reference{scratch_.data(), degree};
            }
            svs::detail::pause();
        }
    }

    /// @brief Degree of node ``i``.
    ///
    /// Callers must treat this as a hint that may already be stale by the time it returns:
    /// only the degree bundled with the snapshot from ``get_node`` is certified. Upstream
    /// uses ``get_node_degree`` only for reporting and capacity checks, never to bound a
    /// loop over memory, so a stale value is harmless there.
    size_t get_node_degree(Idx i) const { return graph_->get_node_degree(i); }

  private:
    const graph_type* graph_;
    mutable std::vector<Idx> scratch_;
};

static_assert(svs::graphs::ImmutableMemoryGraph<SeqLockGraphView<uint32_t>>);

} // namespace svs::concurrent
