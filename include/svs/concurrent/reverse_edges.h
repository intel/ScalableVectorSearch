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

#include "svs/concurrent/spinlock.h"
#include "svs/lib/segmented_vector.h"

// external
#include "tsl/robin_set.h"

#include <algorithm>
#include <atomic>
#include <concepts>
#include <mutex>
#include <vector>

namespace svs::index::vamana::concurrent::graphs {

///
/// @brief Per-node index of in-neighbors: `R(n)` is the list of nodes that point at `n`.
///
/// Stored as one `std::vector<Idx>` per node, indexed by node id in a grow-stable
/// `SegmentedVector`, with a per-node `SpinLock`. Every operation touches only the target
/// node's list under its own lock.
///
/// `R(n)` is a complete superset of `n`'s in-neighbors: `record` is called unconditionally
/// on every created edge, so it may hold stale (edge later dropped) or duplicate entries,
/// but never misses a live in-edge. Consolidation reads it to find who points at a deleted
/// node, and prunes stale entries via `remove`/`reset_node`.
///
template <std::unsigned_integral Idx> class ReverseEdges {
  public:
    explicit ReverseEdges(size_t num_nodes)
        : lists_(num_nodes)
        , locks_(num_nodes) {}

    void resize(size_t new_size) {
        lists_.resize(new_size);
        locks_.resize(new_size);
    }

    void set_recording(bool on) { recording_.store(on, std::memory_order_release); }

    void record(Idx m, Idx n) {
        if (!recording_.load(std::memory_order_acquire)) {
            return;
        }
        std::lock_guard lock{locks_[n]};
        lists_[n].push_back(m);
    }

    void remove(Idx m, Idx n) {
        std::lock_guard lock{locks_[n]};
        auto& list = lists_[n];
        list.erase(std::remove(list.begin(), list.end(), m), list.end());
    }

    template <typename Deleted>
    void collect(Idx n, tsl::robin_set<size_t>& out, const Deleted& is_deleted) const {
        std::lock_guard lock{locks_[n]};
        for (auto m : lists_[n]) {
            if (!is_deleted(m)) {
                out.insert(m);
            }
        }
    }

    void reset_node(Idx n) {
        std::lock_guard lock{locks_[n]};
        lists_[n].clear();
    }

    void reset() {
        for (size_t i = 0, imax = lists_.size(); i < imax; ++i) {
            lists_[i].clear();
        }
    }

  private:
    lib::SegmentedVector<std::vector<Idx>> lists_;
    mutable lib::SegmentedVector<SpinLock> locks_;
    std::atomic<bool> recording_{true};
};

} // namespace svs::index::vamana::concurrent::graphs
