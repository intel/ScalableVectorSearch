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
#include <memory>
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
template <std::unsigned_integral Idx, typename Alloc = std::allocator<Idx>>
class ReverseEdges {
  public:
    /// Per-node in-neighbor list, allocated through `Alloc` (rebound to `Idx`).
    using idx_allocator = typename std::allocator_traits<Alloc>::template rebind_alloc<Idx>;
    using list_type = std::vector<Idx, idx_allocator>;

  private:
    // Element allocators for the two segmented directories: one holds the per-node
    // `list_type` control blocks, the other the per-node `SpinLock`s.
    using list_dir_allocator =
        typename std::allocator_traits<Alloc>::template rebind_alloc<list_type>;
    using lock_dir_allocator =
        typename std::allocator_traits<Alloc>::template rebind_alloc<SpinLock>;

  public:
    explicit ReverseEdges(size_t num_nodes, const Alloc& alloc = Alloc{})
        : idx_alloc_(alloc)
        , lists_(num_nodes, list_type(idx_alloc_), list_dir_allocator(alloc))
        , locks_(num_nodes, lock_dir_allocator(alloc)) {}

    void resize(size_t new_size) {
        // New per-node lists carry the allocator via the prototype (copy-constructed into
        // each new slot, propagating the allocator through the vector's copy ctor).
        lists_.resize(new_size, list_type(idx_alloc_));
        locks_.resize(new_size);
    }

    void set_recording(bool on) { recording_.store(on, std::memory_order_release); }

    // lists are keeped sorted for fast binary search of elements.
    void record(Idx m, Idx n) {
        if (!recording_.load(std::memory_order_acquire)) {
            return;
        }
        std::lock_guard lock{locks_[n]};
        auto& list = lists_[n];
        // lists_[n].push_back(m);
        auto it = std::lower_bound(list.begin(), list.end(), m);
        if (it != list.end() && *it == m)
            return; // already present
        list.insert(it, m);
    }

    void remove(Idx m, Idx n) {
        std::lock_guard lock{locks_[n]};
        auto& list = lists_[n];
        // list.erase(std::remove(list.begin(), list.end(), m), list.end());
        auto it = std::lower_bound(list.begin(), list.end(), m);
        if (it != list.end() && *it == m)
            list.erase(it);
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
        list_type empty(idx_alloc_);
        lists_[n].swap(empty);
    }

    void reset() {
        for (size_t i = 0, imax = lists_.size(); i < imax; ++i) {
            reset_node(i);
        }
    }

    ///
    /// @brief Bytes allocated by the reverse-edge index.
    ///
    size_t memory_bytes() const {
        size_t elems = 0;
        for (size_t i = 0, imax = lists_.size(); i < imax; ++i) {
            std::lock_guard lock{locks_[i]};
            elems += lists_[i].capacity();
        }
        return lists_.capacity() * sizeof(list_type) +
               locks_.capacity() * sizeof(SpinLock) + elems * sizeof(Idx);
    }

  private:
    // Declared before lists_: the constructor's lists_ initializer reads idx_alloc_.
    idx_allocator idx_alloc_;
    lib::SegmentedVector<list_type, list_dir_allocator> lists_;
    mutable lib::SegmentedVector<SpinLock, lock_dir_allocator> locks_;
    std::atomic<bool> recording_{true};
};

} // namespace svs::index::vamana::concurrent::graphs
