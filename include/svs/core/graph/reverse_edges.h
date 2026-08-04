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

#include "svs/lib/segmented_vector.h"
#include "svs/lib/spinlock.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <concepts>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

namespace svs::graphs {

///
/// @brief Per-node index of in-neighbors, maintained as intrusive linked chains over a
/// shared cell pool.
///
/// For each node ``n``, ``R(n)`` holds every ``m`` for which an edge ``m -> n`` was
/// recorded. It is a complete superset of ``n``'s in-neighbors (may contain stale or
/// duplicate entries; never misses a live in-edge). Consolidation uses it to discover the
/// nodes pointing at a deleted node without scanning the whole graph.
///
/// Storage: a per-node ``head_`` chain head, a shared flat ``pool_`` of cells, and a
/// per-node ``chain_locks_`` array. Cell indices are 64-bit (cells count edges). A cell's
/// ``next`` links it within its node's chain when live, or within the free list when free.
/// Freed cells return to a spinlock-guarded free list and are reused by later records, so
/// the pool does not leak across consolidations.
///
template <std::unsigned_integral Idx> class ReverseEdges {
  public:
    using cell_index = uint64_t;
    static constexpr cell_index NIL = std::numeric_limits<cell_index>::max();

    explicit ReverseEdges(size_t num_nodes)
        : head_(num_nodes, NIL)
        , chain_locks_(num_nodes) {}

    void resize(size_t new_size) {
        head_.resize(new_size, NIL);
        chain_locks_.resize(new_size);
    }

    void set_recording(bool on) { recording_.store(on, std::memory_order_release); }

    void record(Idx m, Idx n) {
        if (!recording_.load(std::memory_order_acquire)) {
            return;
        }
        cell_index slot = alloc_cell();
        pool_[slot].value = m;
        std::lock_guard lock{chain_locks_[n]};
        pool_[slot].next = head_[n];
        head_[n] = slot;
    }

    void remove(Idx m, Idx n) {
        std::lock_guard lock{chain_locks_[n]};
        cell_index* link = &head_[n];
        while (*link != NIL) {
            cell_index cur = *link;
            if (pool_[cur].value == m) {
                *link = pool_[cur].next;
                free_cell(cur);
            } else {
                link = &pool_[cur].next;
            }
        }
    }

    void collect(Idx n, std::vector<Idx>& out) const {
        out.clear();
        std::lock_guard lock{chain_locks_[n]};
        for (cell_index cur = head_[n]; cur != NIL; cur = pool_[cur].next) {
            out.push_back(pool_[cur].value);
        }
    }

    void reset_node(Idx n) {
        std::lock_guard lock{chain_locks_[n]};
        cell_index cur = head_[n];
        while (cur != NIL) {
            cell_index next = pool_[cur].next;
            free_cell(cur);
            cur = next;
        }
        head_[n] = NIL;
    }

    void reset() {
        for (size_t i = 0, imax = head_.size(); i < imax; ++i) {
            head_[i] = NIL;
        }
        for (auto& shard : free_shards_) {
            shard.head = NIL;
        }
        pool_top_.store(0, std::memory_order_relaxed);
        free_count_.store(0, std::memory_order_relaxed);
    }

  private:
    struct Cell {
        Idx value;
        cell_index next;
    };

    // Sharded free list: parallelizes the consolidate-time free/reuse traffic. A gate on
    // free_count_ keeps the build path (free list empty) lock-free.
    static constexpr size_t NUM_FREE_SHARDS = 64;
    struct alignas(64) FreeShard {
        SpinLock lock;
        cell_index head = NIL;
    };

    cell_index alloc_cell() {
        // Fast path: nothing freed yet (e.g. a fresh build) — pure atomic bump, no lock.
        if (free_count_.load(std::memory_order_acquire) != 0) {
            size_t start = alloc_rr_.fetch_add(1, std::memory_order_relaxed);
            for (size_t k = 0; k < NUM_FREE_SHARDS; ++k) {
                auto& shard = free_shards_[(start + k) % NUM_FREE_SHARDS];
                // Racy hint: shard.head is written only under shard.lock, so a stale read
                // is safe — a false empty just probes another shard.
                if (shard.head == NIL) {
                    continue;
                }
                std::lock_guard lock{shard.lock};
                if (shard.head != NIL) {
                    cell_index i = shard.head;
                    shard.head = pool_[i].next;
                    free_count_.fetch_sub(1, std::memory_order_acq_rel);
                    return i;
                }
            }
            // All shards drained concurrently; fall through to bump.
        }
        cell_index i = pool_top_.fetch_add(1, std::memory_order_acq_rel);
        if (i >= pool_.size()) {
            std::lock_guard lock{grow_mutex_};
            while (pool_.size() <= i) {
                pool_.resize(std::max(i + 1, pool_.size() * 2));
            }
        }
        return i;
    }

    void free_cell(cell_index i) {
        auto& shard = free_shards_[i % NUM_FREE_SHARDS];
        std::lock_guard lock{shard.lock};
        pool_[i].next = shard.head;
        shard.head = i;
        free_count_.fetch_add(1, std::memory_order_release);
    }

    lib::SegmentedVector<cell_index> head_;
    mutable lib::SegmentedVector<SpinLock> chain_locks_;
    lib::SegmentedVector<Cell> pool_;
    std::atomic<cell_index> pool_top_{0};
    std::atomic<size_t> free_count_{0};
    std::atomic<size_t> alloc_rr_{0};
    std::array<FreeShard, NUM_FREE_SHARDS> free_shards_{};
    std::mutex grow_mutex_;
    std::atomic<bool> recording_{true};
};

} // namespace svs::graphs
