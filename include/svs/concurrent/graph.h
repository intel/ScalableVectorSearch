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

#include "svs/lib/concurrency/atomic_span.h"
#include "svs/lib/concurrency/seqlock.h"
#include "svs/lib/segmented_vector.h"

#include <atomic>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace svs::concurrent {

///
/// @brief A grow-stable adjacency-list graph with per-node sequence locks.
///
/// Layout mirrors ``svs::graphs::SimpleGraphBase``: node ``i`` owns ``stride =
/// max_degree + 1`` contiguous ``Idx`` slots, the first holding the out-degree and the
/// remainder holding neighbor IDs. Unlike ``SimpleBlockedGraph`` -- whose block
/// *descriptor* array is a ``std::vector`` and therefore reallocates on growth -- nodes
/// live in fixed-size segments held by a ``lib::SegmentedVector``, so the address of a
/// node's slot is stable for that node's lifetime.
///
/// Concurrency contract:
///
/// * **One writer at a time.** ``add_edge`` / ``clear_node`` / ``replace_node`` /
///   ``unsafe_resize`` must be serialized by the caller. Because there is only ever one
///   writer, the per-node sequence counters need no writer-writer serialization and no
///   per-node spinlock is required.
/// * **Many concurrent readers.** ``get_node_atomic`` paired with ``seqlock(i)`` gives a
///   race-free read of node ``i``'s adjacency list concurrently with a writer. See
///   ``svs::concurrent::seqlock_greedy_search`` for the retry protocol.
/// * ``get_node`` performs *plain* (non-atomic) loads and is intended for the writer
///   itself and for build-time code that runs before the graph is published to readers
///   (e.g. ``svs::index::vamana::VamanaBuilder``).
/// * **Shrinking frees segments.** ``unsafe_resize`` to a smaller size may free storage,
///   so the caller must have drained readers first (e.g. via an exclusive lock).
///
/// This type satisfies the *unmodified* ``svs::graphs::MemoryGraph`` concept -- in
/// particular ``add_edge`` keeps its ``size_t`` return -- so upstream ``VamanaBuilder``
/// and ``prune`` work against it with no changes to SVS.
///
template <std::unsigned_integral Idx> class SeqLockGraph {
  public:
    using index_type = Idx;
    using reference = std::span<Idx>;
    using const_reference = std::span<const Idx>;
    // Mirrors ``svs::graphs::SimpleGraphBase``. Not required by the graph concepts, but
    // ``svs::index::vamana::GraphConsolidator`` reads ``Graph::const_value_type``.
    using value_type = std::span<Idx>;
    using const_value_type = std::span<const Idx>;

    /// @brief Default number of nodes per segment.
    static constexpr size_t default_segment_size = 512;

  private:
    ///
    /// @brief One contiguous, never-relocated run of ``segment_size * stride`` slots.
    ///
    /// Slots are zero-initialized on allocation. That matters for concurrent readers: a
    /// reader that observes a torn degree can only ever read neighbor slots that were
    /// zero-initialized or validly written, never indeterminate memory, so it cannot
    /// index out of the dataset before the sequence-lock validation rejects the read.
    ///
    class Segment {
      public:
        Segment() = default;
        explicit Segment(size_t n)
            : n_{n}
            , storage_{new Idx[n]()} {}

        Segment(const Segment& other)
            : n_{other.n_}
            , storage_{other.storage_ ? new Idx[other.n_] : nullptr} {
            if (storage_) {
                std::copy(other.storage_.get(), other.storage_.get() + n_, storage_.get());
            }
        }
        Segment& operator=(const Segment& other) {
            if (this != &other) {
                auto tmp = Segment(other);
                *this = std::move(tmp);
            }
            return *this;
        }
        Segment(Segment&&) noexcept = default;
        Segment& operator=(Segment&&) noexcept = default;
        ~Segment() = default;

        Idx* data() noexcept { return storage_.get(); }
        const Idx* data() const noexcept { return storage_.get(); }

      private:
        size_t n_{0};
        std::unique_ptr<Idx[]> storage_{};
    };

  public:
    SeqLockGraph() = default;

    ///
    /// @brief Construct a graph with ``num_nodes`` nodes, each with capacity
    ///     ``max_degree``.
    ///
    SeqLockGraph(
        size_t num_nodes, size_t max_degree, size_t segment_size = default_segment_size
    )
        : max_degree_{max_degree}
        , stride_{max_degree + 1}
        , segment_size_{segment_size} {
        assert(segment_size_ > 0);
        grow_to(num_nodes);
        n_nodes_.store(num_nodes, std::memory_order_release);
    }

    SeqLockGraph(const SeqLockGraph&) = delete;
    SeqLockGraph& operator=(const SeqLockGraph&) = delete;

    // Hand-written because ``n_nodes_`` is an atomic and so is not implicitly movable.
    // Moving is only ever done before the graph is published to readers.
    SeqLockGraph(SeqLockGraph&& other) noexcept
        : max_degree_{other.max_degree_}
        , stride_{other.stride_}
        , segment_size_{other.segment_size_}
        , n_nodes_{other.n_nodes_.load(std::memory_order_relaxed)}
        , segments_{std::move(other.segments_)}
        , seqlocks_{std::move(other.seqlocks_)} {
        other.n_nodes_.store(0, std::memory_order_relaxed);
    }

    SeqLockGraph& operator=(SeqLockGraph&& other) noexcept {
        if (this != &other) {
            max_degree_ = other.max_degree_;
            stride_ = other.stride_;
            segment_size_ = other.segment_size_;
            n_nodes_.store(
                other.n_nodes_.load(std::memory_order_relaxed), std::memory_order_relaxed
            );
            segments_ = std::move(other.segments_);
            seqlocks_ = std::move(other.seqlocks_);
            other.n_nodes_.store(0, std::memory_order_relaxed);
        }
        return *this;
    }

    ~SeqLockGraph() = default;

    ///// ImmutableMemoryGraph

    /// @brief Maximum out-degree any node can hold.
    size_t max_degree() const noexcept { return max_degree_; }

    /// @brief Number of nodes currently in the graph.
    size_t n_nodes() const noexcept { return n_nodes_.load(std::memory_order_acquire); }

    /// @brief Alias for ``n_nodes`` (used by ``svs::graphs::graphs_equal``).
    size_t num_nodes() const noexcept { return n_nodes(); }

    ///
    /// @brief Return node ``i``'s adjacency list using plain loads.
    ///
    /// Only safe when no writer can be concurrently mutating node ``i``. Use
    /// ``get_node_atomic`` on the concurrent read path.
    ///
    const_reference get_node(Idx i) const noexcept {
        const Idx* base = slot(i);
        return const_reference{base + 1, static_cast<size_t>(base[0])};
    }

    ///
    /// @brief Return node ``i``'s adjacency list as an ``AtomicSpan``.
    ///
    /// Every element access is an atomic relaxed load, so the read is race-free even
    /// while the single writer mutates node ``i``. The *contents* may still be a torn
    /// mixture of pre- and post-write state; pair this with ``seqlock(i)`` to detect and
    /// retry such reads.
    ///
    svs::AtomicSpan<Idx> get_node_atomic(Idx i) const noexcept {
        const Idx* base = slot(i);
        return svs::AtomicSpan<Idx>{base + 1, static_cast<size_t>(load_(base[0]))};
    }

    /// @brief Out-degree of node ``i`` (atomic load).
    size_t get_node_degree(Idx i) const noexcept {
        return static_cast<size_t>(load_(slot(i)[0]));
    }

    /// @brief Prefetch node ``i``'s adjacency list. Performance hint only.
    void prefetch_node(Idx i) const noexcept {
        const Idx* base = slot(i);
        for (size_t offset = 0; offset < stride_ * sizeof(Idx); offset += 64) {
            __builtin_prefetch(reinterpret_cast<const char*>(base) + offset);
        }
    }

    /// @brief Access node ``i``'s sequence-lock counter.
    const svs::SeqLockCounter& seqlock(size_t i) const noexcept { return seqlocks_[i]; }

    ///// MemoryGraph (single writer; caller serializes)

    ///
    /// @brief Add the edge ``src -> dst``, returning ``src``'s out-degree afterwards.
    ///
    /// A no-op (returning the current degree) if the edge already exists, if
    /// ``src == dst``, or if ``src``'s adjacency list is already full.
    ///
    size_t add_edge(Idx src, Idx dst) {
        Idx* base = mutable_slot(src);
        const size_t degree = static_cast<size_t>(base[0]);
        if (src == dst || degree >= max_degree_) {
            return degree;
        }
        for (size_t i = 0; i < degree; ++i) {
            if (base[1 + i] == dst) {
                return degree;
            }
        }
        auto seq = seqlocks_[src].begin_write();
        // Publish the neighbor before the degree that exposes it, so a reader can never
        // see a degree covering a slot that has not been written.
        store_(base[1 + degree], dst);
        store_(base[0], static_cast<Idx>(degree + 1));
        seqlocks_[src].end_write(seq);
        return degree + 1;
    }

    /// @brief Drop every edge out of node ``i``.
    void clear_node(Idx i) {
        Idx* base = mutable_slot(i);
        auto seq = seqlocks_[i].begin_write();
        store_(base[0], Idx{0});
        seqlocks_[i].end_write(seq);
    }

    ///
    /// @brief Overwrite node ``src``'s adjacency list with ``neighbors``.
    ///
    /// The degree is dropped to zero first so that a reader which began before the write
    /// only ever observes a validly-written *prefix* of the list.
    ///
    template <typename R> void replace_node(Idx src, const R& neighbors) {
        Idx* base = mutable_slot(src);
        const size_t n = std::size(neighbors);
        assert(n <= max_degree_);

        auto seq = seqlocks_[src].begin_write();
        store_(base[0], Idx{0});
        size_t k = 0;
        for (auto id : neighbors) {
            store_(base[1 + k], static_cast<Idx>(id));
            ++k;
        }
        store_(base[0], static_cast<Idx>(n));
        seqlocks_[src].end_write(seq);
    }

    ///
    /// @brief Resize the graph to ``new_size`` nodes.
    ///
    /// Growing is safe against concurrent readers: storage is allocated and the sequence
    /// counters extended before the new node count is published. Shrinking may free
    /// segments and therefore requires that readers have been drained.
    ///
    void unsafe_resize(size_t new_size) {
        const size_t current = n_nodes_.load(std::memory_order_relaxed);
        if (new_size == current) {
            return;
        }
        if (new_size < current) {
            n_nodes_.store(new_size, std::memory_order_release);
            seqlocks_.resize(new_size);
            return;
        }
        grow_to(new_size);
        n_nodes_.store(new_size, std::memory_order_release);
    }

    /// @brief Append a single node.
    void add_node() { unsafe_resize(n_nodes() + 1); }

    /// @brief Number of nodes addressable without allocating another segment.
    size_t capacity() const noexcept { return segments_.size() * segment_size_; }

    /// @brief Bytes held by the adjacency storage and sequence counters.
    size_t bytes_reserved() const noexcept {
        return capacity() * stride_ * sizeof(Idx) +
               seqlocks_.capacity() * sizeof(svs::SeqLockCounter);
    }

  private:
    // Allocate segments and sequence counters so that every index < n is addressable.
    // Does not publish the new node count.
    void grow_to(size_t n) {
        while (capacity() < n) {
            segments_.push_back(Segment(segment_size_ * stride_));
        }
        if (seqlocks_.size() < n) {
            seqlocks_.resize(n);
        }
    }

    const Idx* slot(size_t i) const noexcept {
        return segments_[i / segment_size_].data() + (i % segment_size_) * stride_;
    }
    Idx* mutable_slot(size_t i) noexcept {
        return segments_[i / segment_size_].data() + (i % segment_size_) * stride_;
    }

    // Relaxed atomics, not plain accesses. The sequence counter tells a reader whether what
    // it read was *coherent*, but it does not make the reads themselves defined behaviour:
    // a reader legitimately touches slots a writer is modifying, which is a data race
    // unless the accesses are atomic. Relaxed is the weakest ordering that removes the
    // race, and on x86 both of these compile to a plain MOV -- so this costs nothing but
    // buys a well-defined program that ThreadSanitizer can actually verify.
    //
    // Define SVS_CONCURRENT_UNSAFE_PLAIN_GRAPH_ACCESS to drop the atomics. That build is
    // *wrong* on purpose: it exists so the TSan targets can be shown to fail when they
    // should, which is the only way a clean TSan run means anything.
#ifdef SVS_CONCURRENT_UNSAFE_PLAIN_GRAPH_ACCESS
    static void store_(Idx& s, Idx v) noexcept { s = v; }
    static Idx load_(const Idx& s) noexcept { return s; }
#else
    static void store_(Idx& s, Idx v) noexcept {
        std::atomic_ref<Idx>(s).store(v, std::memory_order_relaxed);
    }
    static Idx load_(const Idx& s) noexcept {
        return std::atomic_ref<Idx>(const_cast<Idx&>(s)).load(std::memory_order_relaxed);
    }
#endif

    size_t max_degree_{0};
    size_t stride_{1};
    size_t segment_size_{default_segment_size};
    std::atomic<size_t> n_nodes_{0};
    // Grow-stable: appending a segment never relocates existing segments or the
    // directory that addresses them.
    svs::lib::SegmentedVector<Segment> segments_{};
    svs::SeqLockArray seqlocks_{};
};

} // namespace svs::concurrent
