/*
 * Copyright 2023 Intel Corporation
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

#include "svs/concurrent/blocked_data.h"
#include "svs/concurrent/graph_concepts.h"
#include "svs/concurrent/reverse_edges.h"
#include "svs/concurrent/spinlock.h"
#include "svs/core/data/simple.h"
#include "svs/lib/algorithms.h"
#include "svs/lib/boundscheck.h"
#include "svs/lib/concurrency/atomic_span.h"
#include "svs/lib/concurrency/seqlock.h"
#include "svs/lib/saveload.h"
#include "svs/lib/segmented_vector.h"
#include "svs/lib/threads.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

namespace svs::index::vamana::concurrent::graphs {

//
// We rely on an implicit layout for the graphs where length is stored inline with the
// adjacency list like:
//
// Node 0  :  Len N0 N1 N2 .... Nm
// Node 1  :  Len N0 N1 N2 .... Nm
// Node 2  :  Len N0 N1 N2 .... Nm
// ...
// Node K  :  Len N0 N1 N2 .... Nm
//
// Note that the the length variable `Len` is the same type as the adjacency list entries.
//
// In general, C++'s support for type-punning, even for trivially constructible and
// copyable types leaves quite a bit of head-scratching.
//
// The utilities developed here are meant to help with dealing with the implicit layout
// described above.
//
// Base class for packed graphs.
// Should not be used directly. Rather, one of it's derived classes should be used instead.
//
template <std::unsigned_integral Idx, data::MemoryDataset Data> class SimpleGraphBase {
  public:
    using data_type = Data;

    /// The integer representation used to represent vertices in this graph.
    using index_type = Idx;
    using value_type = std::span<Idx>;
    using const_value_type = AtomicSpan<const Idx>;

    /// Type used to represent mutable adjacency lists externally.
    using reference = std::span<Idx>;
    /// Type used to represent constant adjacency lists externally.
    using const_reference = AtomicSpan<const Idx>;

    ///
    /// @brief Construct an empty graph of the desired size.
    ///
    /// @param num_nodes The number of nodes in the graph.
    /// @param max_degree The maximum degree of the graph.
    ///
    /// Implementation notes: Requires that the memory backing the dataset for this graph
    /// is default constructible.
    ///
    explicit SimpleGraphBase(size_t num_nodes, size_t max_degree)
        : data_{num_nodes, max_degree + 1}
        , max_degree_{lib::narrow<Idx>(max_degree)}
        , seq_counters_(num_nodes)
        , node_locks_(num_nodes) {
        reset();
    }

    // TODO: Constrain template approparitely.
    template <typename Allocator>
    explicit SimpleGraphBase(
        size_t num_nodes, size_t max_degree, const Allocator& allocator
    )
        : data_{num_nodes, max_degree + 1, allocator}
        , max_degree_{lib::narrow<Idx>(max_degree)}
        , seq_counters_(num_nodes)
        , node_locks_(num_nodes) {
        reset();
    }

    explicit SimpleGraphBase(data_type data)
        : data_{std::move(data)}
        , max_degree_{lib::narrow<Idx>(data_.dimensions() - 1)}
        , seq_counters_(data_.size())
        , node_locks_(data_.size()) {}

    std::span<const Idx> raw_row(Idx i) const { return data_.get_datum(i); }

    ///
    /// @brief Return the outward adjacency list for vertex ``i``.
    ///
    /// @param i The vertex to get the ID for.
    ///
    const_reference get_node(Idx i) const {
        // Get the raw data.
        std::span<const Idx> raw_data = data_.get_datum(i);
        Idx num_neighbors = relaxed_load(raw_data.front());
        // Clamp to max_degree to safely handle torn reads of the length field.
        num_neighbors = std::min(num_neighbors, max_degree_);

        // Maybe prefetch the rest of the adjacency list.
        size_t bytes = (1 + num_neighbors) * sizeof(Idx);
        if (bytes > lib::CACHELINE_BYTES) {
            lib::prefetch(std::as_bytes(raw_data).subspan(lib::CACHELINE_BYTES));
        }
        return AtomicSpan<const Idx>(raw_data.data() + 1, num_neighbors);
    }

    ///
    /// @brief Return whether or not the adjacency list has an edge from ``src`` to ``dst``.
    ///
    /// Complexity: Linear in the maximum degree.
    ///
    bool has_edge(Idx src, Idx dst) const {
        for (;;) {
            auto maybe_seq = seq_counters_[src].read_begin();
            if (!maybe_seq) {
                svs::detail::pause();
                continue;
            }
            const auto& list = get_node(src);
            bool found = (std::find(list.begin(), list.end(), dst) != list.end());
            if (seq_counters_[src].read_validate(*maybe_seq)) {
                return found;
            }
            svs::detail::pause();
        }
    }

    ///
    /// @brief Return the current out degree of vertex ``i``.
    ///
    size_t get_node_degree(Idx i) const { return relaxed_load(data_.get_datum(i).front()); }

    ///
    /// @brief Prefetch the adjacency list for node ``i`` into the L1 cache.
    ///
    void prefetch_node(Idx i) const { data_.prefetch(i); }

    ///
    /// @brief Remove all outgoing neighbors from node ``i``.
    ///
    /// *Note*: As an implementation detail, this method doesn't mutate the actual adjacency
    /// list. Instead, it simply sets the number of neighbors to zero.
    ///
    /// The complexity of this operation is `O(1)`.
    ///
    void clear_node(Idx i) {
        std::lock_guard lock{node_locks_[i]};
        auto seq = seq_counters_[i].begin_write();
        relaxed_store(data_.get_datum(i).front(), 0);
        seq_counters_[i].end_write(seq);
    }

    ///
    /// @brief Remove all edges from the graph.
    ///
    void reset() {
        for (size_t i = 0; i < n_nodes(); ++i) {
            clear_node(i);
        }
    }

    ///
    /// @brief Replace the adjacency list for vertex ``i``.
    ///
    /// @param i The vertex whose adjacency list is being modified.
    /// @param new_neighbors The new adjacency list for vertex ``i``.
    ///
    /// Takes at most ``max_degree()`` elements from ``new_neighbors``. May silently drop
    /// any excess neighbors.
    ///
    /// **Preconditions:**
    ///
    /// * All elements of ``new_neighbors`` must be between 0 and ``n_nodes()``
    /// * All elements of ``new_neighbors`` must be unique.
    ///
    void replace_node(Idx i, const std::vector<Idx>& new_neighbors) {
        replace_node(i, std::span{new_neighbors.data(), new_neighbors.size()});
    }

    /// @copydoc replace_node(Idx,const std::vector<Idx>&)
    void replace_node(Idx i, std::span<const Idx> new_neighbors) {
        replace_node_impl(i, new_neighbors);
    }

    /// @copydoc replace_node(Idx,const std::vector<Idx>&)
    void replace_node(Idx i, AtomicSpan<const Idx> new_neighbors) {
        replace_node_impl(i, new_neighbors);
    }

    ///
    /// @brief Add an edge from vertex ``src`` to vertex ``dst``.
    ///
    /// @param src The source vertex.
    /// @param dst The destination vertex.
    ///
    /// @returns The number of out neighbors of ``src`` after ``dst`` is inserted.
    ///
    /// The adjacency list of ``src`` will be left unchanged if:
    /// * ``src == dst`` (no self assignment)
    /// * ``get_node_degree(src) == max_degree()`` (adjacency list is already full)
    /// * ``dst`` is already an out-neighbor of ``src``.
    ///
    AddEdgeResult add_edge(Idx src, Idx dst) {
        // Don't assign a node as its own neighbor.
        if (src == dst) {
            return AddEdgeResult::AlreadyExists;
        }

        if constexpr (checkbounds_v) {
            if (dst >= n_nodes()) {
                throw ANNEXCEPTION(
                    "Trying to assign an edge to node {} when the number of nodes in the "
                    "graph is {}!",
                    dst,
                    n_nodes()
                );
            }
        }

        // Acquire lock — all reads and writes under the lock to prevent
        // concurrent writers from seeing stale state.
        std::lock_guard lock{node_locks_[src]};

        // Check if there's room for the new node.
        std::span<Idx> raw_data = data_.get_datum(src);
        Idx current_size = raw_data.front();
        if (current_size == max_degree_) {
            return AddEdgeResult::Full;
        }

        // At this point, we know there is room.
        // Next, we need to find the position where we will insert the new edge.
        // We fuse this with redundant edge insertion detection since the insertion
        // position will also tell us where the edge would already exist.
        Idx new_size = current_size + 1;
        value_type adjacency_list = raw_data.subspan(1, new_size);

        auto begin = adjacency_list.begin();
        auto end = adjacency_list.end();

        // TODO: Replace with binary search eventually.
        // Blocking issue: legacy loaded graphs need validation of the sorted adjacency
        // lists.
        auto it = std::find(begin, end - 1, dst);
        // auto it = std::lower_bound(begin, end - 1, dst);
        if (it != end - 1 && (*it == dst)) {
            return AddEdgeResult::AlreadyExists;
        }

        auto seq = seq_counters_[src].begin_write();

        // Insert at the new location using atomic stores.
        for (auto dst_it = end - 1, src_it = end - 2; dst_it != it; --dst_it, --src_it) {
            relaxed_store(*dst_it, *src_it);
        }
        relaxed_store(*it, dst);

        // Update the number of neighbors.
        relaxed_store(raw_data.front(), new_size);

        seq_counters_[src].end_write(seq);
        if (reverse_edges_) {
            reverse_edges_->record(src, dst);
        }
        return AddEdgeResult::Added;
    }

    /// Return the maximum out-degree this graph is capable of containing.
    size_t max_degree() const { return max_degree_; }
    /// Return the number of vertices currently in the graph.
    size_t n_nodes() const { return data_.size(); }

    /// Return the maximum number of vertices this graph can hold without
    /// reallocating any of its underlying storage.
    size_t capacity() const {
        return std::min({data_.capacity(), seq_counters_.capacity(), node_locks_.capacity()}
        );
    }

    const data_type& get_data() const { return data_; }
    data_type& get_data() { return data_; }

    // Resizeable API
    void unsafe_resize(size_t new_size) {
        data_.resize(new_size);
        seq_counters_.resize(new_size);
        node_locks_.resize(new_size);
        if (reverse_edges_) {
            reverse_edges_->resize(new_size);
        }
    }
    void add_node() { unsafe_resize(n_nodes() + 1); }

    /// @brief Access the per-node sequence lock counters for concurrent read validation.
    const SeqLockArray& seq_counters() const { return seq_counters_; }

    ///
    /// @brief Enable maintenance of the per-node reverse-edge (in-neighbor) index.
    ///
    /// Off by default (null): every graph mutator hook is then a single null check, and
    /// the static index / compaction scratch graphs pay nothing. Enabled only by the
    /// dynamic index.
    ///
    void enable_reverse_edges() {
        reverse_edges_ = std::make_unique<ReverseEdges<Idx>>(n_nodes());
    }

    ReverseEdges<Idx>* reverse_edges() { return reverse_edges_.get(); }
    const ReverseEdges<Idx>* reverse_edges() const { return reverse_edges_.get(); }

    ///
    /// @brief Rebuild the reverse-edge index from the current (quiescent) graph.
    ///
    /// Records ``src`` into ``R(dst)`` for *every* edge ``src -> dst``. Must be called with
    /// no concurrent graph mutation (load, post-build, post-compact).
    ///
    /// It is tempting to skip edges whose reverse ``dst -> src`` also exists, halving the
    /// index: `gather_work_set` visits `out(d) union R(d)`, so a symmetric in-neighbor is
    /// already covered by `out(d)`. That weaker invariant is *not maintainable*, though.
    /// It reads "for every edge `u -> d`: `u` is in `R(d)` **or** the edge `d -> u`
    /// exists", and the second disjunct is falsified whenever consolidation rewires `d`
    /// and drops `d -> u` -- at which point `u`'s in-edge becomes invisible and a later
    /// deletion of `d` leaves `u` pointing at a retired slot. Recording unconditionally
    /// gives the strictly stronger `R(d) contains in(d)`, which no edge *removal* can
    /// break and which every mutator already preserves on edge creation.
    ///
    template <threads::ThreadPool Pool> void rebuild_reverse_edges(Pool& threadpool) {
        if (!reverse_edges_) {
            return;
        }
        reverse_edges_->reset();
        threads::parallel_for(
            threadpool,
            threads::StaticPartition{threads::UnitRange<size_t>{0, n_nodes()}},
            [&](const auto& is, uint64_t /*tid*/) {
                for (auto src : is) {
                    for (auto dst : get_node(lib::narrow_cast<Idx>(src))) {
                        reverse_edges_->record(
                            lib::narrow_cast<Idx>(src), lib::narrow_cast<Idx>(dst)
                        );
                    }
                }
            }
        );
    }

    ///// Saving
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);
    static constexpr std::string_view serialization_schema = "default_graph";

    lib::SaveTable metadata() const {
        auto table = lib::SaveTable(
            serialization_schema,
            save_version,
            {{"name", "graph"},
             {"max_degree", lib::save(max_degree())},
             {"num_vertices", lib::save(n_nodes())},
             {"eltype", lib::save(datatype_v<Idx>)}}
        );
        return table;
    }

    template <class FileName>
    lib::SaveTable metadata(const FileName& filename, const lib::UUID& uuid) const {
        auto table = metadata();
        table.insert("binary_file", filename);
        table.insert("uuid", uuid.str());
        return table;
    }

    lib::SaveTable save(const lib::SaveContext& ctx) const {
        auto uuid = lib::UUID{};
        auto filename = ctx.generate_name("graph");
        io::save(data_, io::NativeFile(filename), uuid);
        return metadata(lib::save(filename.filename()), uuid);
    }

    void save(std::ostream& os) const { io::save(data_, os); }

  protected:
    template <lib::LazyInvocable<data_type> F, typename... Args>
    static lib::lazy_result_t<F, data_type>
    load(const lib::LoadTable& table, const F& lazy, Args&&... args) {
        // Perform a sanity check on the element type.
        // Make sure we're loading the correct kind.
        auto eltype = lib::load_at<DataType>(table, "eltype");
        if (eltype != datatype_v<Idx>) {
            throw ANNEXCEPTION(
                "Trying to load a graph with adjacency list types {} to a graph with "
                "adjacency list types {}.",
                // Qualified: `svs::index::vamana::name(SlotMetadata)` hides `svs::name`
                // for unqualified lookup from this nested namespace.
                svs::name(eltype),
                svs::name<datatype_v<Idx>>()
            );
        }

        // Now that this is out of the way, resolve the file and load the data.
        auto uuid = lib::load_at<lib::UUID>(table, "uuid");
        auto binaryfile = io::find_uuid(table.context().get_directory(), uuid);
        if (!binaryfile.has_value()) {
            throw ANNEXCEPTION("Could not open file with uuid {}!", uuid.str());
        }
        return lazy(data_type::load(binaryfile.value(), std::forward<Args>(args)...));
    }

    template <lib::LazyInvocable<data_type> F, typename... AllocArgs>
    static lib::lazy_result_t<F, data_type> load(
        const lib::ContextFreeLoadTable& table,
        const F& lazy,
        std::istream& is,
        AllocArgs&&... alloc_args
    ) {
        // Perform a sanity check on the element type.
        // Make sure we're loading the correct kind.
        auto eltype = lib::load_at<DataType>(table, "eltype");
        if (eltype != datatype_v<Idx>) {
            throw ANNEXCEPTION(
                "Trying to load a graph with adjacency list types {} to a graph with "
                "adjacency list types {}.",
                // Qualified: `svs::index::vamana::name(SlotMetadata)` hides `svs::name`
                // for unqualified lookup from this nested namespace.
                svs::name(eltype),
                svs::name<datatype_v<Idx>>()
            );
        }

        size_t num_vertices = lib::load_at<size_t>(table, "num_vertices");
        size_t max_degree = lib::load_at<size_t>(table, "max_degree");

        // Build a table compatible with GenericSerializer
        auto data_table = toml::table{
            {lib::config_schema_key, data::GenericSerializer::serialization_schema},
            {lib::config_version_key, data::GenericSerializer::save_version.str()},
            {"eltype", lib::save(datatype_v<Idx>)},
            {"num_vectors", lib::save(num_vertices)},
            {"dims", lib::save(max_degree + 1)},
        };

        return lazy(
            data_type::load(lib::ContextFreeLoadTable(data_table), is, alloc_args...)
        );
    }

  private:
    // Adjacency-slot element access. Every slot is read by lock-free searches while writers
    // mutate it, so all accesses go through `std::atomic_ref` with relaxed ordering: the
    // per-node sequence-lock counters, not these individual accesses, establish the
    // ordering that makes a read consistent. Relaxed atomics compile to plain loads and
    // stores on the platforms SVS targets, so this costs nothing at runtime -- it only
    // removes the data race that would otherwise make the program ill-formed.
    //
    // `SVS_CONCURRENT_UNSAFE_PLAIN_GRAPH_ACCESS` degrades these to plain accesses. It
    // exists solely as a ThreadSanitizer negative control (see `tests/CMakeLists.txt`): a
    // clean TSan run over this graph only means something if the *same* run reports races
    // once the atomics are taken away. Never define it in a real build.
    static Idx relaxed_load(const Idx& slot) {
#if defined(SVS_CONCURRENT_UNSAFE_PLAIN_GRAPH_ACCESS)
        return slot;
#else
        return std::atomic_ref<Idx>(const_cast<Idx&>(slot)).load(std::memory_order_relaxed);
#endif
    }

    static void relaxed_store(Idx& slot, Idx value) {
#if defined(SVS_CONCURRENT_UNSAFE_PLAIN_GRAPH_ACCESS)
        slot = value;
#else
        std::atomic_ref<Idx>(slot).store(value, std::memory_order_relaxed);
#endif
    }

    template <typename Span> void replace_node_impl(Idx i, const Span& new_neighbors) {
        std::span<const Idx> old_snapshot{};
        std::array<Idx, MAX_STACK_DEGREE> old_buffer;
        Idx old_size = 0;

        std::lock_guard lock{node_locks_[i]};
        std::span<Idx> raw_data = data_.get_datum(i);

        if (reverse_edges_) {
            old_size = relaxed_load(raw_data[0]);
            old_size = std::min({old_size, max_degree_, Idx{MAX_STACK_DEGREE}});
            for (Idx j = 0; j < old_size; ++j) {
                old_buffer[j] = relaxed_load(raw_data[1 + j]);
            }
            old_snapshot = std::span<const Idx>(old_buffer.data(), old_size);
        }

        // Clamp the number of elements to copy to the maximum out degree to correctly
        // handle the case where the caller passes in too many neighbors.
        Idx elements_to_copy =
            std::min(max_degree_, lib::narrow_cast<Idx>(new_neighbors.size()));

        auto seq = seq_counters_[i].begin_write();
        for (Idx j = 0; j < elements_to_copy; ++j) {
            relaxed_store(raw_data[1 + j], new_neighbors[j]);
        }
        relaxed_store(raw_data[0], elements_to_copy);
        seq_counters_[i].end_write(seq);

        if (reverse_edges_) {
            for (Idx j = 0; j < elements_to_copy; ++j) {
                Idx dst = new_neighbors[j];
                bool existed = std::find(old_snapshot.begin(), old_snapshot.end(), dst) !=
                               old_snapshot.end();
                if (!existed) {
                    reverse_edges_->record(i, dst);
                }
            }
        }
    }

    // Upper bound on adjacency degree we snapshot on the stack for the reverse-edge diff.
    // Graphs with larger degree still work; the diff is simply skipped past this bound
    // (over-recording, never under-recording, preserving completeness).
    static constexpr Idx MAX_STACK_DEGREE = 256;

  protected:
    data_type data_;
    Idx max_degree_;
    SeqLockArray seq_counters_;
    // Grow-stable: a concurrent add_points Phase 3 backprop locks node_locks_[other]
    // lock-free while another add grows the array; segmented storage keeps existing
    // locks at stable addresses. See svs/lib/segmented_vector.h.
    lib::SegmentedVector<SpinLock> node_locks_;
    // Per-node in-neighbor index. Null (disabled) unless the owning index enables it.
    std::unique_ptr<ReverseEdges<Idx>> reverse_edges_ = nullptr;
};

/////
///// Concrete implementations.
/////

///
/// @brief Simple graph representation.
///
/// @tparam Idx The integer type used to encode vertices in this graph.
///
/// This data structure represents a graph using a single large allocation and a set maximum
/// degree. Accessing adjacency lists takes `O(1)` time. Only out-bound edges are stored.
///
template <std::unsigned_integral Idx, typename Alloc = HugepageAllocator<Idx>>
class SimpleGraph : public SimpleGraphBase<Idx, data::SimpleData<Idx, Dynamic, Alloc>> {
  public:
    using parent_type = SimpleGraphBase<Idx, data::SimpleData<Idx, Dynamic, Alloc>>;
    using data_type = typename parent_type::data_type;
    using parent_type::get_data;

    ///
    /// @brief Construct a new empty graph.
    ///
    /// @param num_nodes The number of nodes in the graph.
    /// @param max_degree The maximum allowable degree in the graph.
    ///
    explicit SimpleGraph(size_t num_nodes, size_t max_degree)
        : parent_type{num_nodes, max_degree} {}

    explicit SimpleGraph(size_t num_nodes, size_t max_degree, const Alloc& allocator)
        : parent_type{num_nodes, max_degree, allocator} {}

    explicit SimpleGraph(data_type data)
        : parent_type{std::move(data)} {}

    explicit SimpleGraph(parent_type&& parent)
        : parent_type(std::move(parent)) {}

    ///// Loading
    static constexpr SimpleGraph
    load(const lib::LoadTable& table, const Alloc& allocator = {}) {
        auto lazy = lib::Lazy([](data_type data) { return SimpleGraph(std::move(data)); });
        return parent_type::load(table, lazy, allocator);
    }

    template <typename... AllocArgs>
    static constexpr SimpleGraph load(
        const lib::ContextFreeLoadTable& table, std::istream& is, AllocArgs&&... alloc_args
    ) {
        auto lazy = lib::Lazy([](data_type data) { return SimpleGraph(std::move(data)); });
        return parent_type::load(table, lazy, is, std::forward<AllocArgs>(alloc_args)...);
    }

    static constexpr SimpleGraph
    load(const std::filesystem::path& path, const Alloc& allocator = {}) {
        if (data::detail::is_likely_reload(path)) {
            return lib::load_from_disk<SimpleGraph>(path, allocator);
        } else {
            return SimpleGraph(data_type::load(path, allocator));
        }
    }

    template <typename... AllocArgs>
    static constexpr SimpleGraph load(std::istream& is, AllocArgs&&... alloc_args) {
        return lib::load_from_stream<SimpleGraph>(
            is, std::forward<AllocArgs>(alloc_args)...
        );
    }
};

template <typename Idx, typename A1, typename A2>
bool operator==(const SimpleGraph<Idx, A1>& x, const SimpleGraph<Idx, A2>& y) {
    return graphs_equal(x, y);
}

template <std::unsigned_integral Idx>
class SimpleBlockedGraph : public SimpleGraphBase<
                               Idx,
                               SegmentedBlockedData<Idx, Dynamic, HugepageAllocator<Idx>>> {
  public:
    using parent_type =
        SimpleGraphBase<Idx, SegmentedBlockedData<Idx, Dynamic, HugepageAllocator<Idx>>>;
    using data_type = typename parent_type::data_type;

    // Constructors
    SimpleBlockedGraph(size_t num_nodes, size_t max_degree)
        : parent_type{num_nodes, max_degree} {}

    explicit SimpleBlockedGraph(data_type data)
        : parent_type{std::move(data)} {}

    explicit SimpleBlockedGraph(parent_type&& parent)
        : parent_type(std::move(parent)) {}

    ///// Loading
    static constexpr SimpleBlockedGraph load(const lib::LoadTable& table) {
        auto lazy =
            lib::Lazy([](data_type data) { return SimpleBlockedGraph(std::move(data)); });
        return parent_type::load(table, lazy);
    }

    static constexpr SimpleBlockedGraph
    load(const lib::ContextFreeLoadTable& table, std::istream& is) {
        auto lazy =
            lib::Lazy([](data_type data) { return SimpleBlockedGraph(std::move(data)); });
        return parent_type::load(table, lazy, is);
    }

    static constexpr SimpleBlockedGraph load(const std::filesystem::path& path) {
        if (data::detail::is_likely_reload(path)) {
            return lib::load_from_disk<SimpleBlockedGraph>(path);
        } else {
            return SimpleBlockedGraph(data_type::load(path));
        }
    }

    static constexpr SimpleBlockedGraph load(std::istream& is) {
        return lib::load_from_stream<SimpleBlockedGraph>(is);
    }
};

} // namespace svs::index::vamana::concurrent::graphs
