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

#include "svs/concepts/graph.h"
#include "svs/core/data/simple.h"
#include "svs/lib/algorithms.h"
#include "svs/lib/boundscheck.h"
#include "svs/lib/saveload.h"

#include <cassert>
#include <cstdint>
#include <span>
#include <type_traits>

namespace svs::graphs {

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
    using const_value_type = std::span<const Idx>;

    /// Type used to represent mutable adjacency lists externally.
    using reference = std::span<Idx>;
    /// Type used to represent constant adjacency lists externally.
    using const_reference = std::span<const Idx>;

    ///
    /// @brief Construct an emptry graph of the desired size.
    ///
    /// @param num_nodes The number of nodes in the graph.
    /// @param max_degree The maximum degree of the graph.
    ///
    /// Implementation notes: Requires that the memory backing the dataset for this graph
    /// is default constructible.
    ///
    explicit SimpleGraphBase(size_t num_nodes, size_t max_degree)
        : data_{num_nodes, max_degree + 1}
        , max_degree_{lib::narrow<Idx>(max_degree)} {
        reset();
    }

    // TODO: Constrain template approparitely.
    template <typename Allocator>
    explicit SimpleGraphBase(
        size_t num_nodes, size_t max_degree, const Allocator& allocator
    )
        : data_{num_nodes, max_degree + 1, allocator}
        , max_degree_{lib::narrow<Idx>(max_degree)} {
        reset();
    }

    explicit SimpleGraphBase(data_type data)
        : data_{std::move(data)}
        , max_degree_{lib::narrow<Idx>(data_.dimensions() - 1)} {}

    const_reference raw_row(Idx i) const { return data_.get_datum(i); }

    ///
    /// @brief Return the outward adjacency list for vertex ``i``.
    ///
    /// @param i The vertex to get the ID for.
    ///
    const_reference get_node(Idx i) const {
        // Get the raw data.
        std::span<const Idx> raw_data = data_.get_datum(i);
        auto num_neighbors = raw_data.front();

        // Maybe prefetch the rest of the adjacncy list.
        size_t bytes = (1 + num_neighbors) * sizeof(Idx);
        if (bytes > lib::CACHELINE_BYTES) {
            lib::prefetch(std::as_bytes(raw_data).subspan(lib::CACHELINE_BYTES));
        }
        return raw_data.subspan(1, num_neighbors);
    }

    ///
    /// @brief Return whether or not the adjacency list has an edge from ``src`` to ``dst``.
    ///
    /// Complexity: Linear in the maximum degree.
    ///
    bool has_edge(Idx src, Idx dst) const {
        const auto& list = get_node(src);
        auto begin = list.begin();
        auto end = list.end();
        return (std::find(begin, end, dst) != end);
    }

    ///
    /// @brief Return the current out degree of vertex ``i``.
    ///
    size_t get_node_degree(Idx i) const { return data_.get_datum(i).front(); }

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
        Idx& num_neighbors = data_.get_datum(i).front();
        num_neighbors = 0;
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
        std::span<Idx> raw_data = data_.get_datum(i);

        // Clamp the number of elements to copy to the maximum out degree to correctly
        // handle the case where the caller passes in too many neighbors.
        //
        // In this case, we take the first `max_degree_` number of elements.
        Idx elements_to_copy =
            std::min(max_degree_, lib::narrow_cast<Idx>(new_neighbors.size()));

        std::span<const Idx> adjusted_neighbors = new_neighbors.first(elements_to_copy);
        value_type adjacency_list = raw_data.subspan(1, elements_to_copy);

        std::copy(
            adjusted_neighbors.begin(), adjusted_neighbors.end(), adjacency_list.begin()
        );
        raw_data.front() = elements_to_copy;
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
    size_t add_edge(Idx src, Idx dst) {
        // Don't assign a node as its own neighbor.
        if (src == dst) {
            return get_node_degree(src);
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

        // Check if there's room for the new node.
        std::span<Idx> raw_data = data_.get_datum(src);
        Idx current_size = raw_data.front();
        if (current_size == max_degree_) {
            return current_size;
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
            return current_size;
        }

        // Insert at the new location.
        std::copy_backward(it, end - 1, end);
        (*it) = dst;

        // // Assign the new edge and update the number of neighbors.
        // adjacency_list.back() = dst;
        raw_data.front() = new_size;
        return new_size;
    }

    /// Return the maximum out-degree this graph is capable of containing.
    size_t max_degree() const { return max_degree_; }
    /// Return the number of vertices currently in the graph.
    size_t n_nodes() const { return data_.size(); }

    const data_type& get_data() const { return data_; }
    data_type& get_data() { return data_; }

    // Resizeable API
    void unsafe_resize(size_t new_size) { data_.resize(new_size); }
    void add_node() { unsafe_resize(n_nodes() + 1); }

    ///// Saving
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);
    static constexpr std::string_view serialization_schema = "default_graph";
    lib::SaveTable save(const lib::SaveContext& ctx) const {
        auto uuid = lib::UUID{};
        auto filename = ctx.generate_name("graph");
        io::save(data_, io::NativeFile(filename), uuid);
        return lib::SaveTable(
            serialization_schema,
            save_version,
            {{"name", "graph"},
             {"binary_file", lib::save(filename.filename())},
             {"max_degree", lib::save(max_degree())},
             {"num_vertices", lib::save(n_nodes())},
             {"uuid", lib::save(uuid.str())},
             {"eltype", lib::save(datatype_v<Idx>)}}
        );
    }

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
                name(eltype),
                name<datatype_v<Idx>>()
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

  protected:
    data_type data_;
    Idx max_degree_;
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
    /// @brief Consturct a new empty graph.
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

    static constexpr SimpleGraph
    load(const std::filesystem::path& path, const Alloc& allocator = {}) {
        if (data::detail::is_likely_reload(path)) {
            return lib::load_from_disk<SimpleGraph>(path, allocator);
        } else {
            return SimpleGraph(data_type::load(path, allocator));
        }
    }
};

template <typename Idx, typename A1, typename A2>
bool operator==(const SimpleGraph<Idx, A1>& x, const SimpleGraph<Idx, A2>& y) {
    return graphs_equal(x, y);
}

template <std::unsigned_integral Idx>
class SimpleBlockedGraph
    : public SimpleGraphBase<Idx, data::BlockedData<Idx, Dynamic, HugepageAllocator<Idx>>> {
  public:
    using parent_type =
        SimpleGraphBase<Idx, data::BlockedData<Idx, Dynamic, HugepageAllocator<Idx>>>;
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

    static constexpr SimpleBlockedGraph load(const std::filesystem::path& path) {
        if (data::detail::is_likely_reload(path)) {
            return lib::load_from_disk<SimpleBlockedGraph>(path);
        } else {
            return SimpleBlockedGraph(data_type::load(path));
        }
    }
};

} // namespace svs::graphs
