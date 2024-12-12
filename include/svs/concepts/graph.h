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

// !! NOTICE TO MAINTAINERS !!
//
// Due to limitations in the Doxygen -> Breathe -> Sphinx interface, support for documenting
// C++ 20 Concepts is limited.
//
// For now, we need to put the member-wise documentation in a code block in the main
// concept docstring.
//
// Hopefully this situation changes in future versions of these systems.

///
/// @file
///

///
/// @ingroup concepts
/// @defgroup graph_concept_entry Main Graph Concepts.
///

///
/// @ingroup concepts
/// @defgroup graph_concept_public Graph Concept Helpers.
///

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace svs::graphs {

// clang-format off

///
/// @ingroup graph_concept_entry
/// @brief Main concept modeling immutable in-memory graphs.
///
/// @code{.cpp}
/// template <typename T>
/// concept ImmutableMemoryGraph = requires(const T& const_g) {
///     // The encoding of vertices in the graph.
///     // At the time of writing, this is expected to be an integer.
///     // This may be relaxed in the future.
///     typename T::index_type;
///     std::integral<typename T::index_type>;
///
///     // Has `reference` and `const_reference` type aliases.
///     // These types should be at least forward ranges, but preferably random access
///     // ranges.
///     //
///     // Items yieled by the iterators for these ranges should ``index_type``.
///     typename T::reference;
///     typename T::const_reference;
///
///     // Return the maximum degree that this particular implementation of the graph is
///     // capable of supporting.
///     //
///     // If the graph supports unbounded adjacency lists, may return `Dynamic`.
///     { const_g.max_degree() } -> std::convertible_to<size_t>;
///
///     // Return the number of vertices contained in traph.
///     { const_g.n_nodes() } -> std::convertible_to<size_t>;
///
///     // Adjacency list operations.
///     requires requires(typename T::index_type i) {
///         // Return a range over the adjacency list for node ``i``.
///         { const_g.get_node(i) } -> std::same_as<typename T::const_reference>;
///
///         // Return the number of out neighbors for node ``i``.
///         { const_g.get_node_degree(i) } -> std::convertible_to<size_t>;
///
///         // Prefetch the adjacency list for node ``i``.
///         // This is a performance optimization only and may be implemented as a no-op
///         // without affecting correctness.
///         const_g.prefetch_node(i);
///     };
/// };
/// @endcode
///
template <typename T>
concept ImmutableMemoryGraph = requires(const T& const_g) {
    // The encoding of vertices in the graph.
    typename T::index_type;
    requires std::integral<typename T::index_type>;

    // Has `reference` and `const_reference` type aliases.
    typename T::reference;
    typename T::const_reference;

    { const_g.max_degree() } -> std::convertible_to<size_t>;
    { const_g.n_nodes() } -> std::convertible_to<size_t>;

    // Get an adjacency list.
    requires requires(typename T::index_type i) {
        { const_g.get_node(i) } -> std::same_as<typename T::const_reference>;
        { const_g.get_node_degree(i) } -> std::convertible_to<size_t>;
        const_g.prefetch_node(i);
    };
};
// clang-format on

///
/// @ingroup graph_concept_public
/// @brief Obtain the index type used to encode neighbors in the graph type ``G``.
///
template <ImmutableMemoryGraph G> using index_type_t = typename G::index_type;

// clang-format off

///
/// @ingroup graph_concept_entry
/// @brief Concept modeling mutable in-memory graphs.
///
/// @code{.cpp}
/// template <typename T>
/// concept MemoryGraph = requires(T& g, const T& const_g) {
///     // Add an edge to the graph.
///     // Must return the out degree of `src` after adding the edge `src -> dst`.
///     // If adding the edge would result in the graph exceeding its maximum degree,
///     // implementations are free to not add this edge.
///     requires requires(index_type_t<T> src, index_type_t<T> dst) {
///         { g.add_edge(src, dst) } -> std::convertible_to<size_t>;
///     };
///
///     // Completely clear the adjacency list for vertex ``i``.
///     requires requires(index_type_t<T> i) {
///         g.clear_node(i);
///     };
///
///     // Overwrite the adjacency list for `src`.
///     requires requires(
///         index_type_t<T> src,
///         const std::vector<index_type_t<T>>& neighbors_vector,
///         std::span<const index_type_t<T>> neighbors_span
///     ) {
///         g.replace_node(src, neighbors_vector);
///         g.replace_node(src, neighbors_span);
///     };
/// };
/// @endcode
///
template <typename T>
concept MemoryGraph = requires(T& g, const T& const_g) {
    // Adding an edge.
    requires requires(index_type_t<T> src, index_type_t<T> dst) {
        { g.add_edge(src, dst) } -> std::convertible_to<size_t>;
    };

    // Clear adjacency list.
    requires requires(index_type_t<T> i) {
        g.clear_node(i);
    };

    // Overwriting an edge.
    requires requires(
        index_type_t<T> src,
        const std::vector<index_type_t<T>>& neighbors_vector,
        std::span<const index_type_t<T>> neighbors_span
    ) {
        g.replace_node(src, neighbors_vector);
        g.replace_node(src, neighbors_span);
    };
};
// clang-format on

///
/// @ingroup graph_concept_public
/// @brief Compare the equality of two graphs.
///
/// Two graphs are considered equal if:
///
/// * The contain the same number of vertices.
/// * The adjacency lists for each vertex compare equal.
///
template <ImmutableMemoryGraph Graph1, ImmutableMemoryGraph Graph2>
bool graphs_equal(const Graph1& x, const Graph2& y) {
    if (x.num_nodes() != y.num_nodes()) {
        return false;
    }

    for (size_t i = 0, imax = x.num_nodes(); i < imax; ++i) {
        const auto& xa = x.get_node(i);
        const auto& ya = y.get_node(i);
        if (!std::equal(xa.begin(), xa.end(), ya.begin())) {
            return false;
        }
    }
    return true;
}

} // namespace svs::graphs
