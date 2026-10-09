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

#include "svs/core/graph/graph.h"
#include "svs/lib/saveload.h"

// stl
#include <filesystem>
#include <istream>
#include <optional>
#include <string>

namespace svs {

///
/// @brief Loader for SVS graphs.
///
/// @tparam Idx The type used to encode nodes in the graph.
/// @tparam Allocator The type used for allocations in the graph. This must be a valid SVS
/// allocator type.
///
template <typename Idx = uint32_t, typename Allocator = HugepageAllocator<Idx>>
struct GraphLoader {
    // Type aliases
    using return_type = graphs::SimpleGraph<Idx, Allocator>;

    /// @brief Construct a new GraphLoader
    ///
    /// @param path The file path to the graph directory on disk.
    /// @param allocator The allocator instance to use for the graph.
    ///
    /// The saved graph directory will generally be created when saving a graph based
    /// index. The ``path`` argument should be this directory.
    ///
    GraphLoader(const std::filesystem::path& path, const Allocator& allocator = {})
        : path_{path}
        , allocator_{allocator} {}

    /// @brief Load the graph into memory.
    return_type load() const { return return_type::load(path_, allocator_); }

    ///// Members
    std::filesystem::path path_{};
    Allocator allocator_{};
};

///
/// @brief Loader selecting the graph allocator for stream-based index assembly.
///
/// @tparam Idx The type used to encode nodes in the graph.
/// @tparam Allocator The type used for allocations in the graph. This must be a valid SVS
/// allocator type.
///
template <typename Idx = uint32_t, typename Allocator = HugepageAllocator<Idx>>
struct StreamGraphLoader {
    // Type aliases
    using return_type = graphs::SimpleGraph<Idx, Allocator>;

    /// @brief Construct a new StreamGraphLoader
    ///
    /// @param allocator The allocator instance to use for the graph.
    ///
    explicit StreamGraphLoader(const Allocator& allocator = {})
        : allocator_{allocator} {}

    /// @brief Load the graph from the given stream.
    return_type load(std::istream& stream) const {
        return return_type::load(stream, allocator_);
    }

    ///// Members
    Allocator allocator_{};
};

template <typename T> inline constexpr bool is_stream_graph_loader_v = false;
template <typename Idx, typename Allocator>
inline constexpr bool is_stream_graph_loader_v<StreamGraphLoader<Idx, Allocator>> = true;

namespace detail {
// Helper that answers: "Is the first template argument a StreamGraphLoader?"
template <typename... Args> inline constexpr bool first_is_stream_graph_loader_v = false;
template <typename First, typename... Rest>
inline constexpr bool first_is_stream_graph_loader_v<First, Rest...> =
    is_stream_graph_loader_v<std::decay_t<First>>;
} // namespace detail

///
/// @brief Allocate a default graph with the given capacity.
///
/// @param num_nodes The number of vertices in the graph.
/// @param max_degree The maximum degree in the final graph.
/// @param allocator The allocator to use for the graph.
///
template <typename Idx = uint32_t, typename Allocator = HugepageAllocator<Idx>>
graphs::SimpleGraph<Idx, Allocator>
default_graph(size_t num_nodes, size_t max_degree, const Allocator& allocator = {}) {
    return graphs::SimpleGraph<Idx, Allocator>(num_nodes, max_degree, allocator);
}
} // namespace svs
