/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
 */
#pragma once

#include "svs/core/graph/graph.h"
#include "svs/lib/saveload.h"

// stl
#include <filesystem>
#include <optional>
#include <string>

namespace svs {

///
/// @brief Loader for SVS graphs.
///
/// @tparam Idx The type used to encode nodes in the graph.
///
template <typename Idx = uint32_t> struct GraphLoader {
    // Type aliases
    using return_type = graphs::SimpleGraph<Idx, HugepageAllocator<Idx>>;

    /// @brief Construct a new GraphLoader
    ///
    /// @param path The file path to the graph directory on disk.
    ///
    /// The saved graph directory will generally be created when saving a graph based
    /// index. The ``path`` argument should be this directory.
    ///
    GraphLoader(const std::filesystem::path& path)
        : path_{path} {}

    /// @brief Load the graph into memory.
    return_type load() const { return return_type::load(path_); }

    ///// Members
    std::filesystem::path path_{};
};

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
