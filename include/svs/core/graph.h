/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

#pragma once

#include "svs/core/graph/graph.h"
#include "svs/core/graph/io.h"
#include "svs/lib/saveload.h"

// stl
#include <filesystem>
#include <optional>
#include <string>

namespace svs {

///
/// @brief The default graph type.
///
template <typename Idx = uint32_t> using DefaultGraph = graphs::SimpleGraph<Idx>;

///
/// @brief Loader for SVS graphs.
///
/// @tparam Idx The type used to encode nodes in the graph.
/// @tparam Allocator The allocator to use for the graph's memory.
///
template <typename Idx = uint32_t, typename Allocator = HugepageAllocator>
struct GraphLoader {
    /// @brief Default constructor.
    GraphLoader()
        : allocator_{} {}
    GraphLoader(lib::InferPath SVS_UNUSED(tag), Allocator allocator = {})
        : allocator_{std::move(allocator)}
        , path_{std::nullopt} {}

    ///
    /// @brief Construct a new GraphLoader
    ///
    /// @param path The file path to the graph directory on disk.
    /// @param allocator The allocator to use.
    ///
    /// The saved graph diredctory will generally be created when saving a graph based
    /// index. The ``path`` argument should be this directory.
    ///
    GraphLoader(const std::filesystem::path& path, Allocator allocator = {})
        : allocator_{std::move(allocator)}
        , path_{path} {}

    // Return a graph loaded from the provided file.
    DefaultGraph<Idx> unsafe_load_direct() const {
        return io::load_graph<Idx>(*path_, allocator_);
    }

    /// @brief Load the graph into memory.
    DefaultGraph<Idx> load() const {
        // First, check that a file in the optional actually exists.
        if (!path_.has_value()) {
            throw ANNEXCEPTION("Trying to load a graph providing a file path!");
        }

        // If the filename is a directory or ends in "toml" - assume we're trying to do
        // a full reload procedure.
        const auto& path = *path_;
        if (maybe_config_file(path) || std::filesystem::is_directory(path)) {
            auto loader = lib::LoadOverride{[&](const toml::table& table,
                                                const lib::LoadContext& ctx,
                                                const lib::Version& version) {
                return load_from_table(table, ctx, version);
            }};
            return lib::load(loader, path);
        }

        return unsafe_load_direct();
    }

    DefaultGraph<Idx> load_from_table(
        const toml::table& table, const lib::LoadContext& ctx, const lib::Version version
    ) const {
        if (version != lib::Version(0, 0, 0)) {
            throw ANNEXCEPTION("Unhandled version!");
        }

        // Perform a sanity check on the element type.
        // Make sure we're loading the correct kind.
        auto graph_eltype_name = get(table, "eltype").value();
        constexpr auto this_eltype_name = name<datatype_v<Idx>>();
        if (graph_eltype_name != this_eltype_name) {
            throw ANNEXCEPTION(
                "Trying to load a graph with adjacency list types ",
                graph_eltype_name,
                " to a graph with adjacency list types ",
                this_eltype_name
            );
        }

        // Now that this is out of the way, resolve the file and load the data.
        auto uuid = lib::UUID(get(table, "uuid").value());
        auto binaryfile = io::find_uuid(ctx.get_directory(), uuid);
        if (!binaryfile.has_value()) {
            throw ANNEXCEPTION("Could not open file with uuid ", uuid.str(), '!');
        }
        return io::load_graph<Idx>(binaryfile.value(), allocator_);
    }

    ///// Members
    Allocator allocator_;
    std::optional<std::filesystem::path> path_ = {};
};

///
/// @brief Allocate a default graph with the given capacity.
///
/// @param num_nodes The number of vertices in the graph.
/// @param max_degree The maximum degree in the final graph.
/// @param allocator The allocator to use for the graph.
///
template <typename Idx = uint32_t, typename Allocator = HugepageAllocator>
DefaultGraph<Idx> default_graph(
    size_t num_nodes, size_t max_degree, const Allocator& allocator = HugepageAllocator()
) {
    return graphs::SimpleGraph<Idx>(allocator, num_nodes, max_degree);
}
} // namespace svs
