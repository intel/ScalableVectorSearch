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

namespace detail {
// Mapping of back-end data builders to the resolved graphtype.
template <typename Idx, typename Builder> struct GraphMapping;

// Deine for the PolymorphicBuilder
template <typename Idx, typename Allocator>
struct GraphMapping<Idx, data::PolymorphicBuilder<Allocator>> {
    using type = graphs::SimpleGraph<Idx>;
};

// Define for the BlockedBuilder
template <typename Idx> struct GraphMapping<Idx, data::BlockedBuilder> {
    using type = graphs::SimpleBlockedGraph<Idx>;
};

} // namespace detail

///
/// @brief Loader for SVS graphs.
///
/// @tparam Idx The type used to encode nodes in the graph.
/// @tparam Builder The builder for the backing data for the graph.
///
template <
    typename Idx = uint32_t,
    typename Builder = data::PolymorphicBuilder<HugepageAllocator>>
struct GraphLoader {
    // Type aliases
    using return_type = typename detail::GraphMapping<Idx, Builder>::type;

    /// @brief Default constructor.
    GraphLoader()
        : builder_{} {}
    GraphLoader(lib::InferPath SVS_UNUSED(tag))
        : path_{std::nullopt} {}

    GraphLoader(lib::InferPath SVS_UNUSED(tag), Builder builder)
        : builder_{std::move(builder)}
        , path_{std::nullopt} {}

    /// @brief Construct a new GraphLoader
    ///
    /// @param path The file path to the graph directory on disk.
    /// @param allocator The allocator to use.
    ///
    /// The saved graph diredctory will generally be created when saving a graph based
    /// index. The ``path`` argument should be this directory.
    ///
    GraphLoader(const std::filesystem::path& path)
        : path_{path} {}

    GraphLoader(const std::filesystem::path& path, Builder builder)
        : builder_{std::move(builder)}
        , path_{path} {}

    // Return a graph loaded from the provided file.
    return_type unsafe_load_direct() const {
        return io::load_graph<return_type>(path_.value(), builder_);
    }

    /// @brief Load the graph into memory.
    return_type load() const {
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

    return_type load_from_table(
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
        return io::load_graph<return_type>(binaryfile.value(), builder_);
    }

    ///// Members
    Builder builder_{};
    std::optional<std::filesystem::path> path_{};
};

///
/// @brief Allocate a default graph with the given capacity.
///
/// @param num_nodes The number of vertices in the graph.
/// @param max_degree The maximum degree in the final graph.
/// @param allocator The allocator to use for the graph.
///
template <typename Idx = uint32_t, typename Allocator = HugepageAllocator>
graphs::SimpleGraph<Idx> default_graph(
    size_t num_nodes, size_t max_degree, const Allocator& allocator = HugepageAllocator()
) {
    return graphs::SimpleGraph<Idx>(allocator, num_nodes, max_degree);
}
} // namespace svs
