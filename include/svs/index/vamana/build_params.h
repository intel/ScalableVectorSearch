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

// svs
#include "svs/lib/saveload.h"

// stl
#include <cstddef>

namespace svs::index::vamana {

/// @brief Parameters controlling graph construction for the Vamana graph index.
struct VamanaBuildParameters {
    VamanaBuildParameters(
        float alpha_,
        size_t graph_max_degree_,
        size_t window_size_,
        size_t max_candidate_pool_size_,
        bool use_full_search_history_ = true
    )
        : alpha{alpha_}
        , graph_max_degree{graph_max_degree_}
        , window_size{window_size_}
        , max_candidate_pool_size{max_candidate_pool_size_}
        , use_full_search_history{use_full_search_history_} {}

    /// The pruning parameter.
    float alpha;

    /// The maximum degree in the graph. A higher max degree may yield a higher quality
    /// graph in terms of recall for performance, but the memory footprint of the graph is
    /// directly proportional to the maximum degree.
    size_t graph_max_degree;

    /// The search window size to use during graph construction. A higher search window
    /// size will yield a higher quality graph since more overall vertices are considered,
    /// but will increase construction time.
    size_t window_size;

    /// Set a limit on the number of neighbors considered during pruning. In practice, set
    /// this to a high number (at least 5 times greater than the window_size) and forget
    /// about it.
    size_t max_candidate_pool_size;

    /// When building, either the contents of the search buffer can be used or the entire
    /// search history can be used.
    ///
    /// The latter case may yield a slightly better graph as the cost of more search time.
    bool use_full_search_history = true;

    ///// Comparison
    friend bool
    operator==(const VamanaBuildParameters&, const VamanaBuildParameters&) = default;

    ///// Saving and Loading
    static constexpr std::string_view name = "vamana build parameters";

    // Change notes:
    //
    // v0.0.0 - Initial versiohn
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);

    lib::SaveType save(const lib::SaveContext& SVS_UNUSED(ctx)) const {
        auto table = toml::table(
            {{"alpha", prepare(alpha)},
             {"graph_max_degree", prepare(graph_max_degree)},
             {"window_size", prepare(window_size)},
             {"max_candidate_pool_size", prepare(max_candidate_pool_size)},
             {"use_full_search_history", use_full_search_history},
             {"name", name}}
        );
        return std::make_pair(std::move(table), save_version);
    }

    static VamanaBuildParameters load(
        const toml::table& table,
        const lib::LoadContext& SVS_UNUSED(ctx),
        const lib::Version& version
    ) {
        // Perform a name check.
        if (auto this_name = get_checked<std::string>(table, "name"); this_name != name) {
            auto msg = fmt::format(
                "Error deserializing VamanaConfigParameters. Expected name {}, got {}!",
                name,
                this_name
            );
            throw ANNEXCEPTION(msg);
        }

        // Version check
        if (version != lib::Version(0, 0, 0)) {
            throw ANNEXCEPTION("Incompatible version!");
        }

        // Okay - by this point we're satistifed that we're probably deserializing the
        // correct object.
        //
        // Now, we finish loading.
        return VamanaBuildParameters(
            get<float>(table, "alpha"),
            get<size_t>(table, "graph_max_degree"),
            get<size_t>(table, "window_size"),
            get<size_t>(table, "max_candidate_pool_size"),
            get<bool>(table, "use_full_search_history")
        );
    }
};
} // namespace svs::index::vamana
