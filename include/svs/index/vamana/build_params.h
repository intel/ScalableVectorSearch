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
        size_t prune_to_,
        bool use_full_search_history_
    )
        : alpha{alpha_}
        , graph_max_degree{graph_max_degree_}
        , window_size{window_size_}
        , max_candidate_pool_size{max_candidate_pool_size_}
        , prune_to{prune_to_}
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

    /// This is the amount that candidates will be pruned to after certain pruning
    /// procedures. Setting this to less than ``graph_max_degree`` can result in significant
    /// speedups in index building.
    size_t prune_to;

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
    // v0.0.0 - Initial version
    // v0.0.1 - Add the "prune_to" parameter.
    //   * Behavior if loading from v0.0.0: Set "prune_to = graph_max_degree"
    static constexpr lib::Version save_version = lib::Version(0, 0, 1);

    lib::SaveTable save() const {
        return lib::SaveTable(
            save_version,
            {SVS_LIST_SAVE(alpha),
             SVS_LIST_SAVE(graph_max_degree),
             SVS_LIST_SAVE(window_size),
             SVS_LIST_SAVE(max_candidate_pool_size),
             SVS_LIST_SAVE(prune_to),
             SVS_LIST_SAVE(use_full_search_history),
             SVS_LIST_SAVE(name)}
        );
    }

    static VamanaBuildParameters
    load(const toml::table& table, const lib::Version& version) {
        // Perform a name check.
        if (auto this_name = lib::load_at<std::string>(table, "name"); this_name != name) {
            throw ANNEXCEPTION(
                "Error deserializing VamanaConfigParameters. Expected name {}, got {}!",
                name,
                this_name
            );
        }

        // Version check
        if (version > lib::Version(0, 0, 1)) {
            throw ANNEXCEPTION("Incompatible version!");
        }

        // Okay - by this point we're satistifed that we're probably deserializing the
        // correct object.
        //
        // Now, we finish loading.
        auto graph_max_degree = lib::load_at<size_t>(table, "graph_max_degree");

        // Require the presence of the "prune_to" field if the version number is greater
        // than v0.0.0.
        auto prune_to = graph_max_degree;
        if (version > lib::Version(0, 0, 0)) {
            prune_to = lib::load_at<size_t>(table, "prune_to");
        }

        return VamanaBuildParameters(
            SVS_LOAD_MEMBER_AT(table, alpha),
            graph_max_degree,
            SVS_LOAD_MEMBER_AT(table, window_size),
            SVS_LOAD_MEMBER_AT(table, max_candidate_pool_size),
            prune_to,
            SVS_LOAD_MEMBER_AT(table, use_full_search_history)
        );
    }
};
} // namespace svs::index::vamana
