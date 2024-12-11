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

// svs
#include "svs/lib/saveload.h"

// stl
#include <cstddef>

namespace svs::index::vamana {

/// @brief Parameters controlling graph construction for the Vamana graph index.
struct VamanaBuildParameters {
    VamanaBuildParameters() = default;

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
    static constexpr std::string_view serialization_schema = "vamana_build_parameters";

    lib::SaveTable save() const {
        return lib::SaveTable(
            serialization_schema,
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

    static bool check_load_compatibility(std::string_view schema, lib::Version version) {
        return schema == serialization_schema && version <= save_version;
    }

    static VamanaBuildParameters load(const lib::ContextFreeLoadTable& table) {
        // Okay - by this point we're satistifed that we're probably deserializing the
        // correct object.
        //
        // Now, we finish loading.
        auto graph_max_degree = lib::load_at<size_t>(table, "graph_max_degree");

        // Require the presence of the "prune_to" field if the version number is greater
        // than v0.0.0.
        auto prune_to = graph_max_degree;
        if (table.version() > lib::Version(0, 0, 0)) {
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
