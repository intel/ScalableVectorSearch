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
#include "svs/index/vamana/search_buffer.h"
#include "svs/lib/preprocessor.h"
#include "svs/lib/saveload.h"

namespace svs::index::vamana {

/// @brief Runtime parameters controlling the accuracy and performance of index search.
struct VamanaSearchParameters {
  public:
    ///// Parameters controlling the quality of retrieve neighbors.

    /// @brief Configuration of the search buffer.
    ///
    /// Increasing the search window size and capacity generally yields more accurate but
    /// slower search results.
    SearchBufferConfig buffer_config_{};

    /// @brief Enabling of the visited set for search.
    ///
    /// The visited set tracks whether candidates the distance between a query and a
    /// candidate has already been computed. Enabling this feature generally improves
    /// performance in the high-recall or high-neighbor regime.
    bool search_buffer_visited_set_ = false;

    /// @brief The number of iterations ahead to prefetch candidates.
    size_t prefetch_lookahead_ = 4;

    /// @brief Parameter controlling the ramp phase of prefetching.
    size_t prefetch_step_ = 1;

  public:
    VamanaSearchParameters() = default;

    VamanaSearchParameters(
        SearchBufferConfig buffer_config,
        bool search_buffer_visited_set,
        size_t prefetch_lookahead,
        size_t prefetch_step
    )
        : buffer_config_{buffer_config}
        , search_buffer_visited_set_{search_buffer_visited_set}
        , prefetch_lookahead_{prefetch_lookahead}
        , prefetch_step_{prefetch_step} {}

    // Buffer config
    SVS_CHAIN_SETTER_(VamanaSearchParameters, buffer_config);
    SVS_CHAIN_SETTER_(VamanaSearchParameters, search_buffer_visited_set);
    SVS_CHAIN_SETTER_(VamanaSearchParameters, prefetch_lookahead);
    SVS_CHAIN_SETTER_(VamanaSearchParameters, prefetch_step);

    // Version History
    // - v0.0.0:
    //      SearchBufferConfig buffer_config_{};
    //      bool search_buffer_visited_set_ = false;
    // - v0.0.1: Added prefetch parameters. Backwards compatible with defaults.
    //      SearchBufferConfig buffer_config_{};
    //      bool search_buffer_visited_set_ = false;
    //      size_t prefetch_lookahead = 4
    //      size_t prefetch_lookstep = 1
    static constexpr lib::Version save_version{0, 0, 1};
    lib::SaveTable save() const {
        return lib::SaveTable(
            save_version,
            {{"search_window_size", lib::save(buffer_config_.get_search_window_size())},
             {"search_buffer_capacity", lib::save(buffer_config_.get_total_capacity())},
             SVS_LIST_SAVE_(search_buffer_visited_set),
             SVS_LIST_SAVE_(prefetch_lookahead),
             SVS_LIST_SAVE_(prefetch_step)}
        );
    }

    static VamanaSearchParameters
    load_legacy(const toml::table& table, const lib::Version& version) {
        if (version != lib::Version{0, 0, 0}) {
            throw ANNEXCEPTION("Something went wrong when loading VamanaSearchParameters!");
        }
        // Version 0.0.0 lacked the `prefetch_lookahead` and `prefetch_step` fields.
        // Try to use somewhat reasonable defaults.
        return VamanaSearchParameters{
            SearchBufferConfig(
                lib::load_at<size_t>(table, "search_window_size"),
                lib::load_at<size_t>(table, "search_buffer_capacity")
            ),
            SVS_LOAD_MEMBER_AT_(table, search_buffer_visited_set),
            4,
            1};
    }

    static VamanaSearchParameters
    load(const toml::table& table, const lib::Version& version) {
        if (version < save_version) {
            return load_legacy(table, version);
        }

        return VamanaSearchParameters{
            SearchBufferConfig(
                lib::load_at<size_t>(table, "search_window_size"),
                lib::load_at<size_t>(table, "search_buffer_capacity")
            ),
            SVS_LOAD_MEMBER_AT_(table, search_buffer_visited_set),
            SVS_LOAD_MEMBER_AT_(table, prefetch_lookahead),
            SVS_LOAD_MEMBER_AT_(table, prefetch_step),
        };
    }

    friend bool
    operator==(const VamanaSearchParameters&, const VamanaSearchParameters&) = default;
};

} // namespace svs::index::vamana
