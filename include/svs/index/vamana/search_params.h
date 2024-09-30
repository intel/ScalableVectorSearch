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
    static constexpr std::string_view serialization_schema = "vamana_search_parameters";
    lib::SaveTable save() const {
        return lib::SaveTable(
            serialization_schema,
            save_version,
            {{"search_window_size", lib::save(buffer_config_.get_search_window_size())},
             {"search_buffer_capacity", lib::save(buffer_config_.get_total_capacity())},
             SVS_LIST_SAVE_(search_buffer_visited_set),
             SVS_LIST_SAVE_(prefetch_lookahead),
             SVS_LIST_SAVE_(prefetch_step)}
        );
    }

    static bool check_load_compatibility(std::string_view schema, lib::Version version) {
        return schema == serialization_schema && version <= save_version;
    }

    static VamanaSearchParameters load_legacy(const lib::ContextFreeLoadTable& table) {
        assert(table.version() == lib::Version(0, 0, 0));
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

    static VamanaSearchParameters load(const lib::ContextFreeLoadTable& table) {
        if (table.version() < save_version) {
            return load_legacy(table);
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
