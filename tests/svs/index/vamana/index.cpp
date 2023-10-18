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

// Header under test
#include "svs/index/vamana/index.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <string_view>

namespace {

const std::string_view vamana_config_parameters_v0_0_0 = R"(
__version__ = 'v0.0.0'
alpha = 1.2
construction_window_size = 200
default_search_window_size = 0
entry_point = 9426
max_candidates = 1000
max_out_degree = 128
name = 'vamana config parameters'
visited_set = false
)";

const std::string_view vamana_config_parameters_v0_0_1 = R"(
__version__ = 'v0.0.1'
alpha = 1.2
construction_window_size = 200
default_search_window_size = 0
entry_point = 9426
max_candidates = 1000
max_out_degree = 128
name = 'vamana config parameters'
use_full_search_history = false
visited_set = false
)";

} // namespace

CATCH_TEST_CASE("Vamana Config Parameters", "[index][vamana]") {
    using VamanaConfigParameters = svs::index::vamana::VamanaConfigParameters;
    CATCH_SECTION("Loading v0.0.0") {
        auto p = svs::lib::load<VamanaConfigParameters>(
            toml::parse(vamana_config_parameters_v0_0_0)
        );
        CATCH_REQUIRE(p.alpha == 1.2f);
        CATCH_REQUIRE(p.construction_window_size == 200);
        CATCH_REQUIRE(p.search_window_size == 0);
        CATCH_REQUIRE(p.entry_point == 9426);
        CATCH_REQUIRE(p.max_candidates == 1000);
        CATCH_REQUIRE(p.graph_max_degree == 128);
        CATCH_REQUIRE(p.visited_set == false);
        // Automatically applied arguments.
        CATCH_REQUIRE(p.use_full_search_history == true);
        CATCH_REQUIRE(p.prune_to == p.graph_max_degree);
    }

    CATCH_SECTION("Loading v0.0.1") {
        auto p = svs::lib::load<VamanaConfigParameters>(
            toml::parse(vamana_config_parameters_v0_0_1)
        );
        CATCH_REQUIRE(p.alpha == 1.2f);
        CATCH_REQUIRE(p.construction_window_size == 200);
        CATCH_REQUIRE(p.search_window_size == 0);
        CATCH_REQUIRE(p.entry_point == 9426);
        CATCH_REQUIRE(p.max_candidates == 1000);
        CATCH_REQUIRE(p.graph_max_degree == 128);
        CATCH_REQUIRE(p.visited_set == false);
        CATCH_REQUIRE(p.use_full_search_history == false);
        // Automatically applied arguments.
        CATCH_REQUIRE(p.prune_to == p.graph_max_degree);
    }

    CATCH_SECTION("Current version") {
        auto p = VamanaConfigParameters{123, 456, 78.9f, 10, 40, 8, false, 9000, true};
        CATCH_REQUIRE(svs::lib::test_self_save_load_context_free(p));
    }
}
