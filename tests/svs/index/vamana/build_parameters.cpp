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

// Header under test.
#include "svs/index/vamana/build_params.h"

// svs
#include "svs/lib/saveload.h"

// svs_test
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"

namespace {
// Legacy serialization formats.
std::string_view v0_0_0 = R"(
__version__ = 'v0.0.0'
__schema__ = 'vamana_build_parameters'
alpha = 1.2
graph_max_degree = 128
max_candidate_pool_size = 750
name = 'vamana build parameters'
use_full_search_history = true
window_size = 200
)";
} // namespace

CATCH_TEST_CASE("VamanaBuildParameters", "[index][vamana]") {
    CATCH_SECTION("Constructors") {
        svs::index::vamana::VamanaBuildParameters empty;
        CATCH_REQUIRE(empty.alpha == svs::FLOAT_PLACEHOLDER);
        CATCH_REQUIRE(empty.graph_max_degree == svs::VAMANA_GRAPH_MAX_DEGREE_DEFAULT);
        CATCH_REQUIRE(empty.window_size == svs::VAMANA_WINDOW_SIZE_DEFAULT);
        CATCH_REQUIRE(empty.max_candidate_pool_size == svs::UNSIGNED_INTEGER_PLACEHOLDER);
        CATCH_REQUIRE(empty.prune_to == svs::UNSIGNED_INTEGER_PLACEHOLDER);
        CATCH_REQUIRE(
            empty.use_full_search_history == svs::VAMANA_USE_FULL_SEARCH_HISTORY_DEFAULT
        );

        auto p = svs::index::vamana::VamanaBuildParameters{1.2f, 64, 128, 750, 60, true};
        CATCH_REQUIRE(p.alpha == 1.2f);
        CATCH_REQUIRE(p.graph_max_degree == 64);
        CATCH_REQUIRE(p.window_size == 128);
        CATCH_REQUIRE(p.max_candidate_pool_size == 750);
        CATCH_REQUIRE(p.prune_to == 60);
        CATCH_REQUIRE(p.use_full_search_history == true);

        // Check for equality.
        auto u = svs::index::vamana::VamanaBuildParameters{1.2, 64, 128, 750, 60, false};
        CATCH_REQUIRE(p != u);
        u.use_full_search_history = true;
        CATCH_REQUIRE(p == u);
    }

    // Serialization.
    CATCH_SECTION("Serialization") {
        svs_test::prepare_temp_directory();
        auto temp_directory = svs_test::temp_directory();

        auto p = svs::index::vamana::VamanaBuildParameters{1.2, 64, 128, 750, 60, false};
        CATCH_REQUIRE(svs::lib::test_self_save_load(p, temp_directory));
    }

    CATCH_SECTION("Loading Legacy Objects") {
        CATCH_SECTION("v0.0.0") {
            auto table = toml::parse(v0_0_0);
            auto p = svs::lib::load<svs::index::vamana::VamanaBuildParameters>(
                svs::lib::node_view(table)
            );
            CATCH_REQUIRE(p.alpha == 1.2f);
            CATCH_REQUIRE(p.graph_max_degree == 128);
            CATCH_REQUIRE(p.max_candidate_pool_size == 750);
            CATCH_REQUIRE(p.use_full_search_history == true);
            CATCH_REQUIRE(p.window_size == 200);
            // Default parameters
            CATCH_REQUIRE(p.prune_to == 128);
        }
    }
}
