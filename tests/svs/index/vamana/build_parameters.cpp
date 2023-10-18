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
            auto p = svs::lib::load<svs::index::vamana::VamanaBuildParameters>(table);
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
