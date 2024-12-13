/*
 * Copyright 2024 Intel Corporation
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
#include "svs/index/vamana/search_params.h"

// svs
#include "svs/lib/saveload.h"

// svs_test
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"

namespace {
// Legacy serialization formats.
std::string_view v0_0_0 = R"(
__schema__ = 'vamana_search_parameters'
__version__ = 'v0.0.0'
search_buffer_capacity = 100
search_buffer_visited_set = true
search_window_size = 50
)";

const size_t DEFAULT_PREFETCH_LOOKAHEAD = 4;
const size_t DEFAULT_PREFETCH_STEP = 1;

} // namespace

CATCH_TEST_CASE("VamanaSearcmParameters", "[index][vamana]") {
    using VamanaSearchParameters = svs::index::vamana::VamanaSearchParameters;
    CATCH_SECTION("Constructors") {
        auto p = VamanaSearchParameters{};
        CATCH_REQUIRE(p.buffer_config_ == svs::index::vamana::SearchBufferConfig{});
        CATCH_REQUIRE(p.search_buffer_visited_set_ == false);
        CATCH_REQUIRE(p.prefetch_lookahead_ == DEFAULT_PREFETCH_LOOKAHEAD);
        CATCH_REQUIRE(p.prefetch_step_ == DEFAULT_PREFETCH_STEP);

        CATCH_REQUIRE(p.buffer_config(10) == p);
        CATCH_REQUIRE(p.buffer_config_ == svs::index::vamana::SearchBufferConfig{10, 10});

        CATCH_REQUIRE(p.search_buffer_visited_set(true) == p);
        CATCH_REQUIRE(p.search_buffer_visited_set_ == true);

        CATCH_REQUIRE(p.prefetch_lookahead(50) == p);
        CATCH_REQUIRE(p.prefetch_lookahead_ == 50);

        CATCH_REQUIRE(p.prefetch_step(5) == p);
        CATCH_REQUIRE(p.prefetch_step_ == 5);
    }

    // Serialization.
    CATCH_SECTION("Serialization") {
        svs_test::prepare_temp_directory();
        auto temp_directory = svs_test::temp_directory();

        auto p = VamanaSearchParameters{{10, 20}, true, 10, 5};
        CATCH_REQUIRE(svs::lib::test_self_save_load(p, temp_directory));
    }

    CATCH_SECTION("Loading Legacy Objects") {
        CATCH_SECTION("v0.0.0") {
            auto table = toml::parse(v0_0_0);
            auto p =
                svs::lib::load<VamanaSearchParameters>(svs::lib::ContextFreeLoadTable(table)
                );
            CATCH_REQUIRE(
                p.buffer_config_ == svs::index::vamana::SearchBufferConfig{50, 100}
            );
            CATCH_REQUIRE(p.search_buffer_visited_set_ == true);
            CATCH_REQUIRE(p.prefetch_lookahead_ == DEFAULT_PREFETCH_LOOKAHEAD);
            CATCH_REQUIRE(p.prefetch_step_ == DEFAULT_PREFETCH_STEP);
        }
    }
}
