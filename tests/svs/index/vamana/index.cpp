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

// Header under test
#include "svs/index/vamana/index.h"
#include "svs/index/vamana/dynamic_index.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <string_view>

namespace {

const std::string_view vamana_config_parameters_v0_0_0 = R"(
__schema__ = 'vamana_index_parameters'
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
__schema__ = 'vamana_index_parameters'
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

const std::string_view vamana_config_parameters_v0_0_2 = R"(
__schema__ = 'vamana_index_parameters'
__version__ = 'v0.0.2'
alpha = 1.2
construction_window_size = 200
default_search_window_size = 0
entry_point = 9426
max_candidates = 1000
max_out_degree = 128
name = 'vamana config parameters'
use_full_search_history = false
prune_to = 100
visited_set = false
)";

} // namespace

CATCH_TEST_CASE("Vamana Index Parameters", "[index][vamana]") {
    using VamanaIndexParameters = svs::index::vamana::VamanaIndexParameters;
    CATCH_SECTION("Loading v0.0.0") {
        auto table = toml::parse(vamana_config_parameters_v0_0_0);
        auto p = svs::lib::load<VamanaIndexParameters>(svs::lib::node_view(table));
        auto expected = VamanaIndexParameters(
            9426, {1.2f, 128, 200, 1000, 128, true}, {{0, 0}, false, 4, 1}
        );
        CATCH_REQUIRE(p == expected);
    }

    CATCH_SECTION("Loading v0.0.1") {
        auto table = toml::parse(vamana_config_parameters_v0_0_1);
        auto p = svs::lib::load<VamanaIndexParameters>(svs::lib::node_view(table));

        auto expected = VamanaIndexParameters(
            9426, {1.2f, 128, 200, 1000, 128, false}, {{0, 0}, false, 4, 1}
        );

        CATCH_REQUIRE(p == expected);
    }

    CATCH_SECTION("Loading v0.0.2") {
        auto table = toml::parse(vamana_config_parameters_v0_0_2);
        auto p = svs::lib::load<VamanaIndexParameters>(svs::lib::node_view(table));

        auto expected = VamanaIndexParameters(
            9426, {1.2f, 128, 200, 1000, 100, false}, {{0, 0}, false, 4, 1}
        );

        CATCH_REQUIRE(p == expected);
    }

    CATCH_SECTION("Current version") {
        auto p = VamanaIndexParameters{
            128, {12.4f, 478, 13, 4, 10, false}, {{10, 20}, true, 1, 1}
        };
        CATCH_REQUIRE(svs::lib::test_self_save_load_context_free(p));
    }
}

CATCH_TEST_CASE("Vamana Index Logging", "[index][logging]") {
    // Create a small test dataset
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<size_t> external_ids = {0, 1};
    const size_t dim = 2;

    // Create the graph and data structures
    auto graph = svs::graphs::SimpleGraph<size_t>(2, 64);  // 2 nodes, max degree 64
    auto data_view = svs::data::SimpleDataView<float>(data.data(), 2, dim);
    
    // Use node 0 as entry point
    size_t entry_point = 0;

    // Create logging capture
    std::vector<std::string> log_messages;
    void* log_ctx = nullptr;
    auto log_callback = [&log_messages]([[maybe_unused]] void* ctx, const char* level, const char* msg) {
        log_messages.push_back(std::string(level) + ": " + msg);
    };

    // Create threadpool
    auto threadpool = svs::threads::DefaultThreadPool(1);

    auto index = svs::index::vamana::MutableVamanaIndex(
        std::move(graph),           // Graph
        std::move(data_view),       // Data
        entry_point,                // Entry point
        svs::distance::DistanceL2(), // Distance function
        external_ids,               // External IDs
        std::move(threadpool),      // Move the threadpool
        log_ctx,                    // Logger context
        log_callback                // Logger callback
    );

    // Verify the index was created and logging works
    CATCH_REQUIRE(index.size() == 2);
    CATCH_REQUIRE(!log_messages.empty());
}