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
#include "spdlog/sinks/callback_sink.h"
#include "svs/core/logging.h"

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
            128, {12.4f, 478, 13, 4, 10, false}, {{10, 20}, true, 1, 1}};
        CATCH_REQUIRE(svs::lib::test_self_save_load_context_free(p));
    }
}

CATCH_TEST_CASE("Static VamanaIndex Per-Index Logging", "[logging]") {
    // Vector to store captured log messages
    std::vector<std::string> captured_logs;

    // Create a callback sink to capture log messages
    auto callback_sink = std::make_shared<spdlog::sinks::callback_sink_mt>(
        [&captured_logs](const spdlog::details::log_msg& msg) {
            captured_logs.emplace_back(msg.payload.data(), msg.payload.size());
        }
    );
    callback_sink->set_level(spdlog::level::trace); // Capture all log levels

    // Create a logger with the callback sink
    auto test_logger = std::make_shared<spdlog::logger>("test_logger", callback_sink);
    test_logger->set_level(spdlog::level::trace);

    // Create some minimal data
    std::vector<float> data = {1.0f, 2.0f};
    auto graph = svs::graphs::SimpleGraph<uint32_t>(1, 64);
    auto data_view = svs::data::SimpleDataView<float>(data.data(), 1, 1);
    svs::distance::DistanceL2 distance_function;
    uint32_t entry_point = 0;
    auto threadpool = svs::threads::DefaultThreadPool(1);

    // Build the VamanaIndex with the test logger
    svs::index::vamana::VamanaBuildParameters buildParams(1.2, 64, 10, 20, 10, true);
    svs::index::vamana::VamanaIndex index(
        buildParams,
        std::move(graph),
        std::move(data_view),
        entry_point,
        distance_function,
        std::move(threadpool),
        test_logger
    );

    // Verify the internal log messages
    CATCH_REQUIRE(captured_logs[0].find("Number of syncs:") != std::string::npos);
    CATCH_REQUIRE(captured_logs[1].find("Batch Size:") != std::string::npos);
}