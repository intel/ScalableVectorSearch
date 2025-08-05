/*
 * Copyright 2025 Intel Corporation
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

#include "svs/index/flat/flat.h"
#include "svs/core/logging.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// spd log
#include "spdlog/sinks/callback_sink.h"

CATCH_TEST_CASE("FlatIndex Logging Test", "[logging]") {
    // Vector to store captured log messages
    std::vector<std::string> captured_logs;
    std::vector<std::string> global_captured_logs;

    // Create a callback sink to capture log messages
    auto callback_sink = std::make_shared<spdlog::sinks::callback_sink_mt>(
        [&captured_logs](const spdlog::details::log_msg& msg) {
            captured_logs.emplace_back(msg.payload.data(), msg.payload.size());
        }
    );
    callback_sink->set_level(spdlog::level::trace);

    // Set up the FlatIndex with the test logger
    auto test_logger = std::make_shared<spdlog::logger>("test_logger", callback_sink);
    test_logger->set_level(spdlog::level::trace);

    auto global_callback_sink = std::make_shared<spdlog::sinks::callback_sink_mt>(
        [&global_captured_logs](const spdlog::details::log_msg& msg) {
            global_captured_logs.emplace_back(msg.payload.data(), msg.payload.size());
        }
    );
    global_callback_sink->set_level(spdlog::level::trace);
    auto original_logger = svs::logging::get();
    original_logger->sinks().push_back(global_callback_sink);

    std::vector<float> data{1.0f, 2.0f};
    auto dataView = svs::data::SimpleDataView<float>(data.data(), 2, 1);
    svs::distance::DistanceL2 dist;
    auto threadpool = svs::threads::DefaultThreadPool(1);

    svs::index::flat::FlatIndex index(
        std::move(dataView), dist, std::move(threadpool), test_logger
    );

    // Log a message
    test_logger->info("Test FlatIndex Logging");

    // Verify the log output
    CATCH_REQUIRE(global_captured_logs.empty());
    CATCH_REQUIRE(captured_logs.size() == 1);
    CATCH_REQUIRE(captured_logs[0] == "Test FlatIndex Logging");
}
