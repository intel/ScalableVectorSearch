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

#include "svs/index/inverted/memory_based.h"
#include "catch2/catch_test_macros.hpp"
#include "spdlog/sinks/callback_sink.h"
#include "svs-benchmark/datasets.h"
#include "svs/lib/timing.h"
#include "tests/utils/inverted_reference.h"
#include "tests/utils/test_dataset.h"
#include <filesystem>

CATCH_TEST_CASE("InvertedIndex Logging Test", "[logging]") {
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

    // Create a logger with the callback sink
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

    // Setup index
    auto distance = svs::DistanceL2();
    constexpr auto distance_type = svs::distance_type_v<svs::DistanceL2>;
    auto expected_results = test_dataset::inverted::expected_build_results(
        distance_type, svsbenchmark::Uncompressed(svs::DataType::float32)
    );
    auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());
    auto threadpool = svs::threads::DefaultThreadPool(1);
    auto invertedIndex = svs::index::inverted::auto_build(
        expected_results.build_parameters_.value(),
        data,
        distance,
        std::move(threadpool),
        {},
        {},
        {},
        test_logger
    );

    // Verify the internal log messages
    CATCH_REQUIRE(global_captured_logs.empty());
    CATCH_REQUIRE(captured_logs[0].find("Vamana Build Parameters:") != std::string::npos);
    CATCH_REQUIRE(captured_logs[1].find("Number of syncs") != std::string::npos);
}