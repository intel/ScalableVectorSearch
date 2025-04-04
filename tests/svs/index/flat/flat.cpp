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
    CATCH_REQUIRE(captured_logs.size() == 1);
    CATCH_REQUIRE(captured_logs[0] == "Test FlatIndex Logging");
}

CATCH_TEST_CASE("FlatIndex get_distance Test", "[get_distance][distance]") {
    const size_t N = 128;

    CATCH_SECTION("Float32 index and queries") {
        // Create test data
        std::vector<float> data(N, 0.5f);
        auto data_view = svs::data::SimpleDataView<float>(data.data(), 1, N);
        svs::distance::DistanceL2 distance_function;
        auto threadpool = svs::threads::DefaultThreadPool(1);

        // Create index
        svs::index::flat::FlatIndex index(
            std::move(data_view), distance_function, std::move(threadpool)
        );

        // Create test vector
        std::vector<float> test_vector(N, 1.0f);

        // Get distance
        double index_distance = index.get_distance(0, test_vector);

        // Calculate expected distance
        std::span<const float> vector1(data.data(), N);
        std::span<const float> vector2(test_vector.data(), N);
        double expected_distance =
            svs::distance::compute(distance_function, vector2, vector1);

        CATCH_REQUIRE(std::abs(index_distance - expected_distance) < 1e-5);

        // Test with out-of-bounds ID
        CATCH_REQUIRE_THROWS_AS(index.get_distance(999, test_vector), svs::ANNException);

        // Test with dimension mismatch
        std::vector<float> wrong_size_vector(N + 1, 1.0f);
        CATCH_REQUIRE_THROWS_AS(
            index.get_distance(0, wrong_size_vector), svs::ANNException
        );
    }

    CATCH_SECTION("Float16 index and queries") {
        // Create Float16 data
        std::vector<svs::Float16> data(N);
        for (size_t i = 0; i < N; i++) {
            data[i] = svs::Float16(0.5f);
        }

        auto data_view = svs::data::SimpleDataView<svs::Float16>(data.data(), 1, N);
        svs::distance::DistanceL2 distance_function;
        auto threadpool = svs::threads::DefaultThreadPool(1);

        // Create index
        svs::index::flat::FlatIndex index(
            std::move(data_view), distance_function, std::move(threadpool)
        );

        // Create Float16 test vector
        std::vector<svs::Float16> test_vector(N);
        for (size_t i = 0; i < N; i++) {
            test_vector[i] = svs::Float16(1.0f);
        }

        // Get distance
        double index_distance = index.get_distance(0, test_vector);

        // Calculate expected distance
        std::span<const svs::Float16> vector1(data.data(), N);
        std::span<const svs::Float16> vector2(test_vector.data(), N);
        double expected_distance =
            svs::distance::compute(distance_function, vector2, vector1);

        CATCH_REQUIRE(std::abs(index_distance - expected_distance) < 1e-5);
    }
}