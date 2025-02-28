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

#include "svs/index/inverted/memory_based.h"
#include "catch2/catch_test_macros.hpp"
#include "svs-benchmark/datasets.h"
#include "svs/lib/timing.h"
#include "tests/utils/inverted_reference.h"
#include "tests/utils/test_dataset.h"
#include <filesystem>

// Define our log context structure.
struct TestLogCtx {
    std::vector<std::string> logs;
};

static void test_log_callback(void* ctx, const char* level, const char* msg) {
    if (!ctx)
        return;
    auto* logCtx = reinterpret_cast<TestLogCtx*>(ctx);
    logCtx->logs.push_back(std::string(level) + ": " + msg);
}

CATCH_TEST_CASE("InvertedIndex Per-Index Logging", "[logging]") {
    // Set global log callback
    svs::logging::set_global_log_callback(&test_log_callback);
    TestLogCtx testLogContext;

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
        &testLogContext
    );

    // Log a test message
    invertedIndex.log("NOTICE", "Test InvertedIndex Build");

    // Check log context received the message
    CATCH_REQUIRE(testLogContext.logs.size() == 1);
    CATCH_REQUIRE(testLogContext.logs[0] == "NOTICE: Test InvertedIndex Build");
}