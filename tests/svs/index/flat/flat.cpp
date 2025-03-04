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

struct TestLogCtx {
    std::vector<std::string> logs;
};

static void test_log_callback(void* ctx, const char* level, const char* msg) {
    if (!ctx)
        return;
    auto* logCtx = reinterpret_cast<TestLogCtx*>(ctx);
    logCtx->logs.push_back(std::string(level) + ": " + msg);
}

CATCH_TEST_CASE("FlatIndex Per-Index Logging Test", "[logging]") {
    svs::logging::set_global_log_callback(&test_log_callback);
    TestLogCtx testLogContext;

    std::vector<float> data{1.0f, 2.0f};
    auto dataView = svs::data::SimpleDataView<float>(data.data(), 2, 1);
    svs::distance::DistanceL2 dist;
    auto threadpool = svs::threads::DefaultThreadPool(1);

    svs::index::flat::FlatIndex index(
        dataView, dist, std::move(threadpool), &testLogContext
    );

    index.log("NOTICE", "Test FlatIndex Logging");

    CATCH_REQUIRE(testLogContext.logs.size() == 1);
    CATCH_REQUIRE(testLogContext.logs[0] == "NOTICE: Test FlatIndex Logging");
}
