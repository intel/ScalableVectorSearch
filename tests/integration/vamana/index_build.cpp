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

// svs
#include "svs/core/data/simple.h"
#include "svs/core/recall.h"
#include "svs/lib/timing.h"
#include "svs/orchestrators/vamana.h"

// svsbenchmark
#include "svs-benchmark/benchmark.h"

// fmt
#include "fmt/core.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"
#include "tests/utils/vamana_reference.h"

// stl
#include <array>
#include <filesystem>
#include <string>
#include <utility>

// logger
#include "spdlog/sinks/callback_sink.h"
#include "svs/core/logging.h"

namespace {

template <
    typename E,
    size_t D = svs::Dynamic,
    svs::threads::ThreadPool Pool = svs::threads::DefaultThreadPool>
svs::Vamana build_index(
    const svs::index::vamana::VamanaBuildParameters parameters,
    const std::filesystem::path& data_path,
    Pool threadpool,
    svs::DistanceType dist_type
) {
    auto tic = svs::lib::now();
    svs::Vamana index = svs::Vamana::build<E>(
        parameters,
        svs::data::SimpleData<E, D>::load(data_path),
        dist_type,
        std::move(threadpool)
    );

    fmt::print("Indexing time: {}s\n", svs::lib::time_difference(tic));

    return index;
}

template <typename E, size_t D = svs::Dynamic>
svs::Vamana build_index(
    const svs::index::vamana::VamanaBuildParameters parameters,
    const std::filesystem::path& data_path,
    size_t num_threads,
    svs::DistanceType dist_type
) {
    auto tic = svs::lib::now();
    svs::Vamana index = svs::Vamana::build<E>(
        parameters, svs::data::SimpleData<E, D>::load(data_path), dist_type, num_threads
    );

    fmt::print("Indexing time: {}s\n", svs::lib::time_difference(tic));

    // Make sure the number of threads was propagated correctly.
    CATCH_REQUIRE(index.get_num_threads() == num_threads);
    return index;
}
} // namespace

CATCH_TEST_CASE("Uncompressed Vamana Build", "[integration][build][vamana][uncompressed]") {
    auto distances = std::to_array<svs::DistanceType>({svs::L2, svs::MIP, svs::Cosine});

    // How far these results may deviate from previously generated results.
    const double epsilon = 0.005;
    const auto queries = svs::data::SimpleData<float>::load(test_dataset::query_file());
    const auto dataset = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());

    for (auto distance_type : distances) {
        CATCH_REQUIRE(svs_test::prepare_temp_directory());
        size_t num_threads = 2;
        auto expected_result = test_dataset::vamana::expected_build_results(
            distance_type, svsbenchmark::Uncompressed(svs::DataType::float32)
        );
        svs::Vamana index = build_index<float>(
            expected_result.build_parameters_.value(),
            test_dataset::data_svs_file(),
            num_threads,
            distance_type
        );
        CATCH_REQUIRE(
            index.query_types() == std::vector<svs::DataType>{svs::DataType::float32}
        );

        // Test get_distance functionality
        svs::DistanceDispatcher dispatcher(distance_type);
        dispatcher([&](auto dist) {
            svs_test::GetDistanceTester::test(index, dist, dataset);
        });

        auto groundtruth = test_dataset::load_groundtruth(distance_type);
        for (const auto& expected : expected_result.config_and_recall_) {
            auto these_queries = test_dataset::get_test_set(queries, expected.num_queries_);
            auto these_groundtruth =
                test_dataset::get_test_set(groundtruth, expected.num_queries_);
            index.set_search_parameters(expected.search_parameters_);
            auto results = index.search(these_queries, expected.num_neighbors_);
            double recall = svs::k_recall_at_n(
                these_groundtruth, results, expected.num_neighbors_, expected.recall_k_
            );

            fmt::print(
                "Window Size: {}, Expected Recall: {}, Actual Recall: {}\n",
                index.get_search_window_size(),
                expected.recall_,
                recall
            );
            CATCH_REQUIRE(recall > expected.recall_ - epsilon);
            CATCH_REQUIRE(recall < expected.recall_ + epsilon);
        }
    }
}

CATCH_TEST_CASE(
    "Uncompressed Vamana Build With Different Threadpools",
    "[integration][build][vamana][threadpool]"
) {
    auto distances = std::to_array<svs::DistanceType>({svs::L2, svs::MIP, svs::Cosine});

    // How far these results may deviate from previously generated results.
    const double epsilon = 0.005;
    const auto queries = svs::data::SimpleData<float>::load(test_dataset::query_file());
    size_t num_threads = 1;
    for (auto distance_type : distances) {
        CATCH_REQUIRE(svs_test::prepare_temp_directory());
        auto expected_result = test_dataset::vamana::expected_build_results(
            distance_type, svsbenchmark::Uncompressed(svs::DataType::float32)
        );
        svs::Vamana index = build_index<float>(
            expected_result.build_parameters_.value(),
            test_dataset::data_svs_file(),
            svs::threads::DefaultThreadPool(num_threads),
            distance_type
        );
        CATCH_REQUIRE(
            index.query_types() == std::vector<svs::DataType>{svs::DataType::float32}
        );

        auto groundtruth = test_dataset::load_groundtruth(distance_type);
        for (const auto& expected : expected_result.config_and_recall_) {
            auto these_queries = test_dataset::get_test_set(queries, expected.num_queries_);
            auto these_groundtruth =
                test_dataset::get_test_set(groundtruth, expected.num_queries_);
            index.set_search_parameters(expected.search_parameters_);
            auto results = index.search(these_queries, expected.num_neighbors_);
            double recall = svs::k_recall_at_n(
                these_groundtruth, results, expected.num_neighbors_, expected.recall_k_
            );

            fmt::print(
                "Window Size: {}, Expected Recall: {}, Actual Recall: {}\n",
                index.get_search_window_size(),
                expected.recall_,
                recall
            );
            CATCH_REQUIRE(recall > expected.recall_ - epsilon);
            CATCH_REQUIRE(recall < expected.recall_ + epsilon);

            index.set_threadpool(svs::threads::CppAsyncThreadPool(num_threads));

            CATCH_REQUIRE(index.get_num_threads() == num_threads);
            results = index.search(these_queries, expected.num_neighbors_);
            recall = svs::k_recall_at_n(
                these_groundtruth, results, expected.num_neighbors_, expected.recall_k_
            );
            fmt::print(
                "Window Size: {}, Expected Recall: {}, Actual Recall: {}\n",
                index.get_search_window_size(),
                expected.recall_,
                recall
            );
            CATCH_REQUIRE(recall > expected.recall_ - epsilon);
            CATCH_REQUIRE(recall < expected.recall_ + epsilon);

            auto& threadpool =
                index.get_threadpool_handle().get<svs::threads::CppAsyncThreadPool>();
            threadpool.resize(2);
            CATCH_REQUIRE(index.get_num_threads() == 2);
        }
        ++num_threads;
    }
}

// Helper function to create a logger with a callback
std::shared_ptr<spdlog::logger> create_test_logger(std::vector<std::string>& captured_logs
) {
    auto callback_sink = std::make_shared<spdlog::sinks::callback_sink_mt>(
        [&captured_logs](const spdlog::details::log_msg& msg) {
            captured_logs.emplace_back(msg.payload.data(), msg.payload.size());
        }
    );
    callback_sink->set_level(spdlog::level::trace);

    auto logger = std::make_shared<spdlog::logger>("test_logger", callback_sink);
    logger->set_level(spdlog::level::trace);
    return logger;
}

CATCH_TEST_CASE("VamanaIndex Logging Tests", "[logging]") {
    using namespace svs::index::vamana;

    // Test data setup
    std::vector<float> data = {1.0f, 2.0f};
    const size_t dim = 1;
    auto graph = svs::graphs::SimpleGraph<uint32_t>(1, 64);
    auto data_view = svs::data::SimpleDataView<float>(data.data(), 1, dim);
    svs::distance::DistanceL2 distance_function;
    uint32_t entry_point = 0;
    auto threadpool = svs::threads::DefaultThreadPool(1);
    VamanaBuildParameters buildParams(1.2, 64, 10, 20, 10, true);

    CATCH_SECTION("With Custom Logger") {
        std::vector<std::string> captured_logs;
        auto custom_logger = create_test_logger(captured_logs);

        // Create VamanaIndex, which will call the builder and construct
        VamanaIndex vamana_index(
            buildParams,
            std::move(graph),
            std::move(data_view),
            entry_point,
            std::move(distance_function),
            std::move(threadpool),
            custom_logger
        );

        // Verify the custom logger captured the log messages
        CATCH_REQUIRE(captured_logs[0].find("Number of syncs:") != std::string::npos);
        CATCH_REQUIRE(captured_logs[1].find("Batch Size:") != std::string::npos);
        auto default_logger = svs::logging::get();
        CATCH_REQUIRE(vamana_index.get_logger() != default_logger);
    }

    CATCH_SECTION("With Default Logger") {
        // Reset the test data setup
        std::vector<float> data = {1.0f, 2.0f};
        auto graph = svs::graphs::SimpleGraph<uint32_t>(1, 64);
        auto data_view = svs::data::SimpleDataView<float>(data.data(), 1, dim);
        auto threadpool = svs::threads::DefaultThreadPool(1);

        // Create VamanaIndex without passing a custom logger
        VamanaIndex vamana_index(
            buildParams,
            std::move(graph),
            std::move(data_view),
            entry_point,
            std::move(distance_function),
            std::move(threadpool)
        );

        auto default_logger = svs::logging::get();
        CATCH_REQUIRE(vamana_index.get_logger() == default_logger);
    }
}
