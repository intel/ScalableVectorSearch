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

// stl
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

// svs
#include "svs/core/recall.h"
#include "svs/lib/saveload.h"
#include "svs/orchestrators/ivf.h"

// svsbenchmark
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/test.h"

// catch2
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_string.hpp"

// fmt
#include "fmt/core.h"

// tests
#include "tests/utils/ivf_reference.h"
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

namespace {

void run_search(
    svs::IVF& index,
    const svs::data::SimpleData<float>& queries_all,
    const svs::data::SimpleData<uint32_t>& groundtruth_all,
    const std::vector<svsbenchmark::ivf::ConfigAndResult>& expected_results
) {
    // If we make a change that somehow improves accuracy, we'll want to know.
    // Use `epsilon` to add to the expected results to set an upper bound on the
    // achieved accuracy.
    double epsilon = 0.005;

    // Ensure we have at least one entry in the expected results.
    CATCH_REQUIRE(!expected_results.empty());

    const auto queries_in_test_set = expected_results.at(0).num_queries_;

    auto queries = test_dataset::get_test_set(queries_all, queries_in_test_set);
    auto groundtruth = test_dataset::get_test_set(groundtruth_all, queries_in_test_set);

    for (const auto& expected : expected_results) {
        // Update the query set if needed.
        auto num_queries = expected.num_queries_;
        if (num_queries != queries.size()) {
            queries = test_dataset::get_test_set(queries_all, num_queries);
            groundtruth = test_dataset::get_test_set(groundtruth_all, num_queries);
        }

        // Configure the index with the current parameters.
        // Ensure that the result sticks.
        index.set_search_parameters(expected.search_parameters_);
        CATCH_REQUIRE(index.get_search_parameters() == expected.search_parameters_);

        // Float32
        auto results = index.search(queries, expected.num_neighbors_);
        auto recall = svs::k_recall_at_n(
            groundtruth, results, expected.num_neighbors_, expected.recall_k_
        );
        fmt::print(
            "n_probes: {}, Expected Recall: {}, Actual Recall: {}\n",
            index.get_search_parameters().n_probes_,
            expected.recall_,
            recall
        );

        CATCH_REQUIRE(recall > expected.recall_ - epsilon);
        CATCH_REQUIRE(recall < expected.recall_ + epsilon);
    }
}

template <typename T, typename Distance>
void test_search(
    svs::data::SimpleData<T> data,
    const Distance& distance,
    const svs::data::SimpleData<float>& queries,
    const svs::data::SimpleData<uint32_t>& groundtruth,
    const size_t num_inner_threads = 1
) {
    size_t num_threads = 2;

    // Find the expected results for this dataset.
    auto expected_result = test_dataset::ivf::expected_search_results(
        svs::distance_type_v<Distance>, svsbenchmark::Uncompressed(svs::datatype_v<T>)
    );

    auto index = svs::IVF::assemble_from_file<float, svs::BFloat16>(
        test_dataset::clustering_directory(), data, distance, num_threads, num_inner_threads
    );
    CATCH_REQUIRE(index.get_num_threads() == num_threads);

    run_search(index, queries, groundtruth, expected_result.config_and_recall_);
    CATCH_REQUIRE(index.dimensions() == test_dataset::NUM_DIMENSIONS);
}

} // namespace

CATCH_TEST_CASE("IVF Search", "[integration][search][ivf]") {
    namespace ivf = svs::index::ivf;

    auto datafile = test_dataset::data_svs_file();
    auto queries = test_dataset::queries();
    auto gt_l2 = test_dataset::groundtruth_euclidean();
    auto gt_ip = test_dataset::groundtruth_mip();

    auto dist_l2 = svs::distance::DistanceL2();
    auto dist_ip = svs::distance::DistanceIP();

    auto data = svs::data::SimpleData<float>::load(datafile);
    auto data_f16 = svs::index::ivf::convert_data<svs::Float16>(data);
    test_search(data, dist_l2, queries, gt_l2);
    test_search(data, dist_l2, queries, gt_l2, 2);

    test_search(data_f16, dist_ip, queries, gt_ip);
    test_search(data_f16, dist_ip, queries, gt_ip, 2);
}

CATCH_TEST_CASE("IVF get_distance", "[integration][ivf][get_distance]") {
    auto datafile = test_dataset::data_svs_file();
    auto queries = test_dataset::queries();
    auto dist_l2 = svs::distance::DistanceL2();

    auto data = svs::data::SimpleData<float>::load(datafile);

    size_t num_threads = 2;
    auto index = svs::IVF::assemble_from_file<float, svs::BFloat16>(
        test_dataset::clustering_directory(), data, dist_l2, num_threads, 1
    );

    // Test get_distance functionality with strict tolerance
    constexpr double TOLERANCE = 1e-2; // 1% tolerance

    // Test with a few different IDs
    std::vector<size_t> test_ids = {0, 10, 50};
    if (data.size() > 100) {
        test_ids.push_back(100);
    }

    for (size_t test_id : test_ids) {
        if (test_id >= data.size()) {
            continue;
        }

        // Get a query vector
        size_t query_id = std::min<size_t>(5, queries.size() - 1);
        auto query = queries.get_datum(query_id);

        // Get distance from index
        double index_distance = index.get_distance(test_id, query);

        // Compute expected distance from original data
        auto datum = data.get_datum(test_id);
        svs::distance::DistanceL2 dist_copy;
        svs::distance::maybe_fix_argument(dist_copy, query);
        double expected_distance = svs::distance::compute(dist_copy, query, datum);

        // Verify the distance is correct
        double relative_diff =
            std::abs((index_distance - expected_distance) / expected_distance);
        CATCH_REQUIRE(relative_diff < TOLERANCE);
    }

    // Test with out of bounds ID - should throw
    CATCH_REQUIRE_THROWS_AS(
        index.get_distance(data.size() + 1000, queries.get_datum(0)), svs::ANNException
    );
}

CATCH_TEST_CASE(
    "IVF get_distance thread safety", "[integration][ivf][get_distance][thread_safety]"
) {
    auto datafile = test_dataset::data_svs_file();
    auto queries = test_dataset::queries();
    auto dist_l2 = svs::distance::DistanceL2();

    auto data = svs::data::SimpleData<float>::load(datafile);

    size_t num_threads = 2;
    auto index = svs::IVF::assemble_from_file<float, svs::BFloat16>(
        test_dataset::clustering_directory(), data, dist_l2, num_threads, 1
    );

    // Test thread safety of get_distance with concurrent calls
    // The lazy initialization of ID mapping should be thread-safe with std::call_once
    constexpr size_t NUM_TEST_THREADS = 8;
    constexpr size_t CALLS_PER_THREAD = 100;
    constexpr double TOLERANCE = 1e-2;

    // Prepare test data
    std::vector<size_t> test_ids;
    for (size_t i = 0; i < std::min<size_t>(10, data.size()); ++i) {
        test_ids.push_back(i * (data.size() / 10));
    }

    // Pre-compute expected distances for verification
    std::vector<std::vector<double>> expected_distances(test_ids.size());
    for (size_t i = 0; i < test_ids.size(); ++i) {
        expected_distances[i].resize(queries.size());
        auto datum = data.get_datum(test_ids[i]);
        for (size_t q = 0; q < queries.size(); ++q) {
            auto query = queries.get_datum(q);
            svs::distance::DistanceL2 dist_copy;
            svs::distance::maybe_fix_argument(dist_copy, query);
            expected_distances[i][q] = svs::distance::compute(dist_copy, query, datum);
        }
    }

    // Track results and errors from threads
    std::atomic<size_t> success_count{0};
    std::atomic<size_t> error_count{0};
    std::vector<std::thread> threads;
    threads.reserve(NUM_TEST_THREADS);

    // Launch multiple threads that concurrently call get_distance
    for (size_t t = 0; t < NUM_TEST_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            for (size_t call = 0; call < CALLS_PER_THREAD; ++call) {
                size_t id_idx = (t + call) % test_ids.size();
                size_t query_idx = (t * CALLS_PER_THREAD + call) % queries.size();
                size_t test_id = test_ids[id_idx];

                auto query = queries.get_datum(query_idx);
                double index_distance = index.get_distance(test_id, query);
                double expected = expected_distances[id_idx][query_idx];

                double relative_diff = std::abs((index_distance - expected) / expected);
                if (relative_diff < TOLERANCE) {
                    ++success_count;
                } else {
                    ++error_count;
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify all calls succeeded
    CATCH_REQUIRE(error_count == 0);
    CATCH_REQUIRE(success_count == NUM_TEST_THREADS * CALLS_PER_THREAD);
}
