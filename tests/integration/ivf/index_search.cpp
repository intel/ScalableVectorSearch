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
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
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
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"
#include "tests/utils/ivf_reference.h"

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
        test_dataset::clustering_directory(),
        data,
        distance,
        num_threads,
        num_inner_threads
    );
    CATCH_REQUIRE(index.get_num_threads() == num_threads);

    run_search(
        index, queries, groundtruth, expected_result.config_and_recall_
    );
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
