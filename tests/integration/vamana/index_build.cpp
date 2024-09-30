/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
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

namespace {

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

CATCH_TEST_CASE("Uncompressed Vamana Build", "[integration][build][vamana]") {
    auto distances = std::to_array<svs::DistanceType>({svs::L2, svs::MIP, svs::Cosine});

    // How far these results may deviate from previously generated results.
    const double epsilon = 0.005;
    const auto queries = svs::data::SimpleData<float>::load(test_dataset::query_file());
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
