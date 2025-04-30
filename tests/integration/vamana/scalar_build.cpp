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

// svs
#include "svs/core/recall.h"
#include "svs/extensions/vamana/scalar.h"
#include "svs/lib/timing.h"
#include "svs/orchestrators/vamana.h"
#include "svs/quantization/scalar/scalar.h"

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

namespace scalar = svs::quantization::scalar;

namespace {

template <typename T, size_t D = svs::Dynamic, typename Distance>
svs::Vamana build_index(
    const svs::index::vamana::VamanaBuildParameters parameters,
    const std::filesystem::path& data_path,
    const size_t num_threads,
    const Distance& dist_type
) {
    auto tic = svs::lib::now();
    auto index = svs::Vamana::build<float>(
        parameters,
        svs::lib::Lazy([&]() {
            auto data = svs::data::SimpleData<float>::load(data_path);
            return scalar::SQDataset<T, D>::compress(data);
        }),
        dist_type,
        num_threads
    );

    fmt::print("Indexing time: {}s\n", svs::lib::time_difference(tic));
    CATCH_REQUIRE(index.get_num_threads() == num_threads);
    return index;
}

template <typename T, typename Distance> void test_build(const Distance& distance) {
    // How far these results may deviate from previously generated results.
    const double epsilon = 0.01;
    const auto queries = svs::data::SimpleData<float>::load(test_dataset::query_file());
    CATCH_REQUIRE(svs_test::prepare_temp_directory());
    size_t num_threads = 2;

    // Use uncompressed reference results which should be identical
    auto expected_result = test_dataset::vamana::expected_build_results(
        svs::distance_type_v<Distance>, svsbenchmark::Uncompressed(svs::DataType::float32)
    );

    svs::Vamana index = build_index<T, 128>(
        expected_result.build_parameters_.value(),
        test_dataset::data_svs_file(),
        num_threads,
        distance
    );

    auto groundtruth = test_dataset::load_groundtruth(svs::distance_type_v<Distance>);
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

} // namespace

CATCH_TEST_CASE("LeanVec Vamana Build", "[integration][build][vamana][scalar]") {
    test_build<std::int8_t>(svs::DistanceL2());
    test_build<std::int8_t>(svs::DistanceIP());
    test_build<std::int8_t>(svs::DistanceCosineSimilarity());
}
