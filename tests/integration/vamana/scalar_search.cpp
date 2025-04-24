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
#include "svs/lib/saveload.h"
#include "svs/orchestrators/vamana.h"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"
#include "tests/utils/vamana_reference.h"

// svsbenchmark
#include "svs-benchmark/benchmark.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

namespace scalar = svs::quantization::scalar;
namespace {

void run_search(
    svs::Vamana& index,
    const svs::data::SimpleData<float>& queries_all,
    const svs::data::SimpleData<uint32_t>& groundtruth_all,
    const std::vector<svsbenchmark::vamana::ConfigAndResult>& expected_results
) {
    double epsilon = 0.008;
    for (const auto& expected : expected_results) {
        auto num_queries = expected.num_queries_;
        auto queries = test_dataset::get_test_set(queries_all, num_queries);
        auto groundtruth = test_dataset::get_test_set(groundtruth_all, num_queries);

        index.set_search_parameters(expected.search_parameters_);
        CATCH_REQUIRE(index.get_search_parameters() == expected.search_parameters_);

        for (auto num_threads : {1, 2}) {
            index.set_threadpool(svs::threads::DefaultThreadPool(num_threads));
            auto results = index.search(queries, expected.num_neighbors_);
            auto recall = svs::k_recall_at_n(
                groundtruth, results, expected.num_neighbors_, expected.recall_k_
            );
            CATCH_REQUIRE(recall > expected.recall_ - epsilon);
            CATCH_REQUIRE(recall < expected.recall_ + epsilon);
        }
    }
}

template <scalar::IsSQData Data, typename Distance>
void test_search(
    Data data, const Distance& distance, const svs::data::SimpleData<float>& queries
) {
    size_t num_threads = 2;

    // We are able to compare to the uncompressed expected results
    auto expected_results = test_dataset::vamana::expected_search_results(
        svs::distance_type_v<Distance>, svsbenchmark::Uncompressed(svs::DataType::float32)
    );
    auto groundtruth = test_dataset::load_groundtruth(svs::distance_type_v<Distance>);

    // Make a copy of the original data to use for reconstruction comparison.
    auto index = svs::Vamana::assemble<float>(
        test_dataset::vamana_config_file(),
        svs::GraphLoader(test_dataset::graph_file()),
        std::move(data),
        distance,
        num_threads
    );
    CATCH_REQUIRE(index.get_num_threads() == num_threads);

    run_search(index, queries, groundtruth, expected_results.config_and_recall_);
    CATCH_REQUIRE(index.size() == test_dataset::VECTORS_IN_DATA_SET);
    CATCH_REQUIRE(index.dimensions() == test_dataset::NUM_DIMENSIONS);

    svs_test::prepare_temp_directory();
    auto dir = svs_test::temp_directory();

    auto config_dir = dir / "config";
    auto graph_dir = dir / "graph";
    auto data_dir = dir / "data";
    index.save(config_dir, graph_dir, data_dir);

    // Reload
    {
        auto reloaded_data = svs::lib::load_from_disk<Data>(data_dir);
        auto reloaded = svs::Vamana::assemble<float>(
            config_dir,
            svs::GraphLoader(graph_dir),
            std::move(reloaded_data),
            distance,
            num_threads
        );
        CATCH_REQUIRE(reloaded.get_num_threads() == num_threads);
        CATCH_REQUIRE(reloaded.size() == test_dataset::VECTORS_IN_DATA_SET);
        CATCH_REQUIRE(reloaded.dimensions() == test_dataset::NUM_DIMENSIONS);
        run_search(index, queries, groundtruth, expected_results.config_and_recall_);
    }
}
} // namespace

CATCH_TEST_CASE("SQDataset Vamana Search", "[integration][search][vamana][scalar]") {
    namespace vamana = svs::index::vamana;

    const size_t N = 128;
    auto datafile = test_dataset::data_svs_file();
    auto queries = test_dataset::queries();
    auto extents = std::make_tuple(svs::lib::Val<N>(), svs::lib::Val<svs::Dynamic>());

    svs::lib::foreach (extents, [&]<size_t E>(svs::lib::Val<E> /*unused*/) {
        fmt::print("Scalar quantization search - Extent {}\n", E);
        auto data = svs::data::SimpleData<float, E>::load(datafile);
        auto compressed = scalar::SQDataset<std::int8_t, E>::compress(data);

        // Sequential tests
        // clang-format off
        test_search(compressed, svs::distance::DistanceL2(), queries);
        test_search(compressed, svs::distance::DistanceIP(), queries);
        test_search(compressed, svs::distance::DistanceCosineSimilarity(), queries);
        // clang-format on
    });
}