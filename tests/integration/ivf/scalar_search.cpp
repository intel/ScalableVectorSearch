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
#include "svs/extensions/ivf/scalar.h"
#include "svs/lib/saveload.h"
#include "svs/orchestrators/ivf.h"

// tests
#include "tests/utils/ivf_reference.h"
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

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
    svs::IVF& index,
    const svs::data::SimpleData<float>& queries_all,
    const svs::data::SimpleData<uint32_t>& groundtruth_all,
    const std::vector<svsbenchmark::ivf::ConfigAndResult>& expected_results
) {
    double epsilon = 0.05;

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

        index.set_search_parameters(expected.search_parameters_);
        CATCH_REQUIRE(index.get_search_parameters() == expected.search_parameters_);

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

template <typename T, scalar::IsSQData Data, typename Distance>
void test_search(
    Data data, const Distance& distance, const svs::data::SimpleData<float>& queries
) {
    size_t num_threads = 2;

    // We are able to compare to the uncompressed expected results
    auto expected_results = test_dataset::ivf::expected_search_results(
        svs::distance_type_v<Distance>, svsbenchmark::Uncompressed(svs::datatype_v<T>)
    );
    auto groundtruth = test_dataset::load_groundtruth(svs::distance_type_v<Distance>);

    auto index = svs::IVF::assemble_from_file<float, svs::BFloat16>(
        test_dataset::clustering_directory(), data, distance, num_threads
    );
    CATCH_REQUIRE(index.get_num_threads() == num_threads);

    run_search(index, queries, groundtruth, expected_results.config_and_recall_);
    CATCH_REQUIRE(index.dimensions() == test_dataset::NUM_DIMENSIONS);
}
} // namespace

CATCH_TEST_CASE("SQDataset IVF Search", "[integration][search][ivf][scalar]") {
    namespace ivf = svs::index::ivf;

    const size_t N = 128;
    auto datafile = test_dataset::data_svs_file();
    auto queries = test_dataset::queries();
    auto extents = std::make_tuple(svs::lib::Val<N>(), svs::lib::Val<svs::Dynamic>());

    svs::lib::foreach (extents, [&]<size_t E>(svs::lib::Val<E> /*unused*/) {
        fmt::print("Scalar quantization search - Extent {}\n", E);
        auto data = svs::data::SimpleData<float, E>::load(datafile);

        auto compressed = scalar::SQDataset<std::int8_t>::compress(data);
        test_search<float>(compressed, svs::distance::DistanceL2(), queries);
        test_search<svs::Float16>(compressed, svs::distance::DistanceIP(), queries);
    });
}
