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
#include "svs/core/data/simple.h"
#include "svs/core/recall.h"
#include "svs/lib/timing.h"
#include "svs/orchestrators/ivf.h"

// svsbenchmark
#include "svs-benchmark/benchmark.h"

// fmt
#include "fmt/core.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// tests
#include "tests/utils/ivf_reference.h"
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

// stl
#include <array>
#include <filesystem>
#include <string>
#include <utility>

namespace {

template <typename E, size_t D = svs::Dynamic, typename Distance>
auto build_index(
    const svs::index::ivf::IVFBuildParameters parameters,
    const std::filesystem::path& data_path,
    size_t num_threads,
    size_t num_inner_threads,
    const Distance& dist_type
) {
    auto data = svs::data::SimpleData<float, D>::load(data_path);
    auto clustering =
        svs::IVF::build_clustering<E>(parameters, data, dist_type, num_threads);

    return svs::IVF::assemble_from_clustering<float>(
        std::move(clustering), std::move(data), dist_type, num_threads, num_inner_threads
    );
}

template <typename T, typename Distance>
void test_build(const Distance& distance, size_t num_inner_threads = 1) {
    const double epsilon = 0.005;
    const auto queries = svs::data::SimpleData<float>::load(test_dataset::query_file());
    CATCH_REQUIRE(svs_test::prepare_temp_directory());
    size_t num_threads = 2;

    auto expected_result = test_dataset::ivf::expected_build_results(
        svs::distance_type_v<Distance>, svsbenchmark::Uncompressed(svs::datatype_v<T>)
    );
    auto index = build_index<T>(
        expected_result.build_parameters_.value(),
        test_dataset::data_svs_file(),
        num_threads,
        num_inner_threads,
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
            "n_probes: {}, Expected Recall: {}, Actual Recall: {}\n",
            index.get_search_parameters().n_probes_,
            expected.recall_,
            recall
        );
        CATCH_REQUIRE(recall > expected.recall_ - epsilon);
        CATCH_REQUIRE(recall < expected.recall_ + epsilon);
    }
}

} // namespace

CATCH_TEST_CASE("IVF Build/Clustering", "[integration][build][ivf]") {
    //test_build<float>(svs::DistanceL2());
    //test_build<svs::Float16>(svs::DistanceIP());

    test_build<svs::BFloat16>(svs::DistanceL2());
    //test_build<svs::BFloat16>(svs::DistanceIP());

    // With 4 inner threads
    //test_build<svs::BFloat16>(svs::DistanceL2(), 4);
    //test_build<svs::BFloat16>(svs::DistanceIP(), 4);
}
