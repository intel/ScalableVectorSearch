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

// Orchestrator under test
#include "svs/orchestrators/dynamic_vamana.h"

// Core helpers
#include "svs/core/data/simple.h"
#include "svs/core/recall.h"

// Distance dispatcher
#include "svs/core/distance.h"

// Test dataset utilities
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"
#include "tests/utils/vamana_reference.h"

// Catch2
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

// STL
#include <numeric>
#include <vector>

namespace {

template <typename DataLoaderT, typename DistanceT>
void test_build(DataLoaderT&& data_loader, DistanceT distance = DistanceT()) {
    auto expected_result = test_dataset::vamana::expected_build_results(
        distance, svsbenchmark::Uncompressed(svs::DataType::float32)
    );
    auto build_params = expected_result.build_parameters_.value();
    auto queries = svs::data::SimpleData<float>::load(test_dataset::query_file());
    auto groundtruth = test_dataset::load_groundtruth(distance);

    // Prepare IDs (0 .. N-1)
    auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());
    const size_t n = data.size();
    std::vector<size_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);

    size_t num_threads = 2;
    svs::DynamicVamana index = svs::DynamicVamana::build<float>(
        build_params, std::forward<DataLoaderT>(data_loader), ids, distance, num_threads
    );

    // Basic invariants
    CATCH_REQUIRE(index.get_alpha() == Catch::Approx(build_params.alpha));
    CATCH_REQUIRE(index.get_construction_window_size() == build_params.window_size);
    CATCH_REQUIRE(index.get_prune_to() == build_params.prune_to);
    CATCH_REQUIRE(index.get_graph_max_degree() == build_params.graph_max_degree);
    CATCH_REQUIRE(index.get_num_threads() == num_threads);

    // ID checks (spot sample)
    CATCH_REQUIRE(index.has_id(0));
    CATCH_REQUIRE(index.has_id(n / 2));
    CATCH_REQUIRE(index.has_id(n - 1));

    const double epsilon = 0.01; // allow small deviation
    for (const auto& expected : expected_result.config_and_recall_) {
        auto these_queries = test_dataset::get_test_set(queries, expected.num_queries_);
        auto these_groundtruth =
            test_dataset::get_test_set(groundtruth, expected.num_queries_);
        index.set_search_parameters(expected.search_parameters_);
        auto results = index.search(these_queries, expected.num_neighbors_);
        double recall = svs::k_recall_at_n(
            these_groundtruth, results, expected.num_neighbors_, expected.recall_k_
        );
        CATCH_REQUIRE(recall > expected.recall_ - epsilon);
        CATCH_REQUIRE(recall < expected.recall_ + epsilon);
    }
}

} // namespace

CATCH_TEST_CASE("DynamicVamana Build", "[managers][dynamic_vamana][build]") {
    for (auto distance_enum : test_dataset::vamana::available_build_distances()) {
        // SimpleData and distance functor.
        {
            std::string section_name =
                std::string("SimpleData ") + std::string(svs::name(distance_enum));
            CATCH_SECTION(section_name) {
                svs::DistanceDispatcher dispatcher(distance_enum);
                dispatcher([&](auto distance_functor) {
                    test_build(
                        svs::data::SimpleData<float>::load(test_dataset::data_svs_file()),
                        distance_functor
                    );
                });
            }
        }

        // VectorDataLoader and distance enum.
        {
            std::string section_name =
                std::string("VectorDataLoader ") + std::string(svs::name(distance_enum));
            CATCH_SECTION(section_name) {
                test_build(
                    svs::VectorDataLoader<float>(test_dataset::data_svs_file()),
                    distance_enum
                );
            }
        }
    }
}

CATCH_TEST_CASE("DynamicVamana Memory Usage", "[managers][dynamic_vamana]") {
    auto distance = svs::distance::DistanceL2();
    auto expected_result = test_dataset::vamana::expected_build_results(
        distance, svsbenchmark::Uncompressed(svs::DataType::float32)
    );
    auto build_params = expected_result.build_parameters_.value();
    size_t num_threads = 2;

    auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());
    const size_t n = data.size();
    const size_t half = n / 2;
    CATCH_REQUIRE(half > 0);
    CATCH_REQUIRE(n - half > 0);

    // Build the index over the first half of the dataset (external IDs 0 .. half-1).
    auto first_data = svs::data::SimpleData<float>(half, data.dimensions());
    for (size_t i = 0; i < half; ++i) {
        first_data.set_datum(i, data.get_datum(i));
    }
    std::vector<size_t> first_ids(half);
    std::iota(first_ids.begin(), first_ids.end(), 0);

    svs::DynamicVamana index = svs::DynamicVamana::build<float>(
        build_params, std::move(first_data), first_ids, distance, num_threads
    );

    const size_t usage_before = index.get_memory_breakdown().total();
    CATCH_REQUIRE(usage_before > 0);

    // Add the second half of the dataset (external IDs half .. n-1).
    const size_t rest = n - half;
    auto second_data = svs::data::SimpleData<float>(rest, data.dimensions());
    for (size_t i = 0; i < rest; ++i) {
        second_data.set_datum(i, data.get_datum(half + i));
    }
    std::vector<size_t> second_ids(rest);
    std::iota(second_ids.begin(), second_ids.end(), half);
    index.add_points(second_data.cview(), second_ids);

    // Adding points must increase the reported allocation.
    const size_t usage_after = index.get_memory_breakdown().total();
    CATCH_REQUIRE(usage_after > usage_before);
}
