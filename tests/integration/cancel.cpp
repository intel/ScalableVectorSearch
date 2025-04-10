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
#include "svs/core/distance.h"
#include "svs/index/flat/flat.h"
#include "svs/index/index.h"
#include "svs/orchestrators/exhaustive.h"
#include "svs/orchestrators/inverted.h"
#include "svs/orchestrators/vamana.h"

// svs-benchmark
#include "svs-benchmark/datasets.h"

// svs-test
#include "tests/utils/inverted_reference.h"
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"
#include "tests/utils/vamana_reference.h"

// catch2
#include "catch2/catch_test_macros.hpp"

namespace {

CATCH_TEST_CASE("Cancel", "[integration][cancel]") {
    CATCH_SECTION("Inverted Search Cancel") {
        size_t num_threads = 2;
        auto queries = svs::data::SimpleData<float>::load(test_dataset::query_file());
        auto distance = svs::DistanceL2();
        auto strategy = svs::index::inverted::SparseStrategy();

        constexpr svs::DistanceType distance_type =
            svs::distance_type_v<decltype(distance)>;
        auto expected_results = test_dataset::inverted::expected_build_results(
            distance_type, svsbenchmark::Uncompressed(svs::DataType::float32)
        );
        svs::Inverted index = svs::Inverted::build<float>(
            expected_results.build_parameters_.value(),
            svs::data::SimpleData<float, svs::Dynamic>::load(test_dataset::data_svs_file()),
            distance,
            num_threads,
            strategy
        );

        std::atomic<size_t> counter{0};
        auto timeout = [&]() { return ++counter >= 3; };

        const auto& expected = expected_results.config_and_recall_[0];
        const auto& sp = expected.search_parameters_;
        auto these_queries = test_dataset::get_test_set(queries, expected.num_queries_);
        auto groundtruth = test_dataset::load_groundtruth(distance_type);
        auto these_groundtruth =
            test_dataset::get_test_set(groundtruth, expected.num_queries_);
        index.set_search_parameters(sp);
        auto results = index.search(these_queries, expected.num_neighbors_, timeout);
        auto recall = svs::k_recall_at_n(
            these_groundtruth, results, expected.num_neighbors_, expected.recall_k_
        );

        // recall should be very bad due to timeout
        CATCH_REQUIRE(recall < 0.5);
        CATCH_REQUIRE(counter >= 3);
    }

    CATCH_SECTION("Vamana Search Cancel") {
        size_t num_threads = 3;
        auto index = svs::Vamana::assemble<svs::lib::Types<float, svs::Float16>>(
            test_dataset::vamana_config_file(),
            svs::GraphLoader(test_dataset::graph_file()),
            svs::VectorDataLoader<float>(test_dataset::data_svs_file()),
            svs::L2,
            2
        );
        auto expected_results =
            test_dataset::vamana::expected_search_results(
                svs::L2, svsbenchmark::Uncompressed(svs::DataType::float32)
            )
                .config_and_recall_;
        const auto& expected = expected_results.at(0);

        std::atomic<size_t> counter{0};
        auto timeout = [&]() { return ++counter >= 4; };

        const auto queries_all = test_dataset::queries();
        const auto queries_in_test_set = expected_results.at(0).num_queries_;
        auto queries = test_dataset::get_test_set(queries_all, queries_in_test_set);
        auto groundtruth_all = test_dataset::load_groundtruth(svs::L2);
        auto groundtruth = test_dataset::get_test_set(groundtruth_all, queries_in_test_set);
        index.set_search_parameters(expected.search_parameters_);
        index.set_threadpool(svs::threads::DefaultThreadPool(num_threads));
        auto results = index.search(queries, expected.num_neighbors_, timeout);
        auto recall = svs::k_recall_at_n(
            groundtruth, results, expected.num_neighbors_, expected.recall_k_
        );

        // recall should be very bad due to timeout
        CATCH_REQUIRE(recall < 0.5);
        CATCH_REQUIRE(counter >= 4);
    }

    auto queries = test_dataset::queries();
    auto data = svs::load_data<float>(test_dataset::data_svs_file());
    auto groundtruth = test_dataset::groundtruth_euclidean();

    CATCH_SECTION("Flat Index Search Cancel") {
        auto result =
            svs::QueryResult<size_t>(groundtruth.size(), groundtruth.dimensions());
        std::atomic<size_t> counter{0};
        auto timeout = [&]() { return ++counter >= 2; };
        auto index =
            svs::index::flat::FlatIndex(std::move(data), svs::distance::DistanceL2{}, 1);
        svs::index::search_batch_into(index, result.view(), queries.cview(), timeout);

        // recall should be very bad due to timeout
        CATCH_REQUIRE(svs::k_recall_at_n(groundtruth, result) < 0.5);
        CATCH_REQUIRE(counter >= 2);
    }

    CATCH_SECTION("Flat Orchestrator Search Cancel") {
        auto result =
            svs::QueryResult<size_t>(groundtruth.size(), groundtruth.dimensions());
        std::atomic<size_t> counter{0};
        auto timeout = [&]() { return ++counter >= 5; };
        svs::Flat index = svs::Flat::assemble<svs::lib::Types<float, svs::Float16>>(
            svs::VectorDataLoader<float>(test_dataset::data_svs_file()), svs::L2, 2
        );
        svs::index::search_batch_into(index, result.view(), queries.cview(), timeout);

        // recall should be very bad due to timeout
        CATCH_REQUIRE(svs::k_recall_at_n(groundtruth, result) < 0.5);
        CATCH_REQUIRE(counter >= 5);
    }

    CATCH_SECTION("Batch Iterator Search Cancel") {
        auto index = svs::Vamana::assemble<svs::lib::Types<float, svs::Float16>>(
            test_dataset::vamana_config_file(),
            svs::GraphLoader(test_dataset::graph_file()),
            svs::VectorDataLoader<float>(test_dataset::data_svs_file()),
            svs::L2,
            2
        );
        auto expected_results =
            test_dataset::vamana::expected_search_results(
                svs::L2, svsbenchmark::Uncompressed(svs::DataType::float32)
            )
                .config_and_recall_;
        const auto& expected = expected_results.at(0);

        std::atomic<size_t> counter{0};
        auto timeout = [&]() { return ++counter >= 4; };

        const auto queries_all = test_dataset::queries();
        auto queries = test_dataset::get_test_set(queries_all, 1);
        auto groundtruth_all = test_dataset::load_groundtruth(svs::L2);
        auto groundtruth = test_dataset::get_test_set(groundtruth_all, 1);
        auto batchsize = expected.num_neighbors_;

        auto itr = index.batch_iterator(queries.get_datum(0));
        itr.next(batchsize);

        auto results = svs::QueryResultImpl<size_t>(1, expected.num_neighbors_);
        auto neighbors = itr.results();
        for (size_t j = 0; j < expected.num_neighbors_; ++j) {
            results.set(neighbors[j], 0, j);
        }
        auto recall = svs::k_recall_at_n(
            groundtruth, results, expected.num_neighbors_, expected.recall_k_
        );

        CATCH_REQUIRE(recall > 0.6);
        CATCH_REQUIRE(counter == 0);

        itr = index.batch_iterator(queries.get_datum(0));
        itr.next(batchsize, timeout);
        neighbors = itr.results();
        for (size_t j = 0; j < expected.num_neighbors_; ++j) {
            results.set(neighbors[j], 0, j);
        }
        recall = svs::k_recall_at_n(
            groundtruth, results, expected.num_neighbors_, expected.recall_k_
        );

        CATCH_REQUIRE(recall < 0.6);
        CATCH_REQUIRE(counter >= 4);
        counter = 0;

        itr.update(queries.get_datum(0));
        itr.next(batchsize, timeout);
        neighbors = itr.results();
        for (size_t j = 0; j < expected.num_neighbors_; ++j) {
            results.set(neighbors[j], 0, j);
        }
        recall = svs::k_recall_at_n(
            groundtruth, results, expected.num_neighbors_, expected.recall_k_
        );

        CATCH_REQUIRE(recall < 0.6);
        CATCH_REQUIRE(counter >= 4);

        itr.restart_next_search();
        itr.next(batchsize, timeout);
        neighbors = itr.results();
        for (size_t j = 0; j < expected.num_neighbors_; ++j) {
            results.set(neighbors[j], 0, j);
        }
        recall = svs::k_recall_at_n(
            groundtruth, results, expected.num_neighbors_, expected.recall_k_
        );
        CATCH_REQUIRE(recall < 0.6);
        CATCH_REQUIRE(counter >= 4);
    }
}

} // namespace
