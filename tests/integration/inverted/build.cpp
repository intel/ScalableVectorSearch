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
#include "svs/lib/timing.h"
#include "svs/orchestrators/inverted.h"

// svs-benchmark
#include "svs-benchmark/datasets.h"

// tests
#include "tests/utils/inverted_reference.h"
#include "tests/utils/test_dataset.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <filesystem>

namespace {

template <
    typename E,
    typename Distance,
    typename ClusterStrategy,
    svs::threads::ThreadPool Pool,
    size_t D = svs::Dynamic>
svs::Inverted build_index(
    const svs::index::inverted::InvertedBuildParameters& build_parameters,
    const std::filesystem::path& data_path,
    Pool threadpool,
    Distance distance,
    ClusterStrategy strategy
) {
    auto tic = svs::lib::now();
    svs::Inverted index = svs::Inverted::build<E>(
        build_parameters,
        svs::data::SimpleData<E, D>::load(data_path),
        distance,
        std::move(threadpool),
        strategy
    );
    fmt::print("Indexing time: {}s\n", svs::lib::time_difference(tic));
    return index;
}

template <typename Distance, typename Strategy, typename Queries, typename ThreadPoolProto>
void run_test(const Queries& queries, ThreadPoolProto threadpool_proto) {
    auto distance = Distance();
    auto strategy = Strategy();

    // Distance between the obtained results and reference ressults.
    #if defined(__APPLE__)
        const double epsilon = 0.01;
    #else
        const double epsilon = 0.005;
    #endif  // __APPLE__

    constexpr svs::DistanceType distance_type = svs::distance_type_v<decltype(distance)>;
    auto expected_results = test_dataset::inverted::expected_build_results(
        distance_type, svsbenchmark::Uncompressed(svs::DataType::float32)
    );

    svs::Inverted index = build_index<float>(
        expected_results.build_parameters_.value(),
        test_dataset::data_svs_file(),
        svs::threads::as_threadpool(std::move(threadpool_proto)),
        distance,
        strategy
    );

    auto groundtruth = test_dataset::load_groundtruth(distance_type);
    for (const auto& expected : expected_results.config_and_recall_) {
        const auto& sp = expected.search_parameters_;
        CATCH_REQUIRE(index.get_search_parameters() != sp);
        auto these_queries = test_dataset::get_test_set(queries, expected.num_queries_);
        auto these_groundtruth =
            test_dataset::get_test_set(groundtruth, expected.num_queries_);
        index.set_search_parameters(sp);
        CATCH_REQUIRE(index.get_search_parameters() == sp);
        for (size_t num_threads : {1, 2}) {
            index.set_threadpool(svs::threads::DefaultThreadPool(num_threads));
            CATCH_REQUIRE(index.get_num_threads() == num_threads);

            auto results = index.search(these_queries, expected.num_neighbors_);
            double recall = svs::k_recall_at_n(
                these_groundtruth, results, expected.num_neighbors_, expected.recall_k_
            );

            fmt::print(
                "Expected Recall: {}, Actual Recall: {}\n", expected.recall_, recall
            );
            CATCH_REQUIRE(recall > expected.recall_ - epsilon);
            CATCH_REQUIRE(recall < expected.recall_ + epsilon);
            auto& threadpool =
                index.get_threadpool_handle().get<svs::threads::DefaultThreadPool>();
            threadpool.resize(3);
            CATCH_REQUIRE(index.get_num_threads() == 3);
        }
    }
}

} // namespace

CATCH_TEST_CASE("Test Inverted Building", "[integration][build][inverted]") {
    auto queries = svs::data::SimpleData<float>::load(test_dataset::query_file());
    run_test<svs::DistanceL2, svs::index::inverted::SparseStrategy>(queries, 2);
    run_test<svs::DistanceL2, svs::index::inverted::DenseStrategy>(
        queries, svs::threads::DefaultThreadPool(2)
    );
    run_test<svs::DistanceIP, svs::index::inverted::SparseStrategy>(queries, 3);
    run_test<svs::DistanceIP, svs::index::inverted::DenseStrategy>(
        queries, svs::threads::CppAsyncThreadPool(3)
    );
    run_test<svs::DistanceIP, svs::index::inverted::SparseStrategy>(
        queries, svs::threads::QueueThreadPoolWrapper(2)
    );
}
