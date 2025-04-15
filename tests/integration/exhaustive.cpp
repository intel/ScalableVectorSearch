/*
 * Copyright 2023 Intel Corporation
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

#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// svs
#include "svs/core/distance.h"
#include "svs/core/recall.h"
#include "svs/index/flat/flat.h"
#include "svs/lib/array.h"
#include "svs/lib/neighbor.h"
#include "svs/lib/timing.h"

#include "svs/orchestrators/exhaustive.h"

// stl
#include <span>

namespace {

template <typename T> inline constexpr bool is_flat_index_v = false;
template <typename... Args>
inline constexpr bool is_flat_index_v<svs::index::flat::FlatIndex<Args...>> = true;

// Test the predicated search.
// In this test, we predicate out the even indices and only return odd indices.
// The test checks that no even indices occur in the result.
template <typename Index, typename Queries>
void test_predicate(Index& index, const Queries& queries) {
    const size_t num_neighbors = 10;
    auto result = svs::QueryResult<size_t>(queries.size(), num_neighbors);

    // Perform a predicated search.
    // TODO: Expose predicated search through the dispatch pipeline.
    auto predicate = [](size_t data_index) { return (data_index % 2) != 0; };

    index.search(
        result.view(),
        queries.cview(),
        index.get_search_parameters(),
        []() { return false; },
        predicate
    );

    for (size_t i = 0; i < result.n_queries(); ++i) {
        for (size_t j = 0; j < result.n_neighbors(); ++j) {
            CATCH_REQUIRE((result.index(i, j) % 2) != 0);
        }
    }
}

template <
    typename Index,
    typename Queries,
    typename GroundTruth,
    svs::threads::ThreadPool Pool = svs::threads::DefaultThreadPool>
void test_flat(
    Index& index,
    const Queries& queries,
    const GroundTruth& groundtruth,
    svs::DistanceType distance_type
) {
    // Test get distance
    auto dataset = svs::load_data<float>(test_dataset::data_svs_file());
    // Call test get_distance in util.h
    svs_test::GetDistanceTester::test(index, distance_type, queries, dataset);

    CATCH_REQUIRE(index.size() == test_dataset::VECTORS_IN_DATA_SET);
    CATCH_REQUIRE(index.dimensions() == test_dataset::NUM_DIMENSIONS);

    const double expected_recall = 0.9999;

    ///// Make sure setting the data and query batch sizes works.
    auto p = index.get_search_parameters();
    CATCH_REQUIRE(p.data_batch_size_ == 0);
    CATCH_REQUIRE(p.query_batch_size_ == 0);

    index.set_search_parameters({10, 20});
    auto q = index.get_search_parameters();

    CATCH_REQUIRE(q.data_batch_size_ == 10);
    CATCH_REQUIRE(q.query_batch_size_ == 20);
    index.set_search_parameters({0, 0});
    q = index.get_search_parameters();
    CATCH_REQUIRE(q.data_batch_size_ == 0);
    CATCH_REQUIRE(q.query_batch_size_ == 0);

    // Make sure that changing the number of threads works as exected.
    // Should not change the end result.
    auto result = svs::QueryResult<size_t>(groundtruth.size(), groundtruth.dimensions());

    for (auto num_threads : std::array<size_t, 2>{{1, 2}}) {
        index.set_threadpool(Pool(num_threads));
        CATCH_REQUIRE((index.get_num_threads() == num_threads));
        svs::index::search_batch_into(index, result.view(), queries.cview());
        // index.search(queries.cview(), groundtruth.dimensions(), result.view());
        CATCH_REQUIRE(svs::k_recall_at_n(groundtruth, result) > expected_recall);
    }

    // Set different data and query batch sizes.
    index.set_threadpool(Pool(2));
    for (size_t query_batch_size : {0, 10}) {
        for (size_t data_batch_size : {0, 100}) {
            svs::index::search_batch_into_with(
                index, result.view(), queries.cview(), {data_batch_size, query_batch_size}
            );

            CATCH_REQUIRE(svs::k_recall_at_n(groundtruth, result) > expected_recall);
        }
    }

    // Test predicated search.
    if constexpr (is_flat_index_v<Index>) {
        test_predicate(index, queries);
    }
}
} // namespace

/////
///// Flat Index
/////

// Test the single-threaded implementation.
CATCH_TEST_CASE("Flat Index Search", "[integration][exhaustive][index]") {
    auto queries = test_dataset::queries();
    auto data = svs::load_data<float>(test_dataset::data_svs_file());

    std::cout << "Data size: (" << data.size() << ", " << data.dimensions() << ")"
              << std::endl;

    CATCH_SECTION("Flat Index - L2") {
        auto groundtruth = test_dataset::groundtruth_euclidean();
        // test the temporary index.
        {
            auto threadpool = svs::threads::DefaultThreadPool(4);
            auto temp = svs::index::flat::temporary_flat_index(
                data,
                svs::distance::DistanceL2(),
                svs::threads::ThreadPoolReferenceWrapper(threadpool)
            );
            test_flat(temp, queries, groundtruth, svs::L2);
        }

        auto index =
            svs::index::flat::FlatIndex(std::move(data), svs::distance::DistanceL2{}, 1);
        test_flat(index, queries, groundtruth, svs::L2);
    }

    CATCH_SECTION("Flat Index - IP") {
        auto groundtruth = test_dataset::groundtruth_mip();
        auto index =
            svs::index::flat::FlatIndex(std::move(data), svs::distance::DistanceIP{}, 1);
        test_flat(index, queries, groundtruth, svs::MIP);
    }

    CATCH_SECTION("Flat Index - Cosine") {
        auto groundtruth = test_dataset::groundtruth_cosine();
        auto index = svs::index::flat::FlatIndex(
            std::move(data), svs::distance::DistanceCosineSimilarity{}, 1
        );
        test_flat(index, queries, groundtruth, svs::Cosine);
    }

    CATCH_SECTION("Flat Index - Stateful") {
        auto groundtruth = test_dataset::groundtruth_euclidean();
        auto index =
            svs::index::flat::FlatIndex{std::move(data), svs_test::StatefulL2<float>{}, 1};
        test_flat(index, queries, groundtruth, svs::L2);
    }

    CATCH_SECTION("Flat Index With CppAyncThreadPool - IP") {
        auto groundtruth = test_dataset::groundtruth_mip();
        auto index = svs::index::flat::FlatIndex(
            std::move(data),
            svs::distance::DistanceIP{},
            svs::threads::CppAsyncThreadPool(2)
        );
        test_flat<
            decltype(index),
            decltype(queries),
            decltype(groundtruth),
            svs::threads::CppAsyncThreadPool>(index, queries, groundtruth, svs::MIP);
        auto& threadpool =
            index.get_threadpool_handle().get<svs::threads::CppAsyncThreadPool>();
        threadpool.resize(3);
        CATCH_REQUIRE(index.get_num_threads() == 3);
        test_flat<
            decltype(index),
            decltype(queries),
            decltype(groundtruth),
            svs::threads::CppAsyncThreadPool>(index, queries, groundtruth, svs::MIP);
    }

    CATCH_SECTION("Flat Index With QueueThreadPoolWrapper - Cosine") {
        auto groundtruth = test_dataset::groundtruth_cosine();
        auto index = svs::index::flat::FlatIndex(
            std::move(data),
            svs::distance::DistanceCosineSimilarity{},
            svs::threads::QueueThreadPoolWrapper(2)
        );
        test_flat<
            decltype(index),
            decltype(queries),
            decltype(groundtruth),
            svs::threads::QueueThreadPoolWrapper>(index, queries, groundtruth, svs::Cosine);
    }

    CATCH_SECTION("Flat Index With Different Thread Pools - Cosine") {
        auto groundtruth = test_dataset::groundtruth_cosine();
        auto index = svs::index::flat::FlatIndex(
            std::move(data),
            svs::distance::DistanceCosineSimilarity{},
            svs::threads::QueueThreadPoolWrapper(2)
        );
        test_flat<
            decltype(index),
            decltype(queries),
            decltype(groundtruth),
            svs::threads::CppAsyncThreadPool>(index, queries, groundtruth, svs::Cosine);
        test_flat(index, queries, groundtruth, svs::Cosine);
    }
}

/////
///// Flat
/////

CATCH_TEST_CASE(
    "Flat Orchestrator Search", "[integration][exhaustive][orchestrator][get_distance]"
) {
    auto queries = test_dataset::queries();

    // Load data using both the file path method and from a direct file.
    // Use the `DefaultAllocator` to allow implicit copies.
    auto data = svs::load_data<float>(test_dataset::data_svs_file());

    CATCH_SECTION("Euclidean") {
        // From file
        svs::Flat index = svs::Flat::assemble<svs::lib::Types<float, svs::Float16>>(
            svs::VectorDataLoader<float>(test_dataset::data_svs_file()), svs::L2, 2
        );
        CATCH_REQUIRE(index.get_num_threads() == 2);
        CATCH_REQUIRE(
            index.query_types() ==
            std::vector<svs::DataType>{svs::DataType::float32, svs::DataType::float16}
        );
        test_flat(index, queries, test_dataset::groundtruth_euclidean(), svs::L2);

        // Also try float16 as the query to test heterogeneous query handling.
        auto queries_f16 =
            svs::data::SimpleData<svs::Float16>(queries.size(), queries.dimensions());
        svs::data::copy(queries, queries_f16);
        test_flat(index, queries_f16, test_dataset::groundtruth_euclidean(), svs::L2);

        // From Data
        index = svs::Flat::assemble<float>(std::move(data), svs::L2, 2);
        CATCH_REQUIRE(index.get_num_threads() == 2);
        test_flat(index, queries, test_dataset::groundtruth_euclidean(), svs::L2);
    }

    CATCH_SECTION("InnerProduct") {
        // From file
        svs::Flat index = svs::Flat::assemble<float>(
            svs::VectorDataLoader<float>(test_dataset::data_svs_file()), svs::MIP, 2
        );
        CATCH_REQUIRE(index.get_num_threads() == 2);
        test_flat(index, queries, test_dataset::groundtruth_mip(), svs::MIP);

        // From Data
        index = svs::Flat::assemble<float>(std::move(data), svs::MIP, 2);
        CATCH_REQUIRE(index.get_num_threads() == 2);
        test_flat(index, queries, test_dataset::groundtruth_mip(), svs::MIP);
    }

    CATCH_SECTION("Cosine") {
        // From file
        svs::Flat index = svs::Flat::assemble<float>(
            svs::VectorDataLoader<float>(test_dataset::data_svs_file()), svs::Cosine, 2
        );
        CATCH_REQUIRE(index.get_num_threads() == 2);
        test_flat(index, queries, test_dataset::groundtruth_cosine(), svs::Cosine);

        // From Data
        index = svs::Flat::assemble<float>(std::move(data), svs::Cosine, 2);
        CATCH_REQUIRE(index.get_num_threads() == 2);
        test_flat(index, queries, test_dataset::groundtruth_cosine(), svs::Cosine);
    }

    CATCH_SECTION("Cosine With Different Thread Pools From File") {
        svs::Flat index = svs::Flat::assemble<float>(
            svs::VectorDataLoader<float>(test_dataset::data_svs_file()),
            svs::Cosine,
            svs::threads::CppAsyncThreadPool(2)
        );
        CATCH_REQUIRE(index.get_num_threads() == 2);

        auto& threadpool =
            index.get_threadpool_handle().get<svs::threads::CppAsyncThreadPool>();
        threadpool.resize(3);
        CATCH_REQUIRE(index.get_num_threads() == 3);
        test_flat<
            decltype(index),
            decltype(queries),
            decltype(test_dataset::groundtruth_cosine()),
            svs::threads::CppAsyncThreadPool>(
            index, queries, test_dataset::groundtruth_cosine(), svs::Cosine
        );

        index.set_threadpool(svs::threads::DefaultThreadPool(3));
        test_flat<
            decltype(index),
            decltype(queries),
            decltype(test_dataset::groundtruth_cosine()),
            svs::threads::DefaultThreadPool>(
            index, queries, test_dataset::groundtruth_cosine(), svs::Cosine
        );

        test_flat<
            decltype(index),
            decltype(queries),
            decltype(test_dataset::groundtruth_cosine()),
            svs::threads::QueueThreadPoolWrapper>(
            index, queries, test_dataset::groundtruth_cosine(), svs::Cosine
        );
        test_flat<
            decltype(index),
            decltype(queries),
            decltype(test_dataset::groundtruth_cosine()),
            svs::threads::SwitchNativeThreadPool>(
            index, queries, test_dataset::groundtruth_cosine(), svs::Cosine
        );
    }

    CATCH_SECTION("Cosine With Different Thread Pools From Data") {
        svs::Flat index = svs::Flat::assemble<float>(
            std::move(data), svs::Cosine, svs::threads::QueueThreadPoolWrapper(3)
        );
        CATCH_REQUIRE(index.get_num_threads() == 3);
        test_flat<
            decltype(index),
            decltype(queries),
            decltype(test_dataset::groundtruth_cosine()),
            svs::threads::QueueThreadPoolWrapper>(
            index, queries, test_dataset::groundtruth_cosine(), svs::Cosine
        );

        index.set_threadpool(svs::threads::CppAsyncThreadPool(2));
        CATCH_REQUIRE(index.get_num_threads() == 2);
        test_flat<
            decltype(index),
            decltype(queries),
            decltype(test_dataset::groundtruth_cosine()),
            svs::threads::CppAsyncThreadPool>(
            index, queries, test_dataset::groundtruth_cosine(), svs::Cosine
        );

        test_flat<
            decltype(index),
            decltype(queries),
            decltype(test_dataset::groundtruth_cosine()),
            svs::threads::DefaultThreadPool>(
            index, queries, test_dataset::groundtruth_cosine(), svs::Cosine
        );
        test_flat<
            decltype(index),
            decltype(queries),
            decltype(test_dataset::groundtruth_cosine()),
            svs::threads::SwitchNativeThreadPool>(
            index, queries, test_dataset::groundtruth_cosine(), svs::Cosine
        );
    }
}
