/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
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

    index.search(result.view(), queries.cview(), index.get_search_parameters(), predicate);

    for (size_t i = 0; i < result.n_queries(); ++i) {
        for (size_t j = 0; j < result.n_neighbors(); ++j) {
            CATCH_REQUIRE((result.index(i, j) % 2) != 0);
        }
    }
}

template <typename Index, typename Queries, typename GroundTruth>
void test_flat(Index& index, const Queries& queries, const GroundTruth& groundtruth) {
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
        index.set_num_threads(num_threads);
        CATCH_REQUIRE((index.get_num_threads() == num_threads));
        svs::index::search_batch_into(index, result.view(), queries.cview());
        // index.search(queries.cview(), groundtruth.dimensions(), result.view());
        CATCH_REQUIRE(svs::k_recall_at_n(groundtruth, result) > expected_recall);
    }

    // Set different data and query batch sizes.
    index.set_num_threads(2);
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
CATCH_TEST_CASE("Flat Index Search", "[integration][exhaustive]") {
    auto queries = test_dataset::queries();
    auto data = svs::load_data<float>(test_dataset::data_svs_file());

    std::cout << "Data size: (" << data.size() << ", " << data.dimensions() << ")"
              << std::endl;

    CATCH_SECTION("Flat Index - L2") {
        auto groundtruth = test_dataset::groundtruth_euclidean();
        // test the temporary index.
        {
            auto threadpool = svs::threads::NativeThreadPool(4);
            auto temp = svs::index::flat::temporary_flat_index(
                data, svs::distance::DistanceL2(), threadpool
            );
            test_flat(temp, queries, groundtruth);
        }

        auto index =
            svs::index::flat::FlatIndex(std::move(data), svs::distance::DistanceL2{}, 1);
        test_flat(index, queries, groundtruth);
    }

    CATCH_SECTION("Flat Index - IP") {
        auto groundtruth = test_dataset::groundtruth_mip();
        auto index =
            svs::index::flat::FlatIndex(std::move(data), svs::distance::DistanceIP{}, 1);
        test_flat(index, queries, groundtruth);
    }

    CATCH_SECTION("Flat Index - Cosine") {
        auto groundtruth = test_dataset::groundtruth_cosine();
        auto index = svs::index::flat::FlatIndex(
            std::move(data), svs::distance::DistanceCosineSimilarity{}, 1
        );
        test_flat(index, queries, groundtruth);
    }

    CATCH_SECTION("Flat Index - Stateful") {
        auto groundtruth = test_dataset::groundtruth_euclidean();
        auto index =
            svs::index::flat::FlatIndex{std::move(data), svs_test::StatefulL2<float>{}, 1};
        test_flat(index, queries, groundtruth);
    }
}

/////
///// Flat
/////

CATCH_TEST_CASE("Flat Orchestrator Search", "[integration][exhaustive]") {
    auto queries = test_dataset::queries();

    // Load data using both the file path method and from a direct file.
    // Use the `DefaultAllocator` to allow implicit copies.
    auto data = svs::load_data<float>(test_dataset::data_svs_file());

    CATCH_SECTION("Euclidean") {
        // From file
        svs::Flat index = svs::Flat::assemble<float>(
            svs::VectorDataLoader<float>(test_dataset::data_svs_file()), svs::L2, 2
        );
        CATCH_REQUIRE(index.get_num_threads() == 2);
        test_flat(index, queries, test_dataset::groundtruth_euclidean());

        // From Data
        index = svs::Flat::assemble<float>(std::move(data), svs::L2, 2);
        CATCH_REQUIRE(index.get_num_threads() == 2);
        test_flat(index, queries, test_dataset::groundtruth_euclidean());
    }

    CATCH_SECTION("InnerProduct") {
        // From file
        svs::Flat index = svs::Flat::assemble<float>(
            svs::VectorDataLoader<float>(test_dataset::data_svs_file()), svs::MIP, 2
        );
        CATCH_REQUIRE(index.get_num_threads() == 2);
        test_flat(index, queries, test_dataset::groundtruth_mip());

        // From Data
        index = svs::Flat::assemble<float>(std::move(data), svs::MIP, 2);
        CATCH_REQUIRE(index.get_num_threads() == 2);
        test_flat(index, queries, test_dataset::groundtruth_mip());
    }

    CATCH_SECTION("Cosine") {
        // From file
        svs::Flat index = svs::Flat::assemble<float>(
            svs::VectorDataLoader<float>(test_dataset::data_svs_file()), svs::Cosine, 2
        );
        CATCH_REQUIRE(index.get_num_threads() == 2);
        test_flat(index, queries, test_dataset::groundtruth_cosine());

        // From Data
        index = svs::Flat::assemble<float>(std::move(data), svs::Cosine, 2);
        CATCH_REQUIRE(index.get_num_threads() == 2);
        test_flat(index, queries, test_dataset::groundtruth_cosine());
    }
}
