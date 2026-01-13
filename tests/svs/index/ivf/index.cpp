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

// header under test
#include "svs/index/ivf/index.h"

// tests
#include "tests/utils/test_dataset.h"

// catch
#include "catch2/catch_test_macros.hpp"

// svs
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/index/ivf/clustering.h"
#include "svs/index/ivf/hierarchical_kmeans.h"

// stl
#include <numeric>

CATCH_TEST_CASE("IVF Index Single Search", "[ivf][index][single_search]") {
    namespace ivf = svs::index::ivf;

    // Load test data
    auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());
    auto queries = test_dataset::queries();

    size_t num_clusters = 10;
    size_t num_threads = 2;
    size_t num_inner_threads = 2;
    auto distance = svs::distance::DistanceL2();

    // Build clustering
    auto build_params = ivf::IVFBuildParameters(num_clusters, 10, false);
    auto threadpool = svs::threads::SequentialThreadPool();
    auto clustering = ivf::build_clustering<svs::BFloat16>(
        build_params, data, distance, threadpool, false
    );

    // Create clustered dataset
    auto centroids = clustering.centroids();
    using Idx = uint32_t;
    auto cluster = ivf::DenseClusteredDataset<decltype(centroids), Idx, decltype(data)>(
        clustering, data, threadpool, svs::lib::Allocator<std::byte>()
    );

    // Build IVF index
    auto threadpool_for_index = svs::threads::as_threadpool(num_threads);
    using IndexType = ivf::IVFIndex<
        decltype(centroids),
        decltype(cluster),
        decltype(distance),
        decltype(threadpool_for_index)>;

    auto index = IndexType(
        std::move(centroids),
        std::move(cluster),
        distance,
        std::move(threadpool_for_index),
        num_inner_threads
    );

    CATCH_SECTION("Test scratchspace creation") {
        // Test scratchspace with custom parameters
        auto search_params = ivf::IVFSearchParameters();
        search_params.n_probes_ = 5;
        search_params.k_reorder_ = 100;

        auto scratch = index.scratchspace(search_params);

        // Verify scratchspace has correct structure
        CATCH_REQUIRE(scratch.buffer_centroids.capacity() == search_params.n_probes_);
        CATCH_REQUIRE(scratch.buffer_leaves.size() == num_inner_threads);

        // Test default scratchspace
        auto default_scratch = index.scratchspace();
        CATCH_REQUIRE(default_scratch.buffer_leaves.size() == num_inner_threads);
    }

    CATCH_SECTION("Test single query search") {
        size_t num_neighbors = 10;

        // Create scratchspace
        auto search_params = ivf::IVFSearchParameters();
        search_params.n_probes_ = 5;
        search_params.k_reorder_ = 100;
        auto scratch = index.scratchspace(search_params);

        // Perform single search
        auto query = queries.get_datum(0);
        index.search(query, scratch);

        // Verify results
        auto& results_buffer = scratch.buffer_leaves[0];
        CATCH_REQUIRE(results_buffer.size() > 0);
        CATCH_REQUIRE(results_buffer.size() >= num_neighbors);

        // Results should be sorted by distance
        results_buffer.sort();
        for (size_t i = 1; i < results_buffer.size(); ++i) {
            CATCH_REQUIRE(results_buffer[i].distance() >= results_buffer[i - 1].distance());
        }
    }

    CATCH_SECTION("Test scratchspace reusability") {
        auto search_params = ivf::IVFSearchParameters();
        search_params.n_probes_ = 5;
        search_params.k_reorder_ = 100;
        auto scratch = index.scratchspace(search_params);

        // Search with multiple queries using same scratchspace
        for (size_t i = 0; i < std::min<size_t>(5, queries.size()); ++i) {
            auto query = queries.get_datum(i);
            index.search(query, scratch);

            // Verify each search produces results
            CATCH_REQUIRE(scratch.buffer_leaves[0].size() > 0);
        }
    }

    CATCH_SECTION("Compare single search with batch search") {
        size_t num_neighbors = 10;

        auto search_params = ivf::IVFSearchParameters();
        search_params.n_probes_ = 5;
        search_params.k_reorder_ = 100;

        // Single search
        auto scratch = index.scratchspace(search_params);
        auto query = queries.get_datum(0);
        index.search(query, scratch);

        // Extract results from single search (already sorted and ID-converted)
        auto& single_results = scratch.buffer_leaves[0];
        std::vector<size_t> single_ids;
        for (size_t i = 0; i < num_neighbors && i < single_results.size(); ++i) {
            single_ids.push_back(single_results[i].id());
        }

        // Batch search
        auto batch_queries =
            svs::data::ConstSimpleDataView<float>(queries.data(), 1, queries.dimensions());
        auto batch_results = svs::QueryResult<size_t>(1, num_neighbors);
        index.search(batch_results.view(), batch_queries, search_params);

        // Extract results from batch search
        std::vector<size_t> batch_ids;
        for (size_t i = 0; i < num_neighbors; ++i) {
            batch_ids.push_back(batch_results.index(0, i));
        }

        // Results should match
        CATCH_REQUIRE(single_ids.size() == batch_ids.size());
        for (size_t i = 0; i < num_neighbors; ++i) {
            CATCH_REQUIRE(single_ids[i] == batch_ids[i]);
        }
    }
}
