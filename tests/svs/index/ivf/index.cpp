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
#include "tests/utils/utils.h"

// catch
#include "catch2/catch_test_macros.hpp"

// svs
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/index/ivf/clustering.h"
#include "svs/index/ivf/hierarchical_kmeans.h"
#include "svs/lib/saveload.h"

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

CATCH_TEST_CASE("IVF Index Save and Load", "[ivf][index][saveload]") {
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
    auto clustering =
        ivf::build_clustering<float>(build_params, data, distance, threadpool, false);

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

    // Get search results before saving
    auto search_params = ivf::IVFSearchParameters();
    search_params.n_probes_ = 5;
    search_params.k_reorder_ = 100;
    size_t num_neighbors = 10;

    auto batch_queries = svs::data::ConstSimpleDataView<float>(
        queries.data(), queries.size(), queries.dimensions()
    );
    auto original_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
    index.search(original_results.view(), batch_queries, search_params);

    CATCH_SECTION("Save and load IVF index") {
        // Prepare temp directory
        auto tempdir = svs_test::prepare_temp_directory_v2();
        auto config_dir = tempdir / "config";
        auto data_dir = tempdir / "data";

        // Save the index
        index.save(config_dir, data_dir);

        // Verify files exist
        CATCH_REQUIRE(std::filesystem::exists(config_dir));
        CATCH_REQUIRE(std::filesystem::exists(data_dir / "centroids"));
        CATCH_REQUIRE(std::filesystem::exists(data_dir / "clusters"));

        // Load the index
        auto loaded_index = ivf::load_ivf_index<float, float>(
            config_dir,
            data_dir,
            distance,
            svs::threads::as_threadpool(num_threads),
            num_inner_threads
        );

        // Verify index properties
        CATCH_REQUIRE(loaded_index.size() == index.size());
        CATCH_REQUIRE(loaded_index.num_clusters() == index.num_clusters());
        CATCH_REQUIRE(loaded_index.dimensions() == index.dimensions());

        // Search with loaded index
        auto loaded_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        loaded_index.search(loaded_results.view(), batch_queries, search_params);

        // Compare results - should be identical
        for (size_t q = 0; q < queries.size(); ++q) {
            for (size_t i = 0; i < num_neighbors; ++i) {
                CATCH_REQUIRE(loaded_results.index(q, i) == original_results.index(q, i));
                CATCH_REQUIRE(
                    loaded_results.distance(q, i) ==
                    Catch::Approx(original_results.distance(q, i)).epsilon(1e-5)
                );
            }
        }

        // Cleanup
        svs_test::cleanup_temp_directory();
    }

    CATCH_SECTION("Save and load DenseClusteredDataset") {
        // Prepare temp directory
        auto tempdir = svs_test::prepare_temp_directory_v2();

        // Re-create clustering and dense clusters for this section
        auto section_clustering =
            ivf::build_clustering<float>(build_params, data, distance, threadpool, false);
        auto section_centroids = section_clustering.centroids();
        auto dense_clusters =
            ivf::DenseClusteredDataset<decltype(section_centroids), Idx, decltype(data)>(
                section_clustering, data, threadpool, svs::lib::Allocator<std::byte>()
            );

        // Save the dense clusters
        svs::lib::save_to_disk(dense_clusters, tempdir);

        // Verify file exists
        CATCH_REQUIRE(std::filesystem::exists(tempdir / "svs_config.toml"));

        // Load the dense clusters
        auto loaded_clusters = svs::lib::load_from_disk<
            ivf::DenseClusteredDataset<decltype(section_centroids), Idx, decltype(data)>>(
            tempdir, threadpool
        );

        // Verify properties
        CATCH_REQUIRE(loaded_clusters.size() == dense_clusters.size());
        CATCH_REQUIRE(loaded_clusters.dimensions() == dense_clusters.dimensions());
        CATCH_REQUIRE(
            loaded_clusters.get_prefetch_offset() == dense_clusters.get_prefetch_offset()
        );

        // Verify cluster contents
        for (size_t c = 0; c < dense_clusters.size(); ++c) {
            auto& orig_cluster = dense_clusters[c];
            auto& loaded_cluster = loaded_clusters[c];

            CATCH_REQUIRE(orig_cluster.size() == loaded_cluster.size());

            // Verify data and IDs match
            for (size_t i = 0; i < orig_cluster.size(); ++i) {
                CATCH_REQUIRE(orig_cluster.ids_[i] == loaded_cluster.ids_[i]);

                // Verify data values
                auto orig_datum = orig_cluster.get_datum(i);
                auto loaded_datum = loaded_cluster.get_datum(i);
                for (size_t d = 0; d < data.dimensions(); ++d) {
                    CATCH_REQUIRE(
                        orig_datum[d] == Catch::Approx(loaded_datum[d]).epsilon(1e-6)
                    );
                }
            }
        }

        // Cleanup
        svs_test::cleanup_temp_directory();
    }
}
