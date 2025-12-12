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
#include "svs/index/ivf/dynamic_ivf.h"
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/core/query_result.h"
#include "svs/core/recall.h"
#include "svs/index/ivf/clustering.h"
#include "svs/lib/preprocessor.h"
#include "svs/lib/threads.h"
#include "svs/lib/timing.h"
#include "svs/misc/dynamic_helper.h"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

// catch
#include "catch2/catch_test_macros.hpp"

// stl
#include <algorithm>
#include <cmath>
#include <concepts>
#include <random>
#include <sstream>

using Idx = uint32_t;
using Eltype = float;
using QueryEltype = float;
using Distance = svs::distance::DistanceL2;
const size_t N = 128;

const size_t NUM_NEIGHBORS = 10;
const size_t NUM_CLUSTERS = 10;

///
/// Utility Methods
///

template <std::integral I> I div(I i, float fraction) {
    return svs::lib::narrow<I>(std::floor(svs::lib::narrow<float>(i) * fraction));
}

template <typename... Args> std::string stringify(Args&&... args) {
    std::ostringstream stream{};
    ((stream << args), ...);
    return stream.str();
}

///
/// Main Loop.
///

template <typename MutableIndex, typename Queries>
void do_check(
    MutableIndex& index,
    svs::misc::ReferenceDataset<Idx, Eltype, N, Distance>& reference,
    const Queries& queries,
    double operation_time,
    std::string message
) {
    // Compute groundtruth
    auto tic = svs::lib::now();
    auto gt = reference.groundtruth();
    CATCH_REQUIRE(gt.n_neighbors() == NUM_NEIGHBORS);
    CATCH_REQUIRE(gt.n_queries() == queries.size());

    double groundtruth_time = svs::lib::time_difference(tic);

    // Run search
    tic = svs::lib::now();
    auto results = svs::QueryResult<size_t>(gt.n_queries(), NUM_NEIGHBORS);
    auto search_parameters = svs::index::ivf::IVFSearchParameters(
        NUM_CLUSTERS, // n_probes - search all clusters for accuracy
        NUM_NEIGHBORS // k_reorder
    );

    index.search(
        results.view(),
        svs::data::ConstSimpleDataView<QueryEltype>{
            queries.data(), queries.size(), queries.dimensions()},
        search_parameters
    );
    double search_time = svs::lib::time_difference(tic);

    // Extra ID checks
    reference.check_ids(results);
    reference.check_equal_ids(index);

    // compute recall
    double recall = svs::k_recall_at_n(gt, results, NUM_NEIGHBORS, NUM_NEIGHBORS);

    std::cout << "[" << message << "] -- {"
              << "operation: " << operation_time << ", groundtruth: " << groundtruth_time
              << ", search: " << search_time << ", recall: " << recall << "}\n";
}

template <typename MutableIndex, typename Queries>
void test_loop(
    MutableIndex& index,
    svs::misc::ReferenceDataset<Idx, Eltype, N, Distance>& reference,
    const Queries& queries,
    size_t num_points,
    size_t consolidate_every,
    size_t iterations
) {
    size_t consolidate_count = 0;
    for (size_t i = 0; i < iterations; ++i) {
        // Add Points
        {
            auto [points, time] = reference.add_points(index, num_points);
            CATCH_REQUIRE(points <= num_points);
            CATCH_REQUIRE(points > num_points - reference.bucket_size());
            do_check(index, reference, queries, time, stringify("add ", points, " points"));
        }

        // Delete Points
        {
            auto [points, time] = reference.delete_points(index, num_points);
            CATCH_REQUIRE(points <= num_points);
            CATCH_REQUIRE(points > num_points - reference.bucket_size());
            do_check(
                index, reference, queries, time, stringify("delete ", points, " points")
            );
        }

        // Maybe compact.
        ++consolidate_count;
        if (consolidate_count == consolidate_every) {
            auto tic = svs::lib::now();
            // Use a batchsize smaller than the whole dataset to ensure that the compaction
            // algorithm correctly handles this case.
            index.compact(reference.valid() / 10);
            double diff = svs::lib::time_difference(tic);
            do_check(index, reference, queries, diff, "compact");
            consolidate_count = 0;
        }
    }
}

CATCH_TEST_CASE("Testing Dynamic IVF Index", "[dynamic_ivf]") {
#if defined(NDEBUG)
    const float initial_fraction = 0.25;
    const float modify_fraction = 0.05;
#else
    const float initial_fraction = 0.05;
    const float modify_fraction = 0.005;
#endif
    const size_t num_threads = 10;

    // Load the base dataset and queries.
    auto data = svs::data::SimpleData<Eltype, N>::load(test_dataset::data_svs_file());
    auto num_points = data.size();
    auto queries = test_dataset::queries();

    auto reference = svs::misc::ReferenceDataset<Idx, Eltype, N, Distance>(
        std::move(data),
        Distance(),
        num_threads,
        div(num_points, 0.5 * modify_fraction),
        NUM_NEIGHBORS,
        queries,
        0x12345678
    );

    auto num_indices_to_add = div(reference.size(), initial_fraction);

    // Generate initial vectors and indices
    std::vector<Idx> initial_indices{};
    auto initial_data = svs::data::SimpleData<Eltype, N>(num_indices_to_add, N);
    {
        auto [vectors, indices] = reference.generate(num_indices_to_add);
        auto num_points_added = indices.size();
        CATCH_REQUIRE(vectors.size() == num_points_added);
        CATCH_REQUIRE(num_points_added <= num_indices_to_add);
        CATCH_REQUIRE(num_points_added > num_indices_to_add - reference.bucket_size());

        initial_indices = indices;
        if (vectors.size() != num_indices_to_add || indices.size() != num_indices_to_add) {
            throw ANNEXCEPTION("Something went horribly wrong!");
        }

        for (size_t i = 0; i < num_indices_to_add; ++i) {
            initial_data.set_datum(i, vectors.get_datum(i));
        }
    }

    // Build IVF clustering
    auto build_params = svs::index::ivf::IVFBuildParameters(
        NUM_CLUSTERS,
        /* max_iters */ 10,
        /* is_hierarchical */ false
    );

    auto threadpool = svs::threads::SequentialThreadPool();
    auto clustering = svs::index::ivf::build_clustering<Eltype>(
        build_params,
        svs::lib::Lazy([&initial_data]() { return initial_data; }),
        Distance(),
        threadpool,
        /* train_only */ false
    );

    // Create dynamic clusters using DenseClusteredDataset
    auto centroids = clustering.centroids();
    using DataType = svs::data::SimpleData<Eltype, N>;
    auto dense_clusters = svs::index::ivf::DenseClusteredDataset<
        decltype(centroids),
        Idx,
        DataType>(clustering, initial_data, threadpool, svs::lib::Allocator<std::byte>());

    // Create the dynamic IVF index
    auto threadpool_for_index = svs::threads::as_threadpool(num_threads);
    using IndexType = svs::index::ivf::DynamicIVFIndex<
        decltype(centroids),
        decltype(dense_clusters),
        Distance,
        decltype(threadpool_for_index)>;

    auto index = IndexType(
        std::move(centroids),
        std::move(dense_clusters),
        initial_indices,
        Distance(),
        std::move(threadpool_for_index),
        1 // intra_query_threads
    );

    reference.configure_extra_checks(true);
    CATCH_REQUIRE(reference.extra_checks_enabled());

    test_loop(index, reference, queries, div(reference.size(), modify_fraction), 2, 6);
}

CATCH_TEST_CASE("Testing Dynamic IVF Index with BlockedData", "[dynamic_ivf]") {
    // This test verifies that BlockedData allocator works correctly for dynamic operations
    const size_t num_threads = 4;

    // Load data
    auto data = svs::data::SimpleData<Eltype, N>::load(test_dataset::data_svs_file());
    auto queries = test_dataset::queries();

    // Build clustering
    auto build_params = svs::index::ivf::IVFBuildParameters(10, 10, false);
    auto threadpool = svs::threads::SequentialThreadPool();
    auto clustering = svs::index::ivf::build_clustering<Eltype>(
        build_params,
        data,
        Distance(),
        threadpool,
        false
    );

    // Use build_dynamic_ivf which automatically creates BlockedData clusters
    std::vector<size_t> ids(data.size());
    std::iota(ids.begin(), ids.end(), 0);
    
    auto index = svs::index::ivf::build_dynamic_ivf(
        std::move(clustering.centroids_),
        clustering,
        data,
        ids,
        Distance(),
        svs::threads::as_threadpool(num_threads),
        1
    );

    // Test 1: Initial search works
    auto params = svs::index::ivf::IVFSearchParameters(10, NUM_NEIGHBORS);
    auto results = svs::QueryResult<size_t>(queries.size(), NUM_NEIGHBORS);
    
    index.search(
        results.view(),
        svs::data::ConstSimpleDataView<float>{
            queries.data(), queries.size(), queries.dimensions()},
        params
    );
    
    // Verify we got results
    size_t valid_results = 0;
    for (size_t i = 0; i < results.n_queries(); ++i) {
        if (results.index(i, 0) != std::numeric_limits<size_t>::max()) {
            valid_results++;
        }
    }
    CATCH_REQUIRE(valid_results > 0);
    
    // Test 2: Add points (BlockedData's resize capability)
    constexpr size_t num_add = 100;
    std::vector<size_t> new_ids;
    auto new_data = svs::data::SimpleData<Eltype, N>(num_add, N);
    for (size_t i = 0; i < num_add; ++i) {
        new_ids.push_back(data.size() + i);
        new_data.set_datum(i, data.get_datum(i % data.size()));
    }
    
    size_t size_before = index.size();
    index.add_points(new_data, new_ids, false);
    CATCH_REQUIRE(index.size() == size_before + num_add);
    
    // Test 3: Search still works after adding
    index.search(
        results.view(),
        svs::data::ConstSimpleDataView<float>{
            queries.data(), queries.size(), queries.dimensions()},
        params
    );
    
    valid_results = 0;
    for (size_t i = 0; i < results.n_queries(); ++i) {
        if (results.index(i, 0) != std::numeric_limits<size_t>::max()) {
            valid_results++;
        }
    }
    CATCH_REQUIRE(valid_results > 0);
    
    // Test 4: Delete some points
    std::vector<size_t> to_delete;
    for (size_t i = 0; i < 50; ++i) {
        to_delete.push_back(i);
    }
    size_t deleted = index.delete_entries(to_delete);
    CATCH_REQUIRE(deleted == to_delete.size());
    CATCH_REQUIRE(index.size() == size_before + num_add - deleted);
    
    // Test 5: Compact works with BlockedData
    index.compact(1000);
    CATCH_REQUIRE(index.size() == size_before + num_add - deleted);
    
    // Test 6: Search after compaction
    index.search(
        results.view(),
        svs::data::ConstSimpleDataView<float>{
            queries.data(), queries.size(), queries.dimensions()},
        params
    );
    
    valid_results = 0;
    for (size_t i = 0; i < results.n_queries(); ++i) {
        if (results.index(i, 0) != std::numeric_limits<size_t>::max()) {
            valid_results++;
        }
    }
    CATCH_REQUIRE(valid_results > 0);
}

CATCH_TEST_CASE("Dynamic IVF - Edge Cases", "[dynamic_ivf]") {
    const size_t num_threads = 4;
    const size_t num_points = 100;

    // Create a small dataset
    auto data = svs::data::SimpleData<Eltype, N>(num_points, N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < num_points; ++i) {
        std::vector<float> vec(N);
        for (size_t j = 0; j < N; ++j) {
            vec[j] = dist(rng);
        }
        data.set_datum(i, vec);
    }

    // Build clustering with more clusters than points to test empty clusters
    // With the fix, this should now work by using all 100 datapoints for training
    auto build_params = svs::index::ivf::IVFBuildParameters(
        50,   // More clusters than 10% of data (which would be 10 points)
        10,   // max_iters
        false // is_hierarchical
    );

    auto threadpool = svs::threads::SequentialThreadPool();
    auto clustering = svs::index::ivf::build_clustering<Eltype>(
        build_params,
        svs::lib::Lazy([&data]() { return data; }),
        Distance(),
        threadpool,
        false
    );

    // Create dynamic clusters using DenseClusteredDataset
    std::vector<Idx> initial_indices;
    for (size_t c = 0; c < clustering.size(); ++c) {
        for (auto idx : clustering.cluster(c)) {
            initial_indices.push_back(idx);
        }
    }

    auto centroids = clustering.centroids();
    using DataType = svs::data::SimpleData<Eltype, N>;
    auto dense_clusters = svs::index::ivf::DenseClusteredDataset<
        decltype(centroids),
        Idx,
        DataType>(clustering, data, threadpool, svs::lib::Allocator<std::byte>());

    auto threadpool_for_index = svs::threads::as_threadpool(num_threads);
    using IndexType = svs::index::ivf::DynamicIVFIndex<
        decltype(centroids),
        decltype(dense_clusters),
        Distance,
        decltype(threadpool_for_index)>;

    auto index = IndexType(
        std::move(centroids),
        std::move(dense_clusters),
        initial_indices,
        Distance(),
        std::move(threadpool_for_index),
        1
    );

    // Test 1: Search with sparse/empty clusters (should not crash)
    auto query = svs::data::SimpleData<QueryEltype, N>(1, N);
    std::vector<float> query_vec(N);
    for (size_t j = 0; j < N; ++j) {
        query_vec[j] = dist(rng);
    }
    query.set_datum(0, query_vec);

    auto results = svs::QueryResult<size_t>(1, NUM_NEIGHBORS);
    auto search_params = svs::index::ivf::IVFSearchParameters(50, NUM_NEIGHBORS);

    index.search(
        results.view(),
        svs::data::ConstSimpleDataView<QueryEltype>{query.data(), 1, N},
        search_params
    );

    // Verify results are valid (not all max values)
    bool found_valid = false;
    for (size_t i = 0; i < NUM_NEIGHBORS; ++i) {
        if (results.index(0, i) != std::numeric_limits<size_t>::max()) {
            found_valid = true;
            break;
        }
    }
    CATCH_REQUIRE(found_valid);

    // Test 2: Delete and compact
    std::vector<Idx> to_delete;
    for (size_t i = 0; i < 20 && i < initial_indices.size(); ++i) {
        to_delete.push_back(initial_indices[i]);
    }

    index.delete_entries(to_delete);

    index.compact(10);

    // Search after compaction
    index.search(
        results.view(),
        svs::data::ConstSimpleDataView<QueryEltype>{query.data(), 1, N},
        search_params
    );

    CATCH_REQUIRE(results.index(0, 0) != std::numeric_limits<size_t>::max());
}

CATCH_TEST_CASE("Dynamic IVF - Search Parameters Variations", "[dynamic_ivf]") {
    const size_t num_threads = 4;
    auto data = svs::data::SimpleData<Eltype, N>::load(test_dataset::data_svs_file());
    auto queries = test_dataset::queries();

    // Build with standard parameters
    auto build_params = svs::index::ivf::IVFBuildParameters(NUM_CLUSTERS, 10, false);
    auto threadpool = svs::threads::SequentialThreadPool();
    auto clustering = svs::index::ivf::build_clustering<Eltype>(
        build_params,
        svs::lib::Lazy([&data]() { return data; }),
        Distance(),
        threadpool,
        false
    );

    // Create dynamic clusters using DenseClusteredDataset
    std::vector<Idx> indices;
    for (size_t c = 0; c < clustering.size(); ++c) {
        for (auto idx : clustering.cluster(c)) {
            indices.push_back(idx);
        }
    }

    auto centroids = clustering.centroids();
    using DataType = svs::data::SimpleData<Eltype, N>;
    auto dense_clusters = svs::index::ivf::DenseClusteredDataset<
        decltype(centroids),
        Idx,
        DataType>(clustering, data, threadpool, svs::lib::Allocator<std::byte>());

    auto threadpool_for_index = svs::threads::as_threadpool(num_threads);
    using IndexType = svs::index::ivf::DynamicIVFIndex<
        decltype(centroids),
        decltype(dense_clusters),
        Distance,
        decltype(threadpool_for_index)>;

    auto index = IndexType(
        std::move(centroids),
        std::move(dense_clusters),
        indices,
        Distance(),
        std::move(threadpool_for_index),
        1
    );

    auto results = svs::QueryResult<size_t>(queries.size(), NUM_NEIGHBORS);

    // Test with different n_probes values
    std::vector<size_t> probe_counts = {1, 3, 5, NUM_CLUSTERS};
    std::vector<double> recalls;

    for (auto n_probes : probe_counts) {
        auto params = svs::index::ivf::IVFSearchParameters(n_probes, NUM_NEIGHBORS);
        index.search(
            results.view(),
            svs::data::ConstSimpleDataView<QueryEltype>{
                queries.data(), queries.size(), queries.dimensions()},
            params
        );

        // Verify all results are valid
        for (size_t i = 0; i < queries.size(); ++i) {
            for (size_t j = 0; j < NUM_NEIGHBORS; ++j) {
                auto idx = results.index(i, j);
                CATCH_REQUIRE(
                    (idx < data.size() || idx == std::numeric_limits<size_t>::max())
                );
            }
        }
    }
}

CATCH_TEST_CASE("Dynamic IVF - Threading Configurations", "[dynamic_ivf]") {
    auto data = svs::data::SimpleData<Eltype, N>::load(test_dataset::data_svs_file());
    auto queries = test_dataset::queries();

    auto build_params = svs::index::ivf::IVFBuildParameters(NUM_CLUSTERS, 10, false);
    auto threadpool = svs::threads::SequentialThreadPool();
    auto clustering = svs::index::ivf::build_clustering<Eltype>(
        build_params,
        svs::lib::Lazy([&data]() { return data; }),
        Distance(),
        threadpool,
        false
    );


    // Test with different thread configurations
    std::vector<size_t> thread_configs = {1, 2, 4, 8};
    std::vector<size_t> intra_query_configs = {1, 2};

    for (auto num_threads : thread_configs) {
        for (auto intra_threads : intra_query_configs) {
            std::vector<Idx> indices;
            for (size_t c = 0; c < clustering.size(); ++c) {
                for (auto idx : clustering.cluster(c)) {
                    indices.push_back(idx);
                }
            }

            auto centroids_copy = clustering.centroids();
            using DataType = svs::data::SimpleData<Eltype, N>;
            auto dense_clusters = svs::index::ivf::DenseClusteredDataset<
                decltype(centroids_copy),
                Idx,
                DataType>(clustering, data, threadpool, svs::lib::Allocator<std::byte>());

            auto threadpool_for_index = svs::threads::as_threadpool(num_threads);
            using IndexType = svs::index::ivf::DynamicIVFIndex<
                decltype(centroids_copy),
                decltype(dense_clusters),
                Distance,
                decltype(threadpool_for_index)>;

            auto index = IndexType(
                std::move(centroids_copy),
                std::move(dense_clusters),
                indices,
                Distance(),
                std::move(threadpool_for_index),
                intra_threads
            );

            auto results = svs::QueryResult<size_t>(queries.size(), NUM_NEIGHBORS);
            auto params = svs::index::ivf::IVFSearchParameters(NUM_CLUSTERS, NUM_NEIGHBORS);

            index.search(
                results.view(),
                svs::data::ConstSimpleDataView<QueryEltype>{
                    queries.data(), queries.size(), queries.dimensions()},
                params
            );

            // Verify results are consistent
            for (size_t i = 0; i < queries.size(); ++i) {
                for (size_t j = 0; j < NUM_NEIGHBORS; ++j) {
                    auto idx = results.index(i, j);
                    CATCH_REQUIRE(
                        (idx < data.size() || idx == std::numeric_limits<size_t>::max())
                    );
                }
            }
        }
    }
}

CATCH_TEST_CASE("Dynamic IVF - Add/Delete Stress Test", "[dynamic_ivf]") {
    const size_t num_threads = 4;
    auto data = svs::data::SimpleData<Eltype, N>::load(test_dataset::data_svs_file());
    auto queries = test_dataset::queries();

    auto build_params = svs::index::ivf::IVFBuildParameters(NUM_CLUSTERS, 10, false);
    auto threadpool = svs::threads::SequentialThreadPool();

    // Start with half the data
    size_t initial_size = data.size() / 2;
    auto initial_data = svs::data::SimpleData<Eltype, N>(initial_size, N);
    for (size_t i = 0; i < initial_size; ++i) {
        initial_data.set_datum(i, data.get_datum(i));
    }

    auto clustering = svs::index::ivf::build_clustering<Eltype>(
        build_params,
        svs::lib::Lazy([&initial_data]() { return initial_data; }),
        Distance(),
        threadpool,
        false
    );

    // Create dynamic clusters using DenseClusteredDataset
    std::vector<Idx> indices;
    for (size_t c = 0; c < clustering.size(); ++c) {
        for (auto idx : clustering.cluster(c)) {
            indices.push_back(idx);
        }
    }

    auto centroids = clustering.centroids();
    using DataType = svs::data::SimpleData<Eltype, N>;
    auto dense_clusters = svs::index::ivf::DenseClusteredDataset<
        decltype(centroids),
        Idx,
        DataType>(clustering, initial_data, threadpool, svs::lib::Allocator<std::byte>());

    auto threadpool_for_index = svs::threads::as_threadpool(num_threads);
    using IndexType = svs::index::ivf::DynamicIVFIndex<
        decltype(centroids),
        decltype(dense_clusters),
        Distance,
        decltype(threadpool_for_index)>;

    auto index = IndexType(
        std::move(centroids),
        std::move(dense_clusters),
        indices,
        Distance(),
        std::move(threadpool_for_index),
        1
    );

    auto results = svs::QueryResult<size_t>(queries.size(), NUM_NEIGHBORS);
    auto params = svs::index::ivf::IVFSearchParameters(NUM_CLUSTERS, NUM_NEIGHBORS);

    // Test: Rapid add/delete cycles
    std::mt19937 rng(12345);
    std::uniform_int_distribution<size_t> idx_dist(0, indices.size() - 1);

    for (size_t cycle = 0; cycle < 5; ++cycle) {
        // Delete random entries
        std::vector<Idx> deleted;
        for (size_t i = 0; i < 10 && i < indices.size(); ++i) {
            size_t idx = idx_dist(rng) % indices.size();
            deleted.push_back(indices[idx]);
        }
        if (!deleted.empty()) {
            index.delete_entries(deleted);
        }

        // Search after deletion
        index.search(
            results.view(),
            svs::data::ConstSimpleDataView<QueryEltype>{
                queries.data(), queries.size(), queries.dimensions()},
            params
        );

        // Verify deleted IDs don't appear in results
        for (size_t q = 0; q < queries.size(); ++q) {
            for (size_t k = 0; k < NUM_NEIGHBORS; ++k) {
                auto result_id = results.index(q, k);
                for (auto deleted_id : deleted) {
                    CATCH_REQUIRE(result_id != deleted_id);
                }
            }
        }

        // Add new entries
        std::vector<Idx> new_ids;
        auto new_data = svs::data::SimpleData<Eltype, N>(10, N);
        Idx new_base_id = 10000 + cycle * 100;
        for (size_t i = 0; i < 10; ++i) {
            new_ids.push_back(new_base_id + i);
            new_data.set_datum(i, data.get_datum(i % data.size()));
        }
        index.add_points(new_data, new_ids, false);

        // Search after addition
        index.search(
            results.view(),
            svs::data::ConstSimpleDataView<QueryEltype>{
                queries.data(), queries.size(), queries.dimensions()},
            params
        );

        // All results should be valid
        for (size_t q = 0; q < queries.size(); ++q) {
            CATCH_REQUIRE(results.index(q, 0) != std::numeric_limits<size_t>::max());
        }

        // Compact periodically
        if (cycle % 2 == 1) {
            index.compact(50);
        }
    }
}

CATCH_TEST_CASE("Dynamic IVF - Single Query Search", "[dynamic_ivf]") {
    const size_t num_threads = 2;
    auto data = svs::data::SimpleData<Eltype, N>::load(test_dataset::data_svs_file());
    auto queries = test_dataset::queries();

    auto build_params = svs::index::ivf::IVFBuildParameters(NUM_CLUSTERS, 10, false);
    auto threadpool = svs::threads::SequentialThreadPool();
    auto clustering = svs::index::ivf::build_clustering<Eltype>(
        build_params,
        svs::lib::Lazy([&data]() { return data; }),
        Distance(),
        threadpool,
        false
    );

    // Create dynamic clusters using DenseClusteredDataset
    std::vector<Idx> indices;
    for (size_t c = 0; c < clustering.size(); ++c) {
        for (auto idx : clustering.cluster(c)) {
            indices.push_back(idx);
        }
    }

    auto centroids = clustering.centroids();
    using DataType = svs::data::SimpleData<Eltype, N>;
    auto dense_clusters = svs::index::ivf::DenseClusteredDataset<
        decltype(centroids),
        Idx,
        DataType>(clustering, data, threadpool, svs::lib::Allocator<std::byte>());

    auto threadpool_for_index = svs::threads::as_threadpool(num_threads);
    using IndexType = svs::index::ivf::DynamicIVFIndex<
        decltype(centroids),
        decltype(dense_clusters),
        Distance,
        decltype(threadpool_for_index)>;

    auto index = IndexType(
        std::move(centroids),
        std::move(dense_clusters),
        indices,
        Distance(),
        std::move(threadpool_for_index),
        1
    );

    // Test single query search
    auto single_query = svs::data::SimpleData<QueryEltype, N>(1, N);
    single_query.set_datum(0, queries.get_datum(0));

    auto results = svs::QueryResult<size_t>(1, NUM_NEIGHBORS);
    auto params = svs::index::ivf::IVFSearchParameters(NUM_CLUSTERS, NUM_NEIGHBORS);

    index.search(
        results.view(),
        svs::data::ConstSimpleDataView<QueryEltype>{single_query.data(), 1, N},
        params
    );

    // Verify we got valid results
    CATCH_REQUIRE(results.index(0, 0) != std::numeric_limits<size_t>::max());

    // Verify distances are in ascending order
    for (size_t k = 1; k < NUM_NEIGHBORS; ++k) {
        if (results.index(0, k) != std::numeric_limits<size_t>::max()) {
            CATCH_REQUIRE(results.distance(0, k) >= results.distance(0, k - 1));
        }
    }
}

CATCH_TEST_CASE("Dynamic IVF Get Distance", "[index][ivf][dynamic_ivf]") {
    const size_t num_threads = 2;
    const size_t num_points = 200;

    // Create test dataset
    auto data = svs::data::SimpleData<Eltype, N>(num_points, N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < num_points; ++i) {
        std::vector<float> vec(N);
        for (size_t j = 0; j < N; ++j) {
            vec[j] = dist(rng);
        }
        data.set_datum(i, vec);
    }

    // Create queries
    const size_t num_queries = 20;
    auto queries = svs::data::SimpleData<QueryEltype, N>(num_queries, N);
    for (size_t i = 0; i < num_queries; ++i) {
        std::vector<float> vec(N);
        for (size_t j = 0; j < N; ++j) {
            vec[j] = dist(rng);
        }
        queries.set_datum(i, vec);
    }

    // Build IVF clustering
    auto build_params = svs::index::ivf::IVFBuildParameters(
        NUM_CLUSTERS,
        /* max_iters */ 10,
        /* is_hierarchical */ false
    );

    auto threadpool = svs::threads::SequentialThreadPool();
    auto clustering = svs::index::ivf::build_clustering<Eltype>(
        build_params,
        svs::lib::Lazy([&data]() { return data; }),
        Distance(),
        threadpool,
        /* train_only */ false
    );

    // Create dynamic clusters using DenseClusteredDataset
    // Note: This test uses sequential internal IDs, so we can't use the simple helper
    std::vector<Idx> initial_indices; // External IDs in order
    size_t internal_id = 0;           // Sequential internal IDs

    // Build mapping: internal_id -> external_id
    for (size_t c = 0; c < clustering.size(); ++c) {
        const auto& cluster_indices = clustering.cluster(c);
        for (size_t i = 0; i < cluster_indices.size(); ++i) {
            Idx external_id = cluster_indices[i]; // Use clustering index as external ID
            initial_indices.push_back(external_id); // Map internal_id -> external_id
            internal_id++;
        }
    }

    auto centroids = clustering.centroids();
    using DataType = svs::data::SimpleData<Eltype, N>;
    auto dense_clusters = svs::index::ivf::DenseClusteredDataset<
        decltype(centroids),
        Idx,
        DataType>(clustering, data, threadpool, svs::lib::Allocator<std::byte>());

    // Need to update cluster IDs to use sequential internal IDs
    for (size_t c = 0, global_idx = 0; c < dense_clusters.size(); ++c) {
        auto& cluster = dense_clusters[c];
        for (size_t i = 0; i < cluster.ids_.size(); ++i) {
            cluster.ids_[i] = global_idx++;
        }
    }

    // Create the dynamic IVF index
    auto threadpool_for_index = svs::threads::as_threadpool(num_threads);
    using IndexType = svs::index::ivf::DynamicIVFIndex<
        decltype(centroids),
        decltype(dense_clusters),
        Distance,
        decltype(threadpool_for_index)>;

    auto index = IndexType(
        std::move(centroids),
        std::move(dense_clusters),
        initial_indices,
        Distance(),
        std::move(threadpool_for_index),
        1 // intra_query_threads
    );

    // Test get_distance functionality using the standard tester
    CATCH_SECTION("Get Distance Test") {
        // Test with strict tolerance to verify correctness
        constexpr double TOLERANCE = 1e-2; // 1% tolerance, same as flat index

        // Test with a few different IDs
        std::vector<size_t> test_ids = {0, 10, 50};
        if (index.size() > 100) {
            test_ids.push_back(100);
        }

        for (size_t test_id : test_ids) {
            if (test_id >= index.size()) {
                continue;
            }

            // Get a query vector
            size_t query_id = std::min<size_t>(5, queries.size() - 1);
            auto query = queries.get_datum(query_id);

            // Get distance from index
            double index_distance = index.get_distance(test_id, query);

            // Compute expected distance from original data
            // test_id is the external ID which maps to data[test_id]
            auto datum = data.get_datum(test_id);
            Distance dist_copy = Distance();
            svs::distance::maybe_fix_argument(dist_copy, query);
            double expected_distance = svs::distance::compute(dist_copy, query, datum);

            // Verify the distance is correct
            double relative_diff =
                std::abs((index_distance - expected_distance) / expected_distance);
            CATCH_REQUIRE(relative_diff < TOLERANCE);
        }

        // Test with out of bounds ID - should throw
        CATCH_REQUIRE_THROWS_AS(
            index.get_distance(index.size() + 1000, queries.get_datum(0)), svs::ANNException
        );
    }

    // Test get_distance after adding and removing points
    CATCH_SECTION("Get Distance After Modifications") {
        // Test with strict tolerance to verify correctness
        constexpr double TOLERANCE = 1e-2; // 1% tolerance, same as flat index

        // Add some new points
        std::vector<Idx> new_ids = {10000, 10001, 10002};

        // Prepare data for batch insertion
        auto new_data = svs::data::SimpleData<Eltype, N>(new_ids.size(), N);
        for (size_t i = 0; i < new_ids.size(); ++i) {
            new_data.set_datum(i, data.get_datum(i));
        }

        // Add points in batch
        index.add_points(new_data, new_ids);

        // Test get_distance for newly added points
        for (size_t i = 0; i < new_ids.size(); ++i) {
            size_t query_id = std::min<size_t>(7, queries.size() - 1);
            auto query = queries.get_datum(query_id);

            double index_distance = index.get_distance(new_ids[i], query);

            // Compute expected distance from the original data we added
            auto datum = data.get_datum(i);
            Distance dist_copy = Distance();
            svs::distance::maybe_fix_argument(dist_copy, query);
            double expected_distance = svs::distance::compute(dist_copy, query, datum);

            double relative_diff =
                std::abs((index_distance - expected_distance) / expected_distance);
            CATCH_REQUIRE(relative_diff < TOLERANCE);
        }

        // Delete a point
        std::vector<Idx> ids_to_delete = {new_ids[0]};
        index.delete_entries(ids_to_delete);

        // Verify the deleted point throws exception
        CATCH_REQUIRE_THROWS_AS(
            index.get_distance(new_ids[0], queries.get_datum(0)), svs::ANNException
        );

        // Verify other points still work
        for (size_t i = 1; i < new_ids.size(); ++i) {
            size_t query_id = std::min<size_t>(8, queries.size() - 1);
            auto query = queries.get_datum(query_id);

            // Should not throw
            double distance = index.get_distance(new_ids[i], query);
            CATCH_REQUIRE(distance >= 0.0);
        }
    }
}
