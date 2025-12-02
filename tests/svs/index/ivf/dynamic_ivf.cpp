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

    // Create dynamic clusters from the clustering result
    using ClusterType =
        svs::index::ivf::DynamicDenseCluster<svs::data::SimpleData<Eltype, N>, Idx>;

    std::vector<ClusterType> clusters;
    for (size_t c = 0; c < NUM_CLUSTERS; ++c) {
        const auto& cluster_indices = clustering.cluster(c);
        size_t cluster_size = cluster_indices.size();

        ClusterType cluster;
        cluster.data_ = svs::data::SimpleData<Eltype, N>(cluster_size, N);
        cluster.ids_.resize(cluster_size);

        for (size_t i = 0; i < cluster_size; ++i) {
            Idx global_id = cluster_indices[i];
            cluster.data_.set_datum(i, initial_data.get_datum(global_id));
            cluster.ids_[i] = global_id;
        }

        clusters.push_back(std::move(cluster));
    }

    // Create the dynamic IVF index
    auto centroids = clustering.centroids();
    auto threadpool_for_index = svs::threads::as_threadpool(num_threads);
    using IndexType = svs::index::ivf::DynamicIVFIndex<
        decltype(centroids),
        ClusterType,
        Distance,
        decltype(threadpool_for_index)>;

    auto index = IndexType(
        std::move(centroids),
        std::move(clusters),
        initial_indices,
        Distance(),
        std::move(threadpool_for_index),
        1 // intra_query_threads
    );

    reference.configure_extra_checks(true);
    CATCH_REQUIRE(reference.extra_checks_enabled());

    test_loop(index, reference, queries, div(reference.size(), modify_fraction), 2, 6);
}
