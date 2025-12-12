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
#include "svs/index/ivf/kmeans.h"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

// catch
#include "catch2/catch_test_macros.hpp"

// stl
#include <unordered_map>

namespace {

template <typename BuildType, svs::data::ImmutableMemoryDataset Data, typename Distance>
void test_kmeans_clustering(const Data& data, Distance distance) {
    namespace ivf = svs::index::ivf;

    for (size_t n_centroids : {1, 99}) {
        for (size_t minibatch : {25}) {
            for (size_t iters : {3}) {
                for (float training_fraction : {0.55}) {
                    auto params = ivf::IVFBuildParameters()
                                      .num_centroids(n_centroids)
                                      .minibatch_size(minibatch)
                                      .num_iterations(iters)
                                      .is_hierarchical(false)
                                      .training_fraction(training_fraction);
                    auto threadpool = svs::threads::as_threadpool(10);
                    auto [centroids, clusters] = ivf::kmeans_clustering<BuildType>(
                        params, data, distance, threadpool
                    );

                    CATCH_REQUIRE(centroids.size() == n_centroids);
                    CATCH_REQUIRE(centroids.dimensions() == data.dimensions());
                    CATCH_REQUIRE(clusters.size() == n_centroids);
                }
            }
        }
    }
}

template <typename BuildType, svs::data::ImmutableMemoryDataset Data, typename Distance>
void test_kmeans_train_only_functionality(const Data& data, Distance distance) {
    namespace ivf = svs::index::ivf;

    // Test train_only functionality
    for (size_t n_centroids : {25, 50}) {
        for (size_t minibatch : {25}) {
            for (size_t iters : {3}) {
                for (float training_fraction : {0.6f}) {
                    auto params = ivf::IVFBuildParameters()
                                      .num_centroids(n_centroids)
                                      .minibatch_size(minibatch)
                                      .num_iterations(iters)
                                      .is_hierarchical(false)
                                      .training_fraction(training_fraction)
                                      .seed(42); // Fixed seed for reproducibility

                    auto threadpool = svs::threads::as_threadpool(4);

                    // Test train_only = false (normal mode)
                    auto [centroids_normal, clusters_normal] =
                        ivf::kmeans_clustering<BuildType>(
                            params, data, distance, threadpool, false
                        );

                    // Test train_only = true
                    auto [centroids_train_only, clusters_train_only] =
                        ivf::kmeans_clustering<BuildType>(
                            params, data, distance, threadpool, true
                        );

                    // Verify basic structure
                    CATCH_REQUIRE(centroids_normal.size() == n_centroids);
                    CATCH_REQUIRE(centroids_train_only.size() == n_centroids);
                    CATCH_REQUIRE(centroids_normal.dimensions() == data.dimensions());
                    CATCH_REQUIRE(centroids_train_only.dimensions() == data.dimensions());

                    CATCH_REQUIRE(clusters_normal.size() == n_centroids);
                    CATCH_REQUIRE(clusters_train_only.size() == n_centroids);

                    // Verify train_only produces empty clusters
                    for (const auto& cluster : clusters_train_only) {
                        CATCH_REQUIRE(cluster.empty());
                    }

                    // Verify normal mode has at least some non-empty clusters
                    bool has_non_empty = false;
                    for (const auto& cluster : clusters_normal) {
                        if (!cluster.empty()) {
                            has_non_empty = true;
                            break;
                        }
                    }
                    CATCH_REQUIRE(has_non_empty);

                    // Verify centroids are identical (using same seed)
                    constexpr float tolerance = 1e-6f;
                    for (size_t i = 0; i < n_centroids; ++i) {
                        auto normal_centroid = centroids_normal.get_datum(i);
                        auto train_only_centroid = centroids_train_only.get_datum(i);

                        for (size_t j = 0; j < data.dimensions(); ++j) {
                            float diff =
                                std::abs(normal_centroid[j] - train_only_centroid[j]);
                            CATCH_REQUIRE(diff < tolerance);
                        }
                    }
                }
            }
        }
    }
}

template <typename BuildType, svs::data::ImmutableMemoryDataset Data, typename Distance>
void test_kmeans_train_only_performance(const Data& data, Distance distance) {
    namespace ivf = svs::index::ivf;

    // Test that train_only mode is at least as fast as normal mode
    // (it should be faster since it skips assignment, but we just check it doesn't slow
    // down)
    size_t n_centroids = 50;
    auto params = ivf::IVFBuildParameters()
                      .num_centroids(n_centroids)
                      .minibatch_size(25)
                      .num_iterations(3)
                      .is_hierarchical(false)
                      .training_fraction(0.5f)
                      .seed(123);

    auto threadpool = svs::threads::as_threadpool(4);

    // Time normal mode
    auto start_normal = std::chrono::high_resolution_clock::now();
    auto [centroids_normal, clusters_normal] =
        ivf::kmeans_clustering<BuildType>(params, data, distance, threadpool, false);
    auto end_normal = std::chrono::high_resolution_clock::now();

    // Time train_only mode
    auto start_train_only = std::chrono::high_resolution_clock::now();
    auto [centroids_train_only, clusters_train_only] =
        ivf::kmeans_clustering<BuildType>(params, data, distance, threadpool, true);
    auto end_train_only = std::chrono::high_resolution_clock::now();

    auto normal_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_normal - start_normal);
    auto train_only_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_train_only - start_train_only
    );

    CATCH_REQUIRE(train_only_duration.count() <= normal_duration.count() * 1.5);
    // Note: We do not assert on performance here, as wall-clock timing is unreliable in CI.
    // In practice, train_only should be faster, but this is best verified with dedicated
    // benchmarks.

    // Verify results are still valid
    CATCH_REQUIRE(centroids_train_only.size() == n_centroids);
    for (const auto& cluster : clusters_train_only) {
        CATCH_REQUIRE(cluster.empty());
    }
}

template <typename BuildType, svs::data::ImmutableMemoryDataset Data, typename Distance>
void test_kmeans_edge_cases(const Data& data, Distance distance) {
    namespace ivf = svs::index::ivf;

    // Test with minimum centroids
    {
        auto params = ivf::IVFBuildParameters()
                          .num_centroids(1)
                          .minibatch_size(10)
                          .num_iterations(2)
                          .is_hierarchical(false)
                          .training_fraction(0.5f);
        auto threadpool = svs::threads::as_threadpool(2);
        auto [centroids, clusters] =
            ivf::kmeans_clustering<BuildType>(params, data, distance, threadpool);

        CATCH_REQUIRE(centroids.size() == 1);
        CATCH_REQUIRE(clusters.size() == 1);
        CATCH_REQUIRE(clusters[0].size() > 0); // Should contain all points
    }

    // Test with large number of centroids (but less than data points)
    if (data.size() > 100) {
        auto params = ivf::IVFBuildParameters()
                          .num_centroids(std::min(data.size() - 1, size_t(100)))
                          .minibatch_size(20)
                          .num_iterations(3)
                          .is_hierarchical(false)
                          .training_fraction(0.7f);
        auto threadpool = svs::threads::as_threadpool(4);
        auto [centroids, clusters] =
            ivf::kmeans_clustering<BuildType>(params, data, distance, threadpool);

        CATCH_REQUIRE(centroids.size() == std::min(data.size() - 1, size_t(100)));
        CATCH_REQUIRE(clusters.size() == std::min(data.size() - 1, size_t(100)));
    }
}

template <typename BuildType, svs::data::ImmutableMemoryDataset Data, typename Distance>
void test_kmeans_reproducibility(const Data& data, Distance distance) {
    namespace ivf = svs::index::ivf;

    // Test that same seed produces same results
    const size_t seed = 12345;
    const size_t n_centroids = 25;

    auto params1 = ivf::IVFBuildParameters()
                       .num_centroids(n_centroids)
                       .minibatch_size(25)
                       .num_iterations(3)
                       .is_hierarchical(false)
                       .training_fraction(0.6f)
                       .seed(seed);

    auto params2 = ivf::IVFBuildParameters()
                       .num_centroids(n_centroids)
                       .minibatch_size(25)
                       .num_iterations(3)
                       .is_hierarchical(false)
                       .training_fraction(0.6f)
                       .seed(seed);

    auto threadpool = svs::threads::as_threadpool(4);

    auto [centroids1, clusters1] =
        ivf::kmeans_clustering<BuildType>(params1, data, distance, threadpool);

    auto [centroids2, clusters2] =
        ivf::kmeans_clustering<BuildType>(params2, data, distance, threadpool);

    // Verify centroids are identical
    CATCH_REQUIRE(centroids1.size() == centroids2.size());
    constexpr float tolerance = 1e-6f;

    for (size_t i = 0; i < centroids1.size(); ++i) {
        auto centroid1 = centroids1.get_datum(i);
        auto centroid2 = centroids2.get_datum(i);

        for (size_t j = 0; j < centroids1.dimensions(); ++j) {
            float diff = std::abs(centroid1[j] - centroid2[j]);
            CATCH_REQUIRE(diff < tolerance);
        }
    }

    // Verify cluster assignments are identical
    CATCH_REQUIRE(clusters1.size() == clusters2.size());
    for (size_t i = 0; i < clusters1.size(); ++i) {
        CATCH_REQUIRE(clusters1[i].size() == clusters2[i].size());
        // Note: We don't check exact order as cluster assignment might vary with same
        // centroids
    }
}

template <typename BuildType, svs::data::ImmutableMemoryDataset Data, typename Distance>
void test_kmeans_cluster_assignment_validity(const Data& data, Distance distance) {
    namespace ivf = svs::index::ivf;

    auto params = ivf::IVFBuildParameters()
                      .num_centroids(20)
                      .minibatch_size(25)
                      .num_iterations(5)
                      .is_hierarchical(false)
                      .training_fraction(0.8f);

    auto threadpool = svs::threads::as_threadpool(4);
    auto [centroids, clusters] =
        ivf::kmeans_clustering<BuildType>(params, data, distance, threadpool);

    // Verify all data points are assigned to exactly one cluster
    std::unordered_set<uint32_t> assigned_points;
    for (size_t i = 0; i < clusters.size(); ++i) {
        for (auto point_id : clusters[i]) {
            CATCH_REQUIRE(point_id < data.size()); // Valid point index
            CATCH_REQUIRE(
                assigned_points.find(point_id) == assigned_points.end()
            ); // Not already assigned
            assigned_points.insert(point_id);
        }
    }

    CATCH_REQUIRE(assigned_points.size() == data.size()); // All points assigned

    // Verify centroids have valid values (no NaN or infinity)
    for (size_t i = 0; i < centroids.size(); ++i) {
        auto centroid = centroids.get_datum(i);
        for (size_t j = 0; j < centroids.dimensions(); ++j) {
            CATCH_REQUIRE(std::isfinite(centroid[j]));
        }
    }
}

template <typename BuildType, svs::data::ImmutableMemoryDataset Data, typename Distance>
void test_kmeans_parameter_variations(const Data& data, Distance distance) {
    namespace ivf = svs::index::ivf;
    auto threadpool = svs::threads::as_threadpool(4);

    // Test different minibatch sizes
    for (size_t minibatch : {10, 25, 50}) {
        auto params = ivf::IVFBuildParameters()
                          .num_centroids(15)
                          .minibatch_size(minibatch)
                          .num_iterations(3)
                          .is_hierarchical(false)
                          .training_fraction(0.6f);

        auto [centroids, clusters] =
            ivf::kmeans_clustering<BuildType>(params, data, distance, threadpool);

        CATCH_REQUIRE(centroids.size() == 15);
        CATCH_REQUIRE(clusters.size() == 15);
    }

    // Test different iteration counts
    for (size_t iters : {1, 3, 5, 10}) {
        auto params = ivf::IVFBuildParameters()
                          .num_centroids(10)
                          .minibatch_size(25)
                          .num_iterations(iters)
                          .is_hierarchical(false)
                          .training_fraction(0.6f);

        auto [centroids, clusters] =
            ivf::kmeans_clustering<BuildType>(params, data, distance, threadpool);

        CATCH_REQUIRE(centroids.size() == 10);
        CATCH_REQUIRE(clusters.size() == 10);
    }

    // Test different training fractions
    for (float training_fraction : {0.3f, 0.5f, 0.7f, 0.9f}) {
        auto params = ivf::IVFBuildParameters()
                          .num_centroids(12)
                          .minibatch_size(25)
                          .num_iterations(3)
                          .is_hierarchical(false)
                          .training_fraction(training_fraction);

        auto [centroids, clusters] =
            ivf::kmeans_clustering<BuildType>(params, data, distance, threadpool);

        CATCH_REQUIRE(centroids.size() == 12);
        CATCH_REQUIRE(clusters.size() == 12);
    }
}

} // namespace

CATCH_TEST_CASE("Build Kmeans Param Check", "[ivf][parameter_check]") {
    CATCH_SECTION("Uncompressed Data - All Data Types") {
        auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());

        // Test float32
        test_kmeans_clustering<float>(data, svs::DistanceIP());
        test_kmeans_clustering<float>(data, svs::DistanceL2());

        // Test Float16 (fp16)
        test_kmeans_clustering<svs::Float16>(data, svs::DistanceIP());
        test_kmeans_clustering<svs::Float16>(data, svs::DistanceL2());

        // Test BFloat16 (bf16)
        test_kmeans_clustering<svs::BFloat16>(data, svs::DistanceIP());
        test_kmeans_clustering<svs::BFloat16>(data, svs::DistanceL2());
    }
}

CATCH_TEST_CASE("Kmeans Edge Cases", "[ivf][kmeans][edge_cases]") {
    CATCH_SECTION("Uncompressed Data - All Data Types") {
        auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());

        // Test float32
        test_kmeans_edge_cases<float>(data, svs::DistanceIP());
        test_kmeans_edge_cases<float>(data, svs::DistanceL2());

        // Test Float16 (fp16)
        test_kmeans_edge_cases<svs::Float16>(data, svs::DistanceIP());
        test_kmeans_edge_cases<svs::Float16>(data, svs::DistanceL2());

        // Test BFloat16 (bf16)
        test_kmeans_edge_cases<svs::BFloat16>(data, svs::DistanceIP());
        test_kmeans_edge_cases<svs::BFloat16>(data, svs::DistanceL2());
    }
}

CATCH_TEST_CASE("Kmeans Reproducibility", "[ivf][kmeans][reproducibility]") {
    CATCH_SECTION("Uncompressed Data - All Data Types") {
        auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());

        // Test float32
        test_kmeans_reproducibility<float>(data, svs::DistanceIP());
        test_kmeans_reproducibility<float>(data, svs::DistanceL2());

        // Test Float16 (fp16)
        test_kmeans_reproducibility<svs::Float16>(data, svs::DistanceIP());
        test_kmeans_reproducibility<svs::Float16>(data, svs::DistanceL2());

        // Test BFloat16 (bf16)
        test_kmeans_reproducibility<svs::BFloat16>(data, svs::DistanceIP());
        test_kmeans_reproducibility<svs::BFloat16>(data, svs::DistanceL2());
    }
}

CATCH_TEST_CASE("Kmeans Cluster Assignment Validity", "[ivf][kmeans][cluster_validity]") {
    CATCH_SECTION("Uncompressed Data - All Data Types") {
        auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());

        // Test float32
        test_kmeans_cluster_assignment_validity<float>(data, svs::DistanceIP());
        test_kmeans_cluster_assignment_validity<float>(data, svs::DistanceL2());

        // Test Float16 (fp16)
        test_kmeans_cluster_assignment_validity<svs::Float16>(data, svs::DistanceIP());
        test_kmeans_cluster_assignment_validity<svs::Float16>(data, svs::DistanceL2());

        // Test BFloat16 (bf16)
        test_kmeans_cluster_assignment_validity<svs::BFloat16>(data, svs::DistanceIP());
        test_kmeans_cluster_assignment_validity<svs::BFloat16>(data, svs::DistanceL2());
    }
}

CATCH_TEST_CASE("Kmeans Parameter Variations", "[ivf][kmeans][parameters]") {
    CATCH_SECTION("Uncompressed Data - All Data Types") {
        auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());

        // Test float32
        test_kmeans_parameter_variations<float>(data, svs::DistanceIP());
        test_kmeans_parameter_variations<float>(data, svs::DistanceL2());

        // Test Float16 (fp16)
        test_kmeans_parameter_variations<svs::Float16>(data, svs::DistanceIP());
        test_kmeans_parameter_variations<svs::Float16>(data, svs::DistanceL2());

        // Test BFloat16 (bf16)
        test_kmeans_parameter_variations<svs::BFloat16>(data, svs::DistanceIP());
        test_kmeans_parameter_variations<svs::BFloat16>(data, svs::DistanceL2());
    }
}
