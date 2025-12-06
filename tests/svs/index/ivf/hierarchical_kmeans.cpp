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
#include "svs/index/ivf/hierarchical_kmeans.h"

// additional headers for train_only test
#include "svs/index/ivf/index.h"
#include "svs/index/ivf/kmeans.h"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

// catch
#include "catch2/catch_test_macros.hpp"

// stl
#include <cmath>
#include <unordered_map>

namespace {

template <typename BuildType, svs::data::ImmutableMemoryDataset Data, typename Distance>
void test_hierarchical_kmeans_clustering(const Data& data, Distance distance) {
    namespace ivf = svs::index::ivf;

    for (size_t n_centroids : {155}) {
        for (size_t minibatch : {25}) {
            for (size_t iters : {3}) {
                for (float training_fraction : {0.55}) {
                    for (size_t l1_clusters : {0, 9}) {
                        auto params = ivf::IVFBuildParameters()
                                          .num_centroids(n_centroids)
                                          .minibatch_size(minibatch)
                                          .num_iterations(iters)
                                          .is_hierarchical(true)
                                          .training_fraction(training_fraction)
                                          .hierarchical_level1_clusters(l1_clusters);

                        auto threadpool = svs::threads::as_threadpool(10);
                        auto [centroids, clusters] =
                            hierarchical_kmeans_clustering<BuildType>(
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
}

template <typename BuildType, svs::data::ImmutableMemoryDataset Data, typename Distance>
void test_train_only_centroids_match(const Data& data, Distance distance) {
    namespace ivf = svs::index::ivf;

    // Test both flat and hierarchical k-means with different modes
    for (bool is_hierarchical : {false, true}) {
        for (size_t n_centroids : {25}) {
            for (size_t minibatch : {25}) {
                for (size_t iters : {3}) {
                    for (float training_fraction : {0.6}) {
                        auto params = ivf::IVFBuildParameters()
                                          .num_centroids(n_centroids)
                                          .minibatch_size(minibatch)
                                          .num_iterations(iters)
                                          .is_hierarchical(is_hierarchical)
                                          .training_fraction(training_fraction)
                                          .seed(12345); // Fixed seed for reproducibility

                        if (is_hierarchical) {
                            params.hierarchical_level1_clusters(5);
                        }

                        size_t num_threads = 4;

                        // Run with train_only = false (normal mode)
                        auto [centroids_normal, clusters_normal] =
                            ivf::build_clustering<BuildType>(
                                params, data, distance, num_threads, false
                            );

                        // Run with train_only = true
                        auto [centroids_train_only, clusters_train_only] =
                            ivf::build_clustering<BuildType>(
                                params, data, distance, num_threads, true
                            );

                        // Verify centroids are identical
                        CATCH_REQUIRE(
                            centroids_normal.size() == centroids_train_only.size()
                        );
                        CATCH_REQUIRE(
                            centroids_normal.dimensions() ==
                            centroids_train_only.dimensions()
                        );

                        constexpr float tolerance = 1e-6f;
                        for (size_t i = 0; i < centroids_normal.size(); ++i) {
                            auto datum_normal = centroids_normal.get_datum(i);
                            auto datum_train_only = centroids_train_only.get_datum(i);

                            for (size_t j = 0; j < centroids_normal.dimensions(); ++j) {
                                float diff =
                                    std::abs(datum_normal[j] - datum_train_only[j]);
                                CATCH_REQUIRE(diff < tolerance);
                            }
                        }

                        // Verify train_only clusters are empty (as expected)
                        for (const auto& cluster : clusters_train_only) {
                            CATCH_REQUIRE(cluster.empty());
                        }

                        // Verify normal mode has non-empty clusters (at least some)
                        bool has_non_empty_cluster = false;
                        for (const auto& cluster : clusters_normal) {
                            if (!cluster.empty()) {
                                has_non_empty_cluster = true;
                                break;
                            }
                        }
                        CATCH_REQUIRE(has_non_empty_cluster);
                    }
                }
            }
        }
    }
}

template <typename BuildType, svs::data::ImmutableMemoryDataset Data, typename Distance>
void test_hierarchical_kmeans_level1_clusters(const Data& data, Distance distance) {
    namespace ivf = svs::index::ivf;

    // Test different Level 1 cluster configurations
    for (size_t n_centroids : {64, 100}) {
        for (size_t l1_clusters : {0, 4, 8, 16}) { // 0 means auto-calculate
            auto params = ivf::IVFBuildParameters()
                              .num_centroids(n_centroids)
                              .minibatch_size(25)
                              .num_iterations(3)
                              .is_hierarchical(true)
                              .training_fraction(0.6f)
                              .hierarchical_level1_clusters(l1_clusters);

            auto threadpool = svs::threads::as_threadpool(4);
            auto [centroids, clusters] = hierarchical_kmeans_clustering<BuildType>(
                params, data, distance, threadpool
            );

            CATCH_REQUIRE(centroids.size() == n_centroids);
            CATCH_REQUIRE(centroids.dimensions() == data.dimensions());
            CATCH_REQUIRE(clusters.size() == n_centroids);

            // Verify all data points are assigned
            std::unordered_set<uint32_t> assigned_points;
            for (const auto& cluster : clusters) {
                for (auto point_id : cluster) {
                    CATCH_REQUIRE(point_id < data.size());
                    assigned_points.insert(point_id);
                }
            }
            CATCH_REQUIRE(assigned_points.size() == data.size());
        }
    }
}

template <typename BuildType, svs::data::ImmutableMemoryDataset Data, typename Distance>
void test_hierarchical_kmeans_reproducibility(const Data& data, Distance distance) {
    namespace ivf = svs::index::ivf;

    const size_t seed = 98765;
    const size_t n_centroids = 50;
    const size_t l1_clusters = 7;

    auto params1 = ivf::IVFBuildParameters()
                       .num_centroids(n_centroids)
                       .minibatch_size(25)
                       .num_iterations(4)
                       .is_hierarchical(true)
                       .training_fraction(0.7f)
                       .hierarchical_level1_clusters(l1_clusters)
                       .seed(seed);

    auto params2 = ivf::IVFBuildParameters()
                       .num_centroids(n_centroids)
                       .minibatch_size(25)
                       .num_iterations(4)
                       .is_hierarchical(true)
                       .training_fraction(0.7f)
                       .hierarchical_level1_clusters(l1_clusters)
                       .seed(seed);

    auto threadpool = svs::threads::as_threadpool(4);

    auto [centroids1, clusters1] =
        hierarchical_kmeans_clustering<BuildType>(params1, data, distance, threadpool);

    auto [centroids2, clusters2] =
        hierarchical_kmeans_clustering<BuildType>(params2, data, distance, threadpool);

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
}

template <typename BuildType, svs::data::ImmutableMemoryDataset Data, typename Distance>
void test_hierarchical_vs_flat_kmeans(const Data& data, Distance distance) {
    namespace ivf = svs::index::ivf;

    const size_t n_centroids = 36;

    // Flat k-means
    auto flat_params = ivf::IVFBuildParameters()
                           .num_centroids(n_centroids)
                           .minibatch_size(25)
                           .num_iterations(3)
                           .is_hierarchical(false)
                           .training_fraction(0.6f)
                           .seed(555);

    // Hierarchical k-means
    auto hierarchical_params = ivf::IVFBuildParameters()
                                   .num_centroids(n_centroids)
                                   .minibatch_size(25)
                                   .num_iterations(3)
                                   .is_hierarchical(true)
                                   .training_fraction(0.6f)
                                   .hierarchical_level1_clusters(6)
                                   .seed(555);

    auto threadpool = svs::threads::as_threadpool(4);

    auto [flat_centroids, flat_clusters] =
        ivf::kmeans_clustering<BuildType>(flat_params, data, distance, threadpool);

    auto [hierarchical_centroids, hierarchical_clusters] =
        hierarchical_kmeans_clustering<BuildType>(
            hierarchical_params, data, distance, threadpool
        );

    // Both should produce same number of centroids and clusters
    CATCH_REQUIRE(flat_centroids.size() == n_centroids);
    CATCH_REQUIRE(hierarchical_centroids.size() == n_centroids);
    CATCH_REQUIRE(flat_clusters.size() == n_centroids);
    CATCH_REQUIRE(hierarchical_clusters.size() == n_centroids);

    // Both should assign all points
    std::unordered_set<uint32_t> flat_points, hierarchical_points;

    for (const auto& cluster : flat_clusters) {
        for (auto point_id : cluster) {
            flat_points.insert(point_id);
        }
    }

    for (const auto& cluster : hierarchical_clusters) {
        for (auto point_id : cluster) {
            hierarchical_points.insert(point_id);
        }
    }

    CATCH_REQUIRE(flat_points.size() == data.size());
    CATCH_REQUIRE(hierarchical_points.size() == data.size());
}

template <typename BuildType, svs::data::ImmutableMemoryDataset Data, typename Distance>
void test_hierarchical_kmeans_edge_cases(const Data& data, Distance distance) {
    namespace ivf = svs::index::ivf;
    auto threadpool = svs::threads::as_threadpool(4);

    // Test with Level 1 clusters equal to total centroids (degenerate case)
    {
        const size_t n_centroids = 16;
        auto params = ivf::IVFBuildParameters()
                          .num_centroids(n_centroids)
                          .minibatch_size(20)
                          .num_iterations(2)
                          .is_hierarchical(true)
                          .training_fraction(0.5f)
                          .hierarchical_level1_clusters(n_centroids);

        auto [centroids, clusters] =
            hierarchical_kmeans_clustering<BuildType>(params, data, distance, threadpool);

        CATCH_REQUIRE(centroids.size() == n_centroids);
        CATCH_REQUIRE(clusters.size() == n_centroids);
    }

    // Test with very few Level 1 clusters
    {
        const size_t n_centroids = 60;
        auto params = ivf::IVFBuildParameters()
                          .num_centroids(n_centroids)
                          .minibatch_size(25)
                          .num_iterations(3)
                          .is_hierarchical(true)
                          .training_fraction(0.6f)
                          .hierarchical_level1_clusters(2);

        auto [centroids, clusters] =
            hierarchical_kmeans_clustering<BuildType>(params, data, distance, threadpool);

        CATCH_REQUIRE(centroids.size() == n_centroids);
        CATCH_REQUIRE(clusters.size() == n_centroids);
    }

    // Test with different training fractions
    for (float training_fraction : {0.3f, 0.5f, 0.8f, 1.0f}) {
        auto params = ivf::IVFBuildParameters()
                          .num_centroids(24)
                          .minibatch_size(20)
                          .num_iterations(2)
                          .is_hierarchical(true)
                          .training_fraction(training_fraction)
                          .hierarchical_level1_clusters(4);

        auto [centroids, clusters] =
            hierarchical_kmeans_clustering<BuildType>(params, data, distance, threadpool);

        CATCH_REQUIRE(centroids.size() == 24);
        CATCH_REQUIRE(clusters.size() == 24);

        // Verify centroids are valid
        for (size_t i = 0; i < centroids.size(); ++i) {
            auto centroid = centroids.get_datum(i);
            for (size_t j = 0; j < centroids.dimensions(); ++j) {
                CATCH_REQUIRE(std::isfinite(centroid[j]));
            }
        }
    }
}

template <typename BuildType, svs::data::ImmutableMemoryDataset Data, typename Distance>
void test_hierarchical_kmeans_cluster_distribution(const Data& data, Distance distance) {
    namespace ivf = svs::index::ivf;

    // Test that Level 2 clusters are reasonably distributed across Level 1 clusters
    const size_t n_centroids = 48;
    const size_t l1_clusters = 6;

    auto params = ivf::IVFBuildParameters()
                      .num_centroids(n_centroids)
                      .minibatch_size(25)
                      .num_iterations(4)
                      .is_hierarchical(true)
                      .training_fraction(0.7f)
                      .hierarchical_level1_clusters(l1_clusters)
                      .seed(777);

    auto threadpool = svs::threads::as_threadpool(4);
    auto [centroids, clusters] =
        hierarchical_kmeans_clustering<BuildType>(params, data, distance, threadpool);

    CATCH_REQUIRE(centroids.size() == n_centroids);
    CATCH_REQUIRE(clusters.size() == n_centroids);

    // Verify we have some reasonable distribution of cluster sizes
    size_t empty_clusters = 0;
    size_t total_assigned = 0;

    for (const auto& cluster : clusters) {
        if (cluster.empty()) {
            empty_clusters++;
        }
        total_assigned += cluster.size();
    }

    CATCH_REQUIRE(total_assigned == data.size());
    // Allow some empty clusters but not too many (less than half)
    CATCH_REQUIRE(empty_clusters < n_centroids / 2);
}

} // namespace

CATCH_TEST_CASE("Hierarchical Kmeans Param Check", "[ivf][hierarchial_parameter_check]") {
    CATCH_SECTION("Uncompressed Data - All Data Types") {
        auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());

        // Test float32
        test_hierarchical_kmeans_clustering<float>(data, svs::DistanceIP());
        test_hierarchical_kmeans_clustering<float>(data, svs::DistanceL2());

        // Test Float16 (fp16)
        test_hierarchical_kmeans_clustering<svs::Float16>(data, svs::DistanceIP());
        test_hierarchical_kmeans_clustering<svs::Float16>(data, svs::DistanceL2());

        // Test BFloat16 (bf16)
        test_hierarchical_kmeans_clustering<svs::BFloat16>(data, svs::DistanceIP());
        test_hierarchical_kmeans_clustering<svs::BFloat16>(data, svs::DistanceL2());
    }
}

CATCH_TEST_CASE(
    "Hierarchical Kmeans Level1 Clusters", "[ivf][hierarchical_kmeans][level1]"
) {
    CATCH_SECTION("Uncompressed Data - All Data Types") {
        auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());

        // Test float32
        test_hierarchical_kmeans_level1_clusters<float>(data, svs::DistanceIP());
        test_hierarchical_kmeans_level1_clusters<float>(data, svs::DistanceL2());

        // Test Float16 (fp16)
        test_hierarchical_kmeans_level1_clusters<svs::Float16>(data, svs::DistanceIP());
        test_hierarchical_kmeans_level1_clusters<svs::Float16>(data, svs::DistanceL2());

        // Test BFloat16 (bf16)
        test_hierarchical_kmeans_level1_clusters<svs::BFloat16>(data, svs::DistanceIP());
        test_hierarchical_kmeans_level1_clusters<svs::BFloat16>(data, svs::DistanceL2());
    }
}

CATCH_TEST_CASE(
    "Hierarchical Kmeans Reproducibility", "[ivf][hierarchical_kmeans][reproducibility]"
) {
    CATCH_SECTION("Uncompressed Data - All Data Types") {
        auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());

        // Test float32
        test_hierarchical_kmeans_reproducibility<float>(data, svs::DistanceIP());
        test_hierarchical_kmeans_reproducibility<float>(data, svs::DistanceL2());

        // Test Float16 (fp16)
        test_hierarchical_kmeans_reproducibility<svs::Float16>(data, svs::DistanceIP());
        test_hierarchical_kmeans_reproducibility<svs::Float16>(data, svs::DistanceL2());

        // Test BFloat16 (bf16)
        test_hierarchical_kmeans_reproducibility<svs::BFloat16>(data, svs::DistanceIP());
        test_hierarchical_kmeans_reproducibility<svs::BFloat16>(data, svs::DistanceL2());
    }
}

CATCH_TEST_CASE("Hierarchical vs Flat Kmeans", "[ivf][hierarchical_kmeans][comparison]") {
    CATCH_SECTION("Uncompressed Data - All Data Types") {
        auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());

        // Test float32
        test_hierarchical_vs_flat_kmeans<float>(data, svs::DistanceIP());
        test_hierarchical_vs_flat_kmeans<float>(data, svs::DistanceL2());

        // Test Float16 (fp16)
        test_hierarchical_vs_flat_kmeans<svs::Float16>(data, svs::DistanceIP());
        test_hierarchical_vs_flat_kmeans<svs::Float16>(data, svs::DistanceL2());

        // Test BFloat16 (bf16)
        test_hierarchical_vs_flat_kmeans<svs::BFloat16>(data, svs::DistanceIP());
        test_hierarchical_vs_flat_kmeans<svs::BFloat16>(data, svs::DistanceL2());
    }
}

CATCH_TEST_CASE(
    "Hierarchical Kmeans Edge Cases", "[ivf][hierarchical_kmeans][edge_cases]"
) {
    CATCH_SECTION("Uncompressed Data - All Data Types") {
        auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());

        // Test float32
        test_hierarchical_kmeans_edge_cases<float>(data, svs::DistanceIP());
        test_hierarchical_kmeans_edge_cases<float>(data, svs::DistanceL2());

        // Test Float16 (fp16)
        test_hierarchical_kmeans_edge_cases<svs::Float16>(data, svs::DistanceIP());
        test_hierarchical_kmeans_edge_cases<svs::Float16>(data, svs::DistanceL2());

        // Test BFloat16 (bf16)
        test_hierarchical_kmeans_edge_cases<svs::BFloat16>(data, svs::DistanceIP());
        test_hierarchical_kmeans_edge_cases<svs::BFloat16>(data, svs::DistanceL2());
    }
}

CATCH_TEST_CASE(
    "Hierarchical Kmeans Cluster Distribution", "[ivf][hierarchical_kmeans][distribution]"
) {
    CATCH_SECTION("Uncompressed Data - All Data Types") {
        auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());

        // Test float32
        test_hierarchical_kmeans_cluster_distribution<float>(data, svs::DistanceIP());
        test_hierarchical_kmeans_cluster_distribution<float>(data, svs::DistanceL2());

        // Test Float16 (fp16)
        test_hierarchical_kmeans_cluster_distribution<svs::Float16>(
            data, svs::DistanceIP()
        );
        test_hierarchical_kmeans_cluster_distribution<svs::Float16>(
            data, svs::DistanceL2()
        );

        // Test BFloat16 (bf16)
        test_hierarchical_kmeans_cluster_distribution<svs::BFloat16>(
            data, svs::DistanceIP()
        );
        test_hierarchical_kmeans_cluster_distribution<svs::BFloat16>(
            data, svs::DistanceL2()
        );
    }
}

CATCH_TEST_CASE("Train Only Centroids Match", "[ivf][kmeans][train_only]") {
    CATCH_SECTION("Uncompressed Data - All Data Types") {
        auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());

        // Test float32
        test_train_only_centroids_match<float>(data, svs::DistanceIP());
        test_train_only_centroids_match<float>(data, svs::DistanceL2());

        // Test Float16 (fp16)
        test_train_only_centroids_match<svs::Float16>(data, svs::DistanceIP());
        test_train_only_centroids_match<svs::Float16>(data, svs::DistanceL2());

        // Test BFloat16 (bf16)
        test_train_only_centroids_match<svs::BFloat16>(data, svs::DistanceIP());
        test_train_only_centroids_match<svs::BFloat16>(data, svs::DistanceL2());
    }
}
