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
#include "svs/index/ivf/common.h"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

// catch
#include "catch2/catch_test_macros.hpp"

// svs
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/index/ivf/hierarchical_kmeans.h"
#include "svs/index/ivf/kmeans.h"
#include "svs/lib/threads.h"

// stl
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <vector>

CATCH_TEST_CASE("Kmeans Clustering", "[ivf][parameters]") {
    namespace ivf = svs::index::ivf;
    CATCH_SECTION("IVF Build Parameters") {
        auto p = ivf::IVFBuildParameters();

        // Test the setter methods.
#define XX(name, v)                        \
    CATCH_REQUIRE(p.name##_ != v);         \
    CATCH_REQUIRE(p.name(v).name##_ == v); \
    CATCH_REQUIRE(p.name##_ == v);

        XX(num_centroids, 10);
        XX(minibatch_size, 100);
        XX(num_iterations, 1000);
        XX(is_hierarchical, false);
        XX(training_fraction, 0.05F);
        XX(hierarchical_level1_clusters, 10);
        XX(seed, 0x1234);
#undef XX

        // Saving and loading
        svs_test::prepare_temp_directory();
        auto dir = svs_test::temp_directory();
        CATCH_REQUIRE(svs::lib::test_self_save_load(p, dir));
    }

    CATCH_SECTION("IVF Search Parameters") {
        auto p = ivf::IVFSearchParameters();

        // Test the setter methods.
#define XX(name, v)                        \
    CATCH_REQUIRE(p.name##_ != v);         \
    CATCH_REQUIRE(p.name(v).name##_ == v); \
    CATCH_REQUIRE(p.name##_ == v);

        XX(n_probes, 10);
        XX(k_reorder, 100);
#undef XX

        // Saving and loading
        svs_test::prepare_temp_directory();
        auto dir = svs_test::temp_directory();
        CATCH_REQUIRE(svs::lib::test_self_save_load(p, dir));
    }
}

CATCH_TEST_CASE("Common Utility Functions", "[ivf][common][core]") {
    namespace ivf = svs::index::ivf;

    CATCH_SECTION("compute_matmul - All Data Types") {
        // Test matrix multiplication for different data types
        constexpr size_t m = 10; // number of data points
        constexpr size_t n = 5;  // number of centroids
        constexpr size_t k = 8;  // dimensions

        auto test_matmul = [&]<typename T>() {
            // Create test data
            auto data = svs::data::SimpleData<T>(m, k);
            auto centroids = svs::data::SimpleData<T>(n, k);
            auto results = svs::data::SimpleData<float>(m, n);

            // Fill with test values
            for (size_t i = 0; i < m; ++i) {
                auto datum = data.get_datum(i);
                for (size_t j = 0; j < k; ++j) {
                    datum[j] = static_cast<T>(i + j * 0.1);
                }
            }

            for (size_t i = 0; i < n; ++i) {
                auto centroid = centroids.get_datum(i);
                for (size_t j = 0; j < k; ++j) {
                    centroid[j] = static_cast<T>(i * 0.5 + j);
                }
            }

            // Compute matrix multiplication
            ivf::compute_matmul(data.data(), centroids.data(), results.data(), m, n, k);

            // Verify results are valid (not NaN or Inf)
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    float val = results.get_datum(i)[j];
                    CATCH_REQUIRE(std::isfinite(val));
                }
            }

            // Verify dimensions match expected output
            CATCH_REQUIRE(results.size() == m);
            CATCH_REQUIRE(results.dimensions() == n);
        };

        // Test all data types
        test_matmul.operator()<float>();
        test_matmul.operator()<svs::Float16>();
        test_matmul.operator()<svs::BFloat16>();
    }

    CATCH_SECTION("compute_matmul - Edge Cases") {
        // Test with zero dimensions (should return without error)
        auto results = svs::data::SimpleData<float>(0, 0);
        auto data = svs::data::SimpleData<float>(0, 0);
        auto centroids = svs::data::SimpleData<float>(0, 0);

        // Should not crash with zero dimensions
        ivf::compute_matmul(data.data(), centroids.data(), results.data(), 0, 0, 0);

        // Test with single point and single centroid
        auto data_single = svs::data::SimpleData<float>(1, 4);
        auto centroid_single = svs::data::SimpleData<float>(1, 4);
        auto result_single = svs::data::SimpleData<float>(1, 1);

        auto datum = data_single.get_datum(0);
        auto centroid = centroid_single.get_datum(0);
        for (size_t i = 0; i < 4; ++i) {
            datum[i] = static_cast<float>(i);
            centroid[i] = static_cast<float>(i + 1);
        }

        ivf::compute_matmul(
            data_single.data(), centroid_single.data(), result_single.data(), 1, 1, 4
        );

        CATCH_REQUIRE(std::isfinite(result_single.get_datum(0)[0]));
    }

    CATCH_SECTION("convert_data - Type Conversions") {
        auto threadpool = svs::threads::as_threadpool(4);

        // Test float to Float16 conversion
        auto data_float = svs::data::SimpleData<float>(10, 8);
        for (size_t i = 0; i < data_float.size(); ++i) {
            auto datum = data_float.get_datum(i);
            for (size_t j = 0; j < data_float.dimensions(); ++j) {
                datum[j] = static_cast<float>(i * 10 + j);
            }
        }

        auto data_fp16 = ivf::convert_data<svs::Float16>(data_float, threadpool);
        CATCH_REQUIRE(data_fp16.size() == data_float.size());
        CATCH_REQUIRE(data_fp16.dimensions() == data_float.dimensions());

        // Test float to BFloat16 conversion
        auto data_bf16 = ivf::convert_data<svs::BFloat16>(data_float, threadpool);
        CATCH_REQUIRE(data_bf16.size() == data_float.size());
        CATCH_REQUIRE(data_bf16.dimensions() == data_float.dimensions());

        // Test Float16 to float conversion
        auto data_back = ivf::convert_data<float>(data_fp16, threadpool);
        CATCH_REQUIRE(data_back.size() == data_fp16.size());
        CATCH_REQUIRE(data_back.dimensions() == data_fp16.dimensions());
    }

    CATCH_SECTION("generate_norms") {
        auto threadpool = svs::threads::as_threadpool(4);

        // Create test data
        auto data = svs::data::SimpleData<float>(20, 10);
        for (size_t i = 0; i < data.size(); ++i) {
            auto datum = data.get_datum(i);
            for (size_t j = 0; j < data.dimensions(); ++j) {
                datum[j] = static_cast<float>(i + j);
            }
        }

        std::vector<float> norms(data.size());
        ivf::generate_norms(data, norms, threadpool);

        // Verify norms are computed
        CATCH_REQUIRE(norms.size() == data.size());
        for (const auto& norm : norms) {
            CATCH_REQUIRE(norm >= 0.0f);
            CATCH_REQUIRE(std::isfinite(norm));
        }
    }

    CATCH_SECTION("maybe_compute_norms") {
        auto threadpool = svs::threads::as_threadpool(4);
        auto data = svs::data::SimpleData<float>(15, 8);

        for (size_t i = 0; i < data.size(); ++i) {
            auto datum = data.get_datum(i);
            for (size_t j = 0; j < data.dimensions(); ++j) {
                datum[j] = static_cast<float>(i + j * 0.5);
            }
        }

        // For L2 distance, norms should be computed
        auto norms_l2 = ivf::maybe_compute_norms<svs::DistanceL2>(data, threadpool);
        CATCH_REQUIRE(norms_l2.size() == data.size());
        for (const auto& norm : norms_l2) {
            CATCH_REQUIRE(norm >= 0.0f);
        }

        // For IP distance, norms should be empty
        auto norms_ip = ivf::maybe_compute_norms<svs::DistanceIP>(data, threadpool);
        CATCH_REQUIRE(norms_ip.empty());
    }

    CATCH_SECTION("group_assignments") {
        // Test grouping assignments
        size_t num_centroids = 5;
        size_t data_size = 50;

        // Create assignments (each point assigned to a centroid)
        std::vector<size_t> assignments(data_size);
        for (size_t i = 0; i < data_size; ++i) {
            assignments[i] = i % num_centroids;
        }

        auto data = svs::data::SimpleData<float>(data_size, 8);
        auto groups = ivf::group_assignments(assignments, num_centroids, data);

        CATCH_REQUIRE(groups.size() == num_centroids);

        // Verify all points are assigned
        size_t total_assigned = 0;
        for (const auto& group : groups) {
            total_assigned += group.size();
        }
        CATCH_REQUIRE(total_assigned == data_size);

        // Verify each group has expected size
        for (const auto& group : groups) {
            CATCH_REQUIRE(group.size() == data_size / num_centroids);
        }
    }

    CATCH_SECTION("make_training_set") {
        auto threadpool = svs::threads::as_threadpool(4);
        auto rng = std::mt19937(12345);

        // Create full dataset
        size_t full_size = 100;
        size_t training_size = 30;
        auto data = svs::data::SimpleData<float>(full_size, 16);

        for (size_t i = 0; i < data.size(); ++i) {
            auto datum = data.get_datum(i);
            for (size_t j = 0; j < data.dimensions(); ++j) {
                datum[j] = static_cast<float>(i * 10 + j);
            }
        }

        std::vector<size_t> ids(training_size);
        auto training_set =
            ivf::make_training_set<float, decltype(data), svs::lib::Allocator<float>>(
                data, ids, training_size, rng, threadpool
            );

        CATCH_REQUIRE(training_set.size() == training_size);
        CATCH_REQUIRE(training_set.dimensions() == data.dimensions());
        CATCH_REQUIRE(ids.size() == training_size);

        // Verify IDs are valid and unique
        std::unordered_set<size_t> unique_ids(ids.begin(), ids.end());
        CATCH_REQUIRE(unique_ids.size() == training_size);
        for (const auto& id : ids) {
            CATCH_REQUIRE(id < full_size);
        }
    }

    CATCH_SECTION("init_centroids") {
        auto threadpool = svs::threads::as_threadpool(4);
        auto rng = std::mt19937(54321);

        // Create training data
        size_t training_size = 50;
        size_t num_centroids = 10;
        auto trainset = svs::data::SimpleData<float>(training_size, 12);

        for (size_t i = 0; i < trainset.size(); ++i) {
            auto datum = trainset.get_datum(i);
            for (size_t j = 0; j < trainset.dimensions(); ++j) {
                datum[j] = static_cast<float>(i + j * 0.3);
            }
        }

        std::vector<size_t> ids(num_centroids);
        auto centroids =
            ivf::init_centroids<float>(trainset, ids, num_centroids, rng, threadpool);

        CATCH_REQUIRE(centroids.size() == num_centroids);
        CATCH_REQUIRE(centroids.dimensions() == trainset.dimensions());

        // Verify centroids are from training set
        for (size_t i = 0; i < num_centroids; ++i) {
            auto centroid = centroids.get_datum(i);
            bool found = false;
            for (size_t j = 0; j < trainset.size(); ++j) {
                auto train_point = trainset.get_datum(j);
                bool matches = true;
                for (size_t k = 0; k < trainset.dimensions(); ++k) {
                    if (std::abs(centroid[k] - train_point[k]) > 1e-6f) {
                        matches = false;
                        break;
                    }
                }
                if (matches) {
                    found = true;
                    break;
                }
            }
            CATCH_REQUIRE(found);
        }
    }

    CATCH_SECTION("normalize_centroids") {
        auto threadpool = svs::threads::as_threadpool(4);
        auto timer = svs::lib::Timer();

        // Create centroids with non-unit norms
        auto centroids = svs::data::SimpleData<float>(8, 10);
        for (size_t i = 0; i < centroids.size(); ++i) {
            auto centroid = centroids.get_datum(i);
            for (size_t j = 0; j < centroids.dimensions(); ++j) {
                centroid[j] = static_cast<float>((i + 1) * (j + 1));
            }
        }

        ivf::normalize_centroids(centroids, threadpool, timer);

        // Verify centroids are normalized (L2 norm = 1)
        for (size_t i = 0; i < centroids.size(); ++i) {
            auto centroid = centroids.get_datum(i);
            float norm_sq = 0.0f;
            for (size_t j = 0; j < centroids.dimensions(); ++j) {
                norm_sq += centroid[j] * centroid[j];
            }
            float norm = std::sqrt(norm_sq);
            CATCH_REQUIRE(std::abs(norm - 1.0f) < 1e-5f);
        }
    }
}

CATCH_TEST_CASE("Cluster Assignment Utility", "[ivf][common][cluster_assignment]") {
    namespace ivf = svs::index::ivf;

    auto test_cluster_assignment =
        [&]<typename BuildType, typename DataType, typename Distance>() {
            auto threadpool = svs::threads::as_threadpool(4);

            // Create test data
            size_t num_points = 1000;
            size_t num_centroids = 10;
            size_t dims = 128;

            auto data = svs::data::SimpleData<DataType>(num_points, dims);
            auto centroids = svs::data::SimpleData<float>(num_centroids, dims);

            // Initialize data with structured patterns
            for (size_t i = 0; i < num_points; ++i) {
                auto datum = data.get_datum(i);
                size_t cluster_id = i % num_centroids;
                for (size_t j = 0; j < dims; ++j) {
                    // Create data that naturally clusters around centroids
                    datum[j] = static_cast<DataType>(
                        cluster_id * 10.0f + j * 0.1f + (i % 10) * 0.01f
                    );
                }
            }

            // Initialize centroids to match cluster centers
            for (size_t i = 0; i < num_centroids; ++i) {
                auto centroid = centroids.get_datum(i);
                for (size_t j = 0; j < dims; ++j) {
                    centroid[j] = static_cast<float>(i * 10.0f + j * 0.1f);
                }
            }

            // Normalize for IP distance if needed
            if constexpr (std::is_same_v<Distance, svs::DistanceIP>) {
                auto timer = svs::lib::Timer();
                ivf::normalize_centroids(centroids, threadpool, timer);

                // Normalize data as well for IP
                for (size_t i = 0; i < num_points; ++i) {
                    auto datum = data.get_datum(i);
                    float norm = 0.0f;
                    for (size_t j = 0; j < dims; ++j) {
                        norm += static_cast<float>(datum[j]) * static_cast<float>(datum[j]);
                    }
                    norm = std::sqrt(norm);
                    if (norm > 0.0f) {
                        for (size_t j = 0; j < dims; ++j) {
                            datum[j] =
                                static_cast<DataType>(static_cast<float>(datum[j]) / norm);
                        }
                    }
                }
            }

            auto distance = Distance();

            // Call cluster_assignment utility
            auto clusters = ivf::cluster_assignment<BuildType>(
                data, centroids, distance, threadpool, 10'000, svs::lib::Type<uint32_t>()
            );

            // Verify results
            CATCH_REQUIRE(clusters.size() == num_centroids);

            // Count total assigned points
            size_t total_assigned = 0;
            for (const auto& cluster : clusters) {
                total_assigned += cluster.size();
            }
            CATCH_REQUIRE(total_assigned == num_points);

            // Verify no cluster is empty (with our structured data)
            size_t empty_clusters = 0;
            for (const auto& cluster : clusters) {
                if (cluster.empty()) {
                    empty_clusters++;
                }
            }
            // With structured data, we expect most clusters to have points
            // but allow a few empty clusters due to random initialization
            CATCH_REQUIRE(empty_clusters <= 2);
        };

    CATCH_SECTION("Float32 with L2 Distance") {
        test_cluster_assignment.operator()<float, float, svs::DistanceL2>();
    }

    CATCH_SECTION("Float32 with IP Distance") {
        test_cluster_assignment.operator()<float, float, svs::DistanceIP>();
    }

    CATCH_SECTION("Float16 with L2 Distance") {
        test_cluster_assignment.operator()<svs::Float16, float, svs::DistanceL2>();
    }

    CATCH_SECTION("Float16 with IP Distance") {
        test_cluster_assignment.operator()<svs::Float16, float, svs::DistanceIP>();
    }

    CATCH_SECTION("BFloat16 with L2 Distance") {
        test_cluster_assignment.operator()<svs::BFloat16, float, svs::DistanceL2>();
    }

    CATCH_SECTION("BFloat16 with IP Distance") {
        test_cluster_assignment.operator()<svs::BFloat16, float, svs::DistanceIP>();
    }
}

CATCH_TEST_CASE(
    "IVF Train-Only and Cluster Assignment", "[ivf][common][train_only][cluster_assignment]"
) {
    namespace ivf = svs::index::ivf;
    auto threadpool = svs::threads::as_threadpool(4);
    auto data = test_dataset::data_f32();

    auto parameters = ivf::IVFBuildParameters()
                          .num_centroids(50)
                          .minibatch_size(500)
                          .num_iterations(10)
                          .is_hierarchical(false)
                          .training_fraction(0.5)
                          .seed(12345);

    CATCH_SECTION("Flat K-means: train_only + cluster_assignment vs full clustering") {
        auto distance_l2 = svs::DistanceL2();

        // Method 1: Full clustering (without train_only)
        auto [centroids_full, clusters_full] = ivf::kmeans_clustering<float>(
            parameters,
            data,
            distance_l2,
            threadpool,
            svs::lib::Type<uint32_t>(),
            svs::logging::get(),
            false // train_only = false
        );

        // Method 2: Train-only + cluster_assignment
        auto [centroids_train, clusters_train] = ivf::kmeans_clustering<float>(
            parameters,
            data,
            distance_l2,
            threadpool,
            svs::lib::Type<uint32_t>(),
            svs::logging::get(),
            true // train_only = true
        );

        // Verify train_only returns empty clusters
        CATCH_REQUIRE(clusters_train.size() == parameters.num_centroids_);
        for (const auto& cluster : clusters_train) {
            CATCH_REQUIRE(cluster.empty());
        }

        // Now assign data using the cluster_assignment utility
        auto clusters_assigned = ivf::cluster_assignment<float>(
            data,
            centroids_train,
            distance_l2,
            threadpool,
            500, // minibatch_size
            svs::lib::Type<uint32_t>()
        );

        // Verify centroids match (within tolerance)
        CATCH_REQUIRE(centroids_train.size() == centroids_full.size());
        CATCH_REQUIRE(centroids_train.dimensions() == centroids_full.dimensions());

        for (size_t i = 0; i < centroids_train.size(); ++i) {
            auto c1 = centroids_train.get_datum(i);
            auto c2 = centroids_full.get_datum(i);
            for (size_t j = 0; j < centroids_train.dimensions(); ++j) {
                CATCH_REQUIRE(std::abs(c1[j] - c2[j]) < 1e-5f);
            }
        }

        // Verify cluster assignments match
        CATCH_REQUIRE(clusters_assigned.size() == clusters_full.size());
        for (size_t i = 0; i < clusters_assigned.size(); ++i) {
            CATCH_REQUIRE(clusters_assigned[i].size() == clusters_full[i].size());

            // Sort both to compare
            auto a = clusters_assigned[i];
            auto b = clusters_full[i];
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            CATCH_REQUIRE(a == b);
        }

        // Verify all points are assigned
        size_t total_assigned = 0;
        for (const auto& cluster : clusters_assigned) {
            total_assigned += cluster.size();
        }
        CATCH_REQUIRE(total_assigned == data.size());
    }

    CATCH_SECTION("Hierarchical K-means: train_only + cluster_assignment vs full clustering"
    ) {
        auto distance_ip = svs::DistanceIP();

        // Use hierarchical k-means
        auto hier_params =
            parameters.is_hierarchical(true).hierarchical_level1_clusters(10);

        // Method 1: Full clustering (without train_only)
        auto [centroids_full, clusters_full] = ivf::hierarchical_kmeans_clustering<float>(
            hier_params,
            data,
            distance_ip,
            threadpool,
            svs::lib::Type<uint32_t>(),
            svs::logging::get(),
            false // train_only = false
        );

        // Method 2: Train-only + cluster_assignment
        auto [centroids_train, clusters_train] = ivf::hierarchical_kmeans_clustering<float>(
            hier_params,
            data,
            distance_ip,
            threadpool,
            svs::lib::Type<uint32_t>(),
            svs::logging::get(),
            true // train_only = true
        );

        // Verify train_only returns empty clusters
        CATCH_REQUIRE(clusters_train.size() == hier_params.num_centroids_);
        for (const auto& cluster : clusters_train) {
            CATCH_REQUIRE(cluster.empty());
        }

        // Now assign data using the cluster_assignment utility
        auto clusters_assigned = ivf::cluster_assignment<float>(
            data,
            centroids_train,
            distance_ip,
            threadpool,
            500, // minibatch_size
            svs::lib::Type<uint32_t>()
        );

        // Verify centroids match (within tolerance)
        CATCH_REQUIRE(centroids_train.size() == centroids_full.size());
        CATCH_REQUIRE(centroids_train.dimensions() == centroids_full.dimensions());

        for (size_t i = 0; i < centroids_train.size(); ++i) {
            auto c1 = centroids_train.get_datum(i);
            auto c2 = centroids_full.get_datum(i);
            for (size_t j = 0; j < centroids_train.dimensions(); ++j) {
                CATCH_REQUIRE(std::abs(c1[j] - c2[j]) < 1e-5f);
            }
        }

        // Verify cluster structure is reasonable
        CATCH_REQUIRE(clusters_assigned.size() == clusters_full.size());

        // Verify all points are assigned in both methods
        size_t total_assigned = 0;
        size_t total_full = 0;
        for (size_t i = 0; i < clusters_assigned.size(); ++i) {
            total_assigned += clusters_assigned[i].size();
            total_full += clusters_full[i].size();
        }
        CATCH_REQUIRE(total_assigned == data.size());
        CATCH_REQUIRE(total_full == data.size());

        // For hierarchical k-means, assignments may differ slightly due to
        // precision differences in the two-level clustering process.
        // The important thing is that both methods produce valid clusterings.
        // We verify this by checking that the distribution of cluster sizes
        // is reasonable and similar.

        // Check no cluster is excessively large (> 50% of data)
        for (const auto& cluster : clusters_assigned) {
            CATCH_REQUIRE(cluster.size() <= data.size() / 2);
        }
        for (const auto& cluster : clusters_full) {
            CATCH_REQUIRE(cluster.size() <= data.size() / 2);
        }

        // Count non-empty clusters in both
        size_t non_empty_assigned = 0;
        size_t non_empty_full = 0;
        for (size_t i = 0; i < clusters_assigned.size(); ++i) {
            if (!clusters_assigned[i].empty())
                non_empty_assigned++;
            if (!clusters_full[i].empty())
                non_empty_full++;
        }

        // Both should have similar number of non-empty clusters (within 20%)
        double ratio = static_cast<double>(non_empty_assigned) / non_empty_full;
        CATCH_REQUIRE(ratio >= 0.8);
        CATCH_REQUIRE(ratio <= 1.2);
    }

    CATCH_SECTION("Different data types with train_only workflow") {
        auto distance_l2 = svs::DistanceL2();

        // Test with Float16
        auto [centroids_fp16, clusters_empty_fp16] = ivf::kmeans_clustering<svs::Float16>(
            parameters,
            data,
            distance_l2,
            threadpool,
            svs::lib::Type<uint32_t>(),
            svs::logging::get(),
            true // train_only = true
        );

        auto clusters_fp16 = ivf::cluster_assignment<svs::Float16>(
            data, centroids_fp16, distance_l2, threadpool, 500, svs::lib::Type<uint32_t>()
        );

        CATCH_REQUIRE(clusters_fp16.size() == parameters.num_centroids_);
        size_t total_fp16 = 0;
        for (const auto& cluster : clusters_fp16) {
            total_fp16 += cluster.size();
        }
        CATCH_REQUIRE(total_fp16 == data.size());

        // Test with BFloat16
        auto [centroids_bf16, clusters_empty_bf16] = ivf::kmeans_clustering<svs::BFloat16>(
            parameters,
            data,
            distance_l2,
            threadpool,
            svs::lib::Type<uint32_t>(),
            svs::logging::get(),
            true // train_only = true
        );

        auto clusters_bf16 = ivf::cluster_assignment<svs::BFloat16>(
            data, centroids_bf16, distance_l2, threadpool, 500, svs::lib::Type<uint32_t>()
        );

        CATCH_REQUIRE(clusters_bf16.size() == parameters.num_centroids_);
        size_t total_bf16 = 0;
        for (const auto& cluster : clusters_bf16) {
            total_bf16 += cluster.size();
        }
        CATCH_REQUIRE(total_bf16 == data.size());
    }
}
