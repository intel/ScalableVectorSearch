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
#include "svs/lib/threads.h"

// stl
#include <algorithm>
#include <numeric>
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
