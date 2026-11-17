/*
 * Copyright 2026 Intel Corporation
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
#include "svs/core/kmeans_training.h"

// svs
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/lib/threads.h"

// catch
#include "catch2/catch_test_macros.hpp"

// stl
#include <cmath>
#include <random>
#include <vector>

namespace {

// Brute-force nearest centroid, used as an oracle independent of centroid_assignment.
template <typename Distance, typename Data>
size_t brute_force_nearest(
    const Data& data, size_t i, const svs::data::SimpleData<float>& centroids
) {
    auto datum = data.get_datum(i);
    size_t best = 0;
    if constexpr (std::is_same_v<Distance, svs::DistanceIP>) {
        float best_score = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < centroids.size(); ++j) {
            auto c = centroids.get_datum(j);
            float dot = 0;
            for (size_t d = 0; d < datum.size(); ++d) {
                dot += datum[d] * c[d];
            }
            if (dot > best_score) {
                best_score = dot;
                best = j;
            }
        }
    } else {
        float best_dist = std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < centroids.size(); ++j) {
            auto c = centroids.get_datum(j);
            float dist = 0;
            for (size_t d = 0; d < datum.size(); ++d) {
                float diff = datum[d] - c[d];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best = j;
            }
        }
    }
    return best;
}

} // namespace

CATCH_TEST_CASE("Shared Kmeans Training - L2 Centroid Update", "[core][kmeans]") {
    namespace kmeans = svs::kmeans;

    // Two well-separated pairs of points, one per cluster, for an unambiguous assignment.
    auto data = svs::data::SimpleData<float>(4, 2);
    data.set_datum(0, std::vector<float>{0.0F, 0.0F});
    data.set_datum(1, std::vector<float>{0.0F, 2.0F});
    data.set_datum(2, std::vector<float>{10.0F, 0.0F});
    data.set_datum(3, std::vector<float>{10.0F, 2.0F});

    auto centroids = svs::data::SimpleData<float>(2, 2);
    centroids.set_datum(0, std::vector<float>{0.0F, 0.0F});
    centroids.set_datum(1, std::vector<float>{10.0F, 0.0F});

    auto matmul_results = svs::data::SimpleData<float>(10, 2);
    auto rng = std::mt19937(0xc0ffee);
    auto threadpool = svs::threads::as_threadpool(2);
    auto timer = svs::lib::Timer();
    auto distance = svs::DistanceL2();

    // centroid_adjustment divides by (count + 1), not count, so this is not a plain mean.
    auto params = kmeans::KMeansTrainingParameters{1, 10};
    auto result = kmeans::kmeans_training(
        params, data, distance, centroids, matmul_results, rng, threadpool, timer
    );

    CATCH_REQUIRE(result.size() == 2);

    auto c0 = result.get_datum(0);
    auto c1 = result.get_datum(1);
    constexpr float tolerance = 1e-4F;

    CATCH_REQUIRE(std::abs(c0[0] - 0.0F) < tolerance);
    CATCH_REQUIRE(std::abs(c0[1] - (2.0F / 3.0F)) < tolerance);

    CATCH_REQUIRE(std::abs(c1[0] - (30.0F / 3.0F)) < tolerance);
    CATCH_REQUIRE(std::abs(c1[1] - (2.0F / 3.0F)) < tolerance);

    // L2 training does not normalize centroids, so they are not unit-norm in general.
    float norm0 = std::sqrt(c0[0] * c0[0] + c0[1] * c0[1]);
    float norm1 = std::sqrt(c1[0] * c1[0] + c1[1] * c1[1]);
    CATCH_REQUIRE(std::abs(norm0 - 1.0F) > tolerance);
    CATCH_REQUIRE(std::abs(norm1 - 1.0F) > tolerance);
}

CATCH_TEST_CASE("Shared Kmeans Training - IP Centroids Are Unit Norm", "[core][kmeans]") {
    namespace kmeans = svs::kmeans;

    auto data = svs::data::SimpleData<float>(6, 3);
    data.set_datum(0, std::vector<float>{1.0F, 2.0F, 3.0F});
    data.set_datum(1, std::vector<float>{2.0F, 1.0F, 4.0F});
    data.set_datum(2, std::vector<float>{3.0F, 3.0F, 1.0F});
    data.set_datum(3, std::vector<float>{-1.0F, -2.0F, 0.5F});
    data.set_datum(4, std::vector<float>{-2.0F, -1.0F, 1.5F});
    data.set_datum(5, std::vector<float>{-3.0F, -0.5F, 2.0F});

    // Non-unit-norm initial centroids, so normalization is confirmed to happen inside.
    auto centroids = svs::data::SimpleData<float>(2, 3);
    centroids.set_datum(0, data.get_datum(0));
    centroids.set_datum(1, data.get_datum(3));

    auto matmul_results = svs::data::SimpleData<float>(6, 2);
    auto rng = std::mt19937(42);
    auto threadpool = svs::threads::as_threadpool(2);
    auto timer = svs::lib::Timer();
    auto distance = svs::DistanceIP();

    auto params = kmeans::KMeansTrainingParameters{3, 6};
    auto result = kmeans::kmeans_training(
        params, data, distance, centroids, matmul_results, rng, threadpool, timer
    );

    constexpr float tolerance = 1e-5F;
    for (size_t i = 0; i < result.size(); ++i) {
        auto c = result.get_datum(i);
        float norm_sq = 0;
        for (size_t d = 0; d < c.size(); ++d) {
            norm_sq += c[d] * c[d];
        }
        CATCH_REQUIRE(std::abs(std::sqrt(norm_sq) - 1.0F) < tolerance);
    }
}

CATCH_TEST_CASE(
    "Shared Kmeans Training - centroid_assignment Agrees With Brute Force", "[core][kmeans]"
) {
    namespace kmeans = svs::kmeans;

    constexpr size_t num_points = 30;
    constexpr size_t num_centroids = 4;
    constexpr size_t dims = 5;

    auto data = svs::data::SimpleData<float>(num_points, dims);
    for (size_t i = 0; i < num_points; ++i) {
        auto datum = data.get_datum(i);
        for (size_t d = 0; d < dims; ++d) {
            // A deterministic, non-trivial pattern; no symmetry that would create ties.
            datum[d] = static_cast<float>((i * 7 + d * 3) % 11) - 5.0F;
        }
    }

    auto centroids = svs::data::SimpleData<float>(num_centroids, dims);
    for (size_t i = 0; i < num_centroids; ++i) {
        auto c = centroids.get_datum(i);
        for (size_t d = 0; d < dims; ++d) {
            c[d] = static_cast<float>((i * 5 + d * 2) % 7) - 3.0F;
        }
    }

    auto threadpool = svs::threads::as_threadpool(2);
    auto timer = svs::lib::Timer();
    auto batch_range = svs::threads::UnitRange<uint64_t>{0, num_points};
    auto matmul_results = svs::data::SimpleData<float>(num_points, num_centroids);

    auto check_distance = [&]<typename Distance>(Distance distance) {
        auto data_norm = kmeans::maybe_compute_norms<Distance>(data, threadpool);
        auto centroids_norm = kmeans::maybe_compute_norms<Distance>(centroids, threadpool);
        auto assignments = std::vector<size_t>(num_points);

        kmeans::centroid_assignment(
            data,
            data_norm,
            batch_range,
            distance,
            centroids,
            centroids_norm,
            assignments,
            matmul_results,
            threadpool,
            timer
        );

        for (size_t i = 0; i < num_points; ++i) {
            size_t expected = brute_force_nearest<Distance>(data, i, centroids);
            CATCH_REQUIRE(assignments[i] == expected);
        }
    };

    check_distance(svs::DistanceL2());
    check_distance(svs::DistanceIP());
}
