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
#include "svs/core/kmeans_emeans.h"

// svs
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/lib/threads.h"

// catch
#include "catch2/catch_test_macros.hpp"

// stl
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
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

// `batch_size` sizes the preallocated matmul_results buffer (minibatch_size), which need
// not equal data.size().
template <typename Distance>
auto run_kmeans(
    const svs::data::SimpleData<float>& data,
    Distance distance,
    svs::data::SimpleData<float>& centroids,
    size_t num_iter,
    size_t batch_size,
    unsigned seed
) {
    namespace kmeans = svs::kmeans;
    auto matmul_results = svs::data::SimpleData<float>(batch_size, centroids.size());
    auto rng = std::mt19937(seed);
    auto threadpool = svs::threads::as_threadpool(2);
    auto timer = svs::lib::Timer();
    auto params = kmeans::KMeansTrainingParameters{num_iter, batch_size};
    return kmeans::kmeans_training(
        params, data, distance, centroids, matmul_results, rng, threadpool, timer
    );
}

} // namespace

CATCH_TEST_CASE("Shared Kmeans Training", "[core][kmeans]") {
    CATCH_SECTION("L2 centroid update divides by count + 1, and does not normalize") {
        auto data = svs::data::SimpleData<float>(4, 2);
        data.set_datum(0, std::vector<float>{0.0F, 0.0F});
        data.set_datum(1, std::vector<float>{0.0F, 2.0F});
        data.set_datum(2, std::vector<float>{10.0F, 0.0F});
        data.set_datum(3, std::vector<float>{10.0F, 2.0F});

        auto centroids = svs::data::SimpleData<float>(2, 2);
        centroids.set_datum(0, std::vector<float>{0.0F, 0.0F});
        centroids.set_datum(1, std::vector<float>{10.0F, 0.0F});

        auto result = run_kmeans(data, svs::DistanceL2(), centroids, 1, 10, 0xc0ffee);
        CATCH_REQUIRE(result.size() == 2);

        auto c0 = result.get_datum(0);
        auto c1 = result.get_datum(1);
        constexpr float tolerance = 1e-4F;
        CATCH_REQUIRE(std::abs(c0[0] - 0.0F) < tolerance);
        CATCH_REQUIRE(std::abs(c0[1] - (2.0F / 3.0F)) < tolerance);
        CATCH_REQUIRE(std::abs(c1[0] - (30.0F / 3.0F)) < tolerance);
        CATCH_REQUIRE(std::abs(c1[1] - (2.0F / 3.0F)) < tolerance);

        float norm0 = std::sqrt(c0[0] * c0[0] + c0[1] * c0[1]);
        float norm1 = std::sqrt(c1[0] * c1[0] + c1[1] * c1[1]);
        CATCH_REQUIRE(std::abs(norm0 - 1.0F) > tolerance);
        CATCH_REQUIRE(std::abs(norm1 - 1.0F) > tolerance);
    }

    CATCH_SECTION("IP centroids come out unit-norm from non-unit-norm initial centroids") {
        auto data = svs::data::SimpleData<float>(6, 3);
        data.set_datum(0, std::vector<float>{1.0F, 2.0F, 3.0F});
        data.set_datum(1, std::vector<float>{2.0F, 1.0F, 4.0F});
        data.set_datum(2, std::vector<float>{3.0F, 3.0F, 1.0F});
        data.set_datum(3, std::vector<float>{-1.0F, -2.0F, 0.5F});
        data.set_datum(4, std::vector<float>{-2.0F, -1.0F, 1.5F});
        data.set_datum(5, std::vector<float>{-3.0F, -0.5F, 2.0F});

        auto centroids = svs::data::SimpleData<float>(2, 3);
        centroids.set_datum(0, data.get_datum(0));
        centroids.set_datum(1, data.get_datum(3));

        auto result = run_kmeans(data, svs::DistanceIP(), centroids, 3, 6, 42);

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

    CATCH_SECTION("centroid_assignment agrees with brute force, for L2 and IP") {
        namespace kmeans = svs::kmeans;
        constexpr size_t num_points = 30;
        constexpr size_t num_centroids = 4;
        constexpr size_t dims = 5;

        auto data = svs::data::SimpleData<float>(num_points, dims);
        for (size_t i = 0; i < num_points; ++i) {
            auto datum = data.get_datum(i);
            for (size_t d = 0; d < dims; ++d) {
                // A deterministic, non-trivial pattern; no symmetry that creates ties.
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
            auto centroids_norm =
                kmeans::maybe_compute_norms<Distance>(centroids, threadpool);
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
}

namespace {

// Three well-separated clusters, far enough apart that nearest-centroid assignment never
// ties and small perturbations in training never cross a cluster boundary.
svs::data::SimpleData<float> make_three_clusters() {
    auto data = svs::data::SimpleData<float>(15, 2);
    std::array<std::array<float, 2>, 3> centers{
        {{0.0F, 0.0F}, {100.0F, 0.0F}, {0.0F, 100.0F}}};
    for (size_t c = 0; c < 3; ++c) {
        for (size_t i = 0; i < 5; ++i) {
            float jitter = static_cast<float>(i) * 0.1F;
            data.set_datum(
                c * 5 + i,
                std::vector<float>{centers[c][0] + jitter, centers[c][1] + jitter}
            );
        }
    }
    return data;
}

svs::kmeans::EMeansParameters small_emeans_parameters() {
    svs::kmeans::EMeansParameters parameters;
    parameters.epochs = 2;
    parameters.batch_size = 3;
    parameters.alpha0 = 0.5;
    parameters.revival_interval = 5;
    parameters.tau = 0.1;
    return parameters;
}

} // namespace

CATCH_TEST_CASE("EMeans Training", "[core][kmeans][emeans]") {
    CATCH_SECTION("determinism for a fixed seed and thread count") {
        auto data = make_three_clusters();
        auto parameters = small_emeans_parameters();

        auto run = [&]() {
            auto rng = std::mt19937(0xc0ffee);
            auto threadpool = svs::threads::as_threadpool(2);
            auto timer = svs::lib::Timer();
            return svs::kmeans::emeans_train(
                parameters, data, size_t{3}, rng, threadpool, timer
            );
        };

        auto result_a = run();
        auto result_b = run();

        CATCH_REQUIRE(result_a.centroids.size() == result_b.centroids.size());
        for (size_t i = 0; i < result_a.centroids.size(); ++i) {
            auto a = result_a.centroids.get_datum(i);
            auto b = result_b.centroids.get_datum(i);
            for (size_t d = 0; d < a.size(); ++d) {
                CATCH_REQUIRE(a[d] == b[d]);
            }
        }
        CATCH_REQUIRE(result_a.objective == result_b.objective);
    }

    CATCH_SECTION("a forced-dead cluster is revived") {
        auto data = make_three_clusters();
        auto parameters = small_emeans_parameters();

        // Four initial centroids: one near each real cluster, plus a fourth parked far
        // from all data so it is never the nearest centroid for any point and stays dead
        // (EMA count near its 1e-10 floor) until revival splits a donor into it.
        auto initial = svs::data::SimpleData<float>(4, 2);
        initial.set_datum(0, std::vector<float>{0.0F, 0.0F});
        initial.set_datum(1, std::vector<float>{100.0F, 0.0F});
        initial.set_datum(2, std::vector<float>{0.0F, 100.0F});
        initial.set_datum(3, std::vector<float>{1000.0F, 1000.0F});

        auto rng = std::mt19937(42);
        auto threadpool = svs::threads::as_threadpool(2);
        auto timer = svs::lib::Timer();
        auto result =
            svs::kmeans::emeans_train(parameters, data, initial, rng, threadpool, timer);

        // Revival splits a donor near the data into this slot; an unrevived cluster would
        // still sit at (1000, 1000), about 1000 from every real cluster center.
        auto final_dead = result.centroids.get_datum(3);
        float nearest_center = std::numeric_limits<float>::max();
        constexpr std::array<std::pair<float, float>, 3> centers{
            {{0.0F, 0.0F}, {100.0F, 0.0F}, {0.0F, 100.0F}}};
        for (auto [cx, cy] : centers) {
            float dx = final_dead[0] - cx;
            float dy = final_dead[1] - cy;
            nearest_center = std::min(nearest_center, dx * dx + dy * dy);
        }
        CATCH_REQUIRE(nearest_center < 10.0F * 10.0F);
    }

    CATCH_SECTION("resolve_emeans_parameters matches hand-computed scaling-law values") {
        // Hand-computed (Python) from Eq. 6/Eq. 7 with the code's exact coefficients.
        svs::kmeans::EMeansParameters small_n_params;
        small_n_params.epochs = 2;
        auto small_n =
            svs::kmeans::resolve_emeans_parameters(100'000, 1'000, small_n_params);
        CATCH_REQUIRE(small_n.batch_size == 4096);
        CATCH_REQUIRE(std::abs(small_n.alpha0 - 0.3191081273388448) < 1e-6);

        // n > 2e6: the batch law's coefficient switches to 0.035/8.
        svs::kmeans::EMeansParameters large_n_params;
        large_n_params.epochs = 2;
        auto large_n =
            svs::kmeans::resolve_emeans_parameters(5'000'000, 1'000, large_n_params);
        CATCH_REQUIRE(large_n.batch_size == 16384);
        CATCH_REQUIRE(std::abs(large_n.alpha0 - 0.19109765404135304) < 1e-6);
    }

    CATCH_SECTION("batch_size not dividing n exercises the tail batch") {
        auto data = make_three_clusters(); // n = 15
        auto parameters = small_emeans_parameters();
        parameters.batch_size = 4; // 15 / 4 = 3 remainder 3: a tail batch of size 3

        auto rng = std::mt19937(0xc0ffee);
        auto threadpool = svs::threads::as_threadpool(2);
        auto timer = svs::lib::Timer();
        auto result =
            svs::kmeans::emeans_train(parameters, data, size_t{3}, rng, threadpool, timer);

        CATCH_REQUIRE(result.centroids.size() == 3);
        CATCH_REQUIRE(std::isfinite(result.objective));
    }

    // Not a general invariant: the last epoch's assignments come from moving centroids, so
    // the closing mean need not dominate the EMA readout in general. Holds here because
    // this toy set's clusters are so well separated that the assignment never wavers.
    CATCH_SECTION(
        "the final mean pass does not raise the objective (well-separated toy set)"
    ) {
        auto data = make_three_clusters();

        auto with_pass_params = small_emeans_parameters();
        with_pass_params.final_mean_pass = true;
        auto without_pass_params = small_emeans_parameters();
        without_pass_params.final_mean_pass = false;

        auto run = [&](const svs::kmeans::EMeansParameters& parameters) {
            auto rng = std::mt19937(0xc0ffee);
            auto threadpool = svs::threads::as_threadpool(2);
            auto timer = svs::lib::Timer();
            return svs::kmeans::emeans_train(
                parameters, data, size_t{3}, rng, threadpool, timer
            );
        };

        auto with_pass = run(with_pass_params);
        auto without_pass = run(without_pass_params);

        CATCH_REQUIRE(with_pass.objective <= without_pass.objective + 1e-6);
    }

    CATCH_SECTION("k > n throws") {
        auto data = svs::data::SimpleData<float>(3, 2);
        data.set_datum(0, std::vector<float>{0.0F, 0.0F});
        data.set_datum(1, std::vector<float>{1.0F, 0.0F});
        data.set_datum(2, std::vector<float>{0.0F, 1.0F});

        auto parameters = small_emeans_parameters();
        auto rng = std::mt19937(0xc0ffee);
        auto threadpool = svs::threads::as_threadpool(2);
        auto timer = svs::lib::Timer();

        CATCH_REQUIRE_THROWS(
            svs::kmeans::emeans_train(parameters, data, size_t{5}, rng, threadpool, timer)
        );
    }
}
