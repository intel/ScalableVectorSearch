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
#include "svs/quantization/gem/codebook.h"
#include "svs/quantization/gem/codebook_timed.h"

// svs
#include "svs/core/data/simple.h"
#include "svs/core/data/vector_set.h"
#include "svs/lib/threads.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace {
namespace gem = svs::quantization::gem;
constexpr float TOL = 1e-4F;

std::vector<float> random_unit_vector(std::mt19937& rng, size_t dim) {
    std::normal_distribution<float> gauss(0.0F, 1.0F);
    std::vector<float> v(dim);
    double norm2 = 0.0;
    for (auto& x : v) {
        x = gauss(rng);
        norm2 += static_cast<double>(x) * x;
    }
    double norm = std::sqrt(norm2);
    for (auto& x : v) {
        x = static_cast<float>(static_cast<double>(x) / norm);
    }
    return v;
}

std::vector<float> make_random_centers(size_t K, size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<float> centers(K * dim);
    for (size_t k = 0; k < K; ++k) {
        auto c = random_unit_vector(rng, dim);
        std::copy(c.begin(), c.end(), centers.begin() + static_cast<ptrdiff_t>(k * dim));
    }
    return centers;
}

// Fine centroids as a SimpleData, laid out row-major from a flat `centers` buffer.
svs::data::SimpleData<float>
centroids_from(const std::vector<float>& centers, size_t K, size_t dim) {
    svs::data::SimpleData<float> centroids(K, dim);
    for (size_t k = 0; k < K; ++k) {
        centroids.set_datum(k, std::span<const float>(centers.data() + k * dim, dim));
    }
    return centroids;
}

// Each token perturbs a center with a large margin, so even an approximate assigner agrees.
svs::data::VectorSetData<float> make_clustered_corpus(
    const std::vector<float>& centers,
    size_t K,
    size_t dim,
    const std::vector<size_t>& cardinalities,
    uint32_t seed,
    float noise
) {
    std::mt19937 rng(seed);
    size_t num_passages = cardinalities.size();
    size_t total_tokens = 0;
    for (auto c : cardinalities) {
        total_tokens += c;
    }
    svs::data::SimpleData<float> token_matrix(total_tokens, dim);
    std::vector<uint64_t> offsets(num_passages + 1, 0);
    std::uniform_int_distribution<size_t> pick(0, K - 1);
    std::normal_distribution<float> noise_dist(0.0F, noise);

    size_t global = 0;
    for (size_t p = 0; p < num_passages; ++p) {
        offsets[p + 1] = offsets[p] + cardinalities[p];
        for (size_t t = 0; t < cardinalities[p]; ++t) {
            size_t k = pick(rng);
            std::vector<float> row(dim);
            double norm2 = 0.0;
            for (size_t d = 0; d < dim; ++d) {
                row[d] = centers[k * dim + d] + noise_dist(rng);
                norm2 += static_cast<double>(row[d]) * row[d];
            }
            double norm = std::sqrt(norm2);
            for (auto& x : row) {
                x = static_cast<float>(static_cast<double>(x) / norm);
            }
            token_matrix.set_datum(global, row);
            ++global;
        }
    }
    return svs::data::VectorSetData<float>(std::move(token_matrix), std::move(offsets));
}

uint32_t brute_force_best(
    std::span<const float> token, const svs::data::SimpleData<float>& centroids
) {
    uint32_t best = 0;
    float best_dot = -std::numeric_limits<float>::infinity();
    for (size_t j = 0, jmax = centroids.size(); j < jmax; ++j) {
        auto c = centroids.get_datum(j);
        float dot = 0.0F;
        for (size_t d = 0, dmax = token.size(); d < dmax; ++d) {
            dot += token[d] * c[d];
        }
        if (dot > best_dot) {
            best_dot = dot;
            best = static_cast<uint32_t>(j);
        }
    }
    return best;
}

std::vector<uint32_t> flatten_ids(const svs::data::VectorSetData<uint32_t>& ids) {
    std::vector<uint32_t> out;
    out.reserve(ids.total_tokens());
    for (size_t p = 0, pmax = ids.size(); p < pmax; ++p) {
        auto doc = ids.get_datum(p);
        for (size_t t = 0, tmax = doc.cardinality(); t < tmax; ++t) {
            out.push_back(doc.token(t)[0]);
        }
    }
    return out;
}

svs::data::VectorSetData<uint32_t>
rebuild_ids(const std::vector<uint32_t>& flat, std::vector<uint64_t> offsets) {
    svs::data::SimpleData<uint32_t> storage(flat.size(), 1);
    for (size_t i = 0; i < flat.size(); ++i) {
        storage.set_datum(i, std::vector<uint32_t>{flat[i]});
    }
    return svs::data::VectorSetData<uint32_t>(std::move(storage), std::move(offsets));
}
} // namespace

CATCH_TEST_CASE("GEM Codebook - sizing rules", "[quantization][gem]") {
    CATCH_SECTION("colbert_num_partitions") {
        // Exact powers of two: 16 * sqrt(1) == 16, 16 * sqrt(4) == 32.
        CATCH_REQUIRE(gem::colbert_num_partitions(1) == 16);
        CATCH_REQUIRE(gem::colbert_num_partitions(4) == 32);
        // One token below/above the num_tokens=4 boundary (raw crosses exactly 32 at n=4).
        CATCH_REQUIRE(gem::colbert_num_partitions(3) == 16);
        CATCH_REQUIRE(gem::colbert_num_partitions(5) == 32);
        CATCH_REQUIRE_THROWS(gem::colbert_num_partitions(0));
    }

    CATCH_SECTION("colbert_sample_size") {
        // Small corpora: the formula's raw value exceeds num_passages, so it is capped.
        CATCH_REQUIRE(gem::colbert_sample_size(1) == 1);
        CATCH_REQUIRE(gem::colbert_sample_size(10) == 10);
        // Large corpus: 1 + floor(16 * sqrt(120 * 1'000'000)) is well under num_passages.
        size_t num_passages = 1'000'000;
        size_t expected =
            1 + static_cast<size_t>(
                    std::floor(16.0 * std::sqrt(120.0 * static_cast<double>(num_passages)))
                );
        CATCH_REQUIRE(gem::colbert_sample_size(num_passages) == expected);
        CATCH_REQUIRE(expected < num_passages);
        CATCH_REQUIRE_THROWS(gem::colbert_sample_size(0));
        // Perfect-square boundary (921'600 == 960^2); integer sqrt must not be off by one.
        CATCH_REQUIRE(gem::detail::isqrt_floor(921'600) == 960);
        CATCH_REQUIRE(gem::detail::isqrt_floor(921'599) == 959);
        CATCH_REQUIRE(gem::detail::isqrt_floor(921'601) == 960);
    }
}

CATCH_TEST_CASE(
    "GEM Codebook - plain mean update differs from (old + sum) / (count + 1)",
    "[quantization][gem]"
) {
    auto data = svs::data::SimpleData<float>(4, 2);
    data.set_datum(0, std::vector<float>{0.0F, 0.0F});
    data.set_datum(1, std::vector<float>{0.0F, 2.0F});
    data.set_datum(2, std::vector<float>{10.0F, 0.0F});
    data.set_datum(3, std::vector<float>{10.0F, 2.0F});
    auto centroids = svs::data::SimpleData<float>(2, 2);
    centroids.set_datum(0, std::vector<float>{0.0F, 0.0F});
    centroids.set_datum(1, std::vector<float>{10.0F, 0.0F});

    std::vector<size_t> assignments{0, 0, 1, 1};
    std::vector<size_t> counts(2);
    auto threadpool = svs::threads::as_threadpool(2);
    gem::detail::plain_mean_update(data, centroids, assignments, counts, threadpool);

    auto c0 = centroids.get_datum(0);
    auto c1 = centroids.get_datum(1);
    // Plain mean: cluster 0 -> (0,1); cluster 1 -> (10,1).
    CATCH_REQUIRE(std::abs(c0[0] - 0.0F) < TOL);
    CATCH_REQUIRE(std::abs(c0[1] - 1.0F) < TOL);
    CATCH_REQUIRE(std::abs(c1[0] - 10.0F) < TOL);
    CATCH_REQUIRE(std::abs(c1[1] - 1.0F) < TOL);
    // (old + sum) / (count + 1) would give (0 + 0 + 2) / 3 == 0.667, not the plain mean 1.
    CATCH_REQUIRE(std::abs(c0[1] - (2.0F / 3.0F)) > TOL);
}

// One small corpus drives the whole pipeline; each SECTION below asserts a different group
// of properties on the shared `result`/`corpus` instead of rebuilding its own input. Cases
// that genuinely need different input (thread count, byte budget) get their own SECTION
// too, still off this one corpus; only parameters/sizing that need a different seed/shape
// get their own TEST_CASE (sizing rules, plain-mean-update, failure paths).
CATCH_TEST_CASE("GEM Codebook - quantize pipeline", "[quantization][gem]") {
    constexpr size_t DIM = 16;
    constexpr size_t NUM_PASSAGES = 100;
    auto centers = make_random_centers(16, DIM, 0x9999);
    auto corpus = make_clustered_corpus(
        centers, 16, DIM, std::vector<size_t>(NUM_PASSAGES, 2), 0x1357, 0.1F
    );
    gem::CodebookParameters parameters; // num_centroids == 0: use the default rule.
    auto threadpool = svs::threads::as_threadpool(2);
    auto result = gem::quantize(corpus, parameters, threadpool);
    size_t expected_K = gem::colbert_num_partitions(NUM_PASSAGES * 2);

    CATCH_SECTION(
        "Pipeline wiring, default coarse rule, unit-norm, non-increasing objective"
    ) {
        CATCH_REQUIRE(result.K == expected_K);
        CATCH_REQUIRE(result.centroids.size() == expected_K);
        CATCH_REQUIRE(result.centroids.dimensions() == DIM);
        CATCH_REQUIRE(result.fine_centroids->K == expected_K);
        CATCH_REQUIRE(result.fine_centroids->dim == DIM);
        CATCH_REQUIRE(result.centroid_table->K == expected_K);
        CATCH_REQUIRE(result.ids.total_tokens() == NUM_PASSAGES * 2);
        for (size_t p = 0; p < NUM_PASSAGES; ++p) {
            CATCH_REQUIRE(result.ids.cardinality(p) == 2);
        }
        // Default num_coarse rule (num_coarse == 0 in `parameters`).
        CATCH_REQUIRE(result.num_coarse == std::max<size_t>(1, expected_K * 5 / 32));
        CATCH_REQUIRE(result.coarse_of_fine.size() == expected_K);
        for (auto c : result.coarse_of_fine) {
            CATCH_REQUIRE(c < result.num_coarse);
        }
        for (size_t i = 0; i < result.centroids.size(); ++i) {
            auto c = result.centroids.get_datum(i);
            float norm_sq = 0.0F;
            for (size_t d = 0; d < c.size(); ++d) {
                norm_sq += c[d] * c[d];
            }
            CATCH_REQUIRE(std::abs(std::sqrt(norm_sq) - 1.0F) < 1e-4F);
        }

        // Same seed and sampling: more k-means passes cannot raise the final objective.
        auto one_iter_params = parameters;
        one_iter_params.kmeans_iterations = 1;
        auto one_iter = gem::train_codebook(corpus, one_iter_params, threadpool);
        auto four_iter_params = parameters;
        four_iter_params.kmeans_iterations = 4;
        auto four_iter = gem::train_codebook(corpus, four_iter_params, threadpool);
        CATCH_REQUIRE(four_iter.objective <= one_iter.objective + 1e-3);
    }

    CATCH_SECTION("coarse_clusters: explicit num_coarse is honored") {
        gem::CodebookParameters explicit_params = parameters;
        explicit_params.num_coarse = 5;
        auto explicit_coarse =
            gem::coarse_clusters(result.centroids, explicit_params, threadpool);
        CATCH_REQUIRE(explicit_coarse.num_coarse == 5);
        for (auto c : explicit_coarse.coarse_of_fine) {
            CATCH_REQUIRE(c < 5);
        }
    }

    CATCH_SECTION(
        "assign_tokens agrees with brute force and the mismatch checker, both assigners"
    ) {
        for (auto assigner :
             {gem::Assigner::ExactGemm, gem::Assigner::InnerProductMatrix}) {
            gem::CodebookParameters assign_params = parameters;
            assign_params.assigner = assigner;
            auto ids =
                gem::assign_tokens(corpus, result.centroids, assign_params, threadpool);
            auto flat = flatten_ids(ids);

            size_t global = 0;
            for (size_t p = 0; p < corpus.size(); ++p) {
                auto doc = corpus.get_datum(p);
                for (size_t t = 0; t < doc.cardinality(); ++t) {
                    CATCH_REQUIRE(
                        flat[global++] == brute_force_best(doc.token(t), result.centroids)
                    );
                }
            }

            auto report =
                gem::count_assignment_mismatches(corpus, result.centroids, ids, 150, 99);
            CATCH_REQUIRE(report.mismatches == 0);
            CATCH_REQUIRE(report.sample_size == 150);

            // Corrupt a few ids and confirm the checker notices.
            auto offsets =
                std::vector<uint64_t>(ids.offsets().begin(), ids.offsets().end());
            auto corrupted = flat;
            for (size_t i = 0; i < corrupted.size(); i += 37) {
                corrupted[i] = static_cast<uint32_t>((corrupted[i] + 1) % expected_K);
            }
            auto bad_report = gem::count_assignment_mismatches(
                corpus, result.centroids, rebuild_ids(corrupted, offsets), 2000, 99
            );
            CATCH_REQUIRE(bad_report.mismatches > 0);
        }
    }

    // Thread partitioning and batch sizing happen before the per-assigner flush branch, so
    // neither depends on `assigner`; one (the default, InnerProductMatrix) is enough.
    CATCH_SECTION("assign_tokens: thread-count independence and byte-budget batching") {
        auto pool1 = svs::threads::as_threadpool(1);
        auto pool4 = svs::threads::as_threadpool(4);
        auto flat1 =
            flatten_ids(gem::assign_tokens(corpus, result.centroids, parameters, pool1));
        CATCH_REQUIRE(
            flat1 ==
            flatten_ids(gem::assign_tokens(corpus, result.centroids, parameters, pool4))
        );

        // expected_K(128)*4 bytes/row: 256 forces a 1-row batch, 1536 forces a 3-row batch.
        for (size_t budget : {size_t{256}, size_t{1536}}) {
            gem::CodebookParameters tiny = parameters;
            tiny.assign_buffer_bytes = budget;
            CATCH_REQUIRE(
                flatten_ids(gem::assign_tokens(corpus, result.centroids, tiny, pool1)) ==
                flat1
            );
        }
    }

    CATCH_SECTION(
        "quantize_timed reproduces quantize's result exactly and times every phase"
    ) {
        auto [timed, timings] = gem::quantize_timed(corpus, parameters, threadpool);
        CATCH_REQUIRE(timed.K == result.K);
        CATCH_REQUIRE(timed.num_coarse == result.num_coarse);
        CATCH_REQUIRE(timed.objective == result.objective);
        CATCH_REQUIRE(timed.num_sampled_passages == result.num_sampled_passages);
        CATCH_REQUIRE(timed.num_training_tokens == result.num_training_tokens);
        CATCH_REQUIRE(timed.coarse_of_fine == result.coarse_of_fine);
        CATCH_REQUIRE(flatten_ids(timed.ids) == flatten_ids(result.ids));
        for (size_t i = 0; i < timed.centroids.size(); ++i) {
            auto a = timed.centroids.get_datum(i);
            auto b = result.centroids.get_datum(i);
            for (size_t d = 0; d < a.size(); ++d) {
                CATCH_REQUIRE(a[d] == b[d]);
            }
        }
        CATCH_REQUIRE(timings.train_iterations.size() == parameters.kmeans_iterations);
        CATCH_REQUIRE(timings.total >= 0.0);
        CATCH_REQUIRE(timings.sample >= 0.0);
        CATCH_REQUIRE(timings.train >= 0.0);
        CATCH_REQUIRE(timings.normalize >= 0.0);
        CATCH_REQUIRE(timings.assign >= 0.0);
        CATCH_REQUIRE(timings.ground_table >= 0.0);
        CATCH_REQUIRE(timings.coarse >= 0.0);
    }
}

CATCH_TEST_CASE("GEM Codebook - failure paths", "[quantization][gem]") {
    constexpr size_t DIM = 8;
    constexpr size_t K = 4;
    auto threadpool = svs::threads::as_threadpool(1);
    // Shared by every case below that just needs *some* valid corpus/centroids; only the
    // 256-per-centroid bound needs a differently-shaped corpus to trigger that branch.
    auto centers = make_random_centers(K, DIM, 1);
    auto corpus =
        make_clustered_corpus(centers, K, DIM, std::vector<size_t>(10, 4), 2, 0.05F);

    CATCH_SECTION("empty tokens dataset") {
        auto empty = svs::data::VectorSetData<float>(
            svs::data::SimpleData<float>(0, DIM), std::vector<uint64_t>{0}
        );
        CATCH_REQUIRE_THROWS(
            gem::train_codebook(empty, gem::CodebookParameters{}, threadpool)
        );
    }

    CATCH_SECTION("num_centroids above the uint32_t id range") {
        gem::CodebookParameters parameters;
        parameters.num_centroids =
            static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1;
        CATCH_REQUIRE_THROWS(gem::train_codebook(corpus, parameters, threadpool));
    }

    CATCH_SECTION("fewer training tokens than K") {
        gem::CodebookParameters parameters;
        parameters.num_centroids = 1'000; // far more centroids than tokens sampled
        CATCH_REQUIRE_THROWS(gem::train_codebook(corpus, parameters, threadpool));
    }

    CATCH_SECTION("training points exceed faiss's 256-per-centroid bound") {
        // One passage, 600 tokens; colbert_sample_size caps at 1 passage as training set.
        auto dense_corpus = make_clustered_corpus(centers, K, DIM, {600}, 6, 0.05F);
        gem::CodebookParameters parameters;
        parameters.num_centroids = 2; // 600ish training tokens > 2 * 256 == 512
        CATCH_REQUIRE_THROWS(gem::train_codebook(dense_corpus, parameters, threadpool));
    }

    CATCH_SECTION("assign_tokens token/centroid dimension mismatch") {
        svs::data::SimpleData<float> bad_centroids(K, DIM + 1);
        CATCH_REQUIRE_THROWS(
            gem::assign_tokens(corpus, bad_centroids, gem::CodebookParameters{}, threadpool)
        );
    }

    CATCH_SECTION("coarse_clusters num_coarse exceeds K") {
        auto centroids = centroids_from(centers, K, DIM);
        gem::CodebookParameters parameters;
        parameters.num_coarse = K + 1;
        CATCH_REQUIRE_THROWS(gem::coarse_clusters(centroids, parameters, threadpool));
    }
}

CATCH_TEST_CASE("GEM Codebook - k-means algorithm selection", "[quantization][gem]") {
    CATCH_SECTION("default CodebookParameters selects Lloyd") {
        CATCH_REQUIRE(
            gem::CodebookParameters{}.kmeans_algorithm == gem::KMeansAlgorithm::Lloyd
        );
    }

    CATCH_SECTION("EMeans trains K centroids of the right dimension, finite objective") {
        constexpr size_t DIM = 8;
        constexpr size_t K = 4;
        constexpr size_t NUM_PASSAGES = 40;
        auto centers = make_random_centers(K, DIM, 0x2024);
        auto corpus = make_clustered_corpus(
            centers, K, DIM, std::vector<size_t>(NUM_PASSAGES, 2), 0xaaaa, 0.1F
        );
        gem::CodebookParameters parameters;
        parameters.num_centroids = K;
        parameters.kmeans_algorithm = gem::KMeansAlgorithm::EMeans;
        parameters.emeans_epochs = 1;
        auto threadpool = svs::threads::as_threadpool(2);

        auto result = gem::train_codebook(corpus, parameters, threadpool);

        CATCH_REQUIRE(result.centroids.size() == K);
        CATCH_REQUIRE(result.centroids.dimensions() == DIM);
        CATCH_REQUIRE(std::isfinite(result.objective));
    }
}
