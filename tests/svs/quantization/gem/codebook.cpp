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

std::vector<float> make_random_centers(size_t K, size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<float> centers(K * dim);
    for (size_t k = 0; k < K; ++k) {
        auto c = random_unit_vector(rng, dim);
        std::copy(c.begin(), c.end(), centers.begin() + static_cast<ptrdiff_t>(k * dim));
    }
    return centers;
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

CATCH_TEST_CASE("GEM Codebook - colbert_num_partitions", "[quantization][gem]") {
    // Exact powers of two: 16 * sqrt(1) == 16, 16 * sqrt(4) == 32.
    CATCH_REQUIRE(gem::colbert_num_partitions(1) == 16);
    CATCH_REQUIRE(gem::colbert_num_partitions(4) == 32);
    // One token below/above the num_tokens=4 boundary (raw crosses exactly 32 at n=4).
    CATCH_REQUIRE(gem::colbert_num_partitions(3) == 16);
    CATCH_REQUIRE(gem::colbert_num_partitions(5) == 32);
    CATCH_REQUIRE_THROWS(gem::colbert_num_partitions(0));
}

CATCH_TEST_CASE("GEM Codebook - colbert_sample_size", "[quantization][gem]") {
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

CATCH_TEST_CASE(
    "GEM Codebook - train_codebook centroids are unit norm, objective is non-increasing",
    "[quantization][gem]"
) {
    constexpr size_t K = 8;
    constexpr size_t DIM = 16;
    auto centers = make_random_centers(K, DIM, 0x1234);

    std::vector<size_t> cardinalities(50, 6);
    auto corpus = make_clustered_corpus(centers, K, DIM, cardinalities, 0xABCD, 0.05F);

    gem::CodebookParameters parameters;
    parameters.num_centroids = K;
    parameters.seed = 7;

    auto threadpool = svs::threads::as_threadpool(2);

    parameters.kmeans_iterations = 1;
    auto one_iter = gem::train_codebook(corpus, parameters, threadpool);

    parameters.kmeans_iterations = 4;
    auto four_iter = gem::train_codebook(corpus, parameters, threadpool);

    // Same seed and sampling: more passes cannot raise the final objective.
    CATCH_REQUIRE(four_iter.objective <= one_iter.objective + 1e-3);

    for (size_t i = 0; i < four_iter.centroids.size(); ++i) {
        auto c = four_iter.centroids.get_datum(i);
        float norm_sq = 0.0F;
        for (size_t d = 0; d < c.size(); ++d) {
            norm_sq += c[d] * c[d];
        }
        CATCH_REQUIRE(std::abs(std::sqrt(norm_sq) - 1.0F) < 1e-4F);
    }
    CATCH_REQUIRE(four_iter.timings.train_iterations.size() == 4);
}

CATCH_TEST_CASE(
    "GEM Codebook - assign_tokens agrees with brute force, both assigners, both thread "
    "counts",
    "[quantization][gem]"
) {
    constexpr size_t K = 64;
    constexpr size_t DIM = 128;
    auto centers = make_random_centers(K, DIM, 0x5EED);

    std::vector<size_t> cardinalities;
    for (size_t i = 0; i < 300; ++i) {
        cardinalities.push_back(1 + (i % 6));
    }
    auto corpus = make_clustered_corpus(centers, K, DIM, cardinalities, 0xC0FFEE, 0.02F);

    svs::data::SimpleData<float> centroids(K, DIM);
    for (size_t k = 0; k < K; ++k) {
        centroids.set_datum(k, std::span<const float>(centers.data() + k * DIM, DIM));
    }

    gem::CodebookParameters parameters;
    parameters.assign_candidates = 8;

    for (auto assigner : {gem::Assigner::ExactGemm, gem::Assigner::InnerProductMatrix}) {
        parameters.assigner = assigner;

        auto pool1 = svs::threads::as_threadpool(1);
        auto pool4 = svs::threads::as_threadpool(4);
        auto ids1 = gem::assign_tokens(corpus, centroids, parameters, pool1);
        auto ids4 = gem::assign_tokens(corpus, centroids, parameters, pool4);

        auto flat1 = flatten_ids(ids1);
        auto flat4 = flatten_ids(ids4);
        CATCH_REQUIRE(flat1 == flat4);

        // Cardinalities of the id sets match the input token store.
        for (size_t p = 0; p < corpus.size(); ++p) {
            CATCH_REQUIRE(ids1.cardinality(p) == corpus.cardinality(p));
        }

        // Every id agrees with an independent brute-force argmax.
        size_t global = 0;
        for (size_t p = 0; p < corpus.size(); ++p) {
            auto doc = corpus.get_datum(p);
            for (size_t t = 0; t < doc.cardinality(); ++t) {
                uint32_t expected = brute_force_best(doc.token(t), centroids);
                CATCH_REQUIRE(flat1[global] == expected);
                ++global;
            }
        }

        auto report = gem::count_assignment_mismatches(corpus, centroids, ids1, 1000, 99);
        CATCH_REQUIRE(report.mismatches == 0);
        CATCH_REQUIRE(report.sample_size == 1000);

        // Corrupt a few ids and confirm the checker notices.
        auto offsets = std::vector<uint64_t>(ids1.offsets().begin(), ids1.offsets().end());
        auto corrupted = flat1;
        for (size_t i = 0; i < corrupted.size(); i += 37) {
            corrupted[i] = static_cast<uint32_t>((corrupted[i] + 1) % K);
        }
        auto corrupted_ids = rebuild_ids(corrupted, offsets);
        auto bad_report =
            gem::count_assignment_mismatches(corpus, centroids, corrupted_ids, 2000, 99);
        CATCH_REQUIRE(bad_report.mismatches > 0);
    }
}

CATCH_TEST_CASE(
    "GEM Codebook - assign_tokens batch row count comes from a byte budget",
    "[quantization][gem]"
) {
    constexpr size_t K = 64;
    constexpr size_t DIM = 32;
    auto centers = make_random_centers(K, DIM, 0xBEEF);

    std::vector<size_t> cardinalities;
    for (size_t i = 0; i < 40; ++i) {
        cardinalities.push_back(1 + (i % 5));
    }
    auto corpus = make_clustered_corpus(centers, K, DIM, cardinalities, 0xFACE, 0.02F);

    svs::data::SimpleData<float> centroids(K, DIM);
    for (size_t k = 0; k < K; ++k) {
        centroids.set_datum(k, std::span<const float>(centers.data() + k * DIM, DIM));
    }

    auto threadpool = svs::threads::as_threadpool(2);

    for (auto assigner : {gem::Assigner::ExactGemm, gem::Assigner::InnerProductMatrix}) {
        gem::CodebookParameters default_parameters;
        default_parameters.assigner = assigner;
        auto default_ids =
            gem::assign_tokens(corpus, centroids, default_parameters, threadpool);
        auto default_flat = flatten_ids(default_ids);

        // K*4 bytes/row: 256 forces a 1-row batch, 900 forces a 3-row batch.
        for (size_t budget : {size_t{256}, size_t{900}}) {
            gem::CodebookParameters tiny_parameters = default_parameters;
            tiny_parameters.assign_buffer_bytes = budget;
            auto tiny_ids =
                gem::assign_tokens(corpus, centroids, tiny_parameters, threadpool);
            CATCH_REQUIRE(flatten_ids(tiny_ids) == default_flat);
        }
    }
}

CATCH_TEST_CASE("GEM Codebook - coarse_clusters", "[quantization][gem]") {
    constexpr size_t K = 64;
    constexpr size_t DIM = 16;
    auto centers = make_random_centers(K, DIM, 0x777);
    svs::data::SimpleData<float> centroids(K, DIM);
    for (size_t k = 0; k < K; ++k) {
        centroids.set_datum(k, std::span<const float>(centers.data() + k * DIM, DIM));
    }

    auto threadpool = svs::threads::as_threadpool(2);

    CATCH_SECTION("default num_coarse rule") {
        gem::CodebookParameters parameters;
        auto result = gem::coarse_clusters(centroids, parameters, threadpool);
        CATCH_REQUIRE(result.num_coarse == std::max<size_t>(1, K * 5 / 32));
        CATCH_REQUIRE(result.coarse_of_fine.size() == K);
        for (auto c : result.coarse_of_fine) {
            CATCH_REQUIRE(c < result.num_coarse);
        }
    }

    CATCH_SECTION("explicit num_coarse") {
        gem::CodebookParameters parameters;
        parameters.num_coarse = 5;
        auto result = gem::coarse_clusters(centroids, parameters, threadpool);
        CATCH_REQUIRE(result.num_coarse == 5);
        for (auto c : result.coarse_of_fine) {
            CATCH_REQUIRE(c < 5);
        }
    }

    CATCH_SECTION("num_coarse exceeding K throws") {
        gem::CodebookParameters parameters;
        parameters.num_coarse = K + 1;
        CATCH_REQUIRE_THROWS(gem::coarse_clusters(centroids, parameters, threadpool));
    }
}

CATCH_TEST_CASE(
    "GEM Codebook - quantize end to end on a small synthetic set", "[quantization][gem]"
) {
    constexpr size_t DIM = 16;
    constexpr size_t NUM_PASSAGES = 200;
    // 200 passages * 2 tokens == 400; colbert_num_partitions(400) == 256, still exceeded.
    std::vector<size_t> cardinalities(NUM_PASSAGES, 2);
    auto centers = make_random_centers(16, DIM, 0x9999);
    auto corpus = make_clustered_corpus(centers, 16, DIM, cardinalities, 0x1357, 0.1F);

    CATCH_REQUIRE(corpus.total_tokens() == 400);
    CATCH_REQUIRE(gem::colbert_num_partitions(corpus.total_tokens()) == 256);

    gem::CodebookParameters parameters; // num_centroids == 0: use the rule.
    auto threadpool = svs::threads::as_threadpool(2);
    auto result = gem::quantize(corpus, parameters, threadpool);

    CATCH_REQUIRE(result.K == 256);
    CATCH_REQUIRE(result.centroids.size() == 256);
    CATCH_REQUIRE(result.centroids.dimensions() == DIM);
    CATCH_REQUIRE(result.fine_centroids->K == 256);
    CATCH_REQUIRE(result.fine_centroids->dim == DIM);
    CATCH_REQUIRE(result.centroid_table->K == 256);
    CATCH_REQUIRE(result.ids.total_tokens() == 400);
    for (size_t p = 0; p < NUM_PASSAGES; ++p) {
        CATCH_REQUIRE(result.ids.cardinality(p) == 2);
    }
    CATCH_REQUIRE(result.num_coarse == std::max<size_t>(1, 256 * 5 / 32));
    CATCH_REQUIRE(result.coarse_of_fine.size() == 256);
    for (auto c : result.coarse_of_fine) {
        CATCH_REQUIRE(c < result.num_coarse);
    }
    CATCH_REQUIRE(result.timings.train_iterations.size() == parameters.kmeans_iterations);
    CATCH_REQUIRE(result.timings.total >= 0.0);
    CATCH_REQUIRE(result.timings.sample >= 0.0);
    CATCH_REQUIRE(result.timings.train >= 0.0);
    CATCH_REQUIRE(result.timings.normalize >= 0.0);
    CATCH_REQUIRE(result.timings.assign >= 0.0);
    CATCH_REQUIRE(result.timings.ground_table >= 0.0);
    CATCH_REQUIRE(result.timings.coarse >= 0.0);
}

CATCH_TEST_CASE("GEM Codebook - failure paths", "[quantization][gem]") {
    constexpr size_t DIM = 8;
    auto threadpool = svs::threads::as_threadpool(1);

    CATCH_SECTION("empty tokens dataset") {
        svs::data::SimpleData<float> empty_tokens(0, DIM);
        std::vector<uint64_t> offsets{0};
        auto corpus =
            svs::data::VectorSetData<float>(std::move(empty_tokens), std::move(offsets));
        gem::CodebookParameters parameters;
        CATCH_REQUIRE_THROWS(gem::train_codebook(corpus, parameters, threadpool));
    }

    CATCH_SECTION("num_centroids above the uint32_t id range") {
        auto centers = make_random_centers(4, DIM, 1);
        std::vector<size_t> cardinalities(10, 4);
        auto corpus = make_clustered_corpus(centers, 4, DIM, cardinalities, 2, 0.05F);
        gem::CodebookParameters parameters;
        parameters.num_centroids =
            static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1;
        CATCH_REQUIRE_THROWS(gem::train_codebook(corpus, parameters, threadpool));
    }

    CATCH_SECTION("fewer training tokens than K") {
        auto centers = make_random_centers(4, DIM, 3);
        std::vector<size_t> cardinalities(5, 2);
        auto corpus = make_clustered_corpus(centers, 4, DIM, cardinalities, 4, 0.05F);
        gem::CodebookParameters parameters;
        parameters.num_centroids = 1'000; // far more centroids than tokens sampled
        CATCH_REQUIRE_THROWS(gem::train_codebook(corpus, parameters, threadpool));
    }

    CATCH_SECTION("training points exceed faiss's 256-per-centroid bound") {
        // One passage, 600 tokens; colbert_sample_size caps at 1 passage as training set.
        auto centers = make_random_centers(2, DIM, 5);
        std::vector<size_t> cardinalities{600};
        auto corpus = make_clustered_corpus(centers, 2, DIM, cardinalities, 6, 0.05F);
        gem::CodebookParameters parameters;
        parameters.num_centroids = 2; // 600ish training tokens > 2 * 256 == 512
        CATCH_REQUIRE_THROWS(gem::train_codebook(corpus, parameters, threadpool));
    }

    CATCH_SECTION("assign_tokens token/centroid dimension mismatch") {
        auto centers = make_random_centers(4, DIM, 7);
        std::vector<size_t> cardinalities(5, 2);
        auto corpus = make_clustered_corpus(centers, 4, DIM, cardinalities, 8, 0.05F);
        svs::data::SimpleData<float> bad_centroids(4, DIM + 1);
        gem::CodebookParameters parameters;
        CATCH_REQUIRE_THROWS(
            gem::assign_tokens(corpus, bad_centroids, parameters, threadpool)
        );
    }

    CATCH_SECTION("coarse_clusters num_coarse exceeds K") {
        auto centers = make_random_centers(4, DIM, 9);
        svs::data::SimpleData<float> centroids(4, DIM);
        for (size_t k = 0; k < 4; ++k) {
            centroids.set_datum(k, std::span<const float>(centers.data() + k * DIM, DIM));
        }
        gem::CodebookParameters parameters;
        parameters.num_coarse = 5;
        CATCH_REQUIRE_THROWS(gem::coarse_clusters(centroids, parameters, threadpool));
    }
}
