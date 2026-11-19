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

#include "svs/runtime/api_defs.h"
#include "svs/runtime/gem_index.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <vector>

namespace {

using svs::runtime::ErrorCode;
using svs::runtime::GemBuildParameters;
using svs::runtime::GemIndex;
using svs::runtime::GemSearchParameters;
using svs::runtime::IDFilter;
using svs::runtime::Status;

constexpr size_t kDim = 16;
constexpr size_t kNumDocs = 300;
constexpr size_t kNumCentroids = 32;

// Synthetic token corpus: `kNumDocs` documents, 4-12 tokens each, `kDim`-dimensional.
struct TokenCorpus {
    std::vector<float> tokens;
    std::vector<size_t> offsets;
};

TokenCorpus make_corpus(size_t n_docs, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> cardinality_dist(4, 12);
    std::uniform_real_distribution<float> value_dist(-1.0F, 1.0F);

    TokenCorpus corpus;
    corpus.offsets.resize(n_docs + 1, 0);
    for (size_t doc = 0; doc < n_docs; ++doc) {
        size_t cardinality = cardinality_dist(rng);
        corpus.offsets[doc + 1] = corpus.offsets[doc] + cardinality;
        for (size_t t = 0; t < cardinality; ++t) {
            for (size_t d = 0; d < kDim; ++d) {
                corpus.tokens.push_back(value_dist(rng));
            }
        }
    }
    return corpus;
}

GemBuildParameters small_build_params(bool keep_tokens = false) {
    GemBuildParameters params{};
    params.num_centroids = kNumCentroids;
    params.graph_max_degree = 16;
    params.construction_window_size = 32;
    params.keep_tokens_for_rerank = keep_tokens;
    params.num_threads = 1;
    return params;
}

// A filter excluding every id in `excluded`.
struct ExcludeFilter : IDFilter {
    std::set<size_t> excluded;
    bool is_member(size_t id) const override { return excluded.count(id) == 0; }
};

} // namespace

CATCH_TEST_CASE("GemIndex build and search returns sane ids", "[runtime][gem]") {
    auto corpus = make_corpus(kNumDocs, 0x1234);

    GemIndex* index = nullptr;
    Status status = GemIndex::build(
        &index,
        kDim,
        kNumDocs,
        corpus.tokens.data(),
        corpus.offsets.data(),
        small_build_params()
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);
    CATCH_REQUIRE(index->size() == kNumDocs);
    CATCH_REQUIRE(index->dimensions() == kDim);
    CATCH_REQUIRE(index->num_centroids() == kNumCentroids);
    CATCH_REQUIRE(index->get_memory_usage() > 0);

    auto queries = make_corpus(5, 0x5678);
    constexpr size_t k = 10;
    std::vector<float> distances(5 * k);
    std::vector<size_t> labels(5 * k);
    status = index->search(
        5, queries.tokens.data(), queries.offsets.data(), k, distances.data(), labels.data()
    );
    CATCH_REQUIRE(status.ok());

    for (size_t q = 0; q < 5; ++q) {
        bool found_valid = false;
        for (size_t i = 0; i < k; ++i) {
            size_t label = labels[q * k + i];
            if (label != static_cast<size_t>(-1)) {
                CATCH_REQUIRE(label < kNumDocs);
                found_valid = true;
            }
        }
        CATCH_REQUIRE(found_valid);
    }

    GemIndex::destroy(index);
}

CATCH_TEST_CASE("GemIndex search respects an IDFilter", "[runtime][gem]") {
    auto corpus = make_corpus(kNumDocs, 0x2222);

    GemIndex* index = nullptr;
    Status status = GemIndex::build(
        &index,
        kDim,
        kNumDocs,
        corpus.tokens.data(),
        corpus.offsets.data(),
        small_build_params()
    );
    CATCH_REQUIRE(status.ok());

    auto queries = make_corpus(1, 0x3333);
    constexpr size_t k = 20;

    ExcludeFilter filter;
    for (size_t id = 0; id < kNumDocs; ++id) {
        if (id % 2 == 0) {
            filter.excluded.insert(id);
        }
    }

    std::vector<float> distances(k);
    std::vector<size_t> labels(k);
    status = index->search(
        1,
        queries.tokens.data(),
        queries.offsets.data(),
        k,
        distances.data(),
        labels.data(),
        nullptr,
        &filter
    );
    CATCH_REQUIRE(status.ok());

    for (size_t i = 0; i < k; ++i) {
        size_t label = labels[i];
        if (label != static_cast<size_t>(-1)) {
            CATCH_REQUIRE(filter.excluded.count(label) == 0);
        }
    }

    GemIndex::destroy(index);
}

CATCH_TEST_CASE("GemIndex save/load round trip gives identical results", "[runtime][gem]") {
    auto corpus = make_corpus(kNumDocs, 0x4444);

    GemIndex* index = nullptr;
    Status status = GemIndex::build(
        &index,
        kDim,
        kNumDocs,
        corpus.tokens.data(),
        corpus.offsets.data(),
        small_build_params()
    );
    CATCH_REQUIRE(status.ok());

    std::stringstream buffer;
    status = index->save(buffer);
    CATCH_REQUIRE(status.ok());

    GemIndex* reloaded = nullptr;
    status = GemIndex::load(&reloaded, buffer);
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(reloaded != nullptr);
    CATCH_REQUIRE(reloaded->size() == index->size());
    CATCH_REQUIRE(reloaded->dimensions() == index->dimensions());
    CATCH_REQUIRE(reloaded->num_centroids() == index->num_centroids());

    auto queries = make_corpus(4, 0x5555);
    constexpr size_t k = 10;
    std::vector<float> distances_a(4 * k);
    std::vector<size_t> labels_a(4 * k);
    std::vector<float> distances_b(4 * k);
    std::vector<size_t> labels_b(4 * k);

    CATCH_REQUIRE(index
                      ->search(
                          4,
                          queries.tokens.data(),
                          queries.offsets.data(),
                          k,
                          distances_a.data(),
                          labels_a.data()
                      )
                      .ok());
    CATCH_REQUIRE(reloaded
                      ->search(
                          4,
                          queries.tokens.data(),
                          queries.offsets.data(),
                          k,
                          distances_b.data(),
                          labels_b.data()
                      )
                      .ok());

    CATCH_REQUIRE(labels_a == labels_b);
    for (size_t i = 0; i < labels_a.size(); ++i) {
        CATCH_REQUIRE(distances_a[i] == distances_b[i]);
    }

    GemIndex::destroy(index);
    GemIndex::destroy(reloaded);
}

CATCH_TEST_CASE("GemIndex rejects native_rerank without kept tokens", "[runtime][gem]") {
    auto corpus = make_corpus(kNumDocs, 0x6666);

    GemIndex* index = nullptr;
    Status status = GemIndex::build(
        &index,
        kDim,
        kNumDocs,
        corpus.tokens.data(),
        corpus.offsets.data(),
        small_build_params(/*keep_tokens=*/false)
    );
    CATCH_REQUIRE(status.ok());

    auto queries = make_corpus(1, 0x7777);
    constexpr size_t k = 5;
    std::vector<float> distances(k);
    std::vector<size_t> labels(k);

    GemSearchParameters search_params{};
    search_params.native_rerank = true;

    status = index->search(
        1,
        queries.tokens.data(),
        queries.offsets.data(),
        k,
        distances.data(),
        labels.data(),
        &search_params
    );
    CATCH_REQUIRE_FALSE(status.ok());
    CATCH_REQUIRE(status.code == ErrorCode::INVALID_ARGUMENT);

    GemIndex::destroy(index);
}

CATCH_TEST_CASE("GemIndex native_rerank works when tokens are kept", "[runtime][gem]") {
    auto corpus = make_corpus(kNumDocs, 0x8888);

    GemIndex* index = nullptr;
    Status status = GemIndex::build(
        &index,
        kDim,
        kNumDocs,
        corpus.tokens.data(),
        corpus.offsets.data(),
        small_build_params(/*keep_tokens=*/true)
    );
    CATCH_REQUIRE(status.ok());

    auto queries = make_corpus(1, 0x9999);
    constexpr size_t k = 5;
    std::vector<float> distances(k);
    std::vector<size_t> labels(k);

    GemSearchParameters search_params{};
    search_params.native_rerank = true;

    status = index->search(
        1,
        queries.tokens.data(),
        queries.offsets.data(),
        k,
        distances.data(),
        labels.data(),
        &search_params
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(labels[0] < kNumDocs);

    GemIndex::destroy(index);
}
