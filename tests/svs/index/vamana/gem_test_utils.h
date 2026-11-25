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

#pragma once

// Shared fixtures and oracles for the vamana test directory's GEM, MaxSim and
// connectivity tests: a brute-force top-k oracle, a graph degree-bound check, and (in
// svs_test::gem) a unit-norm centroid table plus a deterministic multi-token id-corpus.

// svs
#include "svs/core/data/vector_set_view.h"
#include "svs/core/distance.h"
#include "svs/core/distance/gem.h"
#include "svs/core/distance/gem_router.h"
#include "svs/core/graph.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace svs_test {

// Every document's score against `query` under `dist`, sorted best-first by
// `Dist::compare`. `dist` must already carry any argument-fixing the distance needs.
template <typename Dist, typename Query, typename DocsView>
std::vector<std::pair<uint32_t, float>>
brute_force_top_k(Dist& dist, const Query& query, const DocsView& docs) {
    std::vector<std::pair<uint32_t, float>> scored;
    scored.reserve(docs.size());
    for (uint32_t id = 0; id < docs.size(); ++id) {
        scored.emplace_back(id, svs::distance::compute(dist, query, docs.get_datum(id)));
    }
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
        return typename Dist::compare{}(a.second, b.second);
    });
    return scored;
}

template <typename Idx>
void require_degree_bound(const svs::graphs::SimpleGraph<Idx>& graph, size_t max_degree) {
    for (size_t i = 0; i < graph.n_nodes(); ++i) {
        CATCH_REQUIRE(graph.get_node_degree(static_cast<Idx>(i)) <= max_degree);
    }
}

} // namespace svs_test

namespace svs_test::gem {

inline constexpr size_t K = 16;
inline constexpr size_t DIM = 4;

// K unit vectors in R^DIM, deterministic for a given seed.
inline std::vector<float> make_centroids(uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> coord(-1.0F, 1.0F);
    std::vector<float> centroids(K * DIM);
    for (size_t k = 0; k < K; ++k) {
        double norm = 0.0;
        for (size_t d = 0; d < DIM; ++d) {
            float v = coord(rng);
            centroids[k * DIM + d] = v;
            norm += static_cast<double>(v) * static_cast<double>(v);
        }
        norm = std::sqrt(norm);
        for (size_t d = 0; d < DIM; ++d) {
            centroids[k * DIM + d] = static_cast<float>(centroids[k * DIM + d] / norm);
        }
    }
    return centroids;
}

inline std::shared_ptr<const svs::distance::CentroidTable>
make_centroid_table(const std::vector<float>& centroids) {
    return std::make_shared<const svs::distance::CentroidTable>(
        svs::distance::CentroidTable::from_centroids(
            centroids.data(), K, DIM, svs::distance::Ground::Cos, 1
        )
    );
}

inline std::shared_ptr<const svs::distance::FineCentroids>
make_fine_centroids(const std::vector<float>& centroids) {
    return std::make_shared<const svs::distance::FineCentroids>(
        K, DIM, centroids.data(), 1
    );
}

// A CSR-style corpus of id-token vector sets: cardinality cycles 1..6, ids uniform over K.
struct IdSetCorpus {
    std::vector<uint32_t> ids;
    std::vector<uint64_t> offsets;

    size_t size() const { return offsets.size() - 1; }

    svs::data::ConstVectorSetView<uint32_t, 1> view() const {
        return svs::data::ConstVectorSetView<uint32_t, 1>(
            std::span<const uint32_t>(ids), std::span<const uint64_t>(offsets), 1
        );
    }
};

inline IdSetCorpus make_doc_corpus(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> id_dist(0, static_cast<uint32_t>(K - 1));
    IdSetCorpus corpus;
    corpus.offsets.resize(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        size_t cardinality = 1 + (i % 6);
        corpus.offsets[i + 1] = corpus.offsets[i] + cardinality;
        for (size_t t = 0; t < cardinality; ++t) {
            corpus.ids.push_back(id_dist(rng));
        }
    }
    return corpus;
}

// Fine centroid f goes to coarse cluster f % num_coarse; deterministic, not realistic.
inline std::vector<uint32_t> round_robin_coarse_of_fine(size_t num_coarse) {
    std::vector<uint32_t> out(K);
    for (size_t f = 0; f < K; ++f) {
        out[f] = static_cast<uint32_t>(f % num_coarse);
    }
    return out;
}

} // namespace svs_test::gem
