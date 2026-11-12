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

// Proves a VamanaIndex can be built and searched directly over a
// ConstVectorSetView<uint32_t, 1> of centroid-id vector sets, using QChamfer as both the
// build and search distance.
#include "svs/core/distance/gem.h"
#include "svs/extensions/vamana/gem.h"

// svs
#include "svs/core/data/vector_set_view.h"
#include "svs/core/graph.h"
#include "svs/index/vamana/index.h"
#include "svs/lib/threads.h"

// catch2
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

// stl
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace {

constexpr size_t K = 16;
constexpr size_t DIM = 4;

std::shared_ptr<const svs::distance::CentroidTable> make_centroid_table(uint32_t seed) {
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
    return std::make_shared<const svs::distance::CentroidTable>(
        svs::distance::CentroidTable::from_centroids(
            centroids.data(), K, DIM, svs::distance::Ground::Cos, 1
        )
    );
}

// A deterministic dataset of id vector sets, held CSR-style exactly as
// ConstVectorSetView<uint32_t, 1> expects: cardinality cycles 1..6, ids are uniform over
// the K centroids.
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

IdSetCorpus make_corpus(size_t n, uint32_t seed) {
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

svs::index::vamana::VamanaSearchParameters search_params(size_t window, size_t capacity) {
    return svs::index::vamana::VamanaSearchParameters{{window, capacity}, false, 0, 0};
}

// Brute-force oracle: every document's QChamfer score against `query`, sorted best-first
// (ascending, since QChamfer::compare is std::less<>).
std::vector<std::pair<uint32_t, float>> brute_force_qchamfer(
    svs::distance::QChamfer& dist,
    const svs::data::ConstVectorSet<uint32_t, 1>& query,
    const svs::data::ConstVectorSetView<uint32_t, 1>& docs
) {
    std::vector<std::pair<uint32_t, float>> scored;
    scored.reserve(docs.size());
    for (uint32_t id = 0; id < docs.size(); ++id) {
        float score = svs::distance::compute(dist, query, docs.get_datum(id));
        scored.emplace_back(id, score);
    }
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
        return svs::distance::QChamfer::compare{}(a.second, b.second);
    });
    return scored;
}

} // namespace

CATCH_TEST_CASE("Vamana build and search over a vector-set dataset", "[vamana][gem]") {
    auto table = make_centroid_table(0x5EED);
    svs::distance::QChamfer dist{table};

    const size_t n_docs = 200;
    auto corpus = make_corpus(n_docs, 0xDEADBEEF);
    auto queries = make_corpus(10, 0xC0FFEE);
    auto docs = corpus.view();
    auto query_view = queries.view();

    auto build_parameters =
        svs::index::vamana::VamanaBuildParameters{1.2F, 16, 32, 64, 16, true};

    // Entry point 0 is caller-supplied; the vector-set path adds no medoid logic of its
    // own.
    const uint32_t entry_point = 0;

    // VamanaIndex's build-parameters constructor hard-codes L2/IP/Cosine in
    // verify_and_set_default_index_parameters and throws for any other distance, so
    // qCH (like qEMD) has to go through VamanaBuilder directly, mirroring how
    // VamanaIndex's own build constructor drives it (index.h's build-parameters
    // constructor) minus that distance-type check.
    auto graph = svs::default_graph<uint32_t>(n_docs, build_parameters.graph_max_degree);
    auto build_pool = svs::threads::DefaultThreadPool(1);
    auto builder = svs::index::vamana::VamanaBuilder(
        graph,
        docs,
        dist,
        build_parameters,
        build_pool,
        svs::index::vamana::extensions::estimate_prefetch_parameters(docs)
    );
    builder.construct(build_parameters.alpha, entry_point);

    auto pool = svs::threads::DefaultThreadPool(1);
    auto index = svs::index::vamana::VamanaIndex(
        std::move(graph), docs, entry_point, dist, std::move(pool)
    );

    CATCH_SECTION("Small-window search returns a best-first, internally consistent pool") {
        const size_t capacity = 20;
        auto sp = search_params(capacity, capacity);
        auto scratch = index.scratchspace(sp);

        for (size_t q = 0; q < query_view.size(); ++q) {
            auto query = query_view.get_datum(q);
            index.search(query, scratch);

            size_t valid = scratch.buffer.size();
            CATCH_REQUIRE(valid > 0);
            float previous = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < valid; ++j) {
                const auto& neighbor = scratch.buffer[j];
                float direct =
                    svs::distance::compute(dist, query, docs.get_datum(neighbor.id()));
                CATCH_REQUIRE(
                    static_cast<double>(neighbor.distance()) ==
                    Catch::Approx(direct).epsilon(1e-5).margin(1e-5)
                );
                CATCH_REQUIRE(neighbor.distance() >= previous);
                previous = neighbor.distance();
            }
        }
    }

    CATCH_SECTION("Exhaustive search's top-1 matches brute force") {
        const size_t capacity = n_docs;
        auto sp = search_params(capacity, capacity);
        auto scratch = index.scratchspace(sp);

        for (size_t q = 0; q < query_view.size(); ++q) {
            auto query = query_view.get_datum(q);
            index.search(query, scratch);

            auto oracle = brute_force_qchamfer(dist, query, docs);

            // Centroid ids repeat across documents, so several documents can legitimately
            // tie for the best score; check the distance exactly and the id against every
            // document tied for that best score, rather than oracle[0].first alone.
            const auto& top1 = scratch.buffer[0];
            CATCH_REQUIRE(
                static_cast<double>(top1.distance()) ==
                Catch::Approx(oracle[0].second).epsilon(1e-5).margin(1e-5)
            );
            bool id_ties_for_best =
                std::any_of(oracle.begin(), oracle.end(), [&](const auto& entry) {
                    return entry.first == top1.id() &&
                           std::abs(entry.second - oracle[0].second) < 1e-5F;
                });
            CATCH_REQUIRE(id_ties_for_best);
        }
    }
}
