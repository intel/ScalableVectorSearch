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

// Headers under test: the GEM search-routing hook (GemRoutedInitializer,
// search_initializer override).
#include "svs/core/distance/gem.h"
#include "svs/core/distance/gem_router.h"
#include "svs/extensions/vamana/gem.h"

// svs
#include "svs/core/data/vector_set.h"
#include "svs/core/data/vector_set_view.h"
#include "svs/core/graph.h"
#include "svs/index/vamana/index.h"
#include "svs/lib/threads.h"

// tests
#include "tests/utils/utils.h"

// catch2
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
#include <vector>

namespace {

using svs::distance::CentroidTable;
using svs::distance::FineCentroids;
using svs::distance::GemRouter;
using svs::distance::Ground;
using svs::distance::QChamfer;
using svs::distance::QChamferExactQuery;

constexpr size_t K = 16;
constexpr size_t DIM = 4;

// K unit vectors, deterministic for a given seed; dim == DIM.
std::vector<float> make_centroids(uint32_t seed) {
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

IdSetCorpus make_doc_corpus(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> id_dist(0, static_cast<uint32_t>(K - 1));
    IdSetCorpus corpus;
    corpus.offsets.resize(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        size_t cardinality = 2 + (i % 5);
        corpus.offsets[i + 1] = corpus.offsets[i] + cardinality;
        for (size_t t = 0; t < cardinality; ++t) {
            corpus.ids.push_back(id_dist(rng));
        }
    }
    return corpus;
}

// Raw-float query tokens, one per centroid dimension: what QChamferExactQuery expects.
struct QueryCorpus {
    std::vector<float> tokens;
    std::vector<uint64_t> offsets;

    svs::data::ConstVectorSetView<float, DIM> view() const {
        return svs::data::ConstVectorSetView<float, DIM>(
            std::span<const float>(tokens), std::span<const uint64_t>(offsets), DIM
        );
    }
};

// Noisy centroids, so routing/recall has real nearest-centroid structure to exploit.
QueryCorpus
make_query_corpus(const std::vector<float>& centroids, size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> centroid_dist(0, static_cast<uint32_t>(K - 1));
    std::normal_distribution<float> noise(0.0F, 0.05F);

    QueryCorpus corpus;
    corpus.offsets.resize(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        size_t cardinality = 1 + (i % 3);
        corpus.offsets[i + 1] = corpus.offsets[i] + cardinality;
        for (size_t t = 0; t < cardinality; ++t) {
            uint32_t c = centroid_dist(rng);
            for (size_t d = 0; d < DIM; ++d) {
                corpus.tokens.push_back(centroids[c * DIM + d] + noise(rng));
            }
        }
    }
    return corpus;
}

std::shared_ptr<const CentroidTable> make_table(const std::vector<float>& centroids) {
    return std::make_shared<const CentroidTable>(
        CentroidTable::from_centroids(centroids.data(), K, DIM, Ground::Cos, 1)
    );
}

std::shared_ptr<const FineCentroids> make_fine(const std::vector<float>& centroids) {
    return std::make_shared<const FineCentroids>(K, DIM, centroids.data(), 1);
}

// Deterministic grouping (f % num_coarse); good enough to exercise routing, not realistic.
std::vector<uint32_t> round_robin_coarse_of_fine(size_t num_coarse) {
    std::vector<uint32_t> out(K);
    for (size_t f = 0; f < K; ++f) {
        out[f] = static_cast<uint32_t>(f % num_coarse);
    }
    return out;
}

// Minimal greedy-search test double: records pushed ids, nothing else.
struct RecordingBuffer {
    using index_type = uint32_t;
    std::vector<uint32_t> ids;

    void clear() { ids.clear(); }
    void push_back(uint32_t id) { ids.push_back(id); }
    void sort() { std::sort(ids.begin(), ids.end()); }
};

auto dummy_computer() {
    return [](uint32_t /*id*/) { return 0.0F; };
}

auto id_builder() {
    return [](uint32_t id, float /*dist*/) { return id; };
}

} // namespace

CATCH_TEST_CASE(
    "GemRoutedInitializer with no router matches EntryPointInitializer",
    "[vamana][gem][gem_router]"
) {
    auto centroids = make_centroids(0x1111);
    auto fine = make_fine(centroids);
    QChamferExactQuery distance{fine};

    auto queries = make_query_corpus(centroids, 1, 0x2222);
    auto query = queries.view().get_datum(0);
    distance.fix_argument(query);

    auto graph = svs::default_graph<uint32_t>(10, 32);
    std::vector<uint32_t> entry_points{3};
    auto computer = dummy_computer();
    auto builder = id_builder();
    svs::index::vamana::NullTracker tracker;

    RecordingBuffer buf_plain;
    svs::index::vamana::EntryPointInitializer<uint32_t> plain_init{
        std::span<const uint32_t>(entry_points)};
    plain_init(buf_plain, computer, graph, builder, tracker);

    RecordingBuffer buf_routed;
    svs::index::vamana::GemRoutedInitializer<uint32_t> routed_init{
        &distance, std::span<const uint32_t>(entry_points)};
    routed_init(buf_routed, computer, graph, builder, tracker);

    CATCH_REQUIRE(buf_routed.ids == buf_plain.ids);
}

CATCH_TEST_CASE(
    "GemRoutedInitializer with every cluster unrepresented falls back to the medoid",
    "[vamana][gem][gem_router]"
) {
    auto centroids = make_centroids(0x3333);
    auto fine = make_fine(centroids);

    GemRouter all_norep;
    all_norep.coarse_of_fine = round_robin_coarse_of_fine(4);
    all_norep.representative.assign(4, GemRouter::no_representative);
    all_norep.nprobe = 2;
    auto router_ptr = std::make_shared<const GemRouter>(all_norep);

    QChamferExactQuery distance_no_router{fine};
    QChamferExactQuery distance_all_norep{fine, router_ptr};

    auto queries = make_query_corpus(centroids, 1, 0x4444);
    auto query = queries.view().get_datum(0);
    distance_no_router.fix_argument(query);
    distance_all_norep.fix_argument(query);

    auto graph = svs::default_graph<uint32_t>(10, 32);
    std::vector<uint32_t> entry_points{5};
    auto computer = dummy_computer();
    auto builder = id_builder();
    svs::index::vamana::NullTracker tracker;

    RecordingBuffer buf_no_router;
    svs::index::vamana::GemRoutedInitializer<uint32_t>{
        &distance_no_router, std::span<const uint32_t>(entry_points)}(
        buf_no_router, computer, graph, builder, tracker
    );

    RecordingBuffer buf_all_norep;
    svs::index::vamana::GemRoutedInitializer<uint32_t>{
        &distance_all_norep, std::span<const uint32_t>(entry_points)}(
        buf_all_norep, computer, graph, builder, tracker
    );

    CATCH_REQUIRE(buf_all_norep.ids == buf_no_router.ids);
}

CATCH_TEST_CASE(
    "GemRoutedInitializer with a real router pushes exactly medoid union routed",
    "[vamana][gem][gem_router]"
) {
    auto centroids = make_centroids(0x5555);
    auto fine = make_fine(centroids);

    auto doc_corpus = make_doc_corpus(100, 0x6666);
    auto docs = doc_corpus.view();

    constexpr size_t num_coarse = 4;
    auto router = GemRouter::build(
        round_robin_coarse_of_fine(num_coarse), num_coarse, docs, size_t{1}, size_t{2}
    );
    auto router_ptr = std::make_shared<const GemRouter>(router);

    QChamferExactQuery distance{fine, router_ptr};

    auto queries = make_query_corpus(centroids, 1, 0x7777);
    auto query = queries.view().get_datum(0);
    distance.fix_argument(query);

    std::vector<uint32_t> entry_points{7};
    GemRouter::QueryScratch scratch;
    std::vector<uint32_t> routed;
    router.route_query(distance.table(), distance.query_tokens(), routed, scratch);

    std::vector<uint32_t> expected(entry_points.begin(), entry_points.end());
    expected.insert(expected.end(), routed.begin(), routed.end());
    std::sort(expected.begin(), expected.end());
    expected.erase(std::unique(expected.begin(), expected.end()), expected.end());

    auto graph = svs::default_graph<uint32_t>(100, 32);
    auto computer = dummy_computer();
    auto builder = id_builder();
    svs::index::vamana::NullTracker tracker;

    RecordingBuffer buf;
    svs::index::vamana::GemRoutedInitializer<uint32_t>{
        &distance,
        std::span<const uint32_t>(entry_points)}(buf, computer, graph, builder, tracker);

    CATCH_REQUIRE(buf.ids == expected);
}

CATCH_TEST_CASE(
    "Routed search is not worse than unrouted on a small index", "[vamana][gem][gem_router]"
) {
    auto centroids = make_centroids(0x8888);
    auto table = make_table(centroids);
    auto fine = make_fine(centroids);

    const size_t n_docs = 150;
    auto doc_corpus = make_doc_corpus(n_docs, 0x9999);
    auto docs = doc_corpus.view();
    auto queries = make_query_corpus(centroids, 20, 0xAAAA);
    auto query_view = queries.view();

    QChamfer build_dist{table};
    auto build_parameters =
        svs::index::vamana::VamanaBuildParameters{1.2F, 16, 32, 64, 16, true};
    const uint32_t entry_point = 0;

    auto graph = svs::default_graph<uint32_t>(n_docs, build_parameters.graph_max_degree);
    auto build_pool = svs::threads::DefaultThreadPool(1);
    auto builder = svs::index::vamana::VamanaBuilder(
        graph,
        docs,
        build_dist,
        build_parameters,
        build_pool,
        svs::index::vamana::extensions::estimate_prefetch_parameters(docs)
    );
    builder.construct(build_parameters.alpha, entry_point);

    constexpr size_t num_coarse = 6;
    auto router = GemRouter::build(
        round_robin_coarse_of_fine(num_coarse), num_coarse, docs, size_t{1}, size_t{3}
    );
    auto router_ptr = std::make_shared<const GemRouter>(router);

    auto search_params = [](size_t window) {
        return svs::index::vamana::VamanaSearchParameters{{window, window}, false, 0, 0};
    };

    auto run_search = [&](std::shared_ptr<const GemRouter> maybe_router) {
        QChamferExactQuery dist{fine, std::move(maybe_router)};
        auto pool = svs::threads::DefaultThreadPool(1);
        auto graph_copy = graph;
        auto index = svs::index::vamana::VamanaIndex(
            std::move(graph_copy), docs, entry_point, dist, std::move(pool)
        );
        auto sp = search_params(20);
        auto scratch = index.scratchspace(sp);

        size_t top1_hits = 0;
        for (size_t q = 0; q < query_view.size(); ++q) {
            auto query = query_view.get_datum(q);
            index.search(query, scratch);

            // Brute-force QChamferExactQuery oracle for this query.
            QChamferExactQuery oracle_dist{fine};
            oracle_dist.fix_argument(query);
            uint32_t best_id = 0;
            float best_score = std::numeric_limits<float>::infinity();
            for (uint32_t id = 0; id < docs.size(); ++id) {
                float score =
                    svs::distance::compute(oracle_dist, query, docs.get_datum(id));
                if (score < best_score) {
                    best_score = score;
                    best_id = id;
                }
            }
            if (scratch.buffer[0].id() == best_id) {
                ++top1_hits;
            }
        }
        return top1_hits;
    };

    size_t hits_unrouted = run_search(nullptr);
    size_t hits_routed = run_search(router_ptr);

    CATCH_REQUIRE(hits_routed >= hits_unrouted);
}
