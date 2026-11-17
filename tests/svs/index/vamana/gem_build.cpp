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

// Headers under test: gem_build.h and the QEmd build_entry_points overload (gem.h).
#include "svs/index/vamana/gem_build.h"

// svs
#include "svs/core/data.h"
#include "svs/core/data/vector_set.h"
#include "svs/core/data/vector_set_view.h"
#include "svs/core/distance/gem.h"
#include "svs/core/distance/gem_router.h"
#include "svs/core/graph.h"
#include "svs/extensions/vamana/gem.h"
#include "svs/index/vamana/connectivity.h"
#include "svs/lib/threads.h"

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
using svs::distance::Ground;
using svs::distance::QEmd;

constexpr size_t K = 16;
constexpr size_t DIM = 4;

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

std::shared_ptr<const CentroidTable> make_table(const std::vector<float>& centroids) {
    return std::make_shared<const CentroidTable>(
        CentroidTable::from_centroids(centroids.data(), K, DIM, Ground::Cos, 1)
    );
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

// Fine centroid f goes to coarse cluster f % num_coarse; deterministic, not realistic.
std::vector<uint32_t> round_robin_coarse_of_fine(size_t num_coarse) {
    std::vector<uint32_t> out(K);
    for (size_t f = 0; f < K; ++f) {
        out[f] = static_cast<uint32_t>(f % num_coarse);
    }
    return out;
}

std::vector<std::pair<uint32_t, float>> brute_force_qemd(
    QEmd& dist,
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
        return QEmd::compare{}(a.second, b.second);
    });
    return scored;
}

} // namespace

CATCH_TEST_CASE("gem_insert_order", "[vamana][gem][gem_build]") {
    namespace v = svs::index::vamana;

    svs::distance::GemRouter router;
    // 6 documents; representatives have a duplicate (cluster 2 has no_representative).
    router.coarse_of_fine = {0, 0, 1, 1, 2, 2};
    router.representative = {4, 2, svs::distance::GemRouter::no_representative, 2};
    router.nprobe = 2;

    auto order = v::gem_insert_order<uint32_t>(router, 6);

    // Every document appears exactly once.
    auto sorted_order = order;
    std::sort(sorted_order.begin(), sorted_order.end());
    CATCH_REQUIRE(sorted_order == std::vector<uint32_t>{0, 1, 2, 3, 4, 5});

    // Representatives (deduplicated, ascending, no_representative dropped) come first.
    CATCH_REQUIRE(order.size() >= 2);
    CATCH_REQUIRE(order[0] == 2);
    CATCH_REQUIRE(order[1] == 4);
    // The rest follow ascending.
    CATCH_REQUIRE(
        std::vector<uint32_t>(order.begin() + 2, order.end()) ==
        std::vector<uint32_t>{0, 1, 3, 5}
    );
}

CATCH_TEST_CASE("build_entry_points for QEmd", "[vamana][gem][gem_build]") {
    namespace ext = svs::index::vamana::extensions;
    auto centroids = make_centroids(0xB001);
    auto table = make_table(centroids);

    auto doc_corpus = make_doc_corpus(20, 0xB002);
    auto docs = doc_corpus.view();

    std::vector<uint32_t> entry_points{3};
    std::vector<uint32_t> scratch;

    CATCH_SECTION("No router returns entry_points unchanged") {
        QEmd distance{table};
        auto result = ext::build_entry_points(
            docs, distance, uint32_t{7}, std::span<const uint32_t>(entry_points), scratch
        );
        CATCH_REQUIRE(std::vector<uint32_t>(result.begin(), result.end()) == entry_points);
    }

    CATCH_SECTION("Every cluster unrepresented falls back to entry_points") {
        svs::distance::GemRouter all_norep;
        all_norep.coarse_of_fine = round_robin_coarse_of_fine(4);
        all_norep.representative.assign(4, svs::distance::GemRouter::no_representative);
        all_norep.nprobe = 2;
        auto router_ptr = std::make_shared<const svs::distance::GemRouter>(all_norep);

        QEmd distance{table, router_ptr};
        auto result = ext::build_entry_points(
            docs, distance, uint32_t{7}, std::span<const uint32_t>(entry_points), scratch
        );
        CATCH_REQUIRE(std::vector<uint32_t>(result.begin(), result.end()) == entry_points);
    }

    CATCH_SECTION("A real router returns entry_points union routed, minus node_id") {
        constexpr size_t num_coarse = 4;
        auto router = svs::distance::GemRouter::build(
            round_robin_coarse_of_fine(num_coarse), num_coarse, docs, size_t{1}, size_t{2}
        );
        auto router_ptr = std::make_shared<const svs::distance::GemRouter>(router);
        QEmd distance{table, router_ptr};

        uint32_t node_id = 7;
        auto doc = docs.get_datum(node_id);
        std::vector<uint32_t> doc_ids(doc.data(), doc.data() + doc.cardinality());

        std::vector<uint32_t> routed;
        svs::distance::GemRouter::IdScratch id_scratch;
        router.route_ids(doc_ids, router.nprobe, routed, id_scratch);

        std::vector<uint32_t> expected(entry_points.begin(), entry_points.end());
        expected.insert(expected.end(), routed.begin(), routed.end());
        std::sort(expected.begin(), expected.end());
        expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
        expected.erase(
            std::remove(expected.begin(), expected.end(), node_id), expected.end()
        );

        auto result = ext::build_entry_points(
            docs, distance, node_id, std::span<const uint32_t>(entry_points), scratch
        );
        CATCH_REQUIRE(std::vector<uint32_t>(result.begin(), result.end()) == expected);
    }
}

CATCH_TEST_CASE("build_gem_graph", "[vamana][gem][gem_build]") {
    namespace v = svs::index::vamana;

    auto centroids = make_centroids(0xB003);
    auto table = make_table(centroids);

    const size_t n_docs = 300;
    auto doc_corpus = make_doc_corpus(n_docs, 0xB004);
    auto docs = doc_corpus.view();

    auto params = v::VamanaBuildParameters{1.2F, 16, 32, 64, 16, true};
    const uint32_t entry_point = 0;

    constexpr size_t num_coarse = 8;
    auto router = svs::distance::GemRouter::build(
        round_robin_coarse_of_fine(num_coarse), num_coarse, docs, size_t{1}, size_t{3}
    );
    auto router_ptr = std::make_shared<const svs::distance::GemRouter>(router);

    auto run_build = [&](std::shared_ptr<const svs::distance::GemRouter> maybe_router) {
        QEmd distance{table, std::move(maybe_router)};
        auto pool = svs::threads::DefaultThreadPool(1);
        auto [graph, report] =
            v::build_gem_graph(docs, distance, params, pool, entry_point);

        CATCH_REQUIRE(report.documents == n_docs);
        CATCH_REQUIRE(report.unreachable_after_repair == 0);
        for (size_t i = 0; i < graph.n_nodes(); ++i) {
            CATCH_REQUIRE(
                graph.get_node_degree(static_cast<uint32_t>(i)) <= params.graph_max_degree
            );
        }
        return graph;
    };

    auto graph_unrouted = run_build(nullptr);
    auto graph_routed = run_build(router_ptr);

    // Recall check: routed build's top-1 search must not get worse than the unrouted one.
    auto count_top1_hits = [&](const svs::graphs::SimpleGraph<uint32_t>& graph) {
        QEmd oracle_distance{table};
        size_t hits = 0;
        auto search_buffer = v::SearchBuffer<uint32_t, QEmd::compare>{64};
        for (uint32_t q = 0; q < 20; ++q) {
            search_buffer.clear();
            auto query = docs.get_datum(q);
            auto oracle = brute_force_qemd(oracle_distance, query, docs);

            QEmd search_distance{table};
            auto accessor = svs::data::GetDatumAccessor{};
            v::greedy_search(
                graph,
                docs,
                accessor,
                query,
                search_distance,
                search_buffer,
                v::EntryPointInitializer<uint32_t>{
                    std::span<const uint32_t>(&entry_point, 1)}
            );
            search_buffer.sort();
            if (search_buffer.size() > 0 && search_buffer[0].id() == oracle[0].first) {
                ++hits;
            }
        }
        return hits;
    };

    size_t hits_unrouted = count_top1_hits(graph_unrouted);
    size_t hits_routed = count_top1_hits(graph_routed);
    CATCH_REQUIRE(hits_routed >= hits_unrouted);
}
