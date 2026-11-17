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

// tests
#include "tests/svs/index/vamana/gem_test_utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace {

using svs::distance::QEmd;
using svs_test::gem::make_centroid_table;
using svs_test::gem::make_centroids;
using svs_test::gem::make_doc_corpus;
using svs_test::gem::round_robin_coarse_of_fine;

} // namespace

// Covers the no-router / all-unrepresented fallback for GEM entry-point routing at the
// layer that owns it; gem_routing.cpp's GemRoutedInitializer mirrors it without repeating.
CATCH_TEST_CASE("gem_insert_order and build_entry_points", "[vamana][gem][gem_build]") {
    namespace v = svs::index::vamana;
    namespace ext = v::extensions;

    CATCH_SECTION("gem_insert_order dedups representatives ahead of the rest, ascending") {
        svs::distance::GemRouter router;
        router.coarse_of_fine = {0, 0, 1, 1, 2, 2};
        router.representative = {4, 2, svs::distance::GemRouter::no_representative, 2};
        router.nprobe = 2;

        auto order = v::gem_insert_order<uint32_t>(router, 6);
        auto sorted_order = order;
        std::sort(sorted_order.begin(), sorted_order.end());
        CATCH_REQUIRE(sorted_order == std::vector<uint32_t>{0, 1, 2, 3, 4, 5});
        CATCH_REQUIRE(order.size() >= 2);
        CATCH_REQUIRE(order[0] == 2);
        CATCH_REQUIRE(order[1] == 4);
        CATCH_REQUIRE(
            std::vector<uint32_t>(order.begin() + 2, order.end()) ==
            std::vector<uint32_t>{0, 1, 3, 5}
        );
    }

    auto centroids = make_centroids(0xB001);
    auto table = make_centroid_table(centroids);
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
    auto table = make_centroid_table(centroids);

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

    for (auto maybe_router :
         {std::shared_ptr<const svs::distance::GemRouter>(nullptr), router_ptr}) {
        QEmd distance{table, std::move(maybe_router)};
        auto pool = svs::threads::DefaultThreadPool(1);
        auto [graph, report] =
            v::build_gem_graph(docs, distance, params, pool, entry_point);

        CATCH_REQUIRE(report.documents == n_docs);
        CATCH_REQUIRE(report.unreachable_after_repair == 0);
        svs_test::require_degree_bound(graph, params.graph_max_degree);
    }
}
