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

// Header under test
#include "svs/core/distance/gem_router.h"

// svs
#include "svs/core/data/vector_set_view.h"
#include "svs/lib/threads.h"

// tests
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace {
using svs::distance::GemRouter;

constexpr uint32_t NO_REP = GemRouter::no_representative;

svs::data::ConstVectorSetView<uint32_t, 1>
view_of(const std::vector<uint32_t>& ids, const std::vector<uint64_t>& offsets) {
    return svs::data::ConstVectorSetView<uint32_t, 1>(
        std::span<const uint32_t>(ids), std::span<const uint64_t>(offsets), 1
    );
}

// 6 fine centroids / 4 coarse clusters (cluster 3 empty); doc3/doc4 tie for cluster 2 and
// doc3 (lower id) must win.
std::vector<uint32_t> hand_coarse_of_fine() { return {0, 0, 1, 1, 2, 2}; }
std::vector<uint32_t> hand_ids() { return {0, 0, 2, 0, 1, 1, 2, 3, 4, 4}; }
std::vector<uint64_t> hand_offsets() { return {0, 3, 6, 8, 9, 10}; }

std::vector<uint32_t> random_ids(size_t n, size_t k, std::vector<uint64_t>& offsets_out) {
    std::mt19937 rng(0xC0FFEE);
    std::uniform_int_distribution<uint32_t> id_dist(0, static_cast<uint32_t>(k - 1));
    std::vector<uint32_t> ids;
    offsets_out.assign(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        size_t cardinality = 1 + (i % 7);
        offsets_out[i + 1] = offsets_out[i] + cardinality;
        for (size_t t = 0; t < cardinality; ++t) {
            ids.push_back(id_dist(rng));
        }
    }
    return ids;
}
} // namespace

CATCH_TEST_CASE("GemRouter build", "[distance][gem][gem_router]") {
    auto ids = hand_ids();
    auto offsets = hand_offsets();
    auto docs = view_of(ids, offsets);

    CATCH_SECTION("Representative rule: tie to lower doc id, untouched cluster is NO_REP") {
        auto router =
            GemRouter::build(hand_coarse_of_fine(), /*num_coarse=*/4, docs, size_t{1});
        CATCH_REQUIRE(router.num_fine() == 6);
        CATCH_REQUIRE(router.num_coarse() == 4);
        CATCH_REQUIRE(router.representative[0] == 1);
        CATCH_REQUIRE(router.representative[1] == 2);
        CATCH_REQUIRE(router.representative[2] == 3);      // tie broken to the lower doc id
        CATCH_REQUIRE(router.representative[3] == NO_REP); // untouched cluster
    }

    CATCH_SECTION("Out-of-range coarse id is rejected") {
        auto bad = hand_coarse_of_fine();
        bad[2] = 99; // out of [0, num_coarse)
        CATCH_REQUIRE_THROWS(GemRouter::build(bad, 4, docs, size_t{1}));
    }

    CATCH_SECTION("Result is independent of thread count") {
        constexpr size_t k = 24;
        constexpr size_t num_coarse = 6;
        std::vector<uint64_t> random_offsets;
        auto random_id_list = random_ids(400, k, random_offsets);
        auto random_docs = view_of(random_id_list, random_offsets);

        std::vector<uint32_t> coarse_of_fine(k);
        std::mt19937 rng(0xFEED);
        std::uniform_int_distribution<uint32_t> cluster_dist(0, num_coarse - 1);
        for (auto& c : coarse_of_fine) {
            c = cluster_dist(rng);
        }

        auto router_1 =
            GemRouter::build(coarse_of_fine, num_coarse, random_docs, size_t{1});
        auto router_4 =
            GemRouter::build(coarse_of_fine, num_coarse, random_docs, size_t{4});
        CATCH_REQUIRE(router_1.representative == router_4.representative);
        CATCH_REQUIRE(router_1.coarse_of_fine == router_4.coarse_of_fine);
    }
}

// Covers route_query's whole spec at once (nprobe cap, NO_REP skip, dedup across tokens,
// tie-breaking to the lower cluster id): any hand-picked case is a strict instance of it.
CATCH_TEST_CASE(
    "GemRouter route_query matches a brute-force reference", "[distance][gem][gem_router]"
) {
    constexpr size_t k = 20;
    constexpr size_t num_coarse = 5;
    constexpr size_t n_query = 6;
    constexpr size_t nprobe = 2;

    std::mt19937 rng(0xABCDEF);
    std::uniform_int_distribution<uint32_t> cluster_dist(0, num_coarse - 1);
    std::uniform_real_distribution<float> score_dist(-5.0F, 5.0F);
    std::uniform_int_distribution<uint32_t> rep_dist(0, 99);
    std::bernoulli_distribution has_rep(0.7); // some clusters get NO_REP

    GemRouter router;
    router.coarse_of_fine.resize(k);
    for (auto& c : router.coarse_of_fine) {
        c = cluster_dist(rng);
    }
    router.representative.resize(num_coarse);
    for (auto& r : router.representative) {
        r = has_rep(rng) ? rep_dist(rng) : NO_REP;
    }
    router.nprobe = nprobe;

    std::vector<float> table(k * n_query); // table[f * n_query + i]
    for (auto& v : table) {
        v = score_dist(rng);
    }

    GemRouter::QueryScratch scratch;
    std::vector<uint32_t> out;
    router.route_query(table.data(), n_query, out, scratch);

    // Independent reference implementation of the same spec.
    std::vector<uint32_t> expected;
    for (size_t i = 0; i < n_query; ++i) {
        std::vector<float> coarse_score(
            num_coarse, -std::numeric_limits<float>::infinity()
        );
        for (size_t f = 0; f < k; ++f) {
            coarse_score[router.coarse_of_fine[f]] =
                std::max(coarse_score[router.coarse_of_fine[f]], table[f * n_query + i]);
        }
        std::vector<uint32_t> order(num_coarse);
        for (size_t c = 0; c < num_coarse; ++c) {
            order[c] = static_cast<uint32_t>(c);
        }
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            if (coarse_score[a] != coarse_score[b]) {
                return coarse_score[a] > coarse_score[b];
            }
            return a < b;
        });
        for (size_t j = 0; j < std::min(nprobe, num_coarse); ++j) {
            uint32_t rep = router.representative[order[j]];
            if (rep != NO_REP) {
                expected.push_back(rep);
            }
        }
    }
    std::sort(expected.begin(), expected.end());
    expected.erase(std::unique(expected.begin(), expected.end()), expected.end());

    CATCH_REQUIRE(out == expected);
    CATCH_REQUIRE(std::is_sorted(out.begin(), out.end()));
    CATCH_REQUIRE(std::adjacent_find(out.begin(), out.end()) == out.end());
}

CATCH_TEST_CASE("GemRouter route_ids", "[distance][gem][gem_router]") {
    GemRouter router;
    router.coarse_of_fine = {0, 0, 1, 1, 2, 2};
    router.representative = {100, 200, 300};
    GemRouter::IdScratch scratch;
    std::vector<uint32_t> out;

    CATCH_SECTION("Ranked by count desc, ties to the lower cluster id") {
        std::vector<uint32_t> ids{0, 0, 2, 4};
        router.route_ids(ids, 2, out, scratch);
        CATCH_REQUIRE(out == std::vector<uint32_t>{100, 200});

        router.route_ids(ids, 3, out, scratch);
        CATCH_REQUIRE(out == std::vector<uint32_t>{100, 200, 300});
    }

    CATCH_SECTION("no_representative clusters are skipped") {
        router.representative = {100, NO_REP, 300};
        std::vector<uint32_t> ids{2, 3}; // both tokens map to coarse1
        router.route_ids(ids, 5, out, scratch);
        CATCH_REQUIRE(out.empty());
    }
}

CATCH_TEST_CASE("GemRouter save/load", "[distance][gem][gem_router]") {
    svs_test::prepare_temp_directory();
    auto path = svs_test::temp_directory() / "gem_router.bin";

    GemRouter router;
    router.coarse_of_fine = hand_coarse_of_fine();
    router.representative = {1, 2, 3, NO_REP};
    router.nprobe = 7;
    router.save(path);

    auto loaded = GemRouter::load(path);
    CATCH_REQUIRE(loaded.coarse_of_fine == router.coarse_of_fine);
    CATCH_REQUIRE(loaded.representative == router.representative);
    CATCH_REQUIRE(loaded.nprobe == router.nprobe);

    CATCH_SECTION("A corrupted file is rejected") {
        // Flip a byte inside the magic field.
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        char byte = 0;
        f.read(&byte, 1);
        f.seekp(0);
        byte = static_cast<char>(~byte);
        f.write(&byte, 1);
        f.close();
        CATCH_REQUIRE_THROWS(GemRouter::load(path));
    }
}
