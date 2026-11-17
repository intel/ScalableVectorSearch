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
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace {

using svs::distance::GemRouter;

constexpr uint32_t NO_REP = GemRouter::no_representative;

// 6 fine centroids grouped into 4 coarse clusters, leaving cluster 3 with no member.
std::vector<uint32_t> hand_coarse_of_fine() { return {0, 0, 1, 1, 2, 2}; }

// Hand-built corpus where doc3 and doc4 tie for cluster 2; doc3 (lower id) must win.
struct HandCorpus {
    std::vector<uint32_t> ids{0, 0, 2, 0, 1, 1, 2, 3, 4, 4};
    std::vector<uint64_t> offsets{0, 3, 6, 8, 9, 10};

    svs::data::ConstVectorSetView<uint32_t, 1> view() const {
        return svs::data::ConstVectorSetView<uint32_t, 1>(
            std::span<const uint32_t>(ids), std::span<const uint64_t>(offsets), 1
        );
    }
};

// Larger randomized corpus, for the thread-count-independence check.
struct RandomCorpus {
    std::vector<uint32_t> ids;
    std::vector<uint64_t> offsets;

    svs::data::ConstVectorSetView<uint32_t, 1> view() const {
        return svs::data::ConstVectorSetView<uint32_t, 1>(
            std::span<const uint32_t>(ids), std::span<const uint64_t>(offsets), 1
        );
    }
};

RandomCorpus make_random_corpus(size_t n, size_t k, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> id_dist(0, static_cast<uint32_t>(k - 1));
    RandomCorpus corpus;
    corpus.offsets.resize(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        size_t cardinality = 1 + (i % 7);
        corpus.offsets[i + 1] = corpus.offsets[i] + cardinality;
        for (size_t t = 0; t < cardinality; ++t) {
            corpus.ids.push_back(id_dist(rng));
        }
    }
    return corpus;
}

} // namespace

CATCH_TEST_CASE("GemRouter representative rule", "[distance][gem][gem_router]") {
    auto corpus = HandCorpus();
    auto docs = corpus.view();

    auto router =
        GemRouter::build(hand_coarse_of_fine(), /*num_coarse=*/4, docs, /*num_threads=*/1);

    CATCH_REQUIRE(router.num_fine() == 6);
    CATCH_REQUIRE(router.num_coarse() == 4);
    CATCH_REQUIRE(router.representative[0] == 1);
    CATCH_REQUIRE(router.representative[1] == 2);
    CATCH_REQUIRE(router.representative[2] == 3);      // tie broken to the lower doc id
    CATCH_REQUIRE(router.representative[3] == NO_REP); // untouched cluster
}

CATCH_TEST_CASE(
    "GemRouter representative rule rejects bad input", "[distance][gem][gem_router]"
) {
    auto corpus = HandCorpus();
    auto docs = corpus.view();

    auto bad_coarse_of_fine = hand_coarse_of_fine();
    bad_coarse_of_fine[2] = 99; // out of [0, num_coarse)
    CATCH_REQUIRE_THROWS(GemRouter::build(bad_coarse_of_fine, 4, docs, size_t{1}));
}

CATCH_TEST_CASE(
    "GemRouter build is independent of thread count", "[distance][gem][gem_router]"
) {
    constexpr size_t k = 24;
    constexpr size_t num_coarse = 6;
    auto corpus = make_random_corpus(400, k, 0xC0FFEE);
    auto docs = corpus.view();

    std::vector<uint32_t> coarse_of_fine(k);
    std::mt19937 rng(0xFEED);
    std::uniform_int_distribution<uint32_t> cluster_dist(0, num_coarse - 1);
    for (auto& c : coarse_of_fine) {
        c = cluster_dist(rng);
    }

    auto router_1 = GemRouter::build(coarse_of_fine, num_coarse, docs, size_t{1});
    auto router_4 = GemRouter::build(coarse_of_fine, num_coarse, docs, size_t{4});

    CATCH_REQUIRE(router_1.representative == router_4.representative);
    CATCH_REQUIRE(router_1.coarse_of_fine == router_4.coarse_of_fine);
}

CATCH_TEST_CASE("GemRouter route_query", "[distance][gem][gem_router]") {
    CATCH_SECTION("Hand example: nprobe=1, no_representative is skipped") {
        GemRouter router;
        router.coarse_of_fine = {0, 0, 1, 1};
        router.representative = {10, NO_REP};
        router.nprobe = 1;

        // table[f * n_query + i], K=4, n_query=2.
        std::vector<float> table{
            5.0F,
            1.0F, // f0
            3.0F,
            9.0F, // f1
            2.0F,
            2.0F, // f2
            8.0F,
            0.0F // f3
        };
        GemRouter::QueryScratch scratch;
        std::vector<uint32_t> out;
        router.route_query(table.data(), 2, out, scratch);
        CATCH_REQUIRE(out == std::vector<uint32_t>{10});
    }

    CATCH_SECTION("Duplicate representatives across tokens collapse to one entry") {
        GemRouter router;
        router.coarse_of_fine = {0, 0, 1, 1};
        router.representative = {7, 7};
        router.nprobe = 2; // both clusters always selected
        std::vector<float> table{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
        GemRouter::QueryScratch scratch;
        std::vector<uint32_t> out;
        router.route_query(table.data(), 2, out, scratch);
        CATCH_REQUIRE(out == std::vector<uint32_t>{7});
    }

    CATCH_SECTION("Matches a brute-force reference on a random table") {
        constexpr size_t k = 20;
        constexpr size_t num_coarse = 5;
        constexpr size_t n_query = 6;
        constexpr size_t nprobe = 2;

        std::mt19937 rng(0xABCDEF);
        std::uniform_int_distribution<uint32_t> cluster_dist(0, num_coarse - 1);
        std::uniform_real_distribution<float> score_dist(-5.0F, 5.0F);
        std::uniform_int_distribution<uint32_t> rep_dist(0, 99);
        std::bernoulli_distribution has_rep(0.7);

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

        std::vector<float> table(k * n_query);
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
                coarse_score[router.coarse_of_fine[f]] = std::max(
                    coarse_score[router.coarse_of_fine[f]], table[f * n_query + i]
                );
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
            size_t take = std::min(nprobe, num_coarse);
            for (size_t j = 0; j < take; ++j) {
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
}

CATCH_TEST_CASE("GemRouter route_ids", "[distance][gem][gem_router]") {
    GemRouter router;
    router.coarse_of_fine = {0, 0, 1, 1, 2, 2};
    router.representative = {100, 200, 300};

    CATCH_SECTION("Ranked by count desc, ties to the lower cluster id") {
        std::vector<uint32_t> ids{0, 0, 2, 4};
        GemRouter::IdScratch scratch;
        std::vector<uint32_t> out;

        router.route_ids(ids, 2, out, scratch);
        CATCH_REQUIRE(out == std::vector<uint32_t>{100, 200});

        router.route_ids(ids, 3, out, scratch);
        CATCH_REQUIRE(out == std::vector<uint32_t>{100, 200, 300});
    }

    CATCH_SECTION("no_representative clusters are skipped") {
        GemRouter sparse = router;
        sparse.representative = {100, NO_REP, 300};
        std::vector<uint32_t> ids{2, 3}; // both tokens map to coarse1
        GemRouter::IdScratch scratch;
        std::vector<uint32_t> out;

        sparse.route_ids(ids, 5, out, scratch);
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
        f.seekp(0);
        char byte = 0;
        f.read(&byte, 1);
        f.seekp(0);
        byte = static_cast<char>(~byte);
        f.write(&byte, 1);
        f.close();
        CATCH_REQUIRE_THROWS(GemRouter::load(path));
    }
}
