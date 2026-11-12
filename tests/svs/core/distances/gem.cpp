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
#include "svs/core/distance/gem.h"
#include "svs/extensions/vamana/gem.h"

// svs
#include "svs/concepts/distance.h"
#include "svs/index/vamana/prune.h"

// catch2
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

// stdlib
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using svs::distance::CentroidTable;
using svs::distance::FineCentroids;
using svs::distance::Ground;
using svs::distance::QChamfer;
using svs::distance::QChamferExactQuery;
using svs::distance::QEmd;

// Four unit vectors in 2D: (1,0), (0,1), (-1,0), (0,-1). Their pairwise dot products are
// all 0, 1 or -1, so both ground distances come out to round numbers by hand.
std::shared_ptr<const CentroidTable> make_table(Ground ground) {
    std::vector<float> centroids{1.0F, 0.0F, 0.0F, 1.0F, -1.0F, 0.0F, 0.0F, -1.0F};
    return std::make_shared<const CentroidTable>(
        CentroidTable::from_centroids(centroids.data(), 4, 2, ground, 1)
    );
}

// Token-set span layout (gem_detail::set_size/set_ids): [0] = n, [1..n] = centroid ids,
// rest padding. `pad` lets a test fill the unused tail with garbage to prove it is
// ignored.
std::vector<float> make_set(std::vector<float> ids, size_t width, float pad = -999.0F) {
    std::vector<float> s(width, pad);
    s[0] = static_cast<float>(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        s[1 + i] = ids[i];
    }
    return s;
}

} // namespace

CATCH_TEST_CASE("CentroidTable / FineCentroids", "[distance][gem]") {
    CATCH_SECTION("Cosine ground") {
        auto table = make_table(Ground::Cos);
        CATCH_REQUIRE((*table)(0, 0) == Catch::Approx(0.0F));
        CATCH_REQUIRE((*table)(0, 1) == Catch::Approx(1.0F));
        CATCH_REQUIRE((*table)(0, 2) == Catch::Approx(2.0F));
        CATCH_REQUIRE((*table)(0, 3) == Catch::Approx(1.0F));
        CATCH_REQUIRE((*table)(1, 2) == Catch::Approx(1.0F));
        CATCH_REQUIRE((*table)(1, 3) == Catch::Approx(2.0F));
        CATCH_REQUIRE((*table)(2, 3) == Catch::Approx(1.0F));
    }

    CATCH_SECTION("L2 ground") {
        auto table = make_table(Ground::L2);
        CATCH_REQUIRE((*table)(0, 0) == Catch::Approx(0.0F));
        CATCH_REQUIRE((*table)(0, 1) == Catch::Approx(std::sqrt(2.0F)));
        CATCH_REQUIRE((*table)(0, 2) == Catch::Approx(2.0F));
        CATCH_REQUIRE((*table)(1, 3) == Catch::Approx(2.0F));
        CATCH_REQUIRE((*table)(2, 3) == Catch::Approx(std::sqrt(2.0F)));
    }

    CATCH_SECTION("FineCentroids holds raw rows, not a pairwise table") {
        std::vector<float> flat{1.0F, 0.0F, 0.0F, 1.0F};
        FineCentroids fc{2, 2, flat.data()};
        CATCH_REQUIRE(fc.K == 2);
        CATCH_REQUIRE(fc.dim == 2);
        CATCH_REQUIRE(fc.data[0] == 1.0F);
        CATCH_REQUIRE(fc.data[3] == 1.0F);
    }
}

CATCH_TEST_CASE("QChamfer", "[distance][gem]") {
    auto table = make_table(Ground::Cos);
    QChamfer dist{table};

    CATCH_SECTION("Hand-computed: min over doc per query token") {
        // query ids {0,1}, doc ids {2,3}: table(0,2)=2,table(0,3)=1 -> min 1;
        // table(1,2)=1,table(1,3)=2 -> min 1. total = 1 + 1 = 2.
        auto q = make_set({0.0F, 1.0F}, 6);
        auto d = make_set({2.0F, 3.0F}, 6);
        float score = dist.compute(std::span<const float>(q), std::span<const float>(d));
        CATCH_REQUIRE(score == Catch::Approx(2.0F));
    }

    CATCH_SECTION("Duplicate ids are counted, not merged") {
        // query {0,0} against doc {1}: each of the two query tokens pays table(0,1)=1.
        auto q = make_set({0.0F, 0.0F}, 6);
        auto d = make_set({1.0F}, 6);
        float score = dist.compute(std::span<const float>(q), std::span<const float>(d));
        CATCH_REQUIRE(score == Catch::Approx(2.0F));
    }

    CATCH_SECTION("n = 1") {
        auto q = make_set({0.0F}, 6);
        auto d = make_set({2.0F}, 6);
        float score = dist.compute(std::span<const float>(q), std::span<const float>(d));
        CATCH_REQUIRE(score == Catch::Approx(2.0F));
    }

    CATCH_SECTION("Padding beyond n is ignored") {
        auto q1 = make_set({0.0F, 1.0F}, 8, -999.0F);
        auto q2 = make_set({0.0F, 1.0F}, 8, 777.0F);
        auto d = make_set({2.0F, 3.0F}, 8, 42.0F);
        float s1 = dist.compute(std::span<const float>(q1), std::span<const float>(d));
        float s2 = dist.compute(std::span<const float>(q2), std::span<const float>(d));
        CATCH_REQUIRE(s1 == Catch::Approx(s2));
    }
}

CATCH_TEST_CASE("QEmd", "[distance][gem]") {
    auto table = make_table(Ground::Cos);
    QEmd dist{table};

    CATCH_SECTION("1 vs 1") {
        auto a = make_set({0.0F}, 4);
        auto b = make_set({1.0F}, 4);
        float score = dist.compute(std::span<const float>(a), std::span<const float>(b));
        CATCH_REQUIRE(score == Catch::Approx(1.0F)); // table(0,1)
    }

    CATCH_SECTION("1 vs 2") {
        // One source of mass 1 split 0.5/0.5 to two distinct sinks: 0.5*table(0,1) +
        // 0.5*table(0,2) = 0.5*1 + 0.5*2 = 1.5.
        auto a = make_set({0.0F}, 4);
        auto b = make_set({1.0F, 2.0F}, 4);
        float score = dist.compute(std::span<const float>(a), std::span<const float>(b));
        CATCH_REQUIRE(score == Catch::Approx(1.5F));
    }

    CATCH_SECTION("2 vs 2 with obvious optimal matching") {
        // Masses 0.5/0.5 on each side; matching 0->3, 1->2 costs 0.5*1+0.5*1=1, strictly
        // less than the cross matching 0->2, 1->3 (0.5*2+0.5*2=2).
        auto a = make_set({0.0F, 1.0F}, 4);
        auto b = make_set({2.0F, 3.0F}, 4);
        float score = dist.compute(std::span<const float>(a), std::span<const float>(b));
        CATCH_REQUIRE(score == Catch::Approx(1.0F));
    }

    CATCH_SECTION("Duplicates merge to a single point with full mass") {
        // {0,0} merges to one point of mass 1 at id 0, same as the plain "1 vs 1" case.
        auto a = make_set({0.0F, 0.0F}, 4);
        auto b = make_set({1.0F}, 4);
        float score = dist.compute(std::span<const float>(a), std::span<const float>(b));
        CATCH_REQUIRE(score == Catch::Approx(1.0F));
    }

    CATCH_SECTION("Identical sets give 0") {
        auto a = make_set({0.0F, 1.0F}, 4);
        float score = dist.compute(std::span<const float>(a), std::span<const float>(a));
        CATCH_REQUIRE(score == Catch::Approx(0.0F));
    }

    CATCH_SECTION("Symmetry") {
        auto a = make_set({0.0F}, 4);
        auto b = make_set({1.0F, 2.0F}, 4);
        float ab = dist.compute(std::span<const float>(a), std::span<const float>(b));
        float ba = dist.compute(std::span<const float>(b), std::span<const float>(a));
        CATCH_REQUIRE(ab == Catch::Approx(ba));
    }

    CATCH_SECTION("Padding beyond n is ignored") {
        auto a1 = make_set({0.0F, 1.0F}, 8, -999.0F);
        auto a2 = make_set({0.0F, 1.0F}, 8, 777.0F);
        auto b = make_set({2.0F, 3.0F}, 8, 42.0F);
        float s1 = dist.compute(std::span<const float>(a1), std::span<const float>(b));
        float s2 = dist.compute(std::span<const float>(a2), std::span<const float>(b));
        CATCH_REQUIRE(s1 == Catch::Approx(s2));
    }
}

CATCH_TEST_CASE("QChamferExactQuery", "[distance][gem]") {
    // Raw (non-unit) centroids: c0=(1,0), c1=(0,1), c2=(-1,0).
    std::vector<float> flat{1.0F, 0.0F, 0.0F, 1.0F, -1.0F, 0.0F};
    auto centroids = std::make_shared<const FineCentroids>(3, 2, flat.data());
    QChamferExactQuery dist{centroids};

    // Query tokens t0=(1,0), t1=(0,1); doc uses centroid ids {0,2}.
    std::vector<float> raw_query{2.0F, 1.0F, 0.0F, 0.0F, 1.0F};
    auto doc = make_set({0.0F, 2.0F}, 4);

    svs::distance::maybe_fix_argument(dist, std::span<const float>(raw_query));
    float score =
        dist.compute(std::span<const float>(raw_query), std::span<const float>(doc));

    // Naive reference built directly from the raw query tokens and centroid vectors,
    // independent of build_table/compute_decoded.
    std::vector<std::pair<float, float>> tokens{{1.0F, 0.0F}, {0.0F, 1.0F}};
    std::vector<uint32_t> doc_ids{0, 2};
    double total = 0.0;
    for (auto [tx, ty] : tokens) {
        float best = -std::numeric_limits<float>::infinity();
        for (auto id : doc_ids) {
            float cx = flat[id * 2];
            float cy = flat[id * 2 + 1];
            best = std::max(best, cx * tx + cy * ty);
        }
        total += 1.0 - static_cast<double>(best);
    }
    float expected = static_cast<float>(total / tokens.size());

    CATCH_REQUIRE(score == Catch::Approx(expected));
    CATCH_REQUIRE(score == Catch::Approx(0.5F));
}

CATCH_TEST_CASE("GEM distance traits", "[distance][gem]") {
    CATCH_STATIC_REQUIRE(std::is_same_v<svs::distance::QEmd::compare, std::less<>>);
    CATCH_STATIC_REQUIRE(std::is_same_v<svs::distance::QChamfer::compare, std::less<>>);
    CATCH_STATIC_REQUIRE(std::is_same_v<
                         svs::distance::QChamferExactQuery::compare,
                         std::less<>>);

    CATCH_STATIC_REQUIRE(svs::distance::implicitly_broadcastable<svs::distance::QEmd>());
    CATCH_STATIC_REQUIRE(svs::distance::implicitly_broadcastable<svs::distance::QChamfer>()
    );
    CATCH_STATIC_REQUIRE(
        !svs::distance::implicitly_broadcastable<svs::distance::QChamferExactQuery>()
    );

    CATCH_STATIC_REQUIRE(!svs::distance::fix_argument_mandated<svs::distance::QEmd>());
    CATCH_STATIC_REQUIRE(!svs::distance::fix_argument_mandated<svs::distance::QChamfer>());
    CATCH_STATIC_REQUIRE(
        svs::distance::fix_argument_mandated<svs::distance::QChamferExactQuery>()
    );
}

CATCH_TEST_CASE("GEM Vamana extensions", "[distance][gem][vamana]") {
    CATCH_SECTION("Prune strategy") {
        static_assert(std::is_same_v<
                      svs::index::vamana::prune_strategy_t<svs::distance::QEmd>,
                      svs::index::vamana::ProgressivePruneStrategy>);
        static_assert(std::is_same_v<
                      svs::index::vamana::prune_strategy_t<svs::distance::QChamfer>,
                      svs::index::vamana::ProgressivePruneStrategy>);
    }

    // A default-constructed Distance member would not compile for QEmd/QChamfer (their
    // only constructor takes the CentroidTable), so these overloads existing and
    // compiling already proves they do not fall back to DefaultBuildAdaptor's generic
    // default-construct path. The compute() check below additionally proves the adaptor's
    // distance_ is wired to the real table passed in, not some other instance.
    CATCH_SECTION("build_adaptor carries the real CentroidTable through") {
        auto table = make_table(Ground::Cos);
        QEmd qemd{table};
        int dummy_data = 0;
        auto adaptor = svs::index::vamana::extensions::build_adaptor(dummy_data, qemd);
        auto a = make_set({0.0F}, 4);
        auto b = make_set({1.0F}, 4);
        float score = adaptor.general_distance().compute(
            std::span<const float>(a), std::span<const float>(b)
        );
        CATCH_REQUIRE(score == Catch::Approx(1.0F)); // table(0,1), same as the QEmd test.

        QChamfer qchamfer{table};
        auto adaptor2 = svs::index::vamana::extensions::build_adaptor(dummy_data, qchamfer);
        auto q = make_set({0.0F, 1.0F}, 6);
        auto d = make_set({2.0F, 3.0F}, 6);
        float score2 = adaptor2.general_distance().compute(
            std::span<const float>(q), std::span<const float>(d)
        );
        CATCH_REQUIRE(score2 == Catch::Approx(2.0F));
    }
}
