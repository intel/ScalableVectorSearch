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
#include "svs/core/data/vector_set_view.h"
#include "svs/index/vamana/prune.h"
#include "svs/lib/bfloat16.h"

// catch2
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

// stdlib
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
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

// K unit vectors of width dim, row-major: deterministic per (k, dim, seed) so the
// from_centroids tiling test below is reproducible across runs.
std::vector<float> random_unit_centroids(size_t k, size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0F, 1.0F);
    std::vector<float> flat(k * dim);
    for (size_t i = 0; i < k; ++i) {
        float* row = flat.data() + i * dim;
        double norm2 = 0.0;
        for (size_t d = 0; d < dim; ++d) {
            row[d] = gauss(rng);
            norm2 += static_cast<double>(row[d]) * row[d];
        }
        double norm = std::sqrt(norm2);
        for (size_t d = 0; d < dim; ++d) {
            row[d] = static_cast<float>(row[d] / norm);
        }
    }
    return flat;
}

// from_centroids now rounds every centroid component to bfloat16 (round to nearest, ties
// to even) before computing, so a reference built from the unrounded input would differ
// by bf16's ~0.4% rounding error, not just summation order; round here too so the
// pre-existing 1e-4 margin still bounds only the latter.
std::vector<float> round_to_bf16_vec(const std::vector<float>& flat) {
    std::vector<float> out(flat.size());
    for (size_t i = 0; i < flat.size(); ++i) {
        out[i] = static_cast<float>(svs::round_to_bfloat16(flat[i]));
    }
    return out;
}

// Reference ground-cost table computed with double-precision dot products and the same
// formula from_centroids applies, independent of InnerProductMatrix/tiling.
std::vector<double>
naive_ground_table(const std::vector<float>& flat, size_t k, size_t dim, Ground ground) {
    std::vector<double> t(k * k, 0.0);
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < k; ++j) {
            if (i == j) {
                continue;
            }
            double dot = 0.0;
            for (size_t d = 0; d < dim; ++d) {
                dot += static_cast<double>(flat[i * dim + d]) * flat[j * dim + d];
            }
            t[i * k + j] = ground == Ground::Cos
                               ? std::max(0.0, 1.0 - dot)
                               : std::sqrt(std::max(0.0, 2.0 - 2.0 * dot));
        }
    }
    return t;
}

// Owns a 1-D id vector set contiguously, for exercising QEmd/QChamfer/QChamferExactQuery's
// only (vector-set) compute() overloads.
struct IdSet {
    std::vector<uint32_t> ids;

    svs::data::ConstVectorSet<uint32_t, 1> view() const {
        return svs::data::ConstVectorSet<uint32_t, 1>(
            std::span<const uint32_t>(ids), ids.size(), 1
        );
    }
};

IdSet make_id_set(std::vector<uint32_t> ids) { return IdSet{std::move(ids)}; }

// Owns a raw-query vector set for QChamferExactQuery: `n` raw fp32 tokens of width `dim`,
// row-major, contiguously.
struct RawQuerySet {
    std::vector<float> tokens;
    size_t dim;

    svs::data::ConstVectorSet<float> view() const {
        return svs::data::ConstVectorSet<float>(
            std::span<const float>(tokens), tokens.size() / dim, dim
        );
    }
};

// Compile-time guard: the removed fixed-width padded-row layout must not come back.
// QChamfer's only compute() overload requires ContiguousVectorSet, so a type that fails
// that concept (a bare span, the old layout's carrier) cannot satisfy it either.
static_assert(!svs::data::ContiguousVectorSet<std::span<const float>>);

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

    CATCH_SECTION("FineCentroids rounds to bfloat16, round to nearest even") {
        CATCH_STATIC_REQUIRE(std::is_same_v<
                             std::decay_t<decltype(std::declval<FineCentroids&>().data[0])>,
                             svs::BFloat16>);
        // None of these are exactly representable in bfloat16: the test only holds if
        // fc.data actually rounds (matching svs::round_to_bfloat16), not merely stores
        // float32 unchanged or truncates like BFloat16's own constructor would.
        std::vector<float> flat{0.1F, -0.1F, 1.0F / 3.0F, -2.0F / 3.0F};
        FineCentroids fc{2, 2, flat.data()};
        for (size_t i = 0; i < flat.size(); ++i) {
            CATCH_REQUIRE(fc.data[i] == svs::round_to_bfloat16(flat[i]));
        }
    }

    CATCH_SECTION("FineCentroids::load round-trips through the unchanged float32 file "
                  "format") {
        std::vector<float> flat{0.1F, 0.2F, 0.3F, -0.4F, 2.5F, -3.5F};
        auto path =
            std::filesystem::temp_directory_path() / "svs_gem_fine_centroids_test.bin";
        {
            std::ofstream out(path, std::ios::binary);
            out.write(
                reinterpret_cast<const char*>(flat.data()),
                static_cast<std::streamsize>(flat.size() * sizeof(float))
            );
        }
        auto fc = FineCentroids::load(path, 2);
        std::filesystem::remove(path);
        CATCH_REQUIRE(fc.K == 3);
        CATCH_REQUIRE(fc.dim == 2);
        for (size_t i = 0; i < flat.size(); ++i) {
            CATCH_REQUIRE(fc.data[i] == svs::round_to_bfloat16(flat[i]));
        }
    }
}

CATCH_TEST_CASE("CentroidTable::from_centroids tiling", "[distance][gem]") {
    // K values exercising a single partial tile (1, 2, 7), a single full tile (255 off
    // by one, 256 exact, 257 off by one), and several tiles (600); at a small (3) and a
    // production (128) dimension, both ground distances, built with 1 and 4 threads.
    for (size_t k :
         {size_t{1},
          size_t{2},
          size_t{7},
          size_t{255},
          size_t{256},
          size_t{257},
          size_t{600}}) {
        for (size_t dim : {size_t{3}, size_t{128}}) {
            for (Ground ground : {Ground::Cos, Ground::L2}) {
                auto flat = random_unit_centroids(
                    k,
                    dim,
                    static_cast<uint32_t>(
                        k * 1009 + dim * 31 + (ground == Ground::Cos ? 1 : 2)
                    )
                );
                // from_centroids rounds internally; round the reference's input the same
                // way so the comparison below isolates summation-order error, not bf16
                // rounding error.
                auto reference =
                    naive_ground_table(round_to_bf16_vec(flat), k, dim, ground);

                auto one_thread =
                    CentroidTable::from_centroids(flat.data(), k, dim, ground, 1);
                auto four_threads =
                    CentroidTable::from_centroids(flat.data(), k, dim, ground, 4);

                for (size_t i = 0; i < k; ++i) {
                    CATCH_REQUIRE(
                        one_thread(static_cast<uint32_t>(i), static_cast<uint32_t>(i)) ==
                        0.0F
                    );
                    for (size_t j = 0; j < k; ++j) {
                        auto ui = static_cast<uint32_t>(i);
                        auto uj = static_cast<uint32_t>(j);
                        // Thread-count independence: bit for bit, not approximate.
                        CATCH_REQUIRE(one_thread(ui, uj) == four_threads(ui, uj));
                        // Exact symmetry by construction: bit for bit.
                        CATCH_REQUIRE(one_thread(ui, uj) == one_thread(uj, ui));
                        // Against a double-precision reference: summation order differs,
                        // so only approximate agreement is expected.
                        CATCH_REQUIRE(
                            static_cast<double>(one_thread(ui, uj)) ==
                            Catch::Approx(reference[i * k + j]).margin(1e-4)
                        );
                    }
                }
            }
        }
    }
}

CATCH_TEST_CASE("QChamfer", "[distance][gem]") {
    auto table = make_table(Ground::Cos);
    QChamfer dist{table};

    CATCH_SECTION("Hand-computed: min over doc per query token") {
        // query ids {0,1}, doc ids {2,3}: table(0,2)=2,table(0,3)=1 -> min 1;
        // table(1,2)=1,table(1,3)=2 -> min 1. total = 1 + 1 = 2.
        auto q = make_id_set({0, 1});
        auto d = make_id_set({2, 3});
        float score = dist.compute(q.view(), d.view());
        CATCH_REQUIRE(score == Catch::Approx(2.0F));
    }

    CATCH_SECTION("Duplicate ids are counted, not merged") {
        // query {0,0} against doc {1}: each of the two query tokens pays table(0,1)=1.
        auto q = make_id_set({0, 0});
        auto d = make_id_set({1});
        float score = dist.compute(q.view(), d.view());
        CATCH_REQUIRE(score == Catch::Approx(2.0F));
    }

    CATCH_SECTION("Cardinality 1") {
        auto q = make_id_set({0});
        auto d = make_id_set({2});
        float score = dist.compute(q.view(), d.view());
        CATCH_REQUIRE(score == Catch::Approx(2.0F));
    }

    CATCH_SECTION("Different cardinalities") {
        // query {0,0,1} vs doc {2}: table(0,2)=2 (twice, one per query token) +
        // table(1,2)=1 = 5.
        auto q = make_id_set({0, 0, 1});
        auto d = make_id_set({2});
        float score = dist.compute(q.view(), d.view());
        CATCH_REQUIRE(score == Catch::Approx(5.0F));
    }

    CATCH_SECTION("dimensions() != 1 throws") {
        std::vector<uint32_t> raw{0, 1, 2, 3};
        auto bad =
            svs::data::ConstVectorSet<uint32_t>(std::span<const uint32_t>(raw), 2, 2);
        auto good = make_id_set({0});
        CATCH_REQUIRE_THROWS_AS(dist.compute(bad, good.view()), std::invalid_argument);
        CATCH_REQUIRE_THROWS_AS(dist.compute(good.view(), bad), std::invalid_argument);
    }

    CATCH_SECTION("Generic svs::distance::compute dispatches to the vector-set overload") {
        auto q = make_id_set({0, 1});
        auto d = make_id_set({2, 3});
        float via_generic = svs::distance::compute(dist, q.view(), d.view());
        float direct = dist.compute(q.view(), d.view());
        CATCH_REQUIRE(via_generic == Catch::Approx(direct));
    }
}

CATCH_TEST_CASE("QEmd", "[distance][gem]") {
    auto table = make_table(Ground::Cos);
    QEmd dist{table};

    CATCH_SECTION("1 vs 1") {
        auto a = make_id_set({0});
        auto b = make_id_set({1});
        float score = dist.compute(a.view(), b.view());
        CATCH_REQUIRE(score == Catch::Approx(1.0F)); // table(0,1)
    }

    CATCH_SECTION("1 vs 2") {
        // One source of mass 1 split 0.5/0.5 to two distinct sinks: 0.5*table(0,1) +
        // 0.5*table(0,2) = 0.5*1 + 0.5*2 = 1.5.
        auto a = make_id_set({0});
        auto b = make_id_set({1, 2});
        float score = dist.compute(a.view(), b.view());
        CATCH_REQUIRE(score == Catch::Approx(1.5F));
    }

    CATCH_SECTION("2 vs 2 with obvious optimal matching") {
        // Masses 0.5/0.5 on each side; matching 0->3, 1->2 costs 0.5*1+0.5*1=1, strictly
        // less than the cross matching 0->2, 1->3 (0.5*2+0.5*2=2).
        auto a = make_id_set({0, 1});
        auto b = make_id_set({2, 3});
        float score = dist.compute(a.view(), b.view());
        CATCH_REQUIRE(score == Catch::Approx(1.0F));
    }

    CATCH_SECTION("Duplicates merge to a single point with full mass") {
        // {0,0} merges to one point of mass 1 at id 0, same as the plain "1 vs 1" case.
        auto a = make_id_set({0, 0});
        auto b = make_id_set({1});
        float score = dist.compute(a.view(), b.view());
        CATCH_REQUIRE(score == Catch::Approx(1.0F));
    }

    CATCH_SECTION("Identical sets give 0") {
        auto a = make_id_set({0, 1});
        float score = dist.compute(a.view(), a.view());
        CATCH_REQUIRE(score == Catch::Approx(0.0F));
    }

    CATCH_SECTION("Symmetry") {
        auto a = make_id_set({0});
        auto b = make_id_set({1, 2});
        float ab = dist.compute(a.view(), b.view());
        float ba = dist.compute(b.view(), a.view());
        CATCH_REQUIRE(ab == Catch::Approx(ba));
    }

    CATCH_SECTION("dimensions() != 1 throws") {
        std::vector<uint32_t> raw{0, 1, 2, 3};
        auto bad =
            svs::data::ConstVectorSet<uint32_t>(std::span<const uint32_t>(raw), 2, 2);
        auto good = make_id_set({0});
        CATCH_REQUIRE_THROWS_AS(dist.compute(bad, good.view()), std::invalid_argument);
        CATCH_REQUIRE_THROWS_AS(dist.compute(good.view(), bad), std::invalid_argument);
    }

    CATCH_SECTION("Generic svs::distance::compute dispatches to the vector-set overload") {
        auto a = make_id_set({0});
        auto b = make_id_set({1, 2});
        float via_generic = svs::distance::compute(dist, a.view(), b.view());
        float direct = dist.compute(a.view(), b.view());
        CATCH_REQUIRE(via_generic == Catch::Approx(direct));
    }
}

CATCH_TEST_CASE("QChamferExactQuery", "[distance][gem]") {
    // Raw (non-unit) centroids: c0=(1,0), c1=(0,1), c2=(-1,0).
    std::vector<float> flat{1.0F, 0.0F, 0.0F, 1.0F, -1.0F, 0.0F};
    auto centroids = std::make_shared<const FineCentroids>(3, 2, flat.data());

    // Query tokens t0=(1,0), t1=(0,1); doc uses centroid ids {0,2}.
    RawQuerySet query_set{{1.0F, 0.0F, 0.0F, 1.0F}, 2};
    auto doc_set = make_id_set({0, 2});

    CATCH_SECTION("Hand-computed: naive reference over raw query tokens") {
        QChamferExactQuery dist{centroids};
        svs::distance::maybe_fix_argument(dist, query_set.view());
        float score = dist.compute(query_set.view(), doc_set.view());

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

    CATCH_SECTION("Generic maybe_fix_argument/compute dispatch to the vector-set overload"
    ) {
        QChamferExactQuery dist{centroids};
        svs::distance::maybe_fix_argument(dist, query_set.view());
        float direct = dist.compute(query_set.view(), doc_set.view());
        float via_generic = svs::distance::compute(dist, query_set.view(), doc_set.view());
        CATCH_REQUIRE(via_generic == Catch::Approx(direct));
    }

    CATCH_SECTION("fix_argument: query dimension mismatch throws") {
        QChamferExactQuery dist{centroids};
        RawQuerySet wrong_dim{{1.0F, 0.0F, 0.0F}, 3};
        CATCH_REQUIRE_THROWS_AS(
            svs::distance::maybe_fix_argument(dist, wrong_dim.view()), std::invalid_argument
        );
    }

    CATCH_SECTION("compute: doc dimensions() != 1 throws") {
        QChamferExactQuery dist{centroids};
        svs::distance::maybe_fix_argument(dist, query_set.view());
        std::vector<uint32_t> raw{0, 1, 2, 3};
        auto bad_doc =
            svs::data::ConstVectorSet<uint32_t>(std::span<const uint32_t>(raw), 2, 2);
        CATCH_REQUIRE_THROWS_AS(
            dist.compute(query_set.view(), bad_doc), std::invalid_argument
        );
    }
}

CATCH_TEST_CASE(
    "GEM bf16 rounding: non-representable centroid and query values", "[distance][gem]"
) {
    // None of these are exactly representable in bfloat16: unlike every other fixture in
    // this file (chosen to be bf16-exact so their hand-computed expectations hold
    // unchanged), this case exists specifically to exercise rounding.
    std::vector<float> flat{0.1F, 0.2F, -0.3F, 0.4F, 1.0F / 3.0F, -2.0F / 3.0F};
    size_t k = 3;
    size_t dim = 2;

    CATCH_SECTION("CentroidTable::from_centroids matches a bf16-rounded naive reference") {
        auto reference = naive_ground_table(round_to_bf16_vec(flat), k, dim, Ground::Cos);
        auto table = CentroidTable::from_centroids(flat.data(), k, dim, Ground::Cos, 1);
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < k; ++j) {
                auto ui = static_cast<uint32_t>(i);
                auto uj = static_cast<uint32_t>(j);
                CATCH_REQUIRE(
                    static_cast<double>(table(ui, uj)) ==
                    Catch::Approx(reference[i * k + j]).margin(1e-4)
                );
            }
        }
    }

    CATCH_SECTION("QChamferExactQuery matches a naive reference over bf16-rounded values") {
        auto centroids = std::make_shared<const FineCentroids>(k, dim, flat.data());
        QChamferExactQuery dist{centroids};

        std::vector<float> query{0.15F, -0.25F};
        RawQuerySet query_set{query, dim};
        auto doc_set = make_id_set({0, 2});

        svs::distance::maybe_fix_argument(dist, query_set.view());
        float score = dist.compute(query_set.view(), doc_set.view());

        // Naive reference over the same bf16-rounded centroid and query values (round to
        // nearest, ties to even on both sides), independent of fix_argument_decoded and
        // the kernel.
        auto rounded_centroids = round_to_bf16_vec(flat);
        float qx = static_cast<float>(svs::round_to_bfloat16(query[0]));
        float qy = static_cast<float>(svs::round_to_bfloat16(query[1]));
        std::vector<uint32_t> doc_ids{0, 2};
        float best = -std::numeric_limits<float>::infinity();
        for (auto id : doc_ids) {
            float cx = rounded_centroids[id * dim];
            float cy = rounded_centroids[id * dim + 1];
            best = std::max(best, cx * qx + cy * qy);
        }
        float expected = 1.0F - best; // One query token: the "mean" is just this term.

        CATCH_REQUIRE(score == Catch::Approx(expected));
    }
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
        auto a = make_id_set({0});
        auto b = make_id_set({1});
        float score = adaptor.general_distance().compute(a.view(), b.view());
        CATCH_REQUIRE(score == Catch::Approx(1.0F)); // table(0,1), same as the QEmd test.

        QChamfer qchamfer{table};
        auto adaptor2 = svs::index::vamana::extensions::build_adaptor(dummy_data, qchamfer);
        auto q = make_id_set({0, 1});
        auto d = make_id_set({2, 3});
        float score2 = adaptor2.general_distance().compute(q.view(), d.view());
        CATCH_REQUIRE(score2 == Catch::Approx(2.0F));
    }
}
