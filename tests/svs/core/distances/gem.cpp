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
#include "svs/core/distance/transport_simplex.h"
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
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
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

// Four unit vectors in 2D: (1,0), (0,1), (-1,0), (0,-1), so their pairwise ground
// distances come out to round numbers by hand.
std::shared_ptr<const CentroidTable> make_table(Ground ground) {
    std::vector<float> centroids{1.0F, 0.0F, 0.0F, 1.0F, -1.0F, 0.0F, 0.0F, -1.0F};
    return std::make_shared<const CentroidTable>(
        CentroidTable::from_centroids(centroids.data(), 4, 2, ground, 1)
    );
}

// K random unit vectors of width dim, row-major, deterministic per (k, dim, seed).
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

// Reference ground-cost table via double-precision dot products over bf16-rounded
// inputs (from_centroids rounds internally too), independent of InnerProductMatrix/tiling.
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
                dot += static_cast<double>(svs::round_to_bfloat16(flat[i * dim + d])) *
                       static_cast<double>(svs::round_to_bfloat16(flat[j * dim + d]));
            }
            t[i * k + j] = ground == Ground::Cos
                               ? std::max(0.0, 1.0 - dot)
                               : std::sqrt(std::max(0.0, 2.0 - 2.0 * dot));
        }
    }
    return t;
}

// Owns a 1-D id vector set contiguously, for QEmd/QChamfer/QChamferExactQuery's only
// (vector-set) compute() overloads.
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
static_assert(!svs::data::ContiguousVectorSet<std::span<const float>>);

// Independent reference: successive shortest augmenting paths via SPFA over the standard
// min-cost-flow edge-list graph. Deliberately a different algorithm from TransportSimplex.
double reference_transport_cost(
    size_t ua,
    const std::vector<double>& supply,
    size_t ub,
    const std::vector<double>& demand,
    const std::vector<double>& cost // ua * ub, row-major
) {
    constexpr double inf = std::numeric_limits<double>::infinity();
    size_t s = ua + ub, t = ua + ub + 1, n = ua + ub + 2;
    std::vector<int64_t> eto;
    std::vector<double> ecap, ecost;
    std::vector<std::vector<size_t>> adj(n);
    auto add_edge = [&](size_t from, size_t to, double cap, double c) {
        adj[from].push_back(eto.size());
        eto.push_back(static_cast<int64_t>(to));
        ecap.push_back(cap);
        ecost.push_back(c);
        adj[to].push_back(eto.size());
        eto.push_back(static_cast<int64_t>(from));
        ecap.push_back(0.0);
        ecost.push_back(-c);
    };
    for (size_t i = 0; i < ua; ++i) {
        add_edge(s, i, supply[i], 0.0);
    }
    for (size_t j = 0; j < ub; ++j) {
        add_edge(ua + j, t, demand[j], 0.0);
    }
    for (size_t i = 0; i < ua; ++i) {
        for (size_t j = 0; j < ub; ++j) {
            add_edge(i, ua + j, inf, cost[i * ub + j]);
        }
    }

    double total = 0.0;
    for (;;) {
        std::vector<double> dist(n, inf);
        std::vector<int64_t> pred_edge(n, -1);
        std::vector<bool> in_queue(n, false);
        std::deque<size_t> queue{s};
        dist[s] = 0.0;
        in_queue[s] = true;
        while (!queue.empty()) {
            size_t u = queue.front();
            queue.pop_front();
            in_queue[u] = false;
            for (size_t e : adj[u]) {
                if (ecap[e] <= 1e-12) {
                    continue;
                }
                size_t v = static_cast<size_t>(eto[e]);
                if (dist[u] + ecost[e] < dist[v] - 1e-12) {
                    dist[v] = dist[u] + ecost[e];
                    pred_edge[v] = static_cast<int64_t>(e);
                    if (!in_queue[v]) {
                        queue.push_back(v);
                        in_queue[v] = true;
                    }
                }
            }
        }
        if (dist[t] == inf) {
            break;
        }
        std::vector<size_t> path;
        for (size_t v = t; v != s; v = static_cast<size_t>(eto[path.back() ^ 1])) {
            path.push_back(static_cast<size_t>(pred_edge[v]));
        }
        double bottleneck = inf;
        for (size_t e : path) {
            bottleneck = std::min(bottleneck, ecap[e]);
        }
        for (size_t e : path) {
            ecap[e] -= bottleneck;
            ecap[e ^ 1] += bottleneck;
            total += bottleneck * ecost[e];
        }
    }
    return total;
}

} // namespace

CATCH_TEST_CASE("CentroidTable / FineCentroids", "[distance][gem]") {
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
        // fc.data actually rounds, not merely stores float32 unchanged.
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
    // K spans a partial tile, a tile boundary (255/256/257), and several tiles (600), at
    // both dims/grounds/thread-counts; the Gaussian input also exercises bf16 rounding.
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
                auto reference = naive_ground_table(flat, k, dim, ground);
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
                        // Thread-count independence and symmetry: bit for bit.
                        CATCH_REQUIRE(one_thread(ui, uj) == four_threads(ui, uj));
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
    CATCH_SECTION("Random oracle: naive min-over-doc sum, duplicate ids included") {
        std::mt19937 rng(0xC4A4FE2u);
        std::uniform_int_distribution<size_t> n_dist(1, 8);
        std::uniform_int_distribution<uint32_t> id_dist(0, 5);
        std::uniform_real_distribution<float> cost_dist(0.0F, 5.0F);
        size_t k = 6;
        for (int trial = 0; trial < 30; ++trial) {
            std::vector<float> table_flat(k * k);
            for (auto& c : table_flat) {
                c = cost_dist(rng);
            }
            auto table = std::make_shared<const CentroidTable>(k, table_flat);
            QChamfer dist{table};

            std::vector<uint32_t> q(n_dist(rng)), d(n_dist(rng));
            for (auto& id : q) {
                id = id_dist(rng);
            }
            for (auto& id : d) {
                id = id_dist(rng);
            }
            double reference = 0.0;
            for (auto qi : q) {
                float best = std::numeric_limits<float>::infinity();
                for (auto di : d) {
                    best = std::min(best, (*table)(qi, di));
                }
                reference += best;
            }

            float score = dist.compute(make_id_set(q).view(), make_id_set(d).view());
            CATCH_REQUIRE(
                static_cast<double>(score) == Catch::Approx(reference).margin(1e-4)
            );
        }
    }
}

CATCH_TEST_CASE("QEmd", "[distance][gem]") {
    QEmd dist{make_table(Ground::Cos)};

    CATCH_SECTION("1 vs 1 takes the direct table lookup, bypassing the solver") {
        float score = dist.compute(make_id_set({0}).view(), make_id_set({1}).view());
        CATCH_REQUIRE(score == Catch::Approx(1.0F)); // table(0,1)
    }

    CATCH_SECTION("Duplicates merge to a single point with full mass") {
        float score = dist.compute(make_id_set({0, 0}).view(), make_id_set({1}).view());
        CATCH_REQUIRE(score == Catch::Approx(1.0F)); // same as plain 1 vs 1
    }

    // Not exercised by the random oracle below, which never draws a == b.
    CATCH_SECTION("Identical sets give 0") {
        auto a = make_id_set({0, 1});
        CATCH_REQUIRE(dist.compute(a.view(), a.view()) == Catch::Approx(0.0F));
    }

    // Covers compute() end to end (merge_duplicates, runtime ISA dispatch, the solver)
    // against the independent reference over manually-merged weights, plus symmetry.
    CATCH_SECTION("Random oracle against the independent reference, with duplicate ids") {
        std::mt19937 rng(0x5151u);
        std::uniform_int_distribution<size_t> n_dist(2, 10);
        std::uniform_int_distribution<uint32_t> id_dist(0, 5);
        std::uniform_real_distribution<float> cost_dist(0.0F, 5.0F);
        size_t k = 6;
        auto merge = [](const std::vector<uint32_t>& ids) {
            std::map<uint32_t, double> w;
            for (auto id : ids) {
                w[id] += 1.0 / static_cast<double>(ids.size());
            }
            std::vector<uint32_t> uids;
            std::vector<double> uw;
            for (auto [id, weight] : w) {
                uids.push_back(id);
                uw.push_back(weight);
            }
            return std::pair{uids, uw};
        };

        for (int trial = 0; trial < 20; ++trial) {
            std::vector<float> table_flat(k * k);
            for (auto& c : table_flat) {
                c = cost_dist(rng);
            }
            for (size_t i = 0; i < k; ++i) {
                table_flat[i * k + i] = 0.0F;
                for (size_t j = 0; j < i; ++j) {
                    table_flat[i * k + j] = table_flat[j * k + i];
                }
            }
            auto table = std::make_shared<const CentroidTable>(k, table_flat);
            QEmd dist{table};

            std::vector<uint32_t> ids_a(n_dist(rng)), ids_b(n_dist(rng));
            for (auto& id : ids_a) {
                id = id_dist(rng);
            }
            for (auto& id : ids_b) {
                id = id_dist(rng);
            }
            auto [uids_a, uw_a] = merge(ids_a);
            auto [uids_b, uw_b] = merge(ids_b);
            std::vector<double> cost(uids_a.size() * uids_b.size());
            for (size_t i = 0; i < uids_a.size(); ++i) {
                for (size_t j = 0; j < uids_b.size(); ++j) {
                    cost[i * uids_b.size() + j] = (*table)(uids_a[i], uids_b[j]);
                }
            }
            double reference =
                reference_transport_cost(uids_a.size(), uw_a, uids_b.size(), uw_b, cost);

            float ab = dist.compute(make_id_set(ids_a).view(), make_id_set(ids_b).view());
            float ba = dist.compute(make_id_set(ids_b).view(), make_id_set(ids_a).view());
            CATCH_REQUIRE(static_cast<double>(ab) == Catch::Approx(reference).margin(1e-5));
            CATCH_REQUIRE(ba == Catch::Approx(ab));
        }
    }
}

CATCH_TEST_CASE("QEmd transportation solve: random cross-checks", "[distance][gem]") {
    using svs::distance::AVX_AVAILABILITY;
    using svs::distance::QEmdImpl;

    std::mt19937 rng(0x7EA5CADEu);
    std::uniform_int_distribution<size_t> size_dist(1, 40);
    std::uniform_real_distribution<double> cost_dist(0.0, 10.0);

    // Checks compute_decoded's NONE path against the reference and every ISA path against
    // NONE. `tiny_cost_range >= 0`/`uniform_weights` force the degenerate (tied) case.
    auto run_case = [&](size_t ua,
                        size_t ub,
                        double weight_spread,
                        bool uniform_weights = false,
                        int tiny_cost_range = -1,
                        int repeated_levels = 0,
                        bool row_index_cost = false) {
        std::vector<double> w_a(ua), w_b(ub);
        if (uniform_weights) {
            std::fill(w_a.begin(), w_a.end(), 1.0 / static_cast<double>(ua));
            std::fill(w_b.begin(), w_b.end(), 1.0 / static_cast<double>(ub));
        } else if (repeated_levels > 0) {
            // Weights from a small integer set, normalized: ties the walk without making
            // every weight equal, unlike uniform_weights.
            std::uniform_int_distribution<int> level_dist(1, repeated_levels);
            for (auto* w : {&w_a, &w_b}) {
                double total = 0.0;
                for (auto& x : *w) {
                    x = static_cast<double>(level_dist(rng));
                    total += x;
                }
                for (auto& x : *w) {
                    x /= total;
                }
            }
        } else {
            for (auto* w : {&w_a, &w_b}) {
                double total = 0.0;
                std::uniform_real_distribution<double> wd(0.01, weight_spread);
                for (auto& x : *w) {
                    x = wd(rng);
                    total += x;
                }
                for (auto& x : *w) {
                    x /= total; // Same normalization merge_sorted_duplicates relies on.
                }
            }
        }
        size_t k = std::max(ua, ub);
        std::vector<uint32_t> ids_a(ua), ids_b(ub);
        std::vector<double> cost(ua * ub);
        std::vector<float> table_flat(k * k, 0.0F);
        std::uniform_int_distribution<int> tiny_cost_dist(0, std::max(tiny_cost_range, 0));
        for (size_t i = 0; i < ua; ++i) {
            ids_a[i] = static_cast<uint32_t>(i);
            for (size_t j = 0; j < ub; ++j) {
                // row_index_cost: row 0 is cheapest for every column, by construction.
                double c = row_index_cost         ? static_cast<double>(i)
                           : tiny_cost_range >= 0 ? static_cast<double>(tiny_cost_dist(rng))
                                                  : cost_dist(rng);
                cost[i * ub + j] = c;
                table_flat[i * k + j] = static_cast<float>(c);
            }
        }
        for (size_t j = 0; j < ub; ++j) {
            ids_b[j] = static_cast<uint32_t>(j);
        }
        CentroidTable table(k, table_flat);
        double max_cost = *std::max_element(cost.begin(), cost.end());

        double reference = reference_transport_cost(ua, w_a, ub, w_b, cost);
        float none_result = QEmdImpl<AVX_AVAILABILITY::NONE>::compute_decoded(
            table, ids_a.data(), w_a.data(), ua, ids_b.data(), w_b.data(), ub
        );
        CATCH_REQUIRE(
            static_cast<double>(none_result) == Catch::Approx(reference).margin(1e-6)
        );

        // Tolerance vs. NONE: the optimized solver's derived optimality-gap bound, plus 2
        // float ulps of none_result for the float cast/accumulation NONE itself already
        // did.
        auto isa_margin = [&](double none) {
            double gap =
                svs::distance::TransportSimplex<AVX_AVAILABILITY::AVX2>::error_bound(
                    ua, ub, max_cost, 1.0
                );
            double ulp = 2 * static_cast<double>(std::numeric_limits<float>::epsilon()) *
                         std::max(1.0, std::abs(none));
            return gap + ulp;
        };

#if defined(__x86_64__)
        if (svs::detail::avx_runtime_flags.is_avx2_supported()) {
            float avx2_result = QEmdImpl<AVX_AVAILABILITY::AVX2>::compute_decoded(
                table, ids_a.data(), w_a.data(), ua, ids_b.data(), w_b.data(), ub
            );
            CATCH_REQUIRE(
                static_cast<double>(avx2_result) ==
                Catch::Approx(static_cast<double>(none_result))
                    .margin(isa_margin(static_cast<double>(none_result)))
            );
        }
        if (svs::detail::avx_runtime_flags.is_avx512f_supported()) {
            float avx512_result = QEmdImpl<AVX_AVAILABILITY::AVX512>::compute_decoded(
                table, ids_a.data(), w_a.data(), ua, ids_b.data(), w_b.data(), ub
            );
            CATCH_REQUIRE(
                static_cast<double>(avx512_result) ==
                Catch::Approx(static_cast<double>(none_result))
                    .margin(isa_margin(static_cast<double>(none_result)))
            );
        }
#endif
    };

    CATCH_SECTION("Random problems") {
        for (int trial = 0; trial < 64; ++trial) {
            run_case(size_dist(rng), size_dist(rng), 1.0);
        }
    }
    CATCH_SECTION("Single node on one side") {
        for (int trial = 0; trial < 5; ++trial) {
            run_case(1, size_dist(rng), 1.0);
            run_case(size_dist(rng), 1, 1.0);
        }
    }
    CATCH_SECTION("Large weight dynamic range") {
        for (int trial = 0; trial < 10; ++trial) {
            run_case(size_dist(rng), size_dist(rng), 1e6);
        }
    }
    // Merges the "many equal costs" and "uniform-weight tiny-integer-cost" families: both
    // are highly degenerate (many tied reduced costs), just via a different knob.
    CATCH_SECTION("Degenerate: equal costs and tiny integer cost ranges") {
        std::uniform_int_distribution<size_t> big_size_dist(1, 64);
        struct Config {
            bool uniform_weights;
            int tiny_cost_range;
        };
        for (Config cfg : {Config{false, 0}, Config{true, 2}}) {
            for (int trial = 0; trial < 15; ++trial) {
                run_case(
                    big_size_dist(rng),
                    big_size_dist(rng),
                    1.0,
                    cfg.uniform_weights,
                    cfg.tiny_cost_range
                );
            }
        }
    }
    CATCH_SECTION("Larger shapes: dense, lopsided, transpose, and the Dynamic bucket") {
        // 4x256/256x4 exercise the transpose; 5x300/300x5 cross into the Dynamic bucket
        // (side > 256); 1x256/256x1 are single-source/sink; the last two are tied costs.
        run_case(100, 100, 1.0);
        run_case(4, 256, 1.0);
        run_case(256, 4, 1.0);
        run_case(3, 200, 1.0);
        run_case(5, 300, 1.0);
        run_case(300, 5, 1.0);
        run_case(1, 256, 1.0);
        run_case(256, 1, 1.0);
        run_case(100, 100, 1.0, /*uniform_weights=*/true, /*tiny_cost_range=*/0);
        run_case(100, 100, 1.0, /*uniform_weights=*/true, /*tiny_cost_range=*/1);
    }
    // Ties in weights and costs are where the start tree's walk adds zero-flow arcs and
    // where the cheapest-row column grouping collapses; both threaten strong feasibility.
    CATCH_SECTION("Start-tree degeneracy: tied costs, repeated weights, extreme shapes") {
        for (int tiny_cost_range : {0, 1}) {
            for (bool uniform_weights : {false, true}) {
                int repeated_levels = uniform_weights ? 0 : 3;
                for (size_t n : {size_t{2}, size_t{7}, size_t{64}}) {
                    // With uniform weights, supply and demand tie at every walk step.
                    run_case(n, n, 1.0, uniform_weights, tiny_cost_range, repeated_levels);
                }
                // 300 keeps the Dynamic (uint32 id) bucket's 1xn/nx1 cases tied too.
                for (size_t n :
                     {size_t{2}, size_t{7}, size_t{64}, size_t{256}, size_t{300}}) {
                    run_case(1, n, 1.0, uniform_weights, tiny_cost_range, repeated_levels);
                    run_case(n, 1, 1.0, uniform_weights, tiny_cost_range, repeated_levels);
                }
            }
        }

        // Uniform weights only: other weights raise the reference's augmenting-path count
        // toward ua+ub-1 and dominate the section's runtime.
        run_case(256, 256, 1.0, /*uniform_weights=*/true, /*tiny_cost_range=*/0);
        run_case(256, 256, 1.0, /*uniform_weights=*/true, /*tiny_cost_range=*/1);
        run_case(257, 257, 1.0, /*uniform_weights=*/true, /*tiny_cost_range=*/0);
        run_case(257, 257, 1.0, /*uniform_weights=*/true, /*tiny_cost_range=*/1);

        // Tied, lopsided, and into the Dynamic bucket on the larger side.
        run_case(3, 300, 1.0, false, /*tiny_cost_range=*/1, /*repeated_levels=*/3);
        run_case(300, 3, 1.0, false, /*tiny_cost_range=*/1, /*repeated_levels=*/3);

        // Row 0 is cheapest for every column, so every column lands in one group.
        run_case(64, 64, 1.0, false, -1, 3, /*row_index_cost=*/true);
        run_case(256, 64, 1.0, false, -1, 3, /*row_index_cost=*/true);
    }
}

CATCH_TEST_CASE("QChamferExactQuery", "[distance][gem]") {
    std::vector<float> flat2x2{1.0F, 0.0F, 0.0F, 1.0F};
    QChamferExactQuery dist{std::make_shared<const FineCentroids>(2, 2, flat2x2.data())};

    CATCH_SECTION("fix_argument: query dimension mismatch throws") {
        RawQuerySet wrong_dim{{1.0F, 0.0F, 0.0F}, 3};
        CATCH_REQUIRE_THROWS_AS(
            svs::distance::maybe_fix_argument(dist, wrong_dim.view()), std::invalid_argument
        );
    }

    // Uniform(-1,1) draws are, with overwhelming probability, not exactly representable
    // in bfloat16, so this oracle exercises the rounding path without a hardcoded value.
    CATCH_SECTION("Random oracle: naive reference over non-bf16-representable values") {
        std::mt19937 rng(0xB16u);
        std::uniform_real_distribution<float> val_dist(-1.0F, 1.0F);
        std::uniform_int_distribution<uint32_t> id_dist(0, 3);
        size_t k = 4, dim = 3;
        for (int trial = 0; trial < 15; ++trial) {
            std::vector<float> flat(k * dim);
            for (auto& x : flat) {
                x = val_dist(rng);
            }
            auto centroids = std::make_shared<const FineCentroids>(k, dim, flat.data());
            QChamferExactQuery trial_dist{centroids};

            size_t n_query = 1 + (trial % 3);
            std::vector<float> tokens(n_query * dim);
            for (auto& x : tokens) {
                x = val_dist(rng);
            }
            RawQuerySet query_set{tokens, dim};
            svs::distance::maybe_fix_argument(trial_dist, query_set.view());

            std::vector<uint32_t> doc_ids(2 + (trial % 2));
            for (auto& id : doc_ids) {
                id = id_dist(rng);
            }
            float score = trial_dist.compute(query_set.view(), make_id_set(doc_ids).view());

            // Independent of fix_argument_decoded and the kernel: rounds both sides
            // itself, then takes the max dot per query token and averages 1 - max.
            double total = 0.0;
            for (size_t i = 0; i < n_query; ++i) {
                float best = -std::numeric_limits<float>::infinity();
                for (auto id : doc_ids) {
                    double dot = 0.0;
                    for (size_t d = 0; d < dim; ++d) {
                        double cv = svs::round_to_bfloat16(flat[id * dim + d]);
                        double qv = svs::round_to_bfloat16(tokens[i * dim + d]);
                        dot += cv * qv;
                    }
                    best = std::max(best, static_cast<float>(dot));
                }
                total += 1.0 - static_cast<double>(best);
            }
            float expected = static_cast<float>(total / static_cast<double>(n_query));
            CATCH_REQUIRE(score == Catch::Approx(expected).margin(1e-4));
        }
    }
}

// Each distance's compute() calls the shared gem_detail::require_id_dimensions from a
// different call site; one section per site, same underlying check.
CATCH_TEST_CASE("id vector set dimensions() != 1 throws", "[distance][gem]") {
    std::vector<uint32_t> raw{0, 1, 2, 3};
    auto bad = svs::data::ConstVectorSet<uint32_t>(std::span<const uint32_t>(raw), 2, 2);
    auto good = make_id_set({0});

    CATCH_SECTION("QEmd") {
        QEmd dist{make_table(Ground::Cos)};
        CATCH_REQUIRE_THROWS_AS(dist.compute(bad, good.view()), std::invalid_argument);
        CATCH_REQUIRE_THROWS_AS(dist.compute(good.view(), bad), std::invalid_argument);
    }
    CATCH_SECTION("QChamfer") {
        QChamfer dist{make_table(Ground::Cos)};
        CATCH_REQUIRE_THROWS_AS(dist.compute(bad, good.view()), std::invalid_argument);
        CATCH_REQUIRE_THROWS_AS(dist.compute(good.view(), bad), std::invalid_argument);
    }
    CATCH_SECTION("QChamferExactQuery (doc side)") {
        std::vector<float> flat{1.0F, 0.0F, 0.0F, 1.0F};
        QChamferExactQuery dist{std::make_shared<const FineCentroids>(2, 2, flat.data())};
        RawQuerySet query_set{{1.0F, 0.0F}, 2};
        svs::distance::maybe_fix_argument(dist, query_set.view());
        CATCH_REQUIRE_THROWS_AS(dist.compute(query_set.view(), bad), std::invalid_argument);
    }
}

CATCH_TEST_CASE(
    "Generic svs::distance::compute dispatches to each distance's compute() overload",
    "[distance][gem]"
) {
    CATCH_SECTION("QEmd") {
        QEmd dist{make_table(Ground::Cos)};
        auto a = make_id_set({0});
        auto b = make_id_set({1, 2});
        CATCH_REQUIRE(
            svs::distance::compute(dist, a.view(), b.view()) ==
            Catch::Approx(dist.compute(a.view(), b.view()))
        );
    }

    CATCH_SECTION("QChamfer") {
        QChamfer dist{make_table(Ground::Cos)};
        auto q = make_id_set({0, 1});
        auto d = make_id_set({2, 3});
        CATCH_REQUIRE(
            svs::distance::compute(dist, q.view(), d.view()) ==
            Catch::Approx(dist.compute(q.view(), d.view()))
        );
    }

    CATCH_SECTION("QChamferExactQuery") {
        std::vector<float> flat{1.0F, 0.0F, 0.0F, 1.0F};
        QChamferExactQuery dist{std::make_shared<const FineCentroids>(2, 2, flat.data())};
        RawQuerySet query_set{{1.0F, 0.0F}, 2};
        auto doc_set = make_id_set({0, 1});
        svs::distance::maybe_fix_argument(dist, query_set.view());
        CATCH_REQUIRE(
            svs::distance::compute(dist, query_set.view(), doc_set.view()) ==
            Catch::Approx(dist.compute(query_set.view(), doc_set.view()))
        );
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

    // Compiling proves these skip DefaultBuildAdaptor's default-construct path (QEmd/
    // QChamfer have no default ctor); compute() below proves distance_ holds the real
    // table.
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
