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
#include "svs/core/distance/maxsim.h"

// svs
#include "svs/lib/float16.h"

// catch2
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

// stdlib
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <span>
#include <type_traits>
#include <vector>

namespace {

// Minimal svs::data::VectorSet model over a vector of row-major tokens; local to this
// test so it does not depend on the ConstVectorSet view another agent is adding.
template <typename T> struct TestVectorSet {
    using element_type = T;
    std::vector<std::vector<T>> tokens;

    size_t cardinality() const { return tokens.size(); }
    size_t dimensions() const { return tokens.empty() ? 0 : tokens[0].size(); }
    std::span<const T> token(size_t i) const { return std::span<const T>(tokens[i]); }
};

template <typename T> double promote(T x) { return static_cast<double>(x); }

// Naive double-precision oracle: sum over query tokens of the max inner product to any
// document token.
template <typename Q, typename D>
double maxsim_reference(const TestVectorSet<Q>& query, const TestVectorSet<D>& document) {
    if (document.cardinality() == 0) {
        return -std::numeric_limits<double>::infinity();
    }
    double total = 0.0;
    for (size_t i = 0; i < query.cardinality(); ++i) {
        double best = -std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < document.cardinality(); ++j) {
            double dot = 0.0;
            for (size_t d = 0; d < query.dimensions(); ++d) {
                dot += promote(query.tokens[i][d]) * promote(document.tokens[j][d]);
            }
            best = std::max(best, dot);
        }
        total += best;
    }
    return total;
}

TestVectorSet<float> make_random_set(size_t cardinality, size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    TestVectorSet<float> set;
    set.tokens.resize(cardinality);
    for (auto& token : set.tokens) {
        token.resize(dim);
        for (auto& v : token) {
            v = dist(rng);
        }
    }
    return set;
}

constexpr double MAXSIM_EPSILON = 1e-4;

} // namespace

CATCH_TEST_CASE("MaxSim distance", "[distance][maxsim]") {
    CATCH_SECTION("Trait members") {
        CATCH_REQUIRE(std::is_same_v<svs::distance::MaxSim::compare, std::greater<>>);
        CATCH_REQUIRE(svs::distance::MaxSim::implicit_broadcast);
    }

    CATCH_SECTION("Hand-computed tiny case") {
        // Query has two orthonormal tokens; document has one token aligned with the
        // first and one aligned with neither, so the per-query maxima are 1 and 0.
        TestVectorSet<float> query{{{1.0F, 0.0F}, {0.0F, 1.0F}}};
        TestVectorSet<float> document{{{1.0F, 0.0F}, {0.0F, 0.0F}}};
        float score = svs::distance::compute(svs::distance::MaxSim{}, query, document);
        CATCH_REQUIRE(score == Catch::Approx(1.0F).epsilon(MAXSIM_EPSILON));
    }

    CATCH_SECTION("Empty document set gives -inf") {
        TestVectorSet<float> query{{{1.0F, 0.0F}}};
        TestVectorSet<float> document{{}};
        float score = svs::distance::compute(svs::distance::MaxSim{}, query, document);
        CATCH_REQUIRE(score == -std::numeric_limits<float>::infinity());
    }

    CATCH_SECTION("Asymmetry") {
        // `a` has two tokens that each max out against `b`'s lone token, so swapping
        // query/document sums over a different number of maxima (2 vs. 1).
        TestVectorSet<float> a{{{1.0F, 0.0F}, {1.0F, 0.0F}}};
        TestVectorSet<float> b{{{1.0F, 0.0F}}};
        float score_ab = svs::distance::compute(svs::distance::MaxSim{}, a, b);
        float score_ba = svs::distance::compute(svs::distance::MaxSim{}, b, a);
        CATCH_REQUIRE(score_ab != Catch::Approx(score_ba));
    }

    CATCH_SECTION("Exact oracle on random sets") {
        std::mt19937 rng(0xC0FFEE);
        std::vector<size_t> cardinalities{1, 2, 3, 7, 9, 17};
        std::vector<size_t> dims{128, 37};
        for (size_t dim : dims) {
            for (size_t nq : cardinalities) {
                for (size_t nd : cardinalities) {
                    auto query = make_random_set(nq, dim, rng);
                    auto document = make_random_set(nd, dim, rng);
                    double expected = maxsim_reference(query, document);
                    float actual =
                        svs::distance::compute(svs::distance::MaxSim{}, query, document);
                    CATCH_REQUIRE(
                        static_cast<double>(actual) == Catch::Approx(expected)
                                                           .epsilon(MAXSIM_EPSILON)
                                                           .margin(MAXSIM_EPSILON)
                    );
                }
            }
        }
    }

    CATCH_SECTION("Mixed element types") {
        // float query against Float16 document: an AVX-accelerated pair the existing
        // inner product supports (IPImpl<N, float, Float16, ...>).
        TestVectorSet<float> query{{{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}}};
        TestVectorSet<svs::Float16> document{
            {{svs::Float16(1.0F), svs::Float16(0.0F), svs::Float16(0.0F)},
             {svs::Float16(0.0F), svs::Float16(0.0F), svs::Float16(1.0F)}}};
        float score = svs::distance::compute(svs::distance::MaxSim{}, query, document);
        // Token 0 of query matches document token 0 exactly (dot = 1); token 1 of query
        // has dot 0 with both document tokens.
        CATCH_REQUIRE(score == Catch::Approx(1.0F).epsilon(MAXSIM_EPSILON));
    }

    CATCH_SECTION("Batch helper matches per-pair compute") {
        std::mt19937 rng(42);
        auto query = make_random_set(5, 64, rng);
        std::vector<TestVectorSet<float>> candidates;
        for (size_t i = 0; i < 6; ++i) {
            candidates.push_back(make_random_set(i + 1, 64, rng));
        }
        std::vector<uint32_t> ids{0, 1, 2, 3, 4, 5};
        std::vector<float> scores(ids.size());
        svs::distance::maxsim_batch<TestVectorSet<float>>(
            query,
            [&](uint32_t id) -> const TestVectorSet<float>& { return candidates[id]; },
            std::span<const uint32_t>(ids),
            std::span<float>(scores)
        );
        for (size_t i = 0; i < ids.size(); ++i) {
            float expected =
                svs::distance::compute(svs::distance::MaxSim{}, query, candidates[ids[i]]);
            CATCH_REQUIRE(scores[i] == expected);
        }
    }
}
