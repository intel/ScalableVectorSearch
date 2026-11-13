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
#include "svs/concepts/distance.h"
#include "svs/lib/bfloat16.h"
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

// Minimal svs::data::ContiguousVectorSet model: a flat, row-major buffer of `cardinality`
// tokens of `dim` elements each. Flat (rather than a vector of per-token vectors) because
// MaxSim's packed-operand kernel needs `data()` to return one contiguous block.
template <typename T> struct TestVectorSet {
    using element_type = T;
    std::vector<T> flat;
    size_t dim = 0;

    size_t cardinality() const { return dim == 0 ? 0 : flat.size() / dim; }
    size_t dimensions() const { return dim; }
    std::span<const T> token(size_t i) const {
        return std::span<const T>(flat.data() + i * dim, dim);
    }
    const T* data() const { return flat.data(); }
};

template <typename T>
TestVectorSet<T> make_set(std::initializer_list<std::initializer_list<T>> tokens) {
    TestVectorSet<T> set;
    set.dim = tokens.begin()->size();
    for (const auto& token : tokens) {
        set.flat.insert(set.flat.end(), token.begin(), token.end());
    }
    return set;
}

template <typename T> double promote(T x) { return static_cast<double>(x); }

// Namespace-scope concept (rather than a function-local requires-expression) so a
// substitution failure inside MaxSim<DocElement>::compute's requires-clause, which
// mentions the enclosing class template's parameter, stays in the immediate context for
// SFINAE purposes.
template <typename DocElement, typename D>
concept CanComputeMaxSim =
    requires(svs::distance::MaxSim<DocElement>& dist, const D& d) { dist.compute(0, d); };

// Naive double-precision oracle: sum over query tokens of the max inner product to any
// document token.
template <typename Q, typename D>
double maxsim_reference(const TestVectorSet<Q>& query, const TestVectorSet<D>& document) {
    if (document.cardinality() == 0) {
        return -std::numeric_limits<double>::infinity();
    }
    double total = 0.0;
    for (size_t i = 0; i < query.cardinality(); ++i) {
        auto q_token = query.token(i);
        double best = -std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < document.cardinality(); ++j) {
            auto d_token = document.token(j);
            double dot = 0.0;
            for (size_t d = 0; d < query.dimensions(); ++d) {
                dot += promote(q_token[d]) * promote(d_token[d]);
            }
            best = std::max(best, dot);
        }
        total += best;
    }
    return total;
}

// Same oracle as `maxsim_reference`, but both the query and the document values are
// first rounded to bfloat16 precision: matches what MaxSim<svs::BFloat16> actually
// computes, since fix_argument rounds the query (pack_rounded_to_bf16) and the document
// is already held at bf16 precision.
double maxsim_reference_bf16(
    const TestVectorSet<float>& query, const TestVectorSet<svs::BFloat16>& document
) {
    if (document.cardinality() == 0) {
        return -std::numeric_limits<double>::infinity();
    }
    double total = 0.0;
    for (size_t i = 0; i < query.cardinality(); ++i) {
        auto q_token = query.token(i);
        double best = -std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < document.cardinality(); ++j) {
            auto d_token = document.token(j);
            double dot = 0.0;
            for (size_t d = 0; d < query.dimensions(); ++d) {
                double qv = promote(svs::round_to_bfloat16(q_token[d]));
                double dv = promote(d_token[d]);
                dot += qv * dv;
            }
            best = std::max(best, dot);
        }
        total += best;
    }
    return total;
}

TestVectorSet<svs::BFloat16>
make_random_bf16_set(size_t cardinality, size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    TestVectorSet<svs::BFloat16> set;
    set.dim = dim;
    set.flat.resize(cardinality * dim);
    for (auto& v : set.flat) {
        v = svs::round_to_bfloat16(dist(rng));
    }
    return set;
}

TestVectorSet<float> make_random_set(size_t cardinality, size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    TestVectorSet<float> set;
    set.dim = dim;
    set.flat.resize(cardinality * dim);
    for (auto& v : set.flat) {
        v = dist(rng);
    }
    return set;
}

// Packs `query` into a fresh MaxSim functor, through the generic maybe_fix_argument hook
// (the protocol every ShouldFix-opted-in distance goes through), then scores `document`.
template <typename D>
float maxsim_score(const TestVectorSet<float>& query, const D& document) {
    svs::distance::MaxSim<typename D::element_type> dist{};
    svs::distance::maybe_fix_argument(dist, query);
    return svs::distance::compute(dist, query, document);
}

constexpr double MAXSIM_EPSILON = 1e-4;

} // namespace

CATCH_TEST_CASE("MaxSim distance", "[distance][maxsim]") {
    CATCH_SECTION("Trait members") {
        CATCH_REQUIRE(std::is_same_v<svs::distance::MaxSim<>::compare, std::greater<>>);
        CATCH_REQUIRE_FALSE(svs::distance::MaxSim<>::implicit_broadcast);
        CATCH_REQUIRE(svs::distance::MaxSim<>::must_fix_argument);
    }

    CATCH_SECTION("Hand-computed tiny case") {
        // Query has two orthonormal tokens; document has one token aligned with the
        // first and one aligned with neither, so the per-query maxima are 1 and 0.
        auto query = make_set<float>({{1.0F, 0.0F}, {0.0F, 1.0F}});
        auto document = make_set<float>({{1.0F, 0.0F}, {0.0F, 0.0F}});
        float score = maxsim_score(query, document);
        CATCH_REQUIRE(score == Catch::Approx(1.0F).epsilon(MAXSIM_EPSILON));
    }

    CATCH_SECTION("Empty document set gives -inf") {
        auto query = make_set<float>({{1.0F, 0.0F}});
        TestVectorSet<float> document;
        document.dim = 2;
        float score = maxsim_score(query, document);
        CATCH_REQUIRE(score == -std::numeric_limits<float>::infinity());
    }

    CATCH_SECTION("Asymmetry") {
        // `a` has two tokens that each max out against `b`'s lone token, so swapping
        // query/document sums over a different number of maxima (2 vs. 1).
        auto a = make_set<float>({{1.0F, 0.0F}, {1.0F, 0.0F}});
        auto b = make_set<float>({{1.0F, 0.0F}});
        float score_ab = maxsim_score(a, b);
        float score_ba = maxsim_score(b, a);
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
                    float actual = maxsim_score(query, document);
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
        // float query against Float16 document.
        auto query = make_set<float>({{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}});
        auto document = make_set<svs::Float16>(
            {{svs::Float16(1.0F), svs::Float16(0.0F), svs::Float16(0.0F)},
             {svs::Float16(0.0F), svs::Float16(0.0F), svs::Float16(1.0F)}}
        );
        float score = maxsim_score(query, document);
        // Token 0 of query matches document token 0 exactly (dot = 1); token 1 of query
        // has dot 0 with both document tokens.
        CATCH_REQUIRE(score == Catch::Approx(1.0F).epsilon(MAXSIM_EPSILON));
    }

    CATCH_SECTION("Dimension mismatch throws") {
        auto query = make_set<float>({{1.0F, 0.0F, 0.0F}});
        auto document = make_set<float>({{1.0F, 0.0F}});
        svs::distance::MaxSim dist{};
        svs::distance::maybe_fix_argument(dist, query);
        CATCH_REQUIRE_THROWS_AS(
            svs::distance::compute(dist, query, document), svs::lib::ANNException
        );
    }

    CATCH_SECTION("Repeated compute calls reuse one fix_argument") {
        std::mt19937 rng(7);
        auto query = make_random_set(5, 32, rng);
        svs::distance::MaxSim dist{};
        svs::distance::maybe_fix_argument(dist, query);
        for (size_t i = 0; i < 4; ++i) {
            auto document = make_random_set(i + 1, 32, rng);
            float direct = svs::distance::compute(dist, query, document);
            float expected = maxsim_reference(query, document);
            CATCH_REQUIRE(
                static_cast<double>(direct) ==
                Catch::Approx(expected).epsilon(MAXSIM_EPSILON).margin(MAXSIM_EPSILON)
            );
        }
    }

    CATCH_SECTION("Hand-computed bfloat16 document case") {
        auto query = make_set<float>({{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}});
        auto document = make_set<svs::BFloat16>(
            {{svs::BFloat16(1.0F), svs::BFloat16(0.0F), svs::BFloat16(0.0F)},
             {svs::BFloat16(0.0F), svs::BFloat16(0.0F), svs::BFloat16(1.0F)}}
        );
        float score = maxsim_score(query, document);
        // Both values are exactly representable in bfloat16, so rounding changes nothing
        // here: token 0 of query matches document token 0 exactly (dot = 1); token 1 has
        // dot 0 with both document tokens.
        CATCH_REQUIRE(score == Catch::Approx(1.0F).epsilon(MAXSIM_EPSILON));
    }

    CATCH_SECTION("Empty bfloat16 document set gives -inf") {
        auto query = make_set<float>({{1.0F, 0.0F}});
        TestVectorSet<svs::BFloat16> document;
        document.dim = 2;
        float score = maxsim_score(query, document);
        CATCH_REQUIRE(score == -std::numeric_limits<float>::infinity());
    }

    CATCH_SECTION("Bfloat16 dimension mismatch throws") {
        auto query = make_set<float>({{1.0F, 0.0F, 0.0F}});
        auto document =
            make_set<svs::BFloat16>({{svs::BFloat16(1.0F), svs::BFloat16(0.0F)}});
        svs::distance::MaxSim<svs::BFloat16> dist{};
        svs::distance::maybe_fix_argument(dist, query);
        CATCH_REQUIRE_THROWS_AS(
            svs::distance::compute(dist, query, document), svs::lib::ANNException
        );
    }

    CATCH_SECTION("Bfloat16 document exact oracle on random sets, rounded query") {
        std::mt19937 rng(0xB10A7);
        std::vector<size_t> cardinalities{1, 2, 3, 7, 9, 17};
        std::vector<size_t> dims{128, 37};
        for (size_t dim : dims) {
            for (size_t nq : cardinalities) {
                for (size_t nd : cardinalities) {
                    auto query = make_random_set(nq, dim, rng);
                    auto document = make_random_bf16_set(nd, dim, rng);
                    double expected = maxsim_reference_bf16(query, document);
                    float actual = maxsim_score(query, document);
                    CATCH_REQUIRE(
                        static_cast<double>(actual) == Catch::Approx(expected)
                                                           .epsilon(MAXSIM_EPSILON)
                                                           .margin(MAXSIM_EPSILON)
                    );
                }
            }
        }
    }

    CATCH_SECTION("Document element type is checked at compile time") {
        using FloatDoc = TestVectorSet<float>;
        using BF16Doc = TestVectorSet<svs::BFloat16>;
        // A MaxSim<float> accepts a float document and rejects a bfloat16 one: the
        // document's element type is part of the functor's type, so a mismatch is a
        // compile error (the `requires` clause on `compute`), not a run-time branch.
        static_assert(CanComputeMaxSim<float, FloatDoc>);
        static_assert(!CanComputeMaxSim<float, BF16Doc>);
        static_assert(CanComputeMaxSim<svs::BFloat16, BF16Doc>);
        static_assert(!CanComputeMaxSim<svs::BFloat16, FloatDoc>);
    }
}
