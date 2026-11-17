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
// tokens of `dim` elements, since MaxSim's packed-operand kernel needs one contiguous
// block.
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

// Same tokens as `make_set`, converting each float literal to T via T's constructor, so a
// test case can write one literal shape and run it against several document types.
template <typename T>
TestVectorSet<T> make_doc(std::initializer_list<std::initializer_list<float>> tokens) {
    TestVectorSet<T> set;
    set.dim = tokens.begin()->size();
    for (const auto& token : tokens) {
        for (float v : token) {
            set.flat.push_back(T(v));
        }
    }
    return set;
}

template <typename T> double promote(T x) { return static_cast<double>(x); }

// Namespace-scope (not a function-local requires-expression) so a substitution failure
// inside MaxSim<DocElement>::compute's requires-clause stays in the immediate context.
template <typename DocElement, typename D>
concept CanComputeMaxSim =
    requires(svs::distance::MaxSim<DocElement>& dist, const D& d) { dist.compute(0, d); };

// Naive double-precision oracle: sum over query tokens of the max inner product to any
// document token. Rounds the query to bf16 first for a BFloat16 document, since
// fix_argument does.
template <typename D>
double
maxsim_reference(const TestVectorSet<float>& query, const TestVectorSet<D>& document) {
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
                double qv = promote(q_token[d]);
                if constexpr (std::is_same_v<D, svs::BFloat16>) {
                    qv = promote(svs::round_to_bfloat16(q_token[d]));
                }
                dot += qv * promote(d_token[d]);
            }
            best = std::max(best, dot);
        }
        total += best;
    }
    return total;
}

// Random query/document generator shared by every element type: BFloat16 rounds
// ties-to-even, float stores the draw unrounded.
template <typename T>
TestVectorSet<T> make_random_set(size_t cardinality, size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    TestVectorSet<T> set;
    set.dim = dim;
    set.flat.resize(cardinality * dim);
    for (auto& v : set.flat) {
        float f = dist(rng);
        if constexpr (std::is_same_v<T, svs::BFloat16>) {
            v = svs::round_to_bfloat16(f);
        } else {
            v = T(f);
        }
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

template <typename D> void check_oracle_on_random_sets(std::mt19937& rng) {
    std::vector<size_t> cardinalities{1, 2, 3, 7, 9, 17};
    std::vector<size_t> dims{128, 37};
    for (size_t dim : dims) {
        for (size_t nq : cardinalities) {
            for (size_t nd : cardinalities) {
                auto query = make_random_set<float>(nq, dim, rng);
                auto document = make_random_set<D>(nd, dim, rng);
                double expected = maxsim_reference(query, document);
                float actual = maxsim_score(query, document);
                CATCH_REQUIRE(
                    static_cast<double>(actual) ==
                    Catch::Approx(expected).epsilon(MAXSIM_EPSILON).margin(MAXSIM_EPSILON)
                );
            }
        }
    }
}

} // namespace

CATCH_TEST_CASE("MaxSim distance", "[distance][maxsim]") {
    CATCH_SECTION("Traits and compile-time document-element-type check") {
        CATCH_REQUIRE(std::is_same_v<svs::distance::MaxSim<>::compare, std::greater<>>);
        CATCH_REQUIRE_FALSE(svs::distance::MaxSim<>::implicit_broadcast);
        CATCH_REQUIRE(svs::distance::MaxSim<>::must_fix_argument);
        // A mismatch is a compile error (the `requires` clause on `compute`), not a
        // run-time branch.
        static_assert(CanComputeMaxSim<float, TestVectorSet<float>>);
        static_assert(!CanComputeMaxSim<float, TestVectorSet<svs::BFloat16>>);
        static_assert(CanComputeMaxSim<svs::BFloat16, TestVectorSet<svs::BFloat16>>);
        static_assert(!CanComputeMaxSim<svs::BFloat16, TestVectorSet<float>>);
    }

    // The n_doc == 0 check is one branch in MaxSim::compute, shared by every document
    // element type; float exercises it.
    CATCH_SECTION("Empty document set gives -inf") {
        auto query = make_set<float>({{1.0F, 0.0F}});
        TestVectorSet<float> document;
        document.dim = 2;
        CATCH_REQUIRE(
            maxsim_score(query, document) == -std::numeric_limits<float>::infinity()
        );
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

    CATCH_SECTION("Exact oracle on random sets, float document") {
        std::mt19937 rng(0xC0FFEE);
        check_oracle_on_random_sets<float>(rng);
    }

    CATCH_SECTION("Exact oracle on random sets, bfloat16 document, rounded query") {
        std::mt19937 rng(0xB10A7);
        check_oracle_on_random_sets<svs::BFloat16>(rng);
    }

    CATCH_SECTION("Mixed element types") {
        // float query against Float16 document: token 0 of query matches document token
        // 0 exactly (dot = 1), token 1 has dot 0 with both document tokens.
        auto query = make_set<float>({{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}});
        auto document = make_doc<svs::Float16>({{1.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 1.0F}});
        float score = maxsim_score(query, document);
        CATCH_REQUIRE(score == Catch::Approx(1.0F).epsilon(MAXSIM_EPSILON));
    }

    // The dimension check is one branch in MaxSim::compute, shared by every document
    // element type; float exercises it.
    CATCH_SECTION("Dimension mismatch throws") {
        auto query = make_set<float>({{1.0F, 0.0F, 0.0F}});
        auto document = make_set<float>({{1.0F, 0.0F}});
        svs::distance::MaxSim<float> dist{};
        svs::distance::maybe_fix_argument(dist, query);
        CATCH_REQUIRE_THROWS_AS(
            svs::distance::compute(dist, query, document), svs::lib::ANNException
        );
    }

    CATCH_SECTION("Repeated compute calls reuse one fix_argument") {
        std::mt19937 rng(7);
        auto query = make_random_set<float>(5, 32, rng);
        svs::distance::MaxSim dist{};
        svs::distance::maybe_fix_argument(dist, query);
        for (size_t i = 0; i < 4; ++i) {
            auto document = make_random_set<float>(i + 1, 32, rng);
            float direct = svs::distance::compute(dist, query, document);
            float expected = maxsim_reference(query, document);
            CATCH_REQUIRE(
                static_cast<double>(direct) ==
                Catch::Approx(expected).epsilon(MAXSIM_EPSILON).margin(MAXSIM_EPSILON)
            );
        }
    }
}
