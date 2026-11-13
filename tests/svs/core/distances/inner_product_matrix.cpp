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
#include "svs/core/distance/inner_product_matrix.h"

// svs
#include "svs/lib/avx_detection.h"

// catch2
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

// stdlib
#include <cstddef>
#include <random>
#include <vector>

namespace {

using svs::distance::AVX_AVAILABILITY;
using svs::distance::InnerProductMatrix;
using svs::distance::InnerProductMatrixImpl;

// float32 accumulation over at most d=129 terms of values in [-1, 1]: summation order
// differs between the generic and SIMD kernels, so the tolerance must absorb that, not
// just single-ULP rounding.
constexpr double REL_EPS = 1e-4;
constexpr double ABS_EPS = 1e-5;

std::vector<float> random_matrix(size_t rows, size_t d, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> out(rows * d);
    for (auto& v : out) {
        v = dist(rng);
    }
    return out;
}

// Naive double-precision reference: out[i * n + j] = <a_i, b_j>.
std::vector<double> reference(
    const std::vector<float>& a, size_t m, const std::vector<float>& b, size_t n, size_t d
) {
    std::vector<double> out(m * n, 0.0);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (size_t k = 0; k < d; ++k) {
                acc +=
                    static_cast<double>(a[i * d + k]) * static_cast<double>(b[j * d + k]);
            }
            out[i * n + j] = acc;
        }
    }
    return out;
}

void require_close(const std::vector<float>& got, const std::vector<double>& expected) {
    CATCH_REQUIRE(got.size() == expected.size());
    for (size_t i = 0; i < got.size(); ++i) {
        CATCH_REQUIRE(
            static_cast<double>(got[i]) ==
            Catch::Approx(expected[i]).epsilon(REL_EPS).margin(ABS_EPS)
        );
    }
}

// m x n x d shapes exercising every tail: row-block remainder, SIMD lane remainder (8 and
// 16 wide), and both small and multi-cache-line `d`.
const std::vector<size_t> ms{1, 2, 3, 7, 8, 9, 33};
const std::vector<size_t> ns{1, 2, 15, 16, 17, 31, 32, 33, 64};
const std::vector<size_t> ds{1, 7, 128, 129};

template <AVX_AVAILABILITY Avx> void check_against_reference() {
    unsigned seed = 0;
    for (size_t d : ds) {
        for (size_t m : ms) {
            for (size_t n : ns) {
                auto a = random_matrix(m, d, seed++);
                auto b = random_matrix(n, d, seed++);
                std::vector<float> out(m * n, -1.0F);
                InnerProductMatrixImpl<Avx>::build(a.data(), m, b.data(), n, d, out.data());
                require_close(out, reference(a, m, b, n, d));
            }
        }
    }
}

template <AVX_AVAILABILITY Avx> void check_against_generic() {
    unsigned seed = 1000;
    for (size_t d : ds) {
        for (size_t m : ms) {
            for (size_t n : ns) {
                auto a = random_matrix(m, d, seed++);
                auto b = random_matrix(n, d, seed++);
                std::vector<float> generic_out(m * n, -1.0F);
                std::vector<float> isa_out(m * n, -2.0F);
                InnerProductMatrixImpl<AVX_AVAILABILITY::NONE>::build(
                    a.data(), m, b.data(), n, d, generic_out.data()
                );
                InnerProductMatrixImpl<Avx>::build(
                    a.data(), m, b.data(), n, d, isa_out.data()
                );
                for (size_t i = 0; i < generic_out.size(); ++i) {
                    CATCH_REQUIRE(
                        static_cast<double>(isa_out[i]) ==
                        Catch::Approx(static_cast<double>(generic_out[i]))
                            .epsilon(REL_EPS)
                            .margin(ABS_EPS)
                    );
                }
            }
        }
    }
}

} // namespace

CATCH_TEST_CASE(
    "InnerProductMatrix generic kernel matches a naive reference",
    "[distance][inner_product_matrix]"
) {
    check_against_reference<AVX_AVAILABILITY::NONE>();
}

CATCH_TEST_CASE(
    "InnerProductMatrix AVX2 kernel matches generic", "[distance][inner_product_matrix]"
) {
    if (!svs::detail::avx_runtime_flags.is_avx2_supported()) {
        return;
    }
    check_against_reference<AVX_AVAILABILITY::AVX2>();
    check_against_generic<AVX_AVAILABILITY::AVX2>();
}

CATCH_TEST_CASE(
    "InnerProductMatrix AVX512 kernel matches generic", "[distance][inner_product_matrix]"
) {
    if (!svs::detail::avx_runtime_flags.is_avx512f_supported()) {
        return;
    }
    check_against_reference<AVX_AVAILABILITY::AVX512>();
    check_against_generic<AVX_AVAILABILITY::AVX512>();
}

CATCH_TEST_CASE(
    "InnerProductMatrix public wrapper equals the reference",
    "[distance][inner_product_matrix]"
) {
    for (size_t d : {1UL, 7UL, 128UL, 129UL}) {
        for (size_t m : {1UL, 9UL, 33UL}) {
            for (size_t n : {1UL, 17UL, 64UL}) {
                auto a = random_matrix(m, d, static_cast<unsigned>(m * 100 + n * 10 + d));
                auto b = random_matrix(n, d, static_cast<unsigned>(d * 100 + m * 10 + n));
                std::vector<float> out(m * n, -1.0F);
                InnerProductMatrix::compute(a.data(), m, b.data(), n, d, out.data());
                require_close(out, reference(a, m, b, n, d));
            }
        }
    }
}

CATCH_TEST_CASE(
    "InnerProductMatrix degenerate shapes leave `out` untouched",
    "[distance][inner_product_matrix]"
) {
    constexpr size_t d = 4;
    auto a = random_matrix(3, d, 7);
    auto b = random_matrix(3, d, 11);

    CATCH_SECTION("m == 0") {
        std::vector<float> out{1.0F, 2.0F, 3.0F};
        auto before = out;
        InnerProductMatrix::compute(a.data(), 0, b.data(), 3, d, out.data());
        CATCH_REQUIRE(out == before);
    }

    CATCH_SECTION("n == 0") {
        std::vector<float> out{1.0F, 2.0F, 3.0F};
        auto before = out;
        InnerProductMatrix::compute(a.data(), 3, b.data(), 0, d, out.data());
        CATCH_REQUIRE(out == before);
    }
}
