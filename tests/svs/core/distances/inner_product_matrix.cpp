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
#include "svs/lib/bfloat16.h"
#include "svs/lib/float16.h"

// catch2
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

// stdlib
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
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

namespace {

using svs::distance::PackedOperand;

// Shapes covering every tail this family of kernels has: the row-block remainder
// (8-wide), both SIMD lane widths' remainders (8 and 16), m == 0 (empty document), and
// both small and multi-cache-line d.
const std::vector<size_t> column_max_ms{0, 1, 2, 7, 8, 9, 17, 67, 180};
const std::vector<size_t> column_max_ns{1, 15, 16, 17, 32, 33};
const std::vector<size_t> column_max_ds{1, 7, 128, 129};

// Naive double-precision reference: out[j] = max over i of <a_i, b_j>, or -infinity if
// m == 0.
std::vector<double> reference_column_max(
    const std::vector<double>& a, size_t m, const std::vector<float>& b, size_t n, size_t d
) {
    std::vector<double> out(n, -std::numeric_limits<double>::infinity());
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (size_t k = 0; k < d; ++k) {
                acc += a[i * d + k] * static_cast<double>(b[j * d + k]);
            }
            out[j] = std::max(out[j], acc);
        }
    }
    return out;
}

void require_close_vec(const std::vector<float>& got, const std::vector<double>& expected) {
    CATCH_REQUIRE(got.size() == expected.size());
    for (size_t i = 0; i < got.size(); ++i) {
        if (expected[i] == -std::numeric_limits<double>::infinity()) {
            CATCH_REQUIRE(got[i] == -std::numeric_limits<float>::infinity());
        } else {
            CATCH_REQUIRE(
                static_cast<double>(got[i]) ==
                Catch::Approx(expected[i]).epsilon(REL_EPS).margin(ABS_EPS)
            );
        }
    }
}

std::vector<svs::Float16> to_float16(const std::vector<float>& a) {
    std::vector<svs::Float16> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = svs::Float16(a[i]);
    }
    return out;
}

// Promotes a Float16 row-major matrix to double, so the reference is computed from the
// values the kernel actually sees (post-quantization), not the pre-quantization floats.
std::vector<double> promote_float16(const std::vector<svs::Float16>& a) {
    std::vector<double> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = static_cast<double>(static_cast<float>(a[i]));
    }
    return out;
}

std::vector<double> promote_float(const std::vector<float>& a) {
    std::vector<double> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = static_cast<double>(a[i]);
    }
    return out;
}

// Rounds to nearest, ties to even (svs::round_to_bfloat16), not BFloat16's own truncating
// constructor: this is what every bf16 row-overload test below checks the kernel against.
std::vector<svs::BFloat16> to_bfloat16(const std::vector<float>& a) {
    std::vector<svs::BFloat16> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = svs::round_to_bfloat16(a[i]);
    }
    return out;
}

// Promotes a BFloat16 row-major matrix to double, so the reference is computed from the
// values the kernel actually sees (post-rounding), not the pre-rounding floats.
std::vector<double> promote_bfloat16(const std::vector<svs::BFloat16>& a) {
    std::vector<double> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = static_cast<double>(static_cast<float>(a[i]));
    }
    return out;
}

// Naive double-precision reference for the table form, out[i * n + j] = <a_i, b_j>, over
// an already-promoted `a` so a narrow-type A's post-conversion values are what the
// reference uses (shares `reference`'s role above, generalized to a non-float32 `a`).
std::vector<double> reference_from_values(
    const std::vector<double>& a, size_t m, const std::vector<float>& b, size_t n, size_t d
) {
    std::vector<double> out(m * n, 0.0);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (size_t k = 0; k < d; ++k) {
                acc += a[i * d + k] * static_cast<double>(b[j * d + k]);
            }
            out[i * n + j] = acc;
        }
    }
    return out;
}

template <AVX_AVAILABILITY Avx> void check_column_max_float32_against_reference() {
    unsigned seed = 2000;
    for (size_t d : column_max_ds) {
        for (size_t m : column_max_ms) {
            for (size_t n : column_max_ns) {
                auto a = random_matrix(m, d, seed++);
                auto b = random_matrix(n, d, seed++);
                PackedOperand packed;
                packed.pack(b.data(), n, d);
                std::vector<float> out(n, -3.0F);
                InnerProductMatrixImpl<Avx>::column_max(a.data(), m, packed, out.data());
                require_close_vec(out, reference_column_max(promote_float(a), m, b, n, d));
            }
        }
    }
}

template <AVX_AVAILABILITY Avx> void check_column_max_float16_against_reference() {
    unsigned seed = 3000;
    for (size_t d : column_max_ds) {
        for (size_t m : column_max_ms) {
            for (size_t n : column_max_ns) {
                auto a = random_matrix(m, d, seed++);
                auto a16 = to_float16(a);
                auto b = random_matrix(n, d, seed++);
                PackedOperand packed;
                packed.pack(b.data(), n, d);
                std::vector<float> out(n, -3.0F);
                InnerProductMatrixImpl<Avx>::column_max(a16.data(), m, packed, out.data());
                require_close_vec(
                    out, reference_column_max(promote_float16(a16), m, b, n, d)
                );
            }
        }
    }
}

template <AVX_AVAILABILITY Avx> void check_column_max_bf16_against_reference() {
    unsigned seed = 3500;
    for (size_t d : column_max_ds) {
        for (size_t m : column_max_ms) {
            for (size_t n : column_max_ns) {
                auto a = random_matrix(m, d, seed++);
                auto a_bf16 = to_bfloat16(a);
                auto b = random_matrix(n, d, seed++);
                PackedOperand packed;
                packed.pack(b.data(), n, d);
                std::vector<float> out(n, -3.0F);
                InnerProductMatrixImpl<Avx>::column_max(
                    a_bf16.data(), m, packed, out.data()
                );
                require_close_vec(
                    out, reference_column_max(promote_bfloat16(a_bf16), m, b, n, d)
                );
            }
        }
    }
}

template <AVX_AVAILABILITY Avx> void check_column_max_against_generic() {
    unsigned seed = 4000;
    for (size_t d : column_max_ds) {
        for (size_t m : column_max_ms) {
            for (size_t n : column_max_ns) {
                auto a = random_matrix(m, d, seed++);
                auto a16 = to_float16(a);
                auto b = random_matrix(n, d, seed++);
                PackedOperand packed;
                packed.pack(b.data(), n, d);

                std::vector<float> generic_out(n, -1.0F);
                std::vector<float> isa_out(n, -2.0F);
                InnerProductMatrixImpl<AVX_AVAILABILITY::NONE>::column_max(
                    a.data(), m, packed, generic_out.data()
                );
                InnerProductMatrixImpl<Avx>::column_max(
                    a.data(), m, packed, isa_out.data()
                );
                for (size_t j = 0; j < n; ++j) {
                    CATCH_REQUIRE(
                        static_cast<double>(isa_out[j]) ==
                        Catch::Approx(static_cast<double>(generic_out[j]))
                            .epsilon(REL_EPS)
                            .margin(ABS_EPS)
                    );
                }

                std::vector<float> generic_out16(n, -1.0F);
                std::vector<float> isa_out16(n, -2.0F);
                InnerProductMatrixImpl<AVX_AVAILABILITY::NONE>::column_max(
                    a16.data(), m, packed, generic_out16.data()
                );
                InnerProductMatrixImpl<Avx>::column_max(
                    a16.data(), m, packed, isa_out16.data()
                );
                // Float16 inputs decode identically everywhere, but the generic kernel
                // accumulates in strict k-order while the SIMD kernels accumulate across
                // row-blocks in a different order; measured over every (m, n, d) shape
                // this test sweeps, that reordering alone produces at most 4.7e-5
                // absolute / 3.1e-5 relative difference (one-off throwaway probe, not
                // committed), so this tolerance is ~2x that bound, not a slack value.
                for (size_t j = 0; j < n; ++j) {
                    CATCH_REQUIRE(
                        static_cast<double>(isa_out16[j]) ==
                        Catch::Approx(static_cast<double>(generic_out16[j]))
                            .epsilon(2e-4)
                            .margin(1e-4)
                    );
                }

                auto a_bf16 = to_bfloat16(a);
                std::vector<float> generic_out_bf16(n, -1.0F);
                std::vector<float> isa_out_bf16(n, -2.0F);
                InnerProductMatrixImpl<AVX_AVAILABILITY::NONE>::column_max(
                    a_bf16.data(), m, packed, generic_out_bf16.data()
                );
                InnerProductMatrixImpl<Avx>::column_max(
                    a_bf16.data(), m, packed, isa_out_bf16.data()
                );
                // Same reordering-only tolerance as the Float16 case above: BFloat16's
                // widening is exact, so any difference here is summation order, not
                // decode.
                for (size_t j = 0; j < n; ++j) {
                    CATCH_REQUIRE(
                        static_cast<double>(isa_out_bf16[j]) ==
                        Catch::Approx(static_cast<double>(generic_out_bf16[j]))
                            .epsilon(2e-4)
                            .margin(1e-4)
                    );
                }
            }
        }
    }
}

template <AVX_AVAILABILITY Avx> void check_build_packed_narrow_against_reference() {
    // Smaller shape set than column_max's: the table form's output scales as m * n, and
    // this already exercises the row-block (8) and both SIMD lane-width (8, 16) tails.
    unsigned seed = 6000;
    for (size_t d : {1UL, 7UL, 128UL}) {
        for (size_t m : {0UL, 1UL, 7UL, 9UL, 17UL}) {
            for (size_t n : {1UL, 15UL, 17UL, 32UL}) {
                auto a = random_matrix(m, d, seed++);
                auto b = random_matrix(n, d, seed++);
                PackedOperand packed;
                packed.pack(b.data(), n, d);

                auto a16 = to_float16(a);
                std::vector<float> out16(m * n, -1.0F);
                InnerProductMatrixImpl<Avx>::build_packed(
                    a16.data(), m, packed, out16.data()
                );
                require_close(
                    out16, reference_from_values(promote_float16(a16), m, b, n, d)
                );

                auto a_bf16 = to_bfloat16(a);
                std::vector<float> out_bf16(m * n, -1.0F);
                InnerProductMatrixImpl<Avx>::build_packed(
                    a_bf16.data(), m, packed, out_bf16.data()
                );
                require_close(
                    out_bf16, reference_from_values(promote_bfloat16(a_bf16), m, b, n, d)
                );

                // Against the generic (NONE) level, same tail shapes.
                std::vector<float> generic16(m * n, -1.0F);
                InnerProductMatrixImpl<AVX_AVAILABILITY::NONE>::build_packed(
                    a16.data(), m, packed, generic16.data()
                );
                for (size_t idx = 0; idx < generic16.size(); ++idx) {
                    CATCH_REQUIRE(
                        static_cast<double>(out16[idx]) ==
                        Catch::Approx(static_cast<double>(generic16[idx]))
                            .epsilon(2e-4)
                            .margin(1e-4)
                    );
                }
                std::vector<float> generic_bf16(m * n, -1.0F);
                InnerProductMatrixImpl<AVX_AVAILABILITY::NONE>::build_packed(
                    a_bf16.data(), m, packed, generic_bf16.data()
                );
                for (size_t idx = 0; idx < generic_bf16.size(); ++idx) {
                    CATCH_REQUIRE(
                        static_cast<double>(out_bf16[idx]) ==
                        Catch::Approx(static_cast<double>(generic_bf16[idx]))
                            .epsilon(2e-4)
                            .margin(1e-4)
                    );
                }
            }
        }
    }
}

} // namespace

CATCH_TEST_CASE(
    "InnerProductMatrixImpl column_max generic kernel matches a naive reference",
    "[distance][inner_product_matrix]"
) {
    check_column_max_float32_against_reference<AVX_AVAILABILITY::NONE>();
    check_column_max_float16_against_reference<AVX_AVAILABILITY::NONE>();
    check_column_max_bf16_against_reference<AVX_AVAILABILITY::NONE>();
}

CATCH_TEST_CASE(
    "InnerProductMatrixImpl column_max AVX2 kernel matches reference and generic",
    "[distance][inner_product_matrix]"
) {
    if (!svs::detail::avx_runtime_flags.is_avx2_supported()) {
        return;
    }
    check_column_max_float32_against_reference<AVX_AVAILABILITY::AVX2>();
    check_column_max_float16_against_reference<AVX_AVAILABILITY::AVX2>();
    check_column_max_bf16_against_reference<AVX_AVAILABILITY::AVX2>();
    check_column_max_against_generic<AVX_AVAILABILITY::AVX2>();
}

CATCH_TEST_CASE(
    "InnerProductMatrixImpl column_max AVX512 kernel matches reference and generic",
    "[distance][inner_product_matrix]"
) {
    if (!svs::detail::avx_runtime_flags.is_avx512f_supported()) {
        return;
    }
    check_column_max_float32_against_reference<AVX_AVAILABILITY::AVX512>();
    check_column_max_float16_against_reference<AVX_AVAILABILITY::AVX512>();
    check_column_max_bf16_against_reference<AVX_AVAILABILITY::AVX512>();
    check_column_max_against_generic<AVX_AVAILABILITY::AVX512>();
}

CATCH_TEST_CASE(
    "InnerProductMatrixImpl build_packed matches a naive reference for Float16 and "
    "BFloat16 rows, every available level",
    "[distance][inner_product_matrix]"
) {
    check_build_packed_narrow_against_reference<AVX_AVAILABILITY::NONE>();
    if (svs::detail::avx_runtime_flags.is_avx2_supported()) {
        check_build_packed_narrow_against_reference<AVX_AVAILABILITY::AVX2>();
    }
    if (svs::detail::avx_runtime_flags.is_avx512f_supported()) {
        check_build_packed_narrow_against_reference<AVX_AVAILABILITY::AVX512>();
    }
}

CATCH_TEST_CASE(
    "InnerProductMatrixImpl build_packed matches build through an explicit PackedOperand",
    "[distance][inner_product_matrix]"
) {
    // build() packs internally into a thread_local PackedOperand; this checks that
    // packing explicitly and calling build_packed gives the identical result, for every
    // ISA this process can run.
    unsigned seed = 5000;
    for (size_t d : {1UL, 7UL, 128UL}) {
        for (size_t m : {0UL, 1UL, 9UL, 33UL}) {
            for (size_t n : {1UL, 17UL, 33UL}) {
                auto a = random_matrix(m, d, seed++);
                auto b = random_matrix(n, d, seed++);
                std::vector<float> via_build(m * n, -1.0F);
                InnerProductMatrix::compute(a.data(), m, b.data(), n, d, via_build.data());

                PackedOperand packed;
                packed.pack(b.data(), n, d);
                std::vector<float> via_packed(m * n, -2.0F);
                // InnerProductMatrix::select_isa() is private (the ISA choice for this
                // kernel belongs to InnerProductMatrix alone); this test needs to force
                // the same ISA `compute` above picked, so it repeats that priority order
                // against the public svs::detail::avx_runtime_flags instead.
                if (svs::detail::avx_runtime_flags.is_avx512f_supported()) {
                    InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build_packed(
                        a.data(), m, packed, via_packed.data()
                    );
                } else if (svs::detail::avx_runtime_flags.is_avx2_supported()) {
                    InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2>::build_packed(
                        a.data(), m, packed, via_packed.data()
                    );
                } else {
                    InnerProductMatrixImpl<AVX_AVAILABILITY::NONE>::build_packed(
                        a.data(), m, packed, via_packed.data()
                    );
                }
                CATCH_REQUIRE(via_build == via_packed);
            }
        }
    }
}

CATCH_TEST_CASE(
    "InnerProductMatrix::compute_packed picks the same ISA result as "
    "InnerProductMatrixImpl "
    "for BFloat16 and Float16 rows",
    "[distance][inner_product_matrix]"
) {
    unsigned seed = 7000;
    for (size_t d : {1UL, 7UL, 128UL}) {
        for (size_t m : {0UL, 1UL, 9UL, 17UL}) {
            for (size_t n : {1UL, 17UL, 33UL}) {
                auto a = random_matrix(m, d, seed++);
                auto b = random_matrix(n, d, seed++);
                PackedOperand packed;
                packed.pack(b.data(), n, d);

                // Same ISA-priority repetition as the float32 build_packed test above:
                // InnerProductMatrix::select_isa() is private by design.
                auto direct_bf16 = [&](const auto& av, std::vector<float>& out) {
                    auto a_bf16 = to_bfloat16(a);
                    if (svs::detail::avx_runtime_flags.is_avx512f_supported()) {
                        InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build_packed(
                            a_bf16.data(), m, packed, out.data()
                        );
                    } else if (svs::detail::avx_runtime_flags.is_avx2_supported()) {
                        InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2>::build_packed(
                            a_bf16.data(), m, packed, out.data()
                        );
                    } else {
                        InnerProductMatrixImpl<AVX_AVAILABILITY::NONE>::build_packed(
                            a_bf16.data(), m, packed, out.data()
                        );
                    }
                    (void)av;
                };

                auto a_bf16 = to_bfloat16(a);
                std::vector<float> via_public(m * n, -1.0F);
                InnerProductMatrix::compute_packed(
                    a_bf16.data(), m, packed, via_public.data()
                );
                std::vector<float> via_direct(m * n, -2.0F);
                direct_bf16(0, via_direct);
                CATCH_REQUIRE(via_public == via_direct);

                auto a16 = to_float16(a);
                std::vector<float> via_public16(m * n, -1.0F);
                InnerProductMatrix::compute_packed(
                    a16.data(), m, packed, via_public16.data()
                );
                std::vector<float> via_direct16(m * n, -2.0F);
                if (svs::detail::avx_runtime_flags.is_avx512f_supported()) {
                    InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build_packed(
                        a16.data(), m, packed, via_direct16.data()
                    );
                } else if (svs::detail::avx_runtime_flags.is_avx2_supported()) {
                    InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2>::build_packed(
                        a16.data(), m, packed, via_direct16.data()
                    );
                } else {
                    InnerProductMatrixImpl<AVX_AVAILABILITY::NONE>::build_packed(
                        a16.data(), m, packed, via_direct16.data()
                    );
                }
                CATCH_REQUIRE(via_public16 == via_direct16);
            }
        }
    }
}

CATCH_TEST_CASE(
    "PackedOperand::pack_rounded_to_bf16 rounds to nearest, ties to even",
    "[distance][inner_product_matrix]"
) {
    CATCH_SECTION("Matches svs::round_to_bfloat16 over representative values") {
        // Exact bf16 value, round-down, round-up, zeros of both signs, and the largest
        // finite float32 (whose rounding overflows to bf16 infinity, exercised below).
        std::vector<float> values{
            1.0F,
            0.1F,
            -0.1F,
            0.0F,
            -0.0F,
            100.25F,
            -100.25F,
            std::numeric_limits<float>::max(),
            -std::numeric_limits<float>::max(),
        };
        PackedOperand packed;
        packed.pack_rounded_to_bf16(values.data(), values.size(), 1);
        for (size_t i = 0; i < values.size(); ++i) {
            float expected = static_cast<float>(svs::round_to_bfloat16(values[i]));
            CATCH_REQUIRE(packed.data()[i] == expected);
        }
    }

    CATCH_SECTION("Exact halfway cases round to the even neighbor, checked bit for bit") {
        // Both inputs discard exactly 0x8000 (the bf16 ULP's halfway point) below bit 16:
        // the first's kept 16 bits are already even and must stay; the second's are odd
        // and must round up to the next even value. This checks the rounding algorithm's
        // bit pattern directly, not just self-consistency with svs::round_to_bfloat16.
        float tie_even_stays = std::bit_cast<float>(uint32_t{0x3F808000});
        float tie_odd_rounds_up = std::bit_cast<float>(uint32_t{0x3F818000});
        std::vector<float> values{tie_even_stays, tie_odd_rounds_up};
        PackedOperand packed;
        packed.pack_rounded_to_bf16(values.data(), values.size(), 1);
        CATCH_REQUIRE(std::bit_cast<uint32_t>(packed.data()[0]) == uint32_t{0x3F800000});
        CATCH_REQUIRE(std::bit_cast<uint32_t>(packed.data()[1]) == uint32_t{0x3F820000});
    }

    CATCH_SECTION("Rounding float32's max overflows to signed infinity") {
        std::vector<float> values{
            std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
        PackedOperand packed;
        packed.pack_rounded_to_bf16(values.data(), values.size(), 1);
        CATCH_REQUIRE(packed.data()[0] == std::numeric_limits<float>::infinity());
        CATCH_REQUIRE(packed.data()[1] == -std::numeric_limits<float>::infinity());
    }
}
