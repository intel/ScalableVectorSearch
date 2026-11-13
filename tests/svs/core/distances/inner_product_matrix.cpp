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
#include "catch2/catch_message.hpp"
#include "catch2/catch_test_macros.hpp"

// stdlib
#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <vector>

// POSIX, guarded: the guard-page test below needs mmap/mprotect/munmap, which this file
// must still compile without on a non-POSIX target.
#if defined(__unix__) || defined(__APPLE__)
#define SVS_TEST_HAS_MMAP 1
#include <sys/mman.h>
#include <unistd.h>
#else
#define SVS_TEST_HAS_MMAP 0
#endif

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

// AMX's tile grouping (16 pairs per k-chunk, two 16-lane tiles accumulated together)
// reorders the sum differently from both the generic k-order loop and AVX-512's 8-row
// register blocking; measured over every shape amx_ms/amx_ns/amx_ds below, that extra
// reordering alone needs the same ~2x-of-observed tolerance already used for the
// Float16/BFloat16 narrow-type comparisons above (check_column_max_against_generic),
// not REL_EPS/ABS_EPS's tighter float32 bound.
constexpr double AMX_REL_EPS = 2e-4;
constexpr double AMX_ABS_EPS = 1e-4;

void require_close_loose(
    const std::vector<float>& got, const std::vector<double>& expected
) {
    CATCH_REQUIRE(got.size() == expected.size());
    for (size_t i = 0; i < got.size(); ++i) {
        CATCH_REQUIRE(
            static_cast<double>(got[i]) ==
            Catch::Approx(expected[i]).epsilon(AMX_REL_EPS).margin(AMX_ABS_EPS)
        );
    }
}

void require_close_vec_loose(
    const std::vector<float>& got, const std::vector<double>& expected
) {
    CATCH_REQUIRE(got.size() == expected.size());
    for (size_t i = 0; i < got.size(); ++i) {
        if (expected[i] == -std::numeric_limits<double>::infinity()) {
            CATCH_REQUIRE(got[i] == -std::numeric_limits<float>::infinity());
        } else {
            CATCH_REQUIRE(
                static_cast<double>(got[i]) ==
                Catch::Approx(expected[i]).epsilon(AMX_REL_EPS).margin(AMX_ABS_EPS)
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

// Rounds to bf16 and back to float32, for building a reference against a B operand that
// was packed with `pack_rounded_to_bf16`: that kernel sees bf16(b), not b, so a reference
// comparing against the pre-rounding float would be wrong by bf16's own rounding error,
// not by summation order.
std::vector<float> to_bfloat16_as_float(const std::vector<float>& b) {
    std::vector<float> out(b.size());
    for (size_t i = 0; i < b.size(); ++i) {
        out[i] = static_cast<float>(svs::round_to_bfloat16(b[i]));
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

namespace {

// Every tail an AMX tile hits: m crosses the 16-row tile boundary (0, 1, 15, 16, 17, 33,
// 67, 180), n crosses the 16-lane tile boundary (1, 15, 16, 17, 32, 33), d crosses the
// 32-element (16-pair) k-chunk boundary and parity (1, 2, 31, 32, 33, 64, 127, 128, 129).
const std::vector<size_t> amx_ms{0, 1, 15, 16, 17, 33, 67, 180};
const std::vector<size_t> amx_ns{1, 15, 16, 17, 32, 33};
const std::vector<size_t> amx_ds{1, 2, 31, 32, 33, 64, 127, 128, 129};

} // namespace

CATCH_TEST_CASE(
    "InnerProductMatrixImpl<AMX> build_packed matches a double-precision reference and "
    "AVX-512, over every tail shape",
    "[distance][inner_product_matrix][amx]"
) {
    if (!svs::detail::amx_tile_bf16_available()) {
        return;
    }
    unsigned seed = 8000;
    for (size_t d : amx_ds) {
        for (size_t m : amx_ms) {
            for (size_t n : amx_ns) {
                auto a = random_matrix(m, d, seed++);
                auto a_bf16 = to_bfloat16(a);
                auto b = random_matrix(n, d, seed++);
                // The kernel computes <bf16(a), bf16(b)>: the reference must round b too,
                // not compare against the pre-rounding float values.
                auto b_rounded = to_bfloat16_as_float(b);
                PackedOperand packed;
                packed.pack_rounded_to_bf16(b.data(), n, d);
                CATCH_REQUIRE(packed.has_tile_layout());

                CATCH_INFO("m=" << m << " n=" << n << " d=" << d);
                std::vector<float> out_amx(m * n, -1.0F);
                InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::build_packed(
                    a_bf16.data(), m, packed, out_amx.data()
                );
                require_close_loose(
                    out_amx,
                    reference_from_values(promote_bfloat16(a_bf16), m, b_rounded, n, d)
                );

                std::vector<float> out_avx512(m * n, -2.0F);
                InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build_packed(
                    a_bf16.data(), m, packed, out_avx512.data()
                );
                for (size_t idx = 0; idx < out_amx.size(); ++idx) {
                    CATCH_REQUIRE(
                        static_cast<double>(out_amx[idx]) ==
                        Catch::Approx(static_cast<double>(out_avx512[idx]))
                            .epsilon(2e-4)
                            .margin(1e-4)
                    );
                }
            }
        }
    }
}

CATCH_TEST_CASE(
    "InnerProductMatrixImpl<AMX> column_max matches a double-precision reference and "
    "AVX-512, over every tail shape",
    "[distance][inner_product_matrix][amx]"
) {
    if (!svs::detail::amx_tile_bf16_available()) {
        return;
    }
    unsigned seed = 9000;
    for (size_t d : amx_ds) {
        for (size_t m : amx_ms) {
            for (size_t n : amx_ns) {
                auto a = random_matrix(m, d, seed++);
                auto a_bf16 = to_bfloat16(a);
                auto b = random_matrix(n, d, seed++);
                auto b_rounded = to_bfloat16_as_float(b);
                PackedOperand packed;
                packed.pack_rounded_to_bf16(b.data(), n, d);
                CATCH_REQUIRE(packed.has_tile_layout());

                std::vector<float> out_amx(n, -3.0F);
                InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::column_max(
                    a_bf16.data(), m, packed, out_amx.data()
                );
                require_close_vec_loose(
                    out_amx,
                    reference_column_max(promote_bfloat16(a_bf16), m, b_rounded, n, d)
                );

                std::vector<float> out_avx512(n, -4.0F);
                InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::column_max(
                    a_bf16.data(), m, packed, out_avx512.data()
                );
                for (size_t j = 0; j < n; ++j) {
                    if (out_avx512[j] == -std::numeric_limits<float>::infinity()) {
                        CATCH_REQUIRE(
                            out_amx[j] == -std::numeric_limits<float>::infinity()
                        );
                    } else {
                        CATCH_REQUIRE(
                            static_cast<double>(out_amx[j]) ==
                            Catch::Approx(static_cast<double>(out_avx512[j]))
                                .epsilon(2e-4)
                                .margin(1e-4)
                        );
                    }
                }
            }
        }
    }
}

CATCH_TEST_CASE(
    "InnerProductMatrixImpl<AMX> forwards float32 and Float16 A to AVX-512 bit for bit",
    "[distance][inner_product_matrix][amx]"
) {
    unsigned seed = 10000;
    for (size_t d : {1UL, 7UL, 128UL, 129UL}) {
        for (size_t m : {0UL, 1UL, 17UL, 33UL}) {
            for (size_t n : {1UL, 17UL, 33UL}) {
                auto a = random_matrix(m, d, seed++);
                auto a16 = to_float16(a);
                auto b = random_matrix(n, d, seed++);
                PackedOperand packed;
                packed.pack(b.data(), n, d);

                std::vector<float> amx_f32(m * n, -1.0F), avx512_f32(m * n, -2.0F);
                InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::build_packed(
                    a.data(), m, packed, amx_f32.data()
                );
                InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build_packed(
                    a.data(), m, packed, avx512_f32.data()
                );
                CATCH_REQUIRE(amx_f32 == avx512_f32);

                std::vector<float> amx_f16(m * n, -1.0F), avx512_f16(m * n, -2.0F);
                InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::build_packed(
                    a16.data(), m, packed, amx_f16.data()
                );
                InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build_packed(
                    a16.data(), m, packed, avx512_f16.data()
                );
                CATCH_REQUIRE(amx_f16 == avx512_f16);

                std::vector<float> amx_cm(n, -1.0F), avx512_cm(n, -2.0F);
                InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::column_max(
                    a.data(), m, packed, amx_cm.data()
                );
                InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::column_max(
                    a.data(), m, packed, avx512_cm.data()
                );
                CATCH_REQUIRE(amx_cm == avx512_cm);

                std::vector<float> via_build_amx(m * n, -1.0F),
                    via_build_avx512(m * n, -2.0F);
                InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::build(
                    a.data(), m, b.data(), n, d, via_build_amx.data()
                );
                InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build(
                    a.data(), m, b.data(), n, d, via_build_avx512.data()
                );
                CATCH_REQUIRE(via_build_amx == via_build_avx512);
            }
        }
    }
}

CATCH_TEST_CASE(
    "InnerProductMatrixImpl<AMX> falls back to AVX-512 bit for bit when B was not "
    "packed with the tile layout",
    "[distance][inner_product_matrix][amx]"
) {
    // PackedOperand::pack (not pack_rounded_to_bf16) never fills the tile layout, even
    // when AMX is compiled in and available: it is the caller's signal "B is not
    // rounded to bf16," and the AMX kernel only ever computes <bf16(a), bf16(b)>.
    unsigned seed = 11000;
    for (size_t d : {1UL, 7UL, 128UL, 129UL}) {
        for (size_t m : {0UL, 1UL, 17UL, 33UL}) {
            for (size_t n : {1UL, 17UL, 33UL}) {
                auto a = random_matrix(m, d, seed++);
                auto a_bf16 = to_bfloat16(a);
                auto b = random_matrix(n, d, seed++);
                PackedOperand packed;
                packed.pack(b.data(), n, d);
                CATCH_REQUIRE(!packed.has_tile_layout());

                std::vector<float> amx_out(m * n, -1.0F), avx512_out(m * n, -2.0F);
                InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::build_packed(
                    a_bf16.data(), m, packed, amx_out.data()
                );
                InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build_packed(
                    a_bf16.data(), m, packed, avx512_out.data()
                );
                CATCH_REQUIRE(amx_out == avx512_out);

                std::vector<float> amx_cm(n, -1.0F), avx512_cm(n, -2.0F);
                InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::column_max(
                    a_bf16.data(), m, packed, amx_cm.data()
                );
                InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::column_max(
                    a_bf16.data(), m, packed, avx512_cm.data()
                );
                CATCH_REQUIRE(amx_cm == avx512_cm);
            }
        }
    }
}

CATCH_TEST_CASE(
    "svs::detail::amx_tile_bf16_available is consistent across concurrent threads",
    "[distance][inner_product_matrix][amx]"
) {
    constexpr size_t num_threads = 8;
    std::vector<std::thread> threads;
    std::array<bool, num_threads> results{};
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&results, t] {
            results[t] = svs::detail::amx_tile_bf16_available();
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    bool first = results[0];
    for (bool r : results) {
        CATCH_REQUIRE(r == first);
    }
    CATCH_REQUIRE(first == svs::detail::amx_tile_bf16_available());
}

CATCH_TEST_CASE(
    "A thread started before AMX permission is ever requested can still run the AMX "
    "kernel",
    "[distance][inner_product_matrix][amx]"
) {
    if (!svs::detail::amx_tile_bf16_available()) {
        return;
    }
    // This test only demonstrates the property if it runs before any other test case in
    // this binary has already cached svs::detail::amx_tile_bf16_available()'s answer;
    // Catch2 runs test cases in a fixed but unspecified order, so instead this spawns a
    // thread that calls straight into the AMX kernel without itself calling
    // amx_tile_bf16_available() first, which is the actual shape of the hazard: a worker
    // thread in SVS's thread pool that never calls the detection function at all, relying
    // only on whichever thread called it first (see fix_argument_decoded in gem.h).
    size_t m = 17, n = 17, d = 70;
    auto a = random_matrix(m, d, 12000);
    auto a_bf16 = to_bfloat16(a);
    auto b = random_matrix(n, d, 12001);
    auto b_rounded = to_bfloat16_as_float(b);
    PackedOperand packed;
    packed.pack_rounded_to_bf16(b.data(), n, d);

    std::vector<float> out(m * n, -1.0F);
    std::thread worker([&] {
        InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::build_packed(
            a_bf16.data(), m, packed, out.data()
        );
    });
    worker.join();
    require_close_loose(
        out, reference_from_values(promote_bfloat16(a_bf16), m, b_rounded, n, d)
    );
}

#if SVS_TEST_HAS_MMAP

namespace {

// Owns one `mmap` region sized to guard exactly `usable_bytes` of real storage against
// one neighbouring `PROT_NONE` page: `end_guarded` places the usable region so its last
// byte is the last byte before the guard page (catches an overrun), `start_guarded`
// places it at the first byte after a leading guard page (catches an underrun). A
// kernel that reads or writes one byte past what the test tells it to touch faults
// (SIGSEGV), which kills the test binary; that crash, not a Catch2 assertion, is this
// test's failure signal.
class GuardedMapping {
  public:
    static GuardedMapping end_guarded(size_t usable_bytes) {
        size_t page = page_size();
        size_t usable_pages = (usable_bytes + page - 1) / page;
        size_t total = (usable_pages + 1) * page;
        void* base = map(total);
        guard(static_cast<char*>(base) + usable_pages * page, page);
        void* ptr = static_cast<char*>(base) + usable_pages * page - usable_bytes;
        return GuardedMapping(base, total, ptr);
    }

    static GuardedMapping start_guarded(size_t usable_bytes) {
        size_t page = page_size();
        size_t usable_pages = (usable_bytes + page - 1) / page;
        size_t total = (usable_pages + 1) * page;
        void* base = map(total);
        guard(base, page);
        void* ptr = static_cast<char*>(base) + page;
        return GuardedMapping(base, total, ptr);
    }

    GuardedMapping(const GuardedMapping&) = delete;
    GuardedMapping& operator=(const GuardedMapping&) = delete;
    GuardedMapping(GuardedMapping&& other) noexcept
        : base_{other.base_}
        , total_{other.total_}
        , ptr_{other.ptr_} {
        other.base_ = nullptr;
    }
    GuardedMapping& operator=(GuardedMapping&&) = delete;

    ~GuardedMapping() {
        if (base_ != nullptr) {
            munmap(base_, total_);
        }
    }

    void* data() const { return ptr_; }

  private:
    GuardedMapping(void* base, size_t total, void* ptr)
        : base_{base}
        , total_{total}
        , ptr_{ptr} {}

    static size_t page_size() { return static_cast<size_t>(sysconf(_SC_PAGESIZE)); }

    static void* map(size_t total) {
        void* base = mmap(
            nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0
        );
        CATCH_REQUIRE(base != MAP_FAILED);
        return base;
    }

    static void guard(void* addr, size_t len) {
        CATCH_REQUIRE(mprotect(addr, len, PROT_NONE) == 0);
    }

    void* base_;
    size_t total_;
    void* ptr_;
};

// Every tail shape the table and column-max kernels' SIMD/tile paths mask: m and n each
// cross the 16-wide tile/lane boundary, d crosses the 32-element (16-pair) AMX k-chunk
// boundary and both parities, matching the AMX tail sweep above but written out
// explicitly per the task's shape list rather than reusing `amx_ms`/`amx_ns`/`amx_ds`.
const std::vector<size_t> guard_ms{1, 15, 16, 17, 33};
const std::vector<size_t> guard_ns{1, 15, 16, 17, 33};
const std::vector<size_t> guard_ds{1, 2, 31, 32, 33, 127, 128, 129};

// Runs both kernel forms (table and column-max) for one (level, element type) pair over
// every (m, n, d) in the guard sweep, with A and `out` each placed, in turn, so a
// one-element overrun or underrun would fault. `kernel` dispatches to
// InnerProductMatrixImpl<Avx>::build_packed/column_max for whichever `T` the caller
// closed over; `pack_b` chooses `PackedOperand::pack` vs. `pack_rounded_to_bf16` the same
// way MaxSim/InnerProductMatrix's own callers do.
template <typename T, typename BuildPacked, typename ColumnMax>
void run_guarded_sweep(
    bool pack_b_rounded, BuildPacked build_packed, ColumnMax column_max
) {
    for (size_t d : guard_ds) {
        for (size_t m : guard_ms) {
            for (size_t n : guard_ns) {
                auto a_src =
                    random_matrix(m, d, static_cast<unsigned>(m * 997 + n * 31 + d));
                std::vector<T> a_host(m * d);
                for (size_t i = 0; i < a_host.size(); ++i) {
                    a_host[i] = T(a_src[i]);
                }
                auto b = random_matrix(n, d, static_cast<unsigned>(d * 997 + m * 31 + n));
                PackedOperand packed;
                if (pack_b_rounded) {
                    packed.pack_rounded_to_bf16(b.data(), n, d);
                } else {
                    packed.pack(b.data(), n, d);
                }

                // (1) A ends exactly at a guard page: an overrun past A's last element
                // faults.
                {
                    GuardedMapping guard = GuardedMapping::end_guarded(m * d * sizeof(T));
                    std::memcpy(guard.data(), a_host.data(), m * d * sizeof(T));
                    const T* a = static_cast<const T*>(guard.data());
                    std::vector<float> table_out(m * n, 0.0F);
                    build_packed(a, m, packed, table_out.data());
                    std::vector<float> cm_out(n, 0.0F);
                    column_max(a, m, packed, cm_out.data());
                }

                // (2) `out` ends exactly at a guard page, for both forms.
                {
                    GuardedMapping guard =
                        GuardedMapping::end_guarded(m * n * sizeof(float));
                    build_packed(
                        a_host.data(), m, packed, static_cast<float*>(guard.data())
                    );
                }
                {
                    GuardedMapping guard = GuardedMapping::end_guarded(n * sizeof(float));
                    column_max(a_host.data(), m, packed, static_cast<float*>(guard.data()));
                }

                // (3) A starts at the first byte after a guard page: an underrun before
                // A's first element faults.
                {
                    GuardedMapping guard = GuardedMapping::start_guarded(m * d * sizeof(T));
                    std::memcpy(guard.data(), a_host.data(), m * d * sizeof(T));
                    const T* a = static_cast<const T*>(guard.data());
                    std::vector<float> table_out(m * n, 0.0F);
                    build_packed(a, m, packed, table_out.data());
                    std::vector<float> cm_out(n, 0.0F);
                    column_max(a, m, packed, cm_out.data());
                }
            }
        }
    }
}

template <AVX_AVAILABILITY Avx, typename T> void run_guarded_level() {
    run_guarded_sweep<T>(
        std::same_as<T, svs::BFloat16>,
        [](const T* a, size_t m, const PackedOperand& b, float* out) {
            InnerProductMatrixImpl<Avx>::build_packed(a, m, b, out);
        },
        [](const T* a, size_t m, const PackedOperand& b, float* out) {
            InnerProductMatrixImpl<Avx>::column_max(a, m, b, out);
        }
    );
}

} // namespace

// The AMX tile kernels (amx_table_bf16 / amx_column_max_bf16, reached only for a bfloat16
// A against a tile-packed B) copy A's tail through a zero-padded stack buffer precisely
// to avoid reading past A's own allocation (see PackedOperand::fill_tile_layout's doc
// comment); this test is the memory-safety check for that claim, and for the AVX-512/AVX2
// masked-tail kernels' equivalent claim. A violation is a SIGSEGV that kills this test
// binary, which is the failure signal, not a CATCH_REQUIRE.
CATCH_TEST_CASE(
    "InnerProductMatrixImpl kernels never read or write outside A or out, including "
    "their masked tails",
    "[distance][inner_product_matrix][amx]"
) {
    CATCH_SECTION("AMX bfloat16 tile kernel") {
        if (!svs::detail::amx_tile_bf16_available()) {
            return;
        }
        run_guarded_level<AVX_AVAILABILITY::AMX, svs::BFloat16>();
    }

    CATCH_SECTION("AVX-512 kernels") {
        if (!svs::detail::avx_runtime_flags.is_avx512f_supported()) {
            return;
        }
        run_guarded_level<AVX_AVAILABILITY::AVX512, float>();
        run_guarded_level<AVX_AVAILABILITY::AVX512, svs::Float16>();
        run_guarded_level<AVX_AVAILABILITY::AVX512, svs::BFloat16>();
    }

    CATCH_SECTION("AVX2 kernels") {
        if (!svs::detail::avx_runtime_flags.is_avx2_supported()) {
            return;
        }
        run_guarded_level<AVX_AVAILABILITY::AVX2, float>();
        run_guarded_level<AVX_AVAILABILITY::AVX2, svs::Float16>();
        run_guarded_level<AVX_AVAILABILITY::AVX2, svs::BFloat16>();
    }
}

#endif // SVS_TEST_HAS_MMAP
