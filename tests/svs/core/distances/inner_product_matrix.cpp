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
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <type_traits>
#include <utility>
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
using svs::distance::PackedOperand;

// float32 tolerance: summation order differs from the SIMD kernels, not just ULP
// rounding. Float16/BFloat16/AMX reorder further still, needing ~2x that.
constexpr double REL_EPS = 1e-4;
constexpr double ABS_EPS = 1e-5;
constexpr double NARROW_REL_EPS = 2e-4;
constexpr double NARROW_ABS_EPS = 1e-4;

std::vector<float> random_matrix(size_t rows, size_t d, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> out(rows * d);
    for (auto& v : out) {
        v = dist(rng);
    }
    return out;
}

// Converts float32 to A's element type as the kernel would see it: BFloat16 rounds
// ties-to-even (not the truncating constructor); float/Float16 use their constructor.
template <typename T> std::vector<T> to_type(const std::vector<float>& a) {
    std::vector<T> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        if constexpr (std::is_same_v<T, svs::BFloat16>) {
            out[i] = svs::round_to_bfloat16(a[i]);
        } else {
            out[i] = T(a[i]);
        }
    }
    return out;
}

// Promotes any of A's element types to double via each type's lossless `operator float`.
template <typename T> std::vector<double> promote_row(const std::vector<T>& a) {
    std::vector<double> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = static_cast<double>(static_cast<float>(a[i]));
    }
    return out;
}

// Rounds B to bf16 and back to float32: the AMX/bf16 kernels see bf16(b), so a
// reference against pre-rounding b would be wrong by rounding error, not ordering.
std::vector<float> to_bfloat16_as_float(const std::vector<float>& b) {
    std::vector<float> out(b.size());
    for (size_t i = 0; i < b.size(); ++i) {
        out[i] = static_cast<float>(svs::round_to_bfloat16(b[i]));
    }
    return out;
}

// out[i * n + j] = <a_i, b_j>, over an already-promoted `a` so a narrow-type A's
// post-conversion values are what the reference uses.
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

// out[j] = max over i of <a_i, b_j>, or -infinity if m == 0.
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

// `expected[i] == -inf` requires `got[i]` exactly -inf rather than Approx-equal
// (column_max's empty-input value); table-form callers never hit that case.
void require_close(
    const std::vector<float>& got,
    const std::vector<double>& expected,
    double rel = REL_EPS,
    double abs_margin = ABS_EPS
) {
    CATCH_REQUIRE(got.size() == expected.size());
    for (size_t i = 0; i < got.size(); ++i) {
        if (expected[i] == -std::numeric_limits<double>::infinity()) {
            CATCH_REQUIRE(got[i] == -std::numeric_limits<float>::infinity());
        } else {
            CATCH_REQUIRE(
                static_cast<double>(got[i]) ==
                Catch::Approx(expected[i]).epsilon(rel).margin(abs_margin)
            );
        }
    }
}

// Shapes hitting every tail class NONE/AVX2/AVX512 need: m crosses the empty (0) and
// 8-row block boundary; n crosses the 16-lane boundary; d crosses the 16-lane boundary.
const std::vector<size_t> ms{0, 1, 2, 7, 8, 9, 17, 67, 180};
const std::vector<size_t> ns{1, 15, 16, 17, 32, 33};
const std::vector<size_t> ds{1, 7, 128, 129};

// Checks build_packed and column_max for A held as T, at ISA level Avx, against a double
// reference and, for Avx != NONE, against the generic kernel.
template <AVX_AVAILABILITY Avx, typename T>
void check_kernels(
    const std::vector<float>& a, const std::vector<float>& b, size_t m, size_t n, size_t d
) {
    auto a_t = to_type<T>(a);
    PackedOperand packed;
    packed.pack(b.data(), n, d);
    constexpr bool narrow = !std::is_same_v<T, float>;
    double rel = narrow ? NARROW_REL_EPS : REL_EPS;
    double abs_margin = narrow ? NARROW_ABS_EPS : ABS_EPS;

    std::vector<float> table(m * n, -1.0F), col_max(n, -3.0F);
    InnerProductMatrixImpl<Avx>::build_packed(a_t.data(), m, packed, table.data());
    InnerProductMatrixImpl<Avx>::column_max(a_t.data(), m, packed, col_max.data());
    require_close(
        table, reference_from_values(promote_row(a_t), m, b, n, d), rel, abs_margin
    );
    require_close(
        col_max, reference_column_max(promote_row(a_t), m, b, n, d), rel, abs_margin
    );

    if constexpr (Avx != AVX_AVAILABILITY::NONE) {
        std::vector<float> generic_table(m * n, -1.0F), generic_col_max(n, -3.0F);
        InnerProductMatrixImpl<AVX_AVAILABILITY::NONE>::build_packed(
            a_t.data(), m, packed, generic_table.data()
        );
        InnerProductMatrixImpl<AVX_AVAILABILITY::NONE>::column_max(
            a_t.data(), m, packed, generic_col_max.data()
        );
        require_close(table, promote_row(generic_table), rel, abs_margin);
        require_close(col_max, promote_row(generic_col_max), rel, abs_margin);
    }
}

template <AVX_AVAILABILITY Avx> void check_level() {
    unsigned seed = 0;
    for (size_t d : ds) {
        for (size_t m : ms) {
            for (size_t n : ns) {
                auto a = random_matrix(m, d, seed++);
                auto b = random_matrix(n, d, seed++);
                check_kernels<Avx, float>(a, b, m, n, d);
                check_kernels<Avx, svs::Float16>(a, b, m, n, d);
                check_kernels<Avx, svs::BFloat16>(a, b, m, n, d);
            }
        }
    }
}

} // namespace

CATCH_TEST_CASE(
    "InnerProductMatrixImpl build_packed and column_max match a double reference, and "
    "the generic kernel, for every A type and available non-AMX ISA",
    "[distance][inner_product_matrix]"
) {
    check_level<AVX_AVAILABILITY::NONE>();
    if (svs::detail::avx_runtime_flags.is_avx2_supported()) {
        check_level<AVX_AVAILABILITY::AVX2>();
    }
    if (svs::detail::avx_runtime_flags.is_avx512f_supported()) {
        check_level<AVX_AVAILABILITY::AVX512>();
    }
}

namespace {

// Every tail an AMX tile hits: m/n cross the 16-wide tile/lane boundary, d crosses the
// 32-element (16-pair) k-chunk boundary and both parities.
const std::vector<size_t> amx_ms{0, 1, 15, 16, 17, 33, 67, 180};
const std::vector<size_t> amx_ns{1, 15, 16, 17, 32, 33};
const std::vector<size_t> amx_ds{1, 2, 31, 32, 33, 64, 127, 128, 129};

// Bit-exact comparison for one (A type, shape): AMX forwards to AVX-512 with no Approx
// tolerance whenever it isn't running its own bf16 tile kernel.
template <typename T>
void require_amx_matches_avx512(
    const T* a, size_t m, const PackedOperand& packed, size_t n
) {
    std::vector<float> table_amx(m * n, -1.0F), table_avx512(m * n, -2.0F);
    InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::build_packed(
        a, m, packed, table_amx.data()
    );
    InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build_packed(
        a, m, packed, table_avx512.data()
    );
    CATCH_REQUIRE(table_amx == table_avx512);

    std::vector<float> cm_amx(n, -1.0F), cm_avx512(n, -2.0F);
    InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::column_max(a, m, packed, cm_amx.data());
    InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::column_max(
        a, m, packed, cm_avx512.data()
    );
    CATCH_REQUIRE(cm_amx == cm_avx512);
}

} // namespace

// Every InnerProductMatrixImpl<AMX> overload besides BFloat16's build_packed/column_max
// is an unconditional one-line forward to AVX-512 with no branch: float32 stands in for
// Float16 too, since both overloads have that identical shape. BFloat16's build_packed/
// column_max branch on an early m==0/n==0 return and on has_tile_layout(); B is plain-
// packed below, so this covers the early return and the non-tile branch, both distinct
// from the float32/Float16 case. The small, fixed shape set is deliberate: none of these
// branches depend on shape beyond m == 0 or n == 0.
CATCH_TEST_CASE(
    "InnerProductMatrixImpl<AMX> forwards to AVX-512 bit for bit except for its own "
    "bf16 tile kernel, which matches a double reference and AVX-512",
    "[distance][inner_product_matrix][amx]"
) {
    unsigned seed = 20000;
    for (size_t m : {0UL, 1UL, 9UL, 33UL}) {
        for (size_t n : {0UL, 1UL, 17UL, 33UL}) {
            size_t d = 7;
            auto a = random_matrix(m, d, seed++);
            auto a_bf16 = to_type<svs::BFloat16>(a);
            auto b = random_matrix(n, d, seed++);
            PackedOperand packed;
            packed.pack(b.data(), n, d);

            require_amx_matches_avx512(a.data(), m, packed, n);
            require_amx_matches_avx512(a_bf16.data(), m, packed, n);

            std::vector<float> build_amx(m * n, -1.0F), build_avx512(m * n, -2.0F);
            InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::build(
                a.data(), m, b.data(), n, d, build_amx.data()
            );
            InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build(
                a.data(), m, b.data(), n, d, build_avx512.data()
            );
            CATCH_REQUIRE(build_amx == build_avx512);
        }
    }

    // AMX's own bf16 tile kernel: reached only when B is packed with
    // pack_rounded_to_bf16 and tile hardware is available. This is the one branch in
    // the AMX code that isn't a plain forward, so it gets the full tail-shape sweep.
    if (!svs::detail::amx_tile_bf16_available()) {
        return;
    }
    seed = 90000;
    for (size_t d : amx_ds) {
        for (size_t m : amx_ms) {
            for (size_t n : amx_ns) {
                auto a_bf16 = to_type<svs::BFloat16>(random_matrix(m, d, seed++));
                auto b = random_matrix(n, d, seed++);
                auto b_rounded = to_bfloat16_as_float(b);
                PackedOperand packed;
                packed.pack_rounded_to_bf16(b.data(), n, d);
                CATCH_REQUIRE(packed.has_tile_layout());

                std::vector<float> table_amx(m * n, -1.0F), table_avx512(m * n, -2.0F);
                InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::build_packed(
                    a_bf16.data(), m, packed, table_amx.data()
                );
                InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build_packed(
                    a_bf16.data(), m, packed, table_avx512.data()
                );
                require_close(
                    table_amx,
                    reference_from_values(promote_row(a_bf16), m, b_rounded, n, d),
                    NARROW_REL_EPS,
                    NARROW_ABS_EPS
                );
                require_close(
                    table_amx, promote_row(table_avx512), NARROW_REL_EPS, NARROW_ABS_EPS
                );

                std::vector<float> cm_amx(n, -3.0F), cm_avx512(n, -4.0F);
                InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::column_max(
                    a_bf16.data(), m, packed, cm_amx.data()
                );
                InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::column_max(
                    a_bf16.data(), m, packed, cm_avx512.data()
                );
                require_close(
                    cm_amx,
                    reference_column_max(promote_row(a_bf16), m, b_rounded, n, d),
                    NARROW_REL_EPS,
                    NARROW_ABS_EPS
                );
                require_close(
                    cm_amx, promote_row(cm_avx512), NARROW_REL_EPS, NARROW_ABS_EPS
                );
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
    for (auto [m, n] : std::vector<std::pair<size_t, size_t>>{{0, 3}, {3, 0}}) {
        std::vector<float> out{1.0F, 2.0F, 3.0F};
        auto before = out;
        InnerProductMatrix::compute(a.data(), m, b.data(), n, d, out.data());
        CATCH_REQUIRE(out == before);
    }
}

namespace {

// Picks the ISA compute()/compute_packed() would pick: select_isa() is private, so this
// repeats its priority order against the public avx_runtime_flags.
template <typename F> void with_highest_isa(F&& f) {
    if (svs::detail::avx_runtime_flags.is_avx512f_supported()) {
        f(std::integral_constant<AVX_AVAILABILITY, AVX_AVAILABILITY::AVX512>{});
    } else if (svs::detail::avx_runtime_flags.is_avx2_supported()) {
        f(std::integral_constant<AVX_AVAILABILITY, AVX_AVAILABILITY::AVX2>{});
    } else {
        f(std::integral_constant<AVX_AVAILABILITY, AVX_AVAILABILITY::NONE>{});
    }
}

} // namespace

// compute_packed's BFloat16 and Float16 overloads share one dispatch call, unconditional
// on T, so BFloat16 alone exercises that dispatch; Float16 would hit the identical branch.
CATCH_TEST_CASE(
    "InnerProductMatrix's public entry points equal the reference and pick the same ISA "
    "result as InnerProductMatrixImpl, for float32 and BFloat16 rows",
    "[distance][inner_product_matrix]"
) {
    unsigned seed = 5000;
    for (size_t d : {1UL, 7UL, 128UL}) {
        for (size_t m : {0UL, 1UL, 9UL, 33UL}) {
            for (size_t n : {1UL, 17UL, 33UL}) {
                auto a = random_matrix(m, d, seed++);
                auto b = random_matrix(n, d, seed++);
                PackedOperand packed;
                packed.pack(b.data(), n, d);

                std::vector<float> via_build(m * n, -1.0F);
                InnerProductMatrix::compute(a.data(), m, b.data(), n, d, via_build.data());
                require_close(via_build, reference_from_values(promote_row(a), m, b, n, d));

                std::vector<float> via_packed(m * n, -2.0F);
                with_highest_isa([&](auto avx_c) {
                    InnerProductMatrixImpl<avx_c.value>::build_packed(
                        a.data(), m, packed, via_packed.data()
                    );
                });
                CATCH_REQUIRE(via_build == via_packed);

                auto a_bf16 = to_type<svs::BFloat16>(a);
                std::vector<float> via_public(m * n, -1.0F);
                InnerProductMatrix::compute_packed(
                    a_bf16.data(), m, packed, via_public.data()
                );
                std::vector<float> via_direct(m * n, -2.0F);
                with_highest_isa([&](auto avx_c) {
                    InnerProductMatrixImpl<avx_c.value>::build_packed(
                        a_bf16.data(), m, packed, via_direct.data()
                    );
                });
                CATCH_REQUIRE(via_public == via_direct);
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
        // Both inputs discard exactly bit 0x8000, bf16's halfway point: the first's kept
        // bits are already even and must stay, the second's are odd and must round up.
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

CATCH_TEST_CASE(
    "AMX tile permission detection is thread-safe", "[distance][inner_product_matrix][amx]"
) {
    CATCH_SECTION("amx_tile_bf16_available is consistent across concurrent threads") {
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
        for (bool r : results) {
            CATCH_REQUIRE(r == results[0]);
        }
    }

    // A worker that never calls amx_tile_bf16_available() itself (as SVS's thread pool
    // doesn't) must still get correct AMX results (see fix_argument_decoded in gem.h).
    CATCH_SECTION("A thread started before permission is requested can still run AMX") {
        if (!svs::detail::amx_tile_bf16_available()) {
            return;
        }
        size_t m = 17, n = 17, d = 70;
        auto a_bf16 = to_type<svs::BFloat16>(random_matrix(m, d, 12000));
        auto b = random_matrix(n, d, 12001);
        auto b_rounded = to_bfloat16_as_float(b);
        PackedOperand packed;
        packed.pack_rounded_to_bf16(b.data(), n, d);

        std::vector<float> out(m * n, -1.0F);
        std::thread([&] {
            InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>::build_packed(
                a_bf16.data(), m, packed, out.data()
            );
        }).join();
        require_close(
            out,
            reference_from_values(promote_row(a_bf16), m, b_rounded, n, d),
            NARROW_REL_EPS,
            NARROW_ABS_EPS
        );
    }
}

#if SVS_TEST_HAS_MMAP

namespace {

// RAII mmap guard: `bytes` of usable storage plus one neighbouring PROT_NONE page;
// `at_end` guards after the usable region (catches overrun), else before it (underrun).
// An out-of-bounds kernel access segfaults, which is this test's failure signal.
class Guarded {
  public:
    Guarded(size_t bytes, bool at_end) {
        size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        size_t pages = (bytes + page - 1) / page;
        total_ = (pages + 1) * page;
        base_ = mmap(
            nullptr, total_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0
        );
        CATCH_REQUIRE(base_ != MAP_FAILED);
        char* guard_page = static_cast<char*>(base_) + (at_end ? pages * page : 0);
        CATCH_REQUIRE(mprotect(guard_page, page, PROT_NONE) == 0);
        ptr_ = at_end ? guard_page - bytes : static_cast<char*>(base_) + page;
    }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
    ~Guarded() { munmap(base_, total_); }
    void* data() const { return ptr_; }

  private:
    void* base_;
    size_t total_;
    void* ptr_;
};

const std::vector<size_t> guard_ms{1, 15, 16, 17, 33};
const std::vector<size_t> guard_ns{1, 15, 16, 17, 33};
const std::vector<size_t> guard_ds{1, 2, 31, 32, 33, 127, 128, 129};

// Runs build_packed and column_max for one (level, type) pair over every guard shape,
// with A and `out` placed at a guard boundary in turn so a one-element over/underrun
// faults. BFloat16 packs B rounded, matching every real caller of a bf16 A; the rest
// pack plain.
template <AVX_AVAILABILITY Avx, typename T> void run_guarded_level() {
    unsigned seed = 0;
    for (size_t d : guard_ds) {
        for (size_t m : guard_ms) {
            for (size_t n : guard_ns) {
                auto a_src = random_matrix(m, d, seed++);
                auto a_host = to_type<T>(a_src);
                auto b = random_matrix(n, d, seed++);
                PackedOperand packed;
                if constexpr (std::is_same_v<T, svs::BFloat16>) {
                    packed.pack_rounded_to_bf16(b.data(), n, d);
                } else {
                    packed.pack(b.data(), n, d);
                }

                for (bool at_end : {true, false}) {
                    Guarded a_guard(m * d * sizeof(T), at_end);
                    std::memcpy(a_guard.data(), a_host.data(), m * d * sizeof(T));
                    const T* a = static_cast<const T*>(a_guard.data());
                    std::vector<float> table_out(m * n, 0.0F), cm_out(n, 0.0F);
                    InnerProductMatrixImpl<Avx>::build_packed(
                        a, m, packed, table_out.data()
                    );
                    InnerProductMatrixImpl<Avx>::column_max(a, m, packed, cm_out.data());
                }
                Guarded table_guard(m * n * sizeof(float), true);
                InnerProductMatrixImpl<Avx>::build_packed(
                    a_host.data(), m, packed, static_cast<float*>(table_guard.data())
                );
                Guarded cm_guard(n * sizeof(float), true);
                InnerProductMatrixImpl<Avx>::column_max(
                    a_host.data(), m, packed, static_cast<float*>(cm_guard.data())
                );
            }
        }
    }
}

} // namespace

// The AMX tile kernels copy A's tail through a zero-padded stack buffer to avoid reading
// past A's own allocation (PackedOperand::fill_tile_layout); a violation segfaults.
CATCH_TEST_CASE(
    "InnerProductMatrixImpl kernels never read or write outside A or out, including "
    "their masked tails",
    "[distance][inner_product_matrix][amx]"
) {
    if (svs::detail::amx_tile_bf16_available()) {
        run_guarded_level<AVX_AVAILABILITY::AMX, svs::BFloat16>();
    }
    if (svs::detail::avx_runtime_flags.is_avx512f_supported()) {
        run_guarded_level<AVX_AVAILABILITY::AVX512, float>();
        run_guarded_level<AVX_AVAILABILITY::AVX512, svs::Float16>();
        run_guarded_level<AVX_AVAILABILITY::AVX512, svs::BFloat16>();
    }
    if (svs::detail::avx_runtime_flags.is_avx2_supported()) {
        run_guarded_level<AVX_AVAILABILITY::AVX2, float>();
        run_guarded_level<AVX_AVAILABILITY::AVX2, svs::Float16>();
        run_guarded_level<AVX_AVAILABILITY::AVX2, svs::BFloat16>();
    }
}

#endif // SVS_TEST_HAS_MMAP
