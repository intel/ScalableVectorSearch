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

#pragma once

// Float row-pricing kernels for TransportSimplex's float-proposes/double-confirms
// pricing split: a cost row stored as float, with a float or double reduced-cost scan.

#include "svs/core/distance/distance_core.h"
#include "svs/lib/preprocessor.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace svs::distance::transport_detail {

// One AVX-512 float register, or two AVX2 float registers: a row padded to a multiple of
// this needs no scalar tail on either ISA.
inline constexpr size_t kPriceLanes = 16;

inline size_t pad_to_lanes(size_t n) {
    return (n + kPriceLanes - 1) / kPriceLanes * kPriceLanes;
}

/// Finds the minimum reduced cost `r_j = (cost_row[j] - u_i) - v[j]` in a cost row and its
/// column. `cost_row` and `v` must both have `ub_padded` valid entries, a multiple of
/// `kPriceLanes`; padding columns (index >= the caller's real ub) must hold +infinity in
/// `cost_row` so they never win. `best_val`/`best_col` are in-out and updated only on a
/// strict improvement, scanning columns ascending, so the first minimum wins on a tie.
/// This primary definition is not dead on x86 despite the AVX2/AVX512 specializations
/// below: non-x86 builds still instantiate `RowMin<AVX2>`/`RowMin<AVX512>` through
/// `QEmdImpl<AVX2>`/`QEmdImpl<AVX512>`.
template <AVX_AVAILABILITY Avx> struct RowMin {
    static void min_in_row(
        const float* cost_row,
        const float* v,
        size_t ub_padded,
        float u_i,
        float& best_val,
        uint32_t& best_col
    ) {
        for (size_t j = 0; j < ub_padded; ++j) {
            float r = (cost_row[j] - u_i) - v[j];
            if (r < best_val) {
                best_val = r;
                best_col = static_cast<uint32_t>(j);
            }
        }
    }

    static void min_in_row(
        const float* cost_row,
        const double* v,
        size_t ub_padded,
        double u_i,
        double& best_val,
        uint32_t& best_col
    ) {
        for (size_t j = 0; j < ub_padded; ++j) {
            double r = (static_cast<double>(cost_row[j]) - u_i) - v[j];
            if (r < best_val) {
                best_val = r;
                best_col = static_cast<uint32_t>(j);
            }
        }
    }
};

// Guarded per ISA: these bodies are not templates, so every TU that sees one compiles its
// intrinsics, and a TU built without that ISA fails loudly.
SVS_VALIDATE_BOOL_ENV(SVS_AVX512_F)
#if SVS_AVX512_F
template <> struct RowMin<AVX_AVAILABILITY::AVX512> {
    static void min_in_row(
        const float* cost_row,
        const float* v,
        size_t ub_padded,
        float u_i,
        float& best_val,
        uint32_t& best_col
    ) {
        __m512 u_vec = _mm512_set1_ps(u_i);
        for (size_t j = 0; j < ub_padded; j += 16) {
            // Two separate _mm512_sub_ps, not an FMA: the generic version rounds the two
            // subtractions independently, and this must match it bit-for-bit.
            __m512 r = _mm512_sub_ps(
                _mm512_sub_ps(_mm512_loadu_ps(cost_row + j), u_vec), _mm512_loadu_ps(v + j)
            );
            // Ordered-quiet: NaN never wins, matching scalar `<`. A zero mask -- the common
            // case once a good candidate is found -- skips the block untouched.
            __mmask16 lt_mask = _mm512_cmp_ps_mask(r, _mm512_set1_ps(best_val), _CMP_LT_OQ);
            if (lt_mask == 0) {
                continue;
            }
            // reduce_min can return a bit pattern no lane had (e.g. +0.0 vs -0.0 on a tie);
            // compress then moves the lowest-index lane equal to that value to position 0.
            float block_min = _mm512_mask_reduce_min_ps(lt_mask, r);
            __mmask16 eq_mask = static_cast<__mmask16>(
                lt_mask & _mm512_cmp_ps_mask(r, _mm512_set1_ps(block_min), _CMP_EQ_OQ)
            );
            best_val = _mm512_cvtss_f32(_mm512_maskz_compress_ps(eq_mask, r));
            best_col = static_cast<uint32_t>(j + std::countr_zero(eq_mask));
        }
    }

    static void min_in_row(
        const float* cost_row,
        const double* v,
        size_t ub_padded,
        double u_i,
        double& best_val,
        uint32_t& best_col
    ) {
        __m512d u_vec = _mm512_set1_pd(u_i);
        // One 8-lane half (mask8 is native width for double); see the float/float overload
        // above for why compress, not reduce_min, supplies best_val.
        auto resolve = [&](__m512d r, size_t col_base) {
            __mmask8 lt_mask = _mm512_cmp_pd_mask(r, _mm512_set1_pd(best_val), _CMP_LT_OQ);
            if (lt_mask == 0) {
                return;
            }
            double block_min = _mm512_mask_reduce_min_pd(lt_mask, r);
            __mmask8 eq_mask = static_cast<__mmask8>(
                lt_mask & _mm512_cmp_pd_mask(r, _mm512_set1_pd(block_min), _CMP_EQ_OQ)
            );
            best_val = _mm512_cvtsd_f64(_mm512_maskz_compress_pd(eq_mask, r));
            best_col = static_cast<uint32_t>(col_base + std::countr_zero(eq_mask));
        };
        for (size_t j = 0; j < ub_padded; j += 16) {
            __m256 c_lo = _mm256_loadu_ps(cost_row + j);
            __m256 c_hi = _mm256_loadu_ps(cost_row + j + 8);
            resolve(
                _mm512_sub_pd(
                    _mm512_sub_pd(_mm512_cvtps_pd(c_lo), u_vec), _mm512_loadu_pd(v + j)
                ),
                j
            );
            resolve(
                _mm512_sub_pd(
                    _mm512_sub_pd(_mm512_cvtps_pd(c_hi), u_vec), _mm512_loadu_pd(v + j + 8)
                ),
                j + 8
            );
        }
    }
};
#endif

SVS_VALIDATE_BOOL_ENV(SVS_AVX2)
#if SVS_AVX2
template <> struct RowMin<AVX_AVAILABILITY::AVX2> {
    static void min_in_row(
        const float* cost_row,
        const float* v,
        size_t ub_padded,
        float u_i,
        float& best_val,
        uint32_t& best_col
    ) {
        __m256 u_vec = _mm256_set1_ps(u_i);
        __m256 inf_vec = _mm256_set1_ps(std::numeric_limits<float>::infinity());
        // AVX2 has no compress instruction, so the winning lane's exact r is read back
        // from a stack buffer on the rare lt_mask != 0 branch, not taken from block_min.
        auto resolve = [&](__m256 r, size_t col_base) {
            __m256 lt_cmp = _mm256_cmp_ps(r, _mm256_set1_ps(best_val), _CMP_LT_OQ);
            auto lt_mask = static_cast<unsigned>(_mm256_movemask_ps(lt_cmp));
            if (lt_mask == 0) {
                return;
            }
            __m256 masked = _mm256_blendv_ps(inf_vec, r, lt_cmp);
            __m128 m = _mm_min_ps(
                _mm256_castps256_ps128(masked), _mm256_extractf128_ps(masked, 1)
            );
            m = _mm_min_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(0, 0, 3, 2)));
            m = _mm_min_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(0, 0, 0, 1)));
            float block_min = _mm_cvtss_f32(m);
            auto eq_mask =
                lt_mask & static_cast<unsigned>(_mm256_movemask_ps(
                              _mm256_cmp_ps(r, _mm256_set1_ps(block_min), _CMP_EQ_OQ)
                          ));
            auto lane = static_cast<unsigned>(std::countr_zero(eq_mask));
            float buf[8];
            _mm256_storeu_ps(buf, r);
            best_val = buf[lane];
            best_col = static_cast<uint32_t>(col_base + lane);
        };
        for (size_t j = 0; j < ub_padded; j += 16) {
            resolve(
                _mm256_sub_ps(
                    _mm256_sub_ps(_mm256_loadu_ps(cost_row + j), u_vec),
                    _mm256_loadu_ps(v + j)
                ),
                j
            );
            resolve(
                _mm256_sub_ps(
                    _mm256_sub_ps(_mm256_loadu_ps(cost_row + j + 8), u_vec),
                    _mm256_loadu_ps(v + j + 8)
                ),
                j + 8
            );
        }
    }

    static void min_in_row(
        const float* cost_row,
        const double* v,
        size_t ub_padded,
        double u_i,
        double& best_val,
        uint32_t& best_col
    ) {
        __m256d u_vec = _mm256_set1_pd(u_i);
        __m256d inf_vec = _mm256_set1_pd(std::numeric_limits<double>::infinity());
        // Same technique as the float/float overload above, at the 4-double width AVX2
        // actually has.
        auto resolve = [&](__m256d r, size_t col_base) {
            __m256d lt_cmp = _mm256_cmp_pd(r, _mm256_set1_pd(best_val), _CMP_LT_OQ);
            auto lt_mask = static_cast<unsigned>(_mm256_movemask_pd(lt_cmp));
            if (lt_mask == 0) {
                return;
            }
            __m256d masked = _mm256_blendv_pd(inf_vec, r, lt_cmp);
            __m128d m = _mm_min_pd(
                _mm256_castpd256_pd128(masked), _mm256_extractf128_pd(masked, 1)
            );
            m = _mm_min_pd(m, _mm_shuffle_pd(m, m, 0x1));
            double block_min = _mm_cvtsd_f64(m);
            auto eq_mask =
                lt_mask & static_cast<unsigned>(_mm256_movemask_pd(
                              _mm256_cmp_pd(r, _mm256_set1_pd(block_min), _CMP_EQ_OQ)
                          ));
            auto lane = static_cast<unsigned>(std::countr_zero(eq_mask));
            double buf[4];
            _mm256_storeu_pd(buf, r);
            best_val = buf[lane];
            best_col = static_cast<uint32_t>(col_base + lane);
        };
        for (size_t j = 0; j < ub_padded; j += 16) {
            for (size_t sub = 0; sub < 16; sub += 4) {
                __m128 c = _mm_loadu_ps(cost_row + j + sub);
                resolve(
                    _mm256_sub_pd(
                        _mm256_sub_pd(_mm256_cvtps_pd(c), u_vec),
                        _mm256_loadu_pd(v + j + sub)
                    ),
                    j + sub
                );
            }
        }
    }
};
#endif

} // namespace svs::distance::transport_detail
