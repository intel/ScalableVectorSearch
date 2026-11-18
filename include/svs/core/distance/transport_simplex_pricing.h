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

#include <cstddef>
#include <cstdint>

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
        float buf[16];
        for (size_t j = 0; j < ub_padded; j += 16) {
            // Two separate _mm512_sub_ps, not an FMA: the generic version rounds the two
            // subtractions independently, and this must match it bit-for-bit.
            __m512 r = _mm512_sub_ps(
                _mm512_sub_ps(_mm512_loadu_ps(cost_row + j), u_vec), _mm512_loadu_ps(v + j)
            );
            _mm512_storeu_ps(buf, r);
            for (size_t lane = 0; lane < 16; ++lane) {
                if (buf[lane] < best_val) {
                    best_val = buf[lane];
                    best_col = static_cast<uint32_t>(j + lane);
                }
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
        double buf[16];
        __m512d u_vec = _mm512_set1_pd(u_i);
        for (size_t j = 0; j < ub_padded; j += 16) {
            __m256 c_lo = _mm256_loadu_ps(cost_row + j);
            __m256 c_hi = _mm256_loadu_ps(cost_row + j + 8);
            __m512d r_lo = _mm512_sub_pd(
                _mm512_sub_pd(_mm512_cvtps_pd(c_lo), u_vec), _mm512_loadu_pd(v + j)
            );
            __m512d r_hi = _mm512_sub_pd(
                _mm512_sub_pd(_mm512_cvtps_pd(c_hi), u_vec), _mm512_loadu_pd(v + j + 8)
            );
            _mm512_storeu_pd(buf, r_lo);
            _mm512_storeu_pd(buf + 8, r_hi);
            for (size_t lane = 0; lane < 16; ++lane) {
                if (buf[lane] < best_val) {
                    best_val = buf[lane];
                    best_col = static_cast<uint32_t>(j + lane);
                }
            }
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
        float buf[16];
        for (size_t j = 0; j < ub_padded; j += 16) {
            __m256 r_lo = _mm256_sub_ps(
                _mm256_sub_ps(_mm256_loadu_ps(cost_row + j), u_vec), _mm256_loadu_ps(v + j)
            );
            __m256 r_hi = _mm256_sub_ps(
                _mm256_sub_ps(_mm256_loadu_ps(cost_row + j + 8), u_vec),
                _mm256_loadu_ps(v + j + 8)
            );
            _mm256_storeu_ps(buf, r_lo);
            _mm256_storeu_ps(buf + 8, r_hi);
            for (size_t lane = 0; lane < 16; ++lane) {
                if (buf[lane] < best_val) {
                    best_val = buf[lane];
                    best_col = static_cast<uint32_t>(j + lane);
                }
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
        __m256d u_vec = _mm256_set1_pd(u_i);
        double buf[16];
        for (size_t j = 0; j < ub_padded; j += 16) {
            for (size_t sub = 0; sub < 16; sub += 4) {
                __m128 c = _mm_loadu_ps(cost_row + j + sub);
                __m256d r = _mm256_sub_pd(
                    _mm256_sub_pd(_mm256_cvtps_pd(c), u_vec), _mm256_loadu_pd(v + j + sub)
                );
                _mm256_storeu_pd(buf + sub, r);
            }
            for (size_t lane = 0; lane < 16; ++lane) {
                if (buf[lane] < best_val) {
                    best_val = buf[lane];
                    best_col = static_cast<uint32_t>(j + lane);
                }
            }
        }
    }
};
#endif

} // namespace svs::distance::transport_detail
