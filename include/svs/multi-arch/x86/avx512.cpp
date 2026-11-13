/*
 * Copyright 2025 Intel Corporation
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

#if defined(__x86_64__)
#include "svs/core/distance/cosine.h"
#include "svs/core/distance/euclidean.h"
#include "svs/core/distance/gem.h"
#include "svs/core/distance/inner_product.h"
#include "svs/core/distance/inner_product_matrix.h"
#include "svs/core/distance/simd_utils.h"

namespace svs::distance {

// TODO: connect with dim_supported_list
DISTANCE_L2_INSTANTIATE_TEMPLATE(64, AVX_AVAILABILITY::AVX512);
DISTANCE_L2_INSTANTIATE_TEMPLATE(96, AVX_AVAILABILITY::AVX512);
DISTANCE_L2_INSTANTIATE_TEMPLATE(100, AVX_AVAILABILITY::AVX512);
DISTANCE_L2_INSTANTIATE_TEMPLATE(128, AVX_AVAILABILITY::AVX512);
DISTANCE_L2_INSTANTIATE_TEMPLATE(160, AVX_AVAILABILITY::AVX512);
DISTANCE_L2_INSTANTIATE_TEMPLATE(200, AVX_AVAILABILITY::AVX512);
DISTANCE_L2_INSTANTIATE_TEMPLATE(512, AVX_AVAILABILITY::AVX512);
DISTANCE_L2_INSTANTIATE_TEMPLATE(768, AVX_AVAILABILITY::AVX512);
DISTANCE_L2_INSTANTIATE_TEMPLATE(Dynamic, AVX_AVAILABILITY::AVX512);

DISTANCE_IP_INSTANTIATE_TEMPLATE(64, AVX_AVAILABILITY::AVX512);
DISTANCE_IP_INSTANTIATE_TEMPLATE(96, AVX_AVAILABILITY::AVX512);
DISTANCE_IP_INSTANTIATE_TEMPLATE(100, AVX_AVAILABILITY::AVX512);
DISTANCE_IP_INSTANTIATE_TEMPLATE(128, AVX_AVAILABILITY::AVX512);
DISTANCE_IP_INSTANTIATE_TEMPLATE(160, AVX_AVAILABILITY::AVX512);
DISTANCE_IP_INSTANTIATE_TEMPLATE(200, AVX_AVAILABILITY::AVX512);
DISTANCE_IP_INSTANTIATE_TEMPLATE(512, AVX_AVAILABILITY::AVX512);
DISTANCE_IP_INSTANTIATE_TEMPLATE(768, AVX_AVAILABILITY::AVX512);
DISTANCE_IP_INSTANTIATE_TEMPLATE(Dynamic, AVX_AVAILABILITY::AVX512);

DISTANCE_CS_INSTANTIATE_TEMPLATE(64, AVX_AVAILABILITY::AVX512);
DISTANCE_CS_INSTANTIATE_TEMPLATE(96, AVX_AVAILABILITY::AVX512);
DISTANCE_CS_INSTANTIATE_TEMPLATE(100, AVX_AVAILABILITY::AVX512);
DISTANCE_CS_INSTANTIATE_TEMPLATE(128, AVX_AVAILABILITY::AVX512);
DISTANCE_CS_INSTANTIATE_TEMPLATE(160, AVX_AVAILABILITY::AVX512);
DISTANCE_CS_INSTANTIATE_TEMPLATE(200, AVX_AVAILABILITY::AVX512);
DISTANCE_CS_INSTANTIATE_TEMPLATE(512, AVX_AVAILABILITY::AVX512);
DISTANCE_CS_INSTANTIATE_TEMPLATE(768, AVX_AVAILABILITY::AVX512);
DISTANCE_CS_INSTANTIATE_TEMPLATE(Dynamic, AVX_AVAILABILITY::AVX512);

template struct QEmdImpl<AVX_AVAILABILITY::AVX512>;
template struct QChamferImpl<AVX_AVAILABILITY::AVX512>;
template struct QChamferExactImpl<AVX_AVAILABILITY::AVX512>;
template struct InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>;

// Register-blocked micro-kernel: inner_product_matrix_row_block rows of A share one pass
// over a lane-block of B, giving that many independent FMA accumulator chains with no
// horizontal reduction (each lane is one output), so the dependency-chain latency of a
// single accumulator is hidden behind the others instead of serializing the loop over d.
void InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build(
    const float* a, size_t m, const float* b, size_t n, size_t d, float* out
) {
    if (m == 0 || n == 0) {
        return;
    }
    constexpr size_t vl = 16;
    constexpr size_t row_block = inner_product_matrix_row_block;

    // Packs B transposed (bt[k * n + j] = b[j * d + k]) once per call: B is the small
    // operand (e.g. one query's tokens, a handful of KiB), so this copy is noise next to
    // the m x n table it lets every row below read with no horizontal reduction.
    thread_local std::vector<float> bt;
    bt.resize(d * n);
    for (size_t j = 0; j < n; ++j) {
        const float* bj = b + j * d;
        for (size_t k = 0; k < d; ++k) {
            bt[k * n + j] = bj[k];
        }
    }

    size_t full_lanes = n - n % vl;
    __mmask16 tail_mask = svs::create_mask<vl>(svs::lib::MaybeStatic(n));

    size_t full_rows = m - m % row_block;
    size_t i = 0;
    for (; i < full_rows; i += row_block) {
        const float* arow[row_block];
        float* orow[row_block];
        for (size_t r = 0; r < row_block; ++r) {
            arow[r] = a + (i + r) * d;
            orow[r] = out + (i + r) * n;
        }

        size_t j = 0;
        for (; j < full_lanes; j += vl) {
            __m512 acc[row_block];
            for (size_t r = 0; r < row_block; ++r) {
                acc[r] = _mm512_setzero_ps();
            }
            for (size_t k = 0; k < d; ++k) {
                __m512 bv = _mm512_loadu_ps(bt.data() + k * n + j);
                for (size_t r = 0; r < row_block; ++r) {
                    acc[r] = _mm512_fmadd_ps(_mm512_set1_ps(arow[r][k]), bv, acc[r]);
                }
            }
            for (size_t r = 0; r < row_block; ++r) {
                _mm512_storeu_ps(orow[r] + j, acc[r]);
            }
        }
        if (j < n) {
            __m512 acc[row_block];
            for (size_t r = 0; r < row_block; ++r) {
                acc[r] = _mm512_setzero_ps();
            }
            for (size_t k = 0; k < d; ++k) {
                __m512 bv = _mm512_maskz_loadu_ps(tail_mask, bt.data() + k * n + j);
                for (size_t r = 0; r < row_block; ++r) {
                    acc[r] = _mm512_fmadd_ps(_mm512_set1_ps(arow[r][k]), bv, acc[r]);
                }
            }
            for (size_t r = 0; r < row_block; ++r) {
                _mm512_mask_storeu_ps(orow[r] + j, tail_mask, acc[r]);
            }
        }
    }

    // Row remainder (m % row_block): same lane logic with a single accumulator per row.
    for (; i < m; ++i) {
        const float* ai = a + i * d;
        float* row = out + i * n;
        size_t j = 0;
        for (; j < full_lanes; j += vl) {
            __m512 acc = _mm512_setzero_ps();
            for (size_t k = 0; k < d; ++k) {
                acc = _mm512_fmadd_ps(
                    _mm512_set1_ps(ai[k]), _mm512_loadu_ps(bt.data() + k * n + j), acc
                );
            }
            _mm512_storeu_ps(row + j, acc);
        }
        if (j < n) {
            __m512 acc = _mm512_setzero_ps();
            for (size_t k = 0; k < d; ++k) {
                acc = _mm512_fmadd_ps(
                    _mm512_set1_ps(ai[k]),
                    _mm512_maskz_loadu_ps(tail_mask, bt.data() + k * n + j),
                    acc
                );
            }
            _mm512_mask_storeu_ps(row + j, tail_mask, acc);
        }
    }
}

} // namespace svs::distance

#endif
