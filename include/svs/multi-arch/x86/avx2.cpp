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
DISTANCE_L2_INSTANTIATE_TEMPLATE(64, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(96, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(100, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(128, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(160, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(200, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(512, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(768, AVX_AVAILABILITY::AVX2);
DISTANCE_L2_INSTANTIATE_TEMPLATE(Dynamic, AVX_AVAILABILITY::AVX2);

DISTANCE_IP_INSTANTIATE_TEMPLATE(64, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(96, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(100, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(128, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(160, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(200, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(512, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(768, AVX_AVAILABILITY::AVX2);
DISTANCE_IP_INSTANTIATE_TEMPLATE(Dynamic, AVX_AVAILABILITY::AVX2);

DISTANCE_CS_INSTANTIATE_TEMPLATE(64, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(96, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(100, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(128, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(160, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(200, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(512, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(768, AVX_AVAILABILITY::AVX2);
DISTANCE_CS_INSTANTIATE_TEMPLATE(Dynamic, AVX_AVAILABILITY::AVX2);

template struct QEmdImpl<AVX_AVAILABILITY::AVX2>;
template struct QChamferImpl<AVX_AVAILABILITY::AVX2>;
template struct QChamferExactImpl<AVX_AVAILABILITY::AVX2>;
template struct InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2>;

// Same register-blocking as avx512.cpp's InnerProductMatrixImpl<AVX512>::build, at AVX2's
// 8-wide registers; AVX2 has no masked load/store, so the tail lane-block uses
// _mm256_maskload_ps/_mm256_maskstore_ps with an all-lanes blend mask built from the bit
// pattern simd_utils.h's ConvertToFloat<8> uses for the same purpose.
void InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2>::build(
    const float* a, size_t m, const float* b, size_t n, size_t d, float* out
) {
    if (m == 0 || n == 0) {
        return;
    }
    constexpr size_t vl = 8;
    constexpr size_t row_block = inner_product_matrix_row_block;

    thread_local std::vector<float> bt;
    bt.resize(d * n);
    for (size_t j = 0; j < n; ++j) {
        const float* bj = b + j * d;
        for (size_t k = 0; k < d; ++k) {
            bt[k * n + j] = bj[k];
        }
    }

    size_t full_lanes = n - n % vl;
    uint8_t tail_bits = svs::create_mask<vl>(svs::lib::MaybeStatic(n));
    __m256i tail_mask = _mm256_castps_si256(svs::simd::create_blend_mask_avx2(tail_bits));

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
            __m256 acc[row_block];
            for (size_t r = 0; r < row_block; ++r) {
                acc[r] = _mm256_setzero_ps();
            }
            for (size_t k = 0; k < d; ++k) {
                __m256 bv = _mm256_loadu_ps(bt.data() + k * n + j);
                for (size_t r = 0; r < row_block; ++r) {
                    acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(arow[r][k]), bv, acc[r]);
                }
            }
            for (size_t r = 0; r < row_block; ++r) {
                _mm256_storeu_ps(orow[r] + j, acc[r]);
            }
        }
        if (j < n) {
            __m256 acc[row_block];
            for (size_t r = 0; r < row_block; ++r) {
                acc[r] = _mm256_setzero_ps();
            }
            for (size_t k = 0; k < d; ++k) {
                __m256 bv = _mm256_maskload_ps(bt.data() + k * n + j, tail_mask);
                for (size_t r = 0; r < row_block; ++r) {
                    acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(arow[r][k]), bv, acc[r]);
                }
            }
            for (size_t r = 0; r < row_block; ++r) {
                _mm256_maskstore_ps(orow[r] + j, tail_mask, acc[r]);
            }
        }
    }

    for (; i < m; ++i) {
        const float* ai = a + i * d;
        float* row = out + i * n;
        size_t j = 0;
        for (; j < full_lanes; j += vl) {
            __m256 acc = _mm256_setzero_ps();
            for (size_t k = 0; k < d; ++k) {
                acc = _mm256_fmadd_ps(
                    _mm256_set1_ps(ai[k]), _mm256_loadu_ps(bt.data() + k * n + j), acc
                );
            }
            _mm256_storeu_ps(row + j, acc);
        }
        if (j < n) {
            __m256 acc = _mm256_setzero_ps();
            for (size_t k = 0; k < d; ++k) {
                acc = _mm256_fmadd_ps(
                    _mm256_set1_ps(ai[k]),
                    _mm256_maskload_ps(bt.data() + k * n + j, tail_mask),
                    acc
                );
            }
            _mm256_maskstore_ps(row + j, tail_mask, acc);
        }
    }
}

} // namespace svs::distance

#endif
