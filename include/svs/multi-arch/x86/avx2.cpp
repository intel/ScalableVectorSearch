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

namespace {

constexpr size_t kRowBlock = inner_product_matrix_row_block;

// Same role as avx512.cpp's accumulate_full/accumulate_tail, at AVX2's 8-wide registers;
// AVX2 has no masked load/store, so the tail lane-block uses _mm256_maskload_ps with an
// all-lanes blend mask built from the bit pattern simd_utils.h's ConvertToFloat<8> uses.
SVS_FORCE_INLINE void accumulate_full(
    const float* const* arow, const float* bt, size_t n, size_t d, size_t j, __m256* acc
) {
    for (size_t k = 0; k < d; ++k) {
        __m256 bv = _mm256_loadu_ps(bt + k * n + j);
        for (size_t r = 0; r < kRowBlock; ++r) {
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(arow[r][k]), bv, acc[r]);
        }
    }
}

SVS_FORCE_INLINE void accumulate_tail(
    const float* const* arow,
    const float* bt,
    size_t n,
    size_t d,
    size_t j,
    __m256i mask,
    __m256* acc
) {
    for (size_t k = 0; k < d; ++k) {
        __m256 bv = _mm256_maskload_ps(bt + k * n + j, mask);
        for (size_t r = 0; r < kRowBlock; ++r) {
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(arow[r][k]), bv, acc[r]);
        }
    }
}

// Pairwise-max reduction; see avx512.cpp's reduce_max for why padding slots are safe to
// include.
SVS_FORCE_INLINE __m256 reduce_max(const __m256* acc) {
    __m256 m01 = _mm256_max_ps(acc[0], acc[1]);
    __m256 m23 = _mm256_max_ps(acc[2], acc[3]);
    __m256 m45 = _mm256_max_ps(acc[4], acc[5]);
    __m256 m67 = _mm256_max_ps(acc[6], acc[7]);
    return _mm256_max_ps(_mm256_max_ps(m01, m23), _mm256_max_ps(m45, m67));
}

} // namespace

// Same register-blocking as avx512.cpp's InnerProductMatrixImpl<AVX512>::build_packed, at
// AVX2's 8-wide registers.
void InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2>::build_packed(
    const float* a, size_t m, const PackedOperand& b, float* out
) {
    size_t n = b.n();
    size_t d = b.d();
    if (m == 0 || n == 0) {
        return;
    }
    constexpr size_t vl = 8;
    const float* bt = b.data();
    size_t full_lanes = n - n % vl;
    uint8_t tail_bits = svs::create_mask<vl>(svs::lib::MaybeStatic(n));
    __m256i tail_mask = _mm256_castps_si256(svs::simd::create_blend_mask_avx2(tail_bits));

    for (size_t i = 0; i < m; i += kRowBlock) {
        size_t rows = std::min(kRowBlock, m - i);
        const float* arow[kRowBlock];
        float* orow[kRowBlock];
        for (size_t r = 0; r < kRowBlock; ++r) {
            arow[r] = a + (i + std::min(r, rows - 1)) * d;
        }
        for (size_t r = 0; r < rows; ++r) {
            orow[r] = out + (i + r) * n;
        }

        size_t j = 0;
        for (; j < full_lanes; j += vl) {
            __m256 acc[kRowBlock];
            for (size_t r = 0; r < kRowBlock; ++r) {
                acc[r] = _mm256_setzero_ps();
            }
            accumulate_full(arow, bt, n, d, j, acc);
            for (size_t r = 0; r < rows; ++r) {
                _mm256_storeu_ps(orow[r] + j, acc[r]);
            }
        }
        if (j < n) {
            __m256 acc[kRowBlock];
            for (size_t r = 0; r < kRowBlock; ++r) {
                acc[r] = _mm256_setzero_ps();
            }
            accumulate_tail(arow, bt, n, d, j, tail_mask, acc);
            for (size_t r = 0; r < rows; ++r) {
                _mm256_maskstore_ps(orow[r] + j, tail_mask, acc[r]);
            }
        }
    }
}

// See avx512.cpp's InnerProductMatrixImpl<AVX512>::build for why this packs into a
// thread_local buffer rather than taking a PackedOperand from the caller.
void InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2>::build(
    const float* a, size_t m, const float* b, size_t n, size_t d, float* out
) {
    if (m == 0 || n == 0) {
        return;
    }
    thread_local PackedOperand packed;
    packed.pack(b, n, d);
    build_packed(a, m, packed, out);
}

// See avx512.cpp's InnerProductMatrixImpl<AVX512>::column_max for the algorithm; this is
// the same reduction at AVX2's 8-wide registers.
void InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2>::column_max(
    const float* a, size_t m, const PackedOperand& b, float* out
) {
    size_t n = b.n();
    size_t d = b.d();
    if (n == 0) {
        return;
    }
    std::fill(out, out + n, -std::numeric_limits<float>::infinity());
    if (m == 0) {
        return;
    }
    constexpr size_t vl = 8;
    const float* bt = b.data();
    size_t full_lanes = n - n % vl;
    uint8_t tail_bits = svs::create_mask<vl>(svs::lib::MaybeStatic(n));
    __m256i tail_mask = _mm256_castps_si256(svs::simd::create_blend_mask_avx2(tail_bits));

    for (size_t i = 0; i < m; i += kRowBlock) {
        size_t rows = std::min(kRowBlock, m - i);
        const float* arow[kRowBlock];
        for (size_t r = 0; r < kRowBlock; ++r) {
            arow[r] = a + (i + std::min(r, rows - 1)) * d;
        }

        size_t j = 0;
        for (; j < full_lanes; j += vl) {
            __m256 acc[kRowBlock];
            for (size_t r = 0; r < kRowBlock; ++r) {
                acc[r] = _mm256_setzero_ps();
            }
            accumulate_full(arow, bt, n, d, j, acc);
            __m256 current = _mm256_loadu_ps(out + j);
            _mm256_storeu_ps(out + j, _mm256_max_ps(reduce_max(acc), current));
        }
        if (j < n) {
            __m256 acc[kRowBlock];
            for (size_t r = 0; r < kRowBlock; ++r) {
                acc[r] = _mm256_setzero_ps();
            }
            accumulate_tail(arow, bt, n, d, j, tail_mask, acc);
            __m256 current = _mm256_maskload_ps(out + j, tail_mask);
            _mm256_maskstore_ps(
                out + j, tail_mask, _mm256_max_ps(reduce_max(acc), current)
            );
        }
    }
}

// Float16 overload: converts one row-block of the document at a time into a float32
// scratch using F16C's 8-wide _mm256_cvtph_ps (available because avx2.cpp is compiled
// for the "haswell" arch in cmake/multi-arch.cmake, and Haswell carries F16C alongside
// AVX2), then runs the identical float32 path above.
void InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2>::column_max(
    const svs::Float16* a, size_t m, const PackedOperand& b, float* out
) {
    size_t n = b.n();
    size_t d = b.d();
    if (n == 0) {
        return;
    }
    std::fill(out, out + n, -std::numeric_limits<float>::infinity());
    if (m == 0) {
        return;
    }
    constexpr size_t vl = 8;
    const float* bt = b.data();
    size_t full_lanes = n - n % vl;
    uint8_t tail_bits = svs::create_mask<vl>(svs::lib::MaybeStatic(n));
    __m256i tail_mask = _mm256_castps_si256(svs::simd::create_blend_mask_avx2(tail_bits));

    thread_local std::vector<float> scratch;
    scratch.resize(kRowBlock * d);

    for (size_t i = 0; i < m; i += kRowBlock) {
        size_t rows = std::min(kRowBlock, m - i);
        for (size_t r = 0; r < rows; ++r) {
            const svs::Float16* src = a + (i + r) * d;
            float* dst = scratch.data() + r * d;
            size_t k = 0;
            for (; k + 8 <= d; k += 8) {
                __m256 fv = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + k))
                );
                _mm256_storeu_ps(dst + k, fv);
            }
            if (k < d) {
                // Safe masked load: a full-width load could read past the row's last
                // element, so only popcount(mask) halfs are copied into a zeroed buffer
                // before conversion, mirroring simd_utils.h's ConvertToFloat<8> tail path.
                uint8_t dbits = svs::create_mask<8>(svs::lib::MaybeStatic(d - k));
                __m128i buffer = _mm_setzero_si128();
                std::memcpy(
                    &buffer, src + k, __builtin_popcount(dbits) * sizeof(svs::Float16)
                );
                __m256 fv = _mm256_cvtph_ps(buffer);
                __m256i dmask =
                    _mm256_castps_si256(svs::simd::create_blend_mask_avx2(dbits));
                _mm256_maskstore_ps(dst + k, dmask, fv);
            }
        }
        const float* arow[kRowBlock];
        for (size_t r = 0; r < kRowBlock; ++r) {
            arow[r] = scratch.data() + std::min(r, rows - 1) * d;
        }

        size_t j = 0;
        for (; j < full_lanes; j += vl) {
            __m256 acc[kRowBlock];
            for (size_t r = 0; r < kRowBlock; ++r) {
                acc[r] = _mm256_setzero_ps();
            }
            accumulate_full(arow, bt, n, d, j, acc);
            __m256 current = _mm256_loadu_ps(out + j);
            _mm256_storeu_ps(out + j, _mm256_max_ps(reduce_max(acc), current));
        }
        if (j < n) {
            __m256 acc[kRowBlock];
            for (size_t r = 0; r < kRowBlock; ++r) {
                acc[r] = _mm256_setzero_ps();
            }
            accumulate_tail(arow, bt, n, d, j, tail_mask, acc);
            __m256 current = _mm256_maskload_ps(out + j, tail_mask);
            _mm256_maskstore_ps(
                out + j, tail_mask, _mm256_max_ps(reduce_max(acc), current)
            );
        }
    }
}

} // namespace svs::distance

#endif
