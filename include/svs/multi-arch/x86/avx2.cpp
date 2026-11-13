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

// Widens one 8-lane block of Float16 to float32 with F16C's native half->single.
SVS_FORCE_INLINE __m256 widen8(const svs::Float16* p) {
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

// Widens one 8-lane block of BFloat16 to float32: bfloat16 is already float32's high 16
// bits, so this is a lossless zero-extend (cvtepu16_epi32) into the top half of each
// 32-bit lane (slli by 16), not a rounding conversion.
SVS_FORCE_INLINE __m256 widen8(const svs::BFloat16* p) {
    __m128i halfwords = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(halfwords), 16));
}

// Tail (< 8 lanes) widening for either 16-bit type: zero-pads through a stack buffer
// sized by sizeof(T) (2 bytes for both Float16 and BFloat16), then reuses widen8.
template <typename T> SVS_FORCE_INLINE __m256 widen8_tail(const T* p, uint8_t mask_bits) {
    __m128i buffer = _mm_setzero_si128();
    std::memcpy(&buffer, p, __builtin_popcount(mask_bits) * sizeof(T));
    return widen8(reinterpret_cast<const T*>(&buffer));
}

// Converts kRowBlock rows of `a` (T is Float16 or BFloat16) starting at row `i` into a
// contiguous float32 scratch, 8 lanes at a time; shared by the table and column-max forms
// below so either narrow type is widened in exactly one place.
template <typename T>
SVS_FORCE_INLINE void
convert_row_block(const T* a, size_t i, size_t rows, size_t d, float* scratch) {
    for (size_t r = 0; r < rows; ++r) {
        const T* src = a + (i + r) * d;
        float* dst = scratch + r * d;
        size_t k = 0;
        for (; k + 8 <= d; k += 8) {
            _mm256_storeu_ps(dst + k, widen8(src + k));
        }
        if (k < d) {
            uint8_t dbits = svs::create_mask<8>(svs::lib::MaybeStatic(d - k));
            __m256i dmask = _mm256_castps_si256(svs::simd::create_blend_mask_avx2(dbits));
            _mm256_maskstore_ps(dst + k, dmask, widen8_tail(src + k, dbits));
        }
    }
}

// Table form for a narrow-element A (Float16 or BFloat16): identical register-blocked
// loop to the float32 build_packed above, sourced from a row-block converted to float32
// by convert_row_block instead of `a` directly.
template <typename T>
void build_packed_narrow(const T* a, size_t m, const PackedOperand& b, float* out) {
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

    thread_local std::vector<float> scratch;
    scratch.resize(kRowBlock * d);

    for (size_t i = 0; i < m; i += kRowBlock) {
        size_t rows = std::min(kRowBlock, m - i);
        convert_row_block(a, i, rows, d, scratch.data());
        const float* arow[kRowBlock];
        float* orow[kRowBlock];
        for (size_t r = 0; r < kRowBlock; ++r) {
            arow[r] = scratch.data() + std::min(r, rows - 1) * d;
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

// Column-max form for a narrow-element A (Float16 or BFloat16): same structure as
// build_packed_narrow above, reduced through reduce_max instead of stored per row.
template <typename T>
void column_max_narrow(const T* a, size_t m, const PackedOperand& b, float* out) {
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
        convert_row_block(a, i, rows, d, scratch.data());
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

// Table form for a Float16 A: see convert_row_block above for the shared widening.
void InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2>::build_packed(
    const svs::Float16* a, size_t m, const PackedOperand& b, float* out
) {
    build_packed_narrow(a, m, b, out);
}

// Table form for a BFloat16 A: see convert_row_block above for the shared widening.
void InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2>::build_packed(
    const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out
) {
    build_packed_narrow(a, m, b, out);
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

// Float16 overload: see convert_row_block above for the shared widening (F16C's 8-wide
// _mm256_cvtph_ps, available because avx2.cpp is compiled for the "haswell" arch in
// cmake/multi-arch.cmake, and Haswell carries F16C alongside AVX2).
void InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2>::column_max(
    const svs::Float16* a, size_t m, const PackedOperand& b, float* out
) {
    column_max_narrow(a, m, b, out);
}

// BFloat16 overload: see convert_row_block above for the shared (exact) widening.
void InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2>::column_max(
    const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out
) {
    column_max_narrow(a, m, b, out);
}

} // namespace svs::distance

#endif
