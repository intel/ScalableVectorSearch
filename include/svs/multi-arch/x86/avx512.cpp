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

namespace {

constexpr size_t kRowBlock = inner_product_matrix_row_block;

// Accumulates kRowBlock rows of A (padded by repeating the last valid row when the
// document has fewer rows left, see callers) against one full 16-lane block of packed B.
// Shared by the table and column-max output forms so this FMA chain is written once.
SVS_FORCE_INLINE void accumulate_full(
    const float* const* arow, const float* bt, size_t n, size_t d, size_t j, __m512* acc
) {
    for (size_t k = 0; k < d; ++k) {
        __m512 bv = _mm512_loadu_ps(bt + k * n + j);
        for (size_t r = 0; r < kRowBlock; ++r) {
            acc[r] = _mm512_fmadd_ps(_mm512_set1_ps(arow[r][k]), bv, acc[r]);
        }
    }
}

// Same as accumulate_full for the trailing lane-block narrower than 16 lanes.
SVS_FORCE_INLINE void accumulate_tail(
    const float* const* arow,
    const float* bt,
    size_t n,
    size_t d,
    size_t j,
    __mmask16 mask,
    __m512* acc
) {
    for (size_t k = 0; k < d; ++k) {
        __m512 bv = _mm512_maskz_loadu_ps(mask, bt + k * n + j);
        for (size_t r = 0; r < kRowBlock; ++r) {
            acc[r] = _mm512_fmadd_ps(_mm512_set1_ps(arow[r][k]), bv, acc[r]);
        }
    }
}

// Pairwise-max reduction of the kRowBlock accumulators into their lane-wise maximum.
// Padding slots (see accumulate_full's comment) repeat a value already present in a real
// slot, so folding them into the tree does not change the result.
SVS_FORCE_INLINE __m512 reduce_max(const __m512* acc) {
    __m512 m01 = _mm512_max_ps(acc[0], acc[1]);
    __m512 m23 = _mm512_max_ps(acc[2], acc[3]);
    __m512 m45 = _mm512_max_ps(acc[4], acc[5]);
    __m512 m67 = _mm512_max_ps(acc[6], acc[7]);
    return _mm512_max_ps(_mm512_max_ps(m01, m23), _mm512_max_ps(m45, m67));
}

// Widens one 16-lane block of Float16 to float32 with AVX-512F's native half->single.
SVS_FORCE_INLINE __m512 widen16(const svs::Float16* p) {
    return _mm512_cvtph_ps(_mm256_loadu_epi16(p));
}

// Widens one 16-lane block of BFloat16 to float32: bfloat16 is already float32's high 16
// bits, so this is a lossless zero-extend (cvtepu16_epi32, AVX-512F) into the top half of
// each 32-bit lane (slli by 16), not a rounding conversion.
SVS_FORCE_INLINE __m512 widen16(const svs::BFloat16* p) {
    __m256i halfwords = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(halfwords), 16));
}

// Tail (< 16 lanes) widening for either 16-bit type: zero-pads through a stack buffer
// sized by sizeof(T) (2 bytes for both Float16 and BFloat16), then reuses widen16. Uses a
// plain memcpy rather than a masked load so BFloat16's tail needs nothing beyond
// AVX-512F (Float16's masked load above already relies on AVX-512BW+VL for
// _mm256_loadu_epi16, a preexisting requirement this does not widen).
template <typename T> SVS_FORCE_INLINE __m512 widen16_tail(const T* p, size_t count) {
    __m256i buffer = _mm256_setzero_si256();
    std::memcpy(&buffer, p, count * sizeof(T));
    return widen16(reinterpret_cast<const T*>(&buffer));
}

// Converts kRowBlock rows of `a` (T is Float16 or BFloat16) starting at row `i` into a
// contiguous float32 scratch, 16 lanes at a time; shared by the table and column-max
// forms below so either narrow type is widened in exactly one place.
template <typename T>
SVS_FORCE_INLINE void
convert_row_block(const T* a, size_t i, size_t rows, size_t d, float* scratch) {
    for (size_t r = 0; r < rows; ++r) {
        const T* src = a + (i + r) * d;
        float* dst = scratch + r * d;
        size_t k = 0;
        for (; k + 16 <= d; k += 16) {
            _mm512_storeu_ps(dst + k, widen16(src + k));
        }
        if (k < d) {
            size_t tail = d - k;
            __mmask16 dmask = svs::create_mask<16>(svs::lib::MaybeStatic(tail));
            _mm512_mask_storeu_ps(dst + k, dmask, widen16_tail(src + k, tail));
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
    constexpr size_t vl = 16;
    const float* bt = b.data();
    size_t full_lanes = n - n % vl;
    __mmask16 tail_mask = svs::create_mask<vl>(svs::lib::MaybeStatic(n));

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
            __m512 acc[kRowBlock];
            for (size_t r = 0; r < kRowBlock; ++r) {
                acc[r] = _mm512_setzero_ps();
            }
            accumulate_full(arow, bt, n, d, j, acc);
            for (size_t r = 0; r < rows; ++r) {
                _mm512_storeu_ps(orow[r] + j, acc[r]);
            }
        }
        if (j < n) {
            __m512 acc[kRowBlock];
            for (size_t r = 0; r < kRowBlock; ++r) {
                acc[r] = _mm512_setzero_ps();
            }
            accumulate_tail(arow, bt, n, d, j, tail_mask, acc);
            for (size_t r = 0; r < rows; ++r) {
                _mm512_mask_storeu_ps(orow[r] + j, tail_mask, acc[r]);
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
    constexpr size_t vl = 16;
    const float* bt = b.data();
    size_t full_lanes = n - n % vl;
    __mmask16 tail_mask = svs::create_mask<vl>(svs::lib::MaybeStatic(n));

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
            __m512 acc[kRowBlock];
            for (size_t r = 0; r < kRowBlock; ++r) {
                acc[r] = _mm512_setzero_ps();
            }
            accumulate_full(arow, bt, n, d, j, acc);
            __m512 current = _mm512_loadu_ps(out + j);
            _mm512_storeu_ps(out + j, _mm512_max_ps(reduce_max(acc), current));
        }
        if (j < n) {
            __m512 acc[kRowBlock];
            for (size_t r = 0; r < kRowBlock; ++r) {
                acc[r] = _mm512_setzero_ps();
            }
            accumulate_tail(arow, bt, n, d, j, tail_mask, acc);
            __m512 current = _mm512_maskz_loadu_ps(tail_mask, out + j);
            _mm512_mask_storeu_ps(
                out + j, tail_mask, _mm512_max_ps(reduce_max(acc), current)
            );
        }
    }
}

} // namespace

// Register-blocked micro-kernel: inner_product_matrix_row_block rows of A share one pass
// over a lane-block of B, giving that many independent FMA accumulator chains with no
// horizontal reduction (each lane is one output), so the dependency-chain latency of a
// single accumulator is hidden behind the others instead of serializing the loop over d.
void InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build_packed(
    const float* a, size_t m, const PackedOperand& b, float* out
) {
    size_t n = b.n();
    size_t d = b.d();
    if (m == 0 || n == 0) {
        return;
    }
    constexpr size_t vl = 16;
    const float* bt = b.data();
    size_t full_lanes = n - n % vl;
    __mmask16 tail_mask = svs::create_mask<vl>(svs::lib::MaybeStatic(n));

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
            __m512 acc[kRowBlock];
            for (size_t r = 0; r < kRowBlock; ++r) {
                acc[r] = _mm512_setzero_ps();
            }
            accumulate_full(arow, bt, n, d, j, acc);
            for (size_t r = 0; r < rows; ++r) {
                _mm512_storeu_ps(orow[r] + j, acc[r]);
            }
        }
        if (j < n) {
            __m512 acc[kRowBlock];
            for (size_t r = 0; r < kRowBlock; ++r) {
                acc[r] = _mm512_setzero_ps();
            }
            accumulate_tail(arow, bt, n, d, j, tail_mask, acc);
            for (size_t r = 0; r < rows; ++r) {
                _mm512_mask_storeu_ps(orow[r] + j, tail_mask, acc[r]);
            }
        }
    }
}

// Table form for a Float16 A: see convert_row_block above for the shared widening.
void InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build_packed(
    const svs::Float16* a, size_t m, const PackedOperand& b, float* out
) {
    build_packed_narrow(a, m, b, out);
}

// Table form for a BFloat16 A: see convert_row_block above for the shared widening.
void InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build_packed(
    const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out
) {
    build_packed_narrow(a, m, b, out);
}

// Packs B into a reused thread_local buffer: the table form repacks on every call (one
// query against all centroids), unlike MaxSim's column_max below which packs once per
// query and calls into this ISA's kernels directly with the already-packed operand.
void InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build(
    const float* a, size_t m, const float* b, size_t n, size_t d, float* out
) {
    if (m == 0 || n == 0) {
        return;
    }
    thread_local PackedOperand packed;
    packed.pack(b, n, d);
    build_packed(a, m, packed, out);
}

// Column-wise maximum of the same table build_packed computes, without ever
// materializing it: each row-block's accumulators are reduced to a per-lane maximum and
// folded into a running maximum held in `out` itself, so one reduction serves the whole
// document for the price of one load/max/store per lane-block per row-block.
void InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::column_max(
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
    constexpr size_t vl = 16;
    const float* bt = b.data();
    size_t full_lanes = n - n % vl;
    __mmask16 tail_mask = svs::create_mask<vl>(svs::lib::MaybeStatic(n));

    for (size_t i = 0; i < m; i += kRowBlock) {
        size_t rows = std::min(kRowBlock, m - i);
        const float* arow[kRowBlock];
        for (size_t r = 0; r < kRowBlock; ++r) {
            arow[r] = a + (i + std::min(r, rows - 1)) * d;
        }

        size_t j = 0;
        for (; j < full_lanes; j += vl) {
            __m512 acc[kRowBlock];
            for (size_t r = 0; r < kRowBlock; ++r) {
                acc[r] = _mm512_setzero_ps();
            }
            accumulate_full(arow, bt, n, d, j, acc);
            __m512 current = _mm512_loadu_ps(out + j);
            _mm512_storeu_ps(out + j, _mm512_max_ps(reduce_max(acc), current));
        }
        if (j < n) {
            __m512 acc[kRowBlock];
            for (size_t r = 0; r < kRowBlock; ++r) {
                acc[r] = _mm512_setzero_ps();
            }
            accumulate_tail(arow, bt, n, d, j, tail_mask, acc);
            __m512 current = _mm512_maskz_loadu_ps(tail_mask, out + j);
            _mm512_mask_storeu_ps(
                out + j, tail_mask, _mm512_max_ps(reduce_max(acc), current)
            );
        }
    }
}

// Float16 overload: see convert_row_block above for the shared widening (AVX-512F's
// native 16-wide _mm512_cvtph_ps).
void InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::column_max(
    const svs::Float16* a, size_t m, const PackedOperand& b, float* out
) {
    column_max_narrow(a, m, b, out);
}

// BFloat16 overload: see convert_row_block above for the shared (exact) widening.
void InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::column_max(
    const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out
) {
    column_max_narrow(a, m, b, out);
}

} // namespace svs::distance

#endif
