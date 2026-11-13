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

// Float16 overload: converts one row-block (kRowBlock rows) of the document at a time
// into a float32 scratch with the widest conversion AVX-512 offers (_mm512_cvtph_ps over
// 16 halfs), then runs the identical float32 path above. Never branches on element type
// per row: the scratch, not a per-row check, is what makes the two types the same kernel.
void InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::column_max(
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
    constexpr size_t vl = 16;
    const float* bt = b.data();
    size_t full_lanes = n - n % vl;
    __mmask16 tail_mask = svs::create_mask<vl>(svs::lib::MaybeStatic(n));

    thread_local std::vector<float> scratch;
    scratch.resize(kRowBlock * d);

    for (size_t i = 0; i < m; i += kRowBlock) {
        size_t rows = std::min(kRowBlock, m - i);
        for (size_t r = 0; r < rows; ++r) {
            const svs::Float16* src = a + (i + r) * d;
            float* dst = scratch.data() + r * d;
            size_t k = 0;
            for (; k + 16 <= d; k += 16) {
                _mm512_storeu_ps(dst + k, _mm512_cvtph_ps(_mm256_loadu_epi16(src + k)));
            }
            if (k < d) {
                __mmask16 dmask = svs::create_mask<16>(svs::lib::MaybeStatic(d - k));
                _mm512_mask_storeu_ps(
                    dst + k,
                    dmask,
                    _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(dmask, src + k))
                );
            }
        }
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

} // namespace svs::distance

#endif
