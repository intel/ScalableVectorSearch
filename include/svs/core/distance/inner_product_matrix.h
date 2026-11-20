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

#include "svs/core/distance/distance_core.h"
#include "svs/core/distance/simd_utils.h"
#include "svs/lib/avx_detection.h"
#include "svs/lib/bfloat16.h"
#include "svs/lib/float16.h"
#include "svs/lib/preprocessor.h"

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace svs::distance {

// One accumulator per (A-row, B-lane-block), no horizontal reduction: measured on this
// machine, 8 independent accumulators is where row-blocking stops buying throughput,
// matching ~4-cycle FMA latency x 2 ports (see the PR description for the sweep).
inline constexpr size_t inner_product_matrix_row_block = 8;

/// Owns B's packed operand: B transposed into a [d][n] float32 buffer, so every row-block
/// of A can stream through it with a plain `loadu` instead of a gather. Packing is split
/// from use (`pack` vs. the kernel entry points below) because the table form repacks B
/// once per call (B is as cheap as the table it produces), while MaxSim packs one query
/// and reuses it against ~1000 documents: repacking per document would cost as much as
/// scoring them.
class PackedOperand {
  public:
    /// Repacks `b` (`n` rows of `d` floats each, row-major) into this object's transposed
    /// buffer. Storage grows only when `n * d` exceeds the current buffer's capacity
    /// (`std::vector::resize` does not reallocate when shrinking or staying within
    /// capacity), so calling this with the same or a smaller shape across repeated calls
    /// costs no allocation.
    void pack(const float* b, size_t n, size_t d) {
        n_ = n;
        d_ = d;
        buffer_.resize(n * d);
        for (size_t j = 0; j < n; ++j) {
            const float* bj = b + j * d;
            for (size_t k = 0; k < d; ++k) {
                buffer_[k * n + j] = bj[k];
            }
        }
    }

    /// Same contract and layout as `pack`, except every value is first rounded to
    /// bfloat16 precision (round to nearest, ties to even) and stored back as float32.
    /// The kernel itself only ever computes <a_i, b_j> on the values it is given,
    /// unrounded; rounding B to the precision A is held at (bf16, see FineCentroids) is
    /// the caller's choice of precision, made here rather than inside the kernel, so a
    /// future caller that wants a different precision for B is never blocked by this one.
    /// A scalar loop (not vectorized): at GEM's production shape this rounds ~4096
    /// values per query, negligible next to the O(K * d) table build that follows.
    void pack_rounded_to_bf16(const float* b, size_t n, size_t d) {
        n_ = n;
        d_ = d;
        buffer_.resize(n * d);
        for (size_t j = 0; j < n; ++j) {
            const float* bj = b + j * d;
            for (size_t k = 0; k < d; ++k) {
                buffer_[k * n + j] = static_cast<float>(svs::round_to_bfloat16(bj[k]));
            }
        }
#if SVS_AMX_COMPILED_IN
        // Only worth building when some TU in this binary can consume it (amx.cpp
        // exists, SVS_AMX_COMPILED_IN) and the CPU/OS actually grant tile use; otherwise
        // this would cost a few KiB and a scalar pass on every call for nothing.
        if (svs::detail::amx_tile_bf16_available()) {
            fill_tile_layout(b, n, d);
        } else {
            has_tile_layout_ = false;
        }
#endif
    }

    size_t n() const { return n_; }
    size_t d() const { return d_; }
    const float* data() const { return buffer_.data(); }

    /// True only right after `pack_rounded_to_bf16` filled the AMX tile layout below;
    /// `pack` never sets this, since its values are not rounded and the AMX kernel only
    /// ever computes <bf16(a), bf16(b)>.
    bool has_tile_layout() const { return has_tile_layout_; }

    /// Pair-interleaved, transposed bf16 tile block for query-lane tile `nt` (16 lanes,
    /// zero-padded past `n`) and d/2-pair tile `kt` (16 pairs, zero-padded past
    /// ceil(d/2)): row `r`'s dword at lane `c` packs
    /// (bf16(b[nt*16+c][2*(kt*16+r)]), bf16(b[nt*16+c][2*(kt*16+r)+1])) low/high, i.e.
    /// exactly the `b` tile operand _tile_dpbf16ps expects (see maxsim.h's contract).
    /// Valid only when `has_tile_layout()`; caller (amx.cpp) owns interpreting it.
    const uint32_t* tile_block(size_t nt, size_t kt) const {
        return tile_buffer_.data() + (nt * kp_tiles_ + kt) * kTileElements;
    }
    size_t n_tiles() const { return n_tiles_; }
    size_t kp_tiles() const { return kp_tiles_; }

  private:
    static constexpr size_t kTileElements = 16 * 16;

    // Builds the tile-layout buffer described by tile_block() above. Zero-padding both
    // the lane tail (n not a multiple of 16) and the pair tail (d odd, or not a multiple
    // of 32) is safe here because B's padding only ever multiplies against A's real
    // values and contributes 0; A's own out-of-bounds tail is handled separately, by the
    // AMX kernel copying A's last partial k-chunk through a zero-padded stack buffer
    // rather than reading past A's allocation.
    void fill_tile_layout(const float* b, size_t n, size_t d) {
        size_t kp = (d + 1) / 2;
        n_tiles_ = (n + 15) / 16;
        kp_tiles_ = (kp + 15) / 16;
        tile_buffer_.assign(n_tiles_ * kp_tiles_ * kTileElements, uint32_t{0});
        for (size_t j = 0; j < n; ++j) {
            const float* bj = b + j * d;
            size_t nt = j / 16;
            size_t lane = j % 16;
            for (size_t k = 0; k < d; ++k) {
                size_t kp_idx = k / 2;
                size_t kt = kp_idx / 16;
                size_t row = kp_idx % 16;
                uint32_t bits = static_cast<uint32_t>(svs::round_to_bfloat16(bj[k]).raw());
                uint32_t* dword =
                    &tile_buffer_[(nt * kp_tiles_ + kt) * kTileElements + row * 16 + lane];
                if (k % 2 == 0) {
                    *dword = (*dword & 0xFFFF0000u) | bits;
                } else {
                    *dword = (*dword & 0x0000FFFFu) | (bits << 16);
                }
            }
        }
        has_tile_layout_ = true;
    }

    std::vector<float> buffer_;
    size_t n_ = 0;
    size_t d_ = 0;
    std::vector<uint32_t> tile_buffer_;
    size_t n_tiles_ = 0;
    size_t kp_tiles_ = 0;
    bool has_tile_layout_ = false;
};

/// Per-ISA kernel family for a large row-major A (m x d) against B, packed once into a
/// `PackedOperand` by the caller. This primary template is the scalar, compiler
/// auto-vectorized generic implementation: both the fallback when no SIMD ISA is
/// available and the reference the ISA-specific specializations below are checked
/// against in tests. Those specializations are declared here and defined in avx2.cpp /
/// avx512.cpp so their intrinsics only compile under their own -march flags, mirroring
/// QChamferExactImpl in gem.h.
///
/// Two output forms share every row of A they stream over:
/// * `build`/`build_packed`: the dense table, out[i * n + j] = <a_i, b_j>.
/// * `column_max`: out[j] = max over i of <a_i, b_j>, i.e. the column-wise maximum of
///   that same table, computed with no table ever materialized.
///
/// Both forms are additionally overloaded on A's element type (float32, svs::Float16 and
/// svs::BFloat16); the narrow-type overloads convert one row-block of A at a time into a
/// float32 scratch and run the identical float32 path, so no element type is ever branched
/// on per row. BFloat16's widening is exact (it is float32's high 16 bits, zero-extended),
/// so unlike Float16 there is no rounding question on A's side; B's precision for a
/// BFloat16 A is the caller's choice, made by packing with
/// `PackedOperand::pack_rounded_to_bf16` instead of `pack`.
template <AVX_AVAILABILITY Avx, typename = void> struct InnerProductMatrixImpl {
    // Shared by every element type A can hold: float32 needs no conversion (the
    // static_cast below is a no-op then), Float16/BFloat16 each widen losslessly to
    // float32 per element. One shape, never a per-row branch on T.
    template <typename T>
    static void
    build_packed_generic(const T* a, size_t m, const PackedOperand& b, float* out) {
        size_t n = b.n();
        size_t d = b.d();
        const float* bt = b.data();
        for (size_t i = 0; i < m; ++i) {
            const T* ai = a + i * d;
            float* row = out + i * n;
            for (size_t j = 0; j < n; ++j) {
                float dot = 0.0F;
                for (size_t k = 0; k < d; ++k) {
                    dot += static_cast<float>(ai[k]) * bt[k * n + j];
                }
                row[j] = dot;
            }
        }
    }

    SVS_NOINLINE static void
    build_packed(const float* a, size_t m, const PackedOperand& b, float* out) {
        build_packed_generic(a, m, b, out);
    }

    SVS_NOINLINE static void
    build_packed(const svs::Float16* a, size_t m, const PackedOperand& b, float* out) {
        build_packed_generic(a, m, b, out);
    }

    SVS_NOINLINE static void
    build_packed(const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out) {
        build_packed_generic(a, m, b, out);
    }

    // Packs B into a reused thread_local buffer so this keeps its no-allocation-per-call
    // property: InnerProductMatrix::compute (GEM's fix_argument path) calls this once per
    // query against all K centroids, not once per document, so repacking here is cheap.
    SVS_NOINLINE static void
    build(const float* a, size_t m, const float* b, size_t n, size_t d, float* out) {
        thread_local PackedOperand packed;
        packed.pack(b, n, d);
        build_packed(a, m, packed, out);
    }

    template <typename T>
    static void
    column_max_generic(const T* a, size_t m, const PackedOperand& b, float* out) {
        size_t n = b.n();
        size_t d = b.d();
        const float* bt = b.data();
        std::fill(out, out + n, -std::numeric_limits<float>::infinity());
        for (size_t i = 0; i < m; ++i) {
            const T* ai = a + i * d;
            for (size_t j = 0; j < n; ++j) {
                float dot = 0.0F;
                for (size_t k = 0; k < d; ++k) {
                    dot += static_cast<float>(ai[k]) * bt[k * n + j];
                }
                out[j] = std::max(out[j], dot);
            }
        }
    }

    SVS_NOINLINE static void
    column_max(const float* a, size_t m, const PackedOperand& b, float* out) {
        column_max_generic(a, m, b, out);
    }

    SVS_NOINLINE static void
    column_max(const svs::Float16* a, size_t m, const PackedOperand& b, float* out) {
        column_max_generic(a, m, b, out);
    }

    SVS_NOINLINE static void
    column_max(const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out) {
        column_max_generic(a, m, b, out);
    }
};

#if defined(__x86_64__)

SVS_VALIDATE_BOOL_ENV(SVS_AVX512_F)
#if SVS_AVX512_F

namespace inner_product_matrix_avx512_detail {

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
inline void build_packed_narrow(const T* a, size_t m, const PackedOperand& b, float* out) {
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
inline void column_max_narrow(const T* a, size_t m, const PackedOperand& b, float* out) {
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

} // namespace inner_product_matrix_avx512_detail

// Partial specializations on a dummy parameter: a full one's explicit instantiation emits
// no inline members, and GCC 11 emits nothing for a second constrained one with equal
// arguments.
template <typename Unused> struct InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512, Unused> {
    // Register-blocked micro-kernel: inner_product_matrix_row_block rows of A share one
    // pass over a lane-block of B, giving that many independent FMA accumulator chains
    // with no horizontal reduction (each lane is one output), so the dependency-chain
    // latency of a single accumulator is hidden behind the others instead of
    // serializing the loop over d.
    SVS_NOINLINE static void
    build_packed(const float* a, size_t m, const PackedOperand& b, float* out) {
        using namespace inner_product_matrix_avx512_detail;
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
    SVS_NOINLINE static void
    build_packed(const svs::Float16* a, size_t m, const PackedOperand& b, float* out) {
        inner_product_matrix_avx512_detail::build_packed_narrow(a, m, b, out);
    }

    // Table form for a BFloat16 A: see convert_row_block above for the shared widening.
    SVS_NOINLINE static void
    build_packed(const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out) {
        inner_product_matrix_avx512_detail::build_packed_narrow(a, m, b, out);
    }

    // Packs B into a reused thread_local buffer: the table form repacks on every call
    // (one query against all centroids), unlike MaxSim's column_max below which packs
    // once per query and calls into this ISA's kernels directly with the already-packed
    // operand.
    SVS_NOINLINE static void
    build(const float* a, size_t m, const float* b, size_t n, size_t d, float* out) {
        if (m == 0 || n == 0) {
            return;
        }
        thread_local PackedOperand packed;
        packed.pack(b, n, d);
        build_packed(a, m, packed, out);
    }

    // Column-wise maximum of the same table build_packed computes, without ever
    // materializing it: each row-block's accumulators are reduced to a per-lane maximum
    // and folded into a running maximum held in `out` itself, so one reduction serves
    // the whole document for the price of one load/max/store per lane-block per
    // row-block.
    SVS_NOINLINE static void
    column_max(const float* a, size_t m, const PackedOperand& b, float* out) {
        using namespace inner_product_matrix_avx512_detail;
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
    SVS_NOINLINE static void
    column_max(const svs::Float16* a, size_t m, const PackedOperand& b, float* out) {
        inner_product_matrix_avx512_detail::column_max_narrow(a, m, b, out);
    }

    // BFloat16 overload: see convert_row_block above for the shared (exact) widening.
    SVS_NOINLINE static void
    column_max(const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out) {
        inner_product_matrix_avx512_detail::column_max_narrow(a, m, b, out);
    }
};
#endif

SVS_VALIDATE_BOOL_ENV(SVS_AVX512_F)
SVS_VALIDATE_BOOL_ENV(SVS_AVX2)
#if !SVS_AVX512_F && SVS_AVX2

namespace inner_product_matrix_avx2_detail {

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

// Pairwise-max reduction; see avx512_detail::reduce_max for why padding slots are safe
// to include.
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
inline void build_packed_narrow(const T* a, size_t m, const PackedOperand& b, float* out) {
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
inline void column_max_narrow(const T* a, size_t m, const PackedOperand& b, float* out) {
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

} // namespace inner_product_matrix_avx2_detail

template <typename Unused> struct InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2, Unused> {
    // Same register-blocking as InnerProductMatrixImpl<AVX512>::build_packed, at AVX2's
    // 8-wide registers.
    SVS_NOINLINE static void
    build_packed(const float* a, size_t m, const PackedOperand& b, float* out) {
        using namespace inner_product_matrix_avx2_detail;
        size_t n = b.n();
        size_t d = b.d();
        if (m == 0 || n == 0) {
            return;
        }
        constexpr size_t vl = 8;
        const float* bt = b.data();
        size_t full_lanes = n - n % vl;
        uint8_t tail_bits = svs::create_mask<vl>(svs::lib::MaybeStatic(n));
        __m256i tail_mask =
            _mm256_castps_si256(svs::simd::create_blend_mask_avx2(tail_bits));

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
    SVS_NOINLINE static void
    build_packed(const svs::Float16* a, size_t m, const PackedOperand& b, float* out) {
        inner_product_matrix_avx2_detail::build_packed_narrow(a, m, b, out);
    }

    // Table form for a BFloat16 A: see convert_row_block above for the shared widening.
    SVS_NOINLINE static void
    build_packed(const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out) {
        inner_product_matrix_avx2_detail::build_packed_narrow(a, m, b, out);
    }

    // See InnerProductMatrixImpl<AVX512>::build for why this packs into a thread_local
    // buffer rather than taking a PackedOperand from the caller.
    SVS_NOINLINE static void
    build(const float* a, size_t m, const float* b, size_t n, size_t d, float* out) {
        if (m == 0 || n == 0) {
            return;
        }
        thread_local PackedOperand packed;
        packed.pack(b, n, d);
        build_packed(a, m, packed, out);
    }

    // See InnerProductMatrixImpl<AVX512>::column_max for the algorithm; this is the same
    // reduction at AVX2's 8-wide registers.
    SVS_NOINLINE static void
    column_max(const float* a, size_t m, const PackedOperand& b, float* out) {
        using namespace inner_product_matrix_avx2_detail;
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
        __m256i tail_mask =
            _mm256_castps_si256(svs::simd::create_blend_mask_avx2(tail_bits));

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

    // Float16 overload: see convert_row_block above for the shared widening (F16C's
    // 8-wide _mm256_cvtph_ps, available because avx2.cpp is compiled for the "haswell"
    // arch in cmake/multi-arch.cmake, and Haswell carries F16C alongside AVX2).
    SVS_NOINLINE static void
    column_max(const svs::Float16* a, size_t m, const PackedOperand& b, float* out) {
        inner_product_matrix_avx2_detail::column_max_narrow(a, m, b, out);
    }

    // BFloat16 overload: see convert_row_block above for the shared (exact) widening.
    SVS_NOINLINE static void
    column_max(const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out) {
        inner_product_matrix_avx2_detail::column_max_narrow(a, m, b, out);
    }
};
#endif

// The scalar-distance kernels (L2/IP/cosine) and QEmd/QChamfer/QChamferExactImpl dispatch
// through a runtime if-chain on svs::detail::avx_runtime_flags (euclidean.h, cosine.h,
// inner_product.h, gem.h), never through a switch or template argument naming
// AVX_AVAILABILITY::AMX; adding the AMX enumerator needs no new instantiation of any of
// them (checked: `git grep -n AVX_AVAILABILITY` finds no exhaustive switch or `AMX`
// branch to add outside InnerProductMatrix). AMX only applies to the InnerProductMatrix
// family below, whose public dispatch is the one exhaustive switch this change extends.

// Declared (full member set, matching every other level) only when SVS_AMX_TILE_BF16
// (__AMX_TILE__ && __AMX_BF16__) is set for the current TU's own compiler flags: only
// amx.cpp is compiled with -mamx-tile -mamx-bf16, same per-TU pattern as
// SVS_AVX512_F/SVS_AVX2 above, not the build-wide SVS_AMX_COMPILED_IN that PackedOperand
// reads. build_packed/column_max for float32 and Float16 A, and for a BFloat16 A whose B
// was not packed with the tile layout, forward to the AVX512 implementation (explicit
// forwarding below, one source of truth); only a BFloat16 A against a tile-packed B runs
// the real tile kernel.
SVS_VALIDATE_BOOL_ENV(SVS_AMX_TILE_BF16)
#if SVS_AMX_TILE_BF16

namespace inner_product_matrix_amx_detail {

// 64-byte tile-config struct the palette-1 AMX instructions read (Intel SDM Vol 1,
// 3.3.7): colsb/rows per tmm register 0-15, reserved bytes must stay zero.
struct TileConfig {
    uint8_t palette_id = 0;
    uint8_t start_row = 0;
    uint8_t reserved[14] = {};
    uint16_t colsb[16] = {};
    uint8_t rows[16] = {};
};

// Releases tile state on every exit path (normal return or exception): leaving tile
// state live costs later context switches and ordinary SSE/AVX code extra save/restore
// work, per Intel's optimization guidance for AMX.
struct TileGuard {
    TileGuard() = default;
    ~TileGuard() { _tile_release(); }
    TileGuard(const TileGuard&) = delete;
    TileGuard& operator=(const TileGuard&) = delete;
};

// tmm register assignment: tmm0 = A's current k-chunk (16 rows x 32 bf16, i.e. 16
// pairs); tmm1/tmm2 = B's two n-lane-tiles for the pair-chunk (pair-interleaved
// transposed bf16, see PackedOperand::tile_block); tmm3/tmm4 = the matching fp32
// accumulators. Two n-tiles in flight so their dpbf16ps chains are independent of each
// other (both read tmm0, but write distinct accumulators), per the task's "several
// accumulator tiles overlap" requirement. Named here for the tile-config array indexing
// below (ordinary C++, any expression is fine); every `_tile_*` call past this point
// uses the literal 0/1/2/3/4 directly, because gcc's tile intrinsics are macros that
// stringify their register argument (`#dst`), so a named constant used there would
// stringify to its identifier text, not a digit, and fail to assemble.
enum Tmm : int { kTmmA = 0, kTmmB0 = 1, kTmmB1 = 2, kTmmAcc0 = 3, kTmmAcc1 = 4 };

// gcc 11's _tile_loadd/_tile_stored/_tile_loadconfig macros expand to inline asm with no
// memory operand and no clobber, so the optimizer believes they neither read nor write
// memory: a plain C++ store just before a load, or a plain C++ read just after a store,
// through the same address has no ordering guarantee and can be reordered around the
// tile instruction (sunk store -> stale/garbage read; hoisted read -> stale/garbage
// value), independent of any `volatile` on the asm itself (that only orders the asm
// relative to *other* volatile operations, not to ordinary memory accesses). Every tile
// memory access in this file goes through one of the wrappers below instead of the bare
// intrinsic, so the dependency is a real operand or clobber the compiler understands
// rather than an accident of how today's code generation happens to schedule things.
// `_tile_zero`/`_tile_dpbf16ps`/`_tile_release` touch only tile *registers*, never
// memory, so they keep using the compiler's intrinsics unwrapped.

// Loads the 64-byte tile configuration from `cfg`, naming the whole struct as the
// input memory this reads (not just a pointer-sized slice of it, which is what gcc's own
// _tile_loadconfig macro's cast effectively does): every `cfg.rows[]`/`cfg.colsb[]` store
// before this call is now visibly a dependency, not an accident of scheduling.
inline void tile_loadconfig_safe(const TileConfig& cfg) {
    __asm__ volatile("ldtilecfg\t%0" ::"m"(cfg));
}

// One generated function per literal tile number actually used: the mnemonic's register
// is spelled into the asm text (`#tmm`), which gcc's own tile intrinsics require to be a
// literal token, so a runtime tile-number parameter is not an option. Kept local to this
// namespace: the macro only generates definitions used a few lines below it, nothing
// external depends on it (include/svs/multi-arch/AGENTS.md's rule is about a macro whose
// defining header is absent from a TU that uses it, which does not apply to a macro
// defined and consumed in the same file).
//
// _strided takes a runtime stride (A's row pitch, which varies with `d`) and so cannot
// name a compile-time memory extent; its "memory" clobber is a full barrier instead,
// forcing every pending store to complete before the load and every cached value to be
// re-read after it.
#define SVS_AMX_DEFINE_TILE_LOADD_STRIDED(tmm)                                        \
    inline void tile_loadd_tmm##tmm##_strided(const void* base, long stride) {        \
        __asm__ volatile("{tileloadd\t(%0,%1,1), %%tmm" #tmm "|tileloadd\t%%tmm" #tmm \
                         ", [%0+%1*1]}" ::"r"(base),                                  \
                         "r"(stride)                                                  \
                         : "memory");                                                 \
    }

// _fixed is for a source whose extent is a compile-time constant 1024 bytes (B's tile
// blocks, always a full 16x64-byte tile; the local zero-padded tail buffer, likewise
// 16x64 bytes): the dummy "m" operand (unused in the asm text) names that exact memory as
// read, so the compiler orders this correctly against its last write without needing a
// blanket barrier.
#define SVS_AMX_DEFINE_TILE_LOADD_FIXED(tmm)                             \
    inline void tile_loadd_tmm##tmm##_fixed(const void* base) {          \
        __asm__ volatile(                                                \
            "{tileloadd\t(%0,%1,1), %%tmm" #tmm "|tileloadd\t%%tmm" #tmm \
            ", [%0+%1*1]}" ::"r"(base),                                  \
            "r"(64L),                                                    \
            "m"(*reinterpret_cast<const char(*)[1024]>(base))            \
        );                                                               \
    }

// Stores are always to a caller-owned 16x16 fp32 buffer (1024 bytes, compile-time
// constant): the dummy "=m" output operand tells the compiler this asm writes that
// memory, so a later plain read of the same buffer is ordered after the store instead of
// possibly being hoisted before it.
#define SVS_AMX_DEFINE_TILE_STORED_FIXED(tmm)                                \
    inline void tile_stored_tmm##tmm##_fixed(void* base) {                   \
        __asm__ volatile("{tilestored\t%%tmm" #tmm                           \
                         ", (%1,%2,1)|tilestored\t[%1+%2*1], %%tmm" #tmm "}" \
                         : "=m"(*reinterpret_cast<char(*)[1024]>(base))      \
                         : "r"(base), "r"(64L));                             \
    }

SVS_AMX_DEFINE_TILE_LOADD_STRIDED(0)
SVS_AMX_DEFINE_TILE_LOADD_FIXED(0)
SVS_AMX_DEFINE_TILE_LOADD_FIXED(1)
SVS_AMX_DEFINE_TILE_LOADD_FIXED(2)
SVS_AMX_DEFINE_TILE_STORED_FIXED(3)
SVS_AMX_DEFINE_TILE_STORED_FIXED(4)

#undef SVS_AMX_DEFINE_TILE_LOADD_STRIDED
#undef SVS_AMX_DEFINE_TILE_LOADD_FIXED
#undef SVS_AMX_DEFINE_TILE_STORED_FIXED

inline void configure_tiles(uint8_t a_acc_rows) {
    TileConfig cfg;
    cfg.palette_id = 1;
    cfg.rows[kTmmA] = a_acc_rows;
    cfg.colsb[kTmmA] = 64;
    cfg.rows[kTmmAcc0] = a_acc_rows;
    cfg.colsb[kTmmAcc0] = 64;
    cfg.rows[kTmmAcc1] = a_acc_rows;
    cfg.colsb[kTmmAcc1] = 64;
    // B's tiles are always a full 16 (pair-)rows: PackedOperand::tile_block zero-pads
    // every tile block to 16x16, so there is no partial-row case to configure for them.
    cfg.rows[kTmmB0] = 16;
    cfg.colsb[kTmmB0] = 64;
    cfg.rows[kTmmB1] = 16;
    cfg.colsb[kTmmB1] = 64;
    tile_loadconfig_safe(cfg);
}

// Computes the dot products of up to 16 rows of A (starting at `a_row0`, already known
// not to read past `d` columns of any row via the tail copy below) against B's n-tiles
// `nt0` (and `nt1` when `has_nt1`) for every k-chunk, leaving the result in tmm3 (and
// tmm4). Tiles must already be configured by configure_tiles() for this call's row count.
//
// A's tail k-chunk (the last one, when d is not a multiple of 32) is copied through a
// zero-padded stack buffer rather than loaded directly: loading straight from `a_row0`
// would read up to 31 bf16 elements past column d-1 of the block's last real row, which
// is caller memory this kernel does not own past `d`.
inline void accumulate_group(
    const svs::BFloat16* a_row0,
    size_t rows,
    size_t stride_bytes,
    size_t full_k_chunks,
    size_t k_tail,
    const PackedOperand& b,
    size_t nt0,
    bool has_nt1,
    size_t nt1
) {
    _tile_zero(3);
    if (has_nt1) {
        _tile_zero(4);
    }
    const auto* base = reinterpret_cast<const char*>(a_row0);
    for (size_t kt = 0; kt < full_k_chunks; ++kt) {
        tile_loadd_tmm0_strided(base + kt * 64, static_cast<long>(stride_bytes));
        tile_loadd_tmm1_fixed(b.tile_block(nt0, kt));
        _tile_dpbf16ps(3, 0, 1);
        if (has_nt1) {
            tile_loadd_tmm2_fixed(b.tile_block(nt1, kt));
            _tile_dpbf16ps(4, 0, 2);
        }
    }
    if (k_tail > 0) {
        alignas(64) uint16_t tail[16][32] = {};
        for (size_t r = 0; r < rows; ++r) {
            std::memcpy(tail[r], base + r * stride_bytes + full_k_chunks * 64, k_tail * 2);
        }
        tile_loadd_tmm0_fixed(tail);
        tile_loadd_tmm1_fixed(b.tile_block(nt0, full_k_chunks));
        _tile_dpbf16ps(3, 0, 1);
        if (has_nt1) {
            tile_loadd_tmm2_fixed(b.tile_block(nt1, full_k_chunks));
            _tile_dpbf16ps(4, 0, 2);
        }
    }
}

// Shared row-block/n-tile-group driver for both output forms below: walks A 16 rows at a
// time (reconfiguring tiles once more only for a final row-block narrower than 16, never
// per row-block in the steady state) and B 32 lanes (two n-tiles) at a time, handing each
// block's two 16x16 fp32 results to `emit(row0, rows, nt0, has_nt1, nt1, c0, c1)`.
template <typename Emit>
inline void for_each_block(
    const svs::BFloat16* a, size_t m, size_t d, const PackedOperand& b, Emit&& emit
) {
    size_t n_tiles = b.n_tiles();
    size_t full_k_chunks = d / 32;
    size_t k_tail = d % 32;
    size_t stride_bytes = d * 2;

    TileGuard guard;
    configure_tiles(16);
    bool configured_full = true;

    for (size_t i = 0; i < m; i += 16) {
        size_t rows = std::min<size_t>(16, m - i);
        if (rows < 16 && configured_full) {
            configure_tiles(static_cast<uint8_t>(rows));
            configured_full = false;
        }
        const svs::BFloat16* row0 = a + i * d;
        for (size_t nt0 = 0; nt0 < n_tiles; nt0 += 2) {
            bool has_nt1 = nt0 + 1 < n_tiles;
            size_t nt1 = has_nt1 ? nt0 + 1 : nt0;
            alignas(64) float c0[16][16];
            alignas(64) float c1[16][16];
            accumulate_group(
                row0, rows, stride_bytes, full_k_chunks, k_tail, b, nt0, has_nt1, nt1
            );
            tile_stored_tmm3_fixed(c0);
            if (has_nt1) {
                tile_stored_tmm4_fixed(c1);
            }
            emit(i, rows, nt0, has_nt1, nt1, c0, c1);
        }
    }
}

// AMX table kernel for a BFloat16 A against a tile-packed B: writes out[i*n+j] exactly
// like the generic/AVX-512 forms, with the same <bf16(a), bf16(b)> values (summed in a
// different order, hence InnerProductMatrix's tolerance-based tests, not a bit-exact
// one).
inline void
amx_table_bf16(const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out) {
    size_t n = b.n();
    for_each_block(
        a,
        m,
        b.d(),
        b,
        [&](size_t row0,
            size_t rows,
            size_t nt0,
            bool has_nt1,
            size_t nt1,
            auto& c0,
            auto& c1) {
            size_t lanes0 = std::min<size_t>(16, n - nt0 * 16);
            for (size_t r = 0; r < rows; ++r) {
                float* dst = out + (row0 + r) * n + nt0 * 16;
                std::memcpy(dst, c0[r], lanes0 * sizeof(float));
            }
            if (has_nt1) {
                size_t lanes1 = std::min<size_t>(16, n - nt1 * 16);
                for (size_t r = 0; r < rows; ++r) {
                    float* dst = out + (row0 + r) * n + nt1 * 16;
                    std::memcpy(dst, c1[r], lanes1 * sizeof(float));
                }
            }
        }
    );
}

// AMX column-max kernel for a BFloat16 A against a tile-packed B: folds each block's two
// 16x16 tiles into the running per-lane maximum with AVX-512 (the AMX level implies
// AVX-512F), reusing the same row-then-lane reduction shape as
// InnerProductMatrixImpl<AVX512>::column_max above.
inline void
amx_column_max_bf16(const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out) {
    size_t n = b.n();
    std::fill(out, out + n, -std::numeric_limits<float>::infinity());
    if (m == 0) {
        return;
    }
    for_each_block(
        a,
        m,
        b.d(),
        b,
        [&](size_t /*row0*/,
            size_t rows,
            size_t nt0,
            bool has_nt1,
            size_t nt1,
            auto& c0,
            auto& c1) {
            auto fold = [&](size_t nt, auto& c) {
                size_t lanes = std::min<size_t>(16, n - nt * 16);
                alignas(64) float colmax[16] = {};
                for (size_t j = 0; j < 16; ++j) {
                    float best = -std::numeric_limits<float>::infinity();
                    for (size_t r = 0; r < rows; ++r) {
                        best = std::max(best, c[r][j]);
                    }
                    colmax[j] = best;
                }
                for (size_t j = 0; j < lanes; ++j) {
                    out[nt * 16 + j] = std::max(out[nt * 16 + j], colmax[j]);
                }
            };
            fold(nt0, c0);
            if (has_nt1) {
                fold(nt1, c1);
            }
        }
    );
}

} // namespace inner_product_matrix_amx_detail

template <typename Unused> struct InnerProductMatrixImpl<AVX_AVAILABILITY::AMX, Unused> {
    // Never the AMX precision case (A is float32, not bf16): one source of truth is the
    // AVX-512 kernel.
    SVS_NOINLINE static void
    build_packed(const float* a, size_t m, const PackedOperand& b, float* out) {
        InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build_packed(a, m, b, out);
    }

    SVS_NOINLINE static void
    build_packed(const svs::Float16* a, size_t m, const PackedOperand& b, float* out) {
        InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build_packed(a, m, b, out);
    }

    SVS_NOINLINE static void
    build_packed(const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out) {
        if (m == 0 || b.n() == 0) {
            return;
        }
        if (!b.has_tile_layout()) {
            InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build_packed(a, m, b, out);
            return;
        }
        inner_product_matrix_amx_detail::amx_table_bf16(a, m, b, out);
    }

    // `a` is always float32 here (InnerProductMatrix::compute's only entry point to this
    // overload): the AMX precision case never applies, so this is a plain forward.
    SVS_NOINLINE static void
    build(const float* a, size_t m, const float* b, size_t n, size_t d, float* out) {
        InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build(a, m, b, n, d, out);
    }

    SVS_NOINLINE static void
    column_max(const float* a, size_t m, const PackedOperand& b, float* out) {
        InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::column_max(a, m, b, out);
    }

    SVS_NOINLINE static void
    column_max(const svs::Float16* a, size_t m, const PackedOperand& b, float* out) {
        InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::column_max(a, m, b, out);
    }

    SVS_NOINLINE static void
    column_max(const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out) {
        if (!b.has_tile_layout()) {
            InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::column_max(a, m, b, out);
            return;
        }
        inner_product_matrix_amx_detail::amx_column_max_bf16(a, m, b, out);
    }
};
#endif

#endif // defined(__x86_64__)

#if defined(__x86_64__)
extern template struct InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>;
extern template struct InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2>;
#endif
// Gated on the build-wide SVS_AMX_COMPILED_IN, not __x86_64__ alone: amx.cpp is only built
// when the compiler accepts -mamx-tile -mamx-bf16 (cmake/multi-arch.cmake's
// SVS_COMPILER_SUPPORTS_AMX check), compiled at -march=cascadelake plus those two flags,
// not at -march=sapphirerapids; when that check fails, no TU anywhere provides
// `template struct InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>;`'s definition, and an
// unconditional `extern template` here would then be an unresolved symbol at link time
// the moment anything calls InnerProductMatrix, not just when AMX runs.
#if SVS_AMX_COMPILED_IN
extern template struct InnerProductMatrixImpl<AVX_AVAILABILITY::AMX>;
#endif

/// Public entry points for the InnerProductMatrixImpl kernel family: the only place that
/// picks an ISA for this kernel. A caller that needs this kernel's output names the form
/// it wants (`compute` for the table, `column_max` for the column-wise maximum) and never
/// names `InnerProductMatrixImpl` or an `AVX_AVAILABILITY` enumerator itself; the layer
/// that owns the ISA choice (this class) is the only one that repeats it.
class InnerProductMatrix {
  public:
    /// Dense inner-product table between a large row-major matrix and a small row-major
    /// matrix, both float32: out[i * n + j] = <a_i, b_j> for i in [0, m), j in [0, n).
    ///
    /// @param a Row-major "A" operand, m rows of d floats each; the large, streamed
    ///     operand (e.g. all centroids).
    /// @param m Number of rows of `a`. 0 is valid and leaves `out` untouched.
    /// @param b Row-major "B" operand, n rows of d floats each; the small operand (e.g.
    ///     one query's tokens). Packed once per call into a thread_local PackedOperand
    ///     this class owns, so `b` itself is read once and not retained.
    /// @param n Number of rows of `b`. 0 is valid and leaves `out` untouched.
    /// @param d Row width shared by `a` and `b`.
    /// @param out Row-major output buffer, caller-owned, at least m * n floats:
    ///     out[i * n + j] holds <a_i, b_j>. Untouched when m == 0 or n == 0.
    static void
    compute(const float* a, size_t m, const float* b, size_t n, size_t d, float* out) {
        if (m == 0 || n == 0) {
            return;
        }
        dispatch([&](auto avx) {
            InnerProductMatrixImpl<decltype(avx)::value>::build(a, m, b, n, d, out);
        });
    }

    /// Column-wise maximum of the same table `compute` would produce, for a float32 A,
    /// against an already-packed B: out[j] = max over i of <a_i, b_j>, for j in [0, n).
    /// No table is ever materialized.
    ///
    /// @param a Row-major "A" operand, m rows of `b.d()` floats each; the large,
    ///     streamed operand (e.g. one document's tokens).
    /// @param m Number of rows of `a`. 0 is valid: every lane of `out` is set to
    ///     -infinity, matching an empty A having no row to contribute a maximum.
    /// @param b A `PackedOperand` already packed by the caller (e.g. once per query),
    ///     read but not modified.
    /// @param out Caller-owned output buffer of exactly `b.n()` floats, overwritten in
    ///     full (every lane, including when m == 0): out[j] holds the column maximum
    ///     for j in [0, b.n()).
    static void column_max(const float* a, size_t m, const PackedOperand& b, float* out) {
        dispatch([&](auto avx) {
            InnerProductMatrixImpl<decltype(avx)::value>::column_max(a, m, b, out);
        });
    }

    /// Same contract as the float32 `column_max` above, for a svs::Float16 A: converts
    /// one row-block of `a` at a time into a float32 scratch and reduces identically.
    static void
    column_max(const svs::Float16* a, size_t m, const PackedOperand& b, float* out) {
        dispatch([&](auto avx) {
            InnerProductMatrixImpl<decltype(avx)::value>::column_max(a, m, b, out);
        });
    }

    /// Same contract as the float32 `column_max` above, for a svs::BFloat16 A.
    static void
    column_max(const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out) {
        dispatch([&](auto avx) {
            InnerProductMatrixImpl<decltype(avx)::value>::column_max(a, m, b, out);
        });
    }

    /// Dense inner-product table (same layout as `compute`) between a large row-major A
    /// and an already-packed B, for an A element type other than float32. Unlike
    /// `compute`, this takes a `PackedOperand` the caller packed itself, so the caller
    /// controls B's precision (e.g. `PackedOperand::pack_rounded_to_bf16` for a bf16 A, so
    /// every ISA level sees the same rounded values on both operands).
    ///
    /// @param a Row-major "A" operand, m rows of `b.d()` elements each.
    /// @param m Number of rows of `a`. 0 is valid and leaves `out` untouched.
    /// @param b A `PackedOperand` already packed by the caller, read but not modified.
    /// @param out Row-major output buffer, caller-owned, at least m * b.n() floats.
    static void
    compute_packed(const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out) {
        if (m == 0 || b.n() == 0) {
            return;
        }
        dispatch([&](auto avx) {
            InnerProductMatrixImpl<decltype(avx)::value>::build_packed(a, m, b, out);
        });
    }

    /// Same contract as `compute_packed` above, for a svs::Float16 A.
    static void
    compute_packed(const svs::Float16* a, size_t m, const PackedOperand& b, float* out) {
        if (m == 0 || b.n() == 0) {
            return;
        }
        dispatch([&](auto avx) {
            InnerProductMatrixImpl<decltype(avx)::value>::build_packed(a, m, b, out);
        });
    }

  private:
    // Picks the ISA once per call and invokes `f` with that choice as a compile-time
    // constant (an std::integral_constant, so `InnerProductMatrixImpl<decltype(avx)::
    // value>` below is a valid template argument), so every public entry point above
    // writes this exhaustive switch zero times of its own.
    template <typename F> static void dispatch(F&& f) {
        switch (select_isa()) {
            case AVX_AVAILABILITY::AMX:
                // select_isa() only returns AMX when SVS_AMX_COMPILED_IN (amx.cpp was
                // built), so the symbol this case names exists only then; this case
                // itself must stay unconditional so -Wswitch under -Werror still catches
                // a 5th enumerator added later. Without SVS_AMX_COMPILED_IN this case is
                // unreachable (select_isa() never returns it) and falls back to AVX512,
                // never naming a specialization that cannot exist in this build.
#if SVS_AMX_COMPILED_IN
                f(std::integral_constant<AVX_AVAILABILITY, AVX_AVAILABILITY::AMX>{});
#else
                f(std::integral_constant<AVX_AVAILABILITY, AVX_AVAILABILITY::AVX512>{});
#endif
                return;
            case AVX_AVAILABILITY::AVX512:
                f(std::integral_constant<AVX_AVAILABILITY, AVX_AVAILABILITY::AVX512>{});
                return;
            case AVX_AVAILABILITY::AVX2:
                f(std::integral_constant<AVX_AVAILABILITY, AVX_AVAILABILITY::AVX2>{});
                return;
            case AVX_AVAILABILITY::NONE:
                f(std::integral_constant<AVX_AVAILABILITY, AVX_AVAILABILITY::NONE>{});
                return;
        }
        // Exhaustive switch over AVX_AVAILABILITY's enumerators, no default: -Wswitch
        // under -Werror turns a 5th enumerator added later into a compile error here,
        // rather than this call silently keeping today's dispatch.
        __builtin_unreachable();
    }

    // AMX is checked first regardless of A's element type or B's packing: every
    // InnerProductMatrixImpl<AMX> member handles "not actually usable for this call"
    // itself (forwarding to AVX512), so this one-branch-per-call choice never needs to
    // know the call's operand types. Gated on SVS_AMX_COMPILED_IN so a build that never
    // compiled amx.cpp never returns an enumerator dispatch() cannot safely name.
    static AVX_AVAILABILITY select_isa() {
#if SVS_AMX_COMPILED_IN
        if (svs::detail::amx_tile_bf16_available()) {
            return AVX_AVAILABILITY::AMX;
        }
#endif
        if (svs::detail::avx_runtime_flags.is_avx512f_supported()) {
            return AVX_AVAILABILITY::AVX512;
        }
        if (svs::detail::avx_runtime_flags.is_avx2_supported()) {
            return AVX_AVAILABILITY::AVX2;
        }
        return AVX_AVAILABILITY::NONE;
    }
};

} // namespace svs::distance
