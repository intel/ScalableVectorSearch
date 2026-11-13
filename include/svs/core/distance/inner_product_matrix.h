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
#include "svs/lib/avx_detection.h"
#include "svs/lib/bfloat16.h"
#include "svs/lib/float16.h"
#include "svs/lib/preprocessor.h"

#include <algorithm>
#include <cstddef>
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
    }

    size_t n() const { return n_; }
    size_t d() const { return d_; }
    const float* data() const { return buffer_.data(); }

  private:
    std::vector<float> buffer_;
    size_t n_ = 0;
    size_t d_ = 0;
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
template <AVX_AVAILABILITY Avx> struct InnerProductMatrixImpl {
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
template <> struct InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512> {
    SVS_NOINLINE static void
    build_packed(const float* a, size_t m, const PackedOperand& b, float* out);
    SVS_NOINLINE static void
    build_packed(const svs::Float16* a, size_t m, const PackedOperand& b, float* out);
    SVS_NOINLINE static void
    build_packed(const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out);
    SVS_NOINLINE static void
    build(const float* a, size_t m, const float* b, size_t n, size_t d, float* out);
    SVS_NOINLINE static void
    column_max(const float* a, size_t m, const PackedOperand& b, float* out);
    SVS_NOINLINE static void
    column_max(const svs::Float16* a, size_t m, const PackedOperand& b, float* out);
    SVS_NOINLINE static void
    column_max(const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out);
};
#endif

SVS_VALIDATE_BOOL_ENV(SVS_AVX512_F)
SVS_VALIDATE_BOOL_ENV(SVS_AVX2)
#if !SVS_AVX512_F && SVS_AVX2
template <> struct InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2> {
    SVS_NOINLINE static void
    build_packed(const float* a, size_t m, const PackedOperand& b, float* out);
    SVS_NOINLINE static void
    build_packed(const svs::Float16* a, size_t m, const PackedOperand& b, float* out);
    SVS_NOINLINE static void
    build_packed(const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out);
    SVS_NOINLINE static void
    build(const float* a, size_t m, const float* b, size_t n, size_t d, float* out);
    SVS_NOINLINE static void
    column_max(const float* a, size_t m, const PackedOperand& b, float* out);
    SVS_NOINLINE static void
    column_max(const svs::Float16* a, size_t m, const PackedOperand& b, float* out);
    SVS_NOINLINE static void
    column_max(const svs::BFloat16* a, size_t m, const PackedOperand& b, float* out);
};
#endif

#endif // defined(__x86_64__)

#if defined(__x86_64__)
extern template struct InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>;
extern template struct InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2>;
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
        // under -Werror turns a 4th enumerator added later into a compile error here,
        // rather than this call silently keeping today's dispatch.
        __builtin_unreachable();
    }

    static AVX_AVAILABILITY select_isa() {
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
