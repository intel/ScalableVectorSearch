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
#include "svs/lib/float16.h"
#include "svs/lib/preprocessor.h"

#include <algorithm>
#include <cstddef>
#include <limits>
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
/// `column_max` is additionally overloaded on A's element type (float32 and
/// svs::Float16); the Float16 overload converts one row-block of A at a time into a
/// float32 scratch and runs the identical float32 path, so the two element types are
/// never branched on per row.
template <AVX_AVAILABILITY Avx> struct InnerProductMatrixImpl {
    SVS_NOINLINE static void
    build_packed(const float* a, size_t m, const PackedOperand& b, float* out) {
        size_t n = b.n();
        size_t d = b.d();
        const float* bt = b.data();
        for (size_t i = 0; i < m; ++i) {
            const float* ai = a + i * d;
            float* row = out + i * n;
            for (size_t j = 0; j < n; ++j) {
                float dot = 0.0F;
                for (size_t k = 0; k < d; ++k) {
                    dot += ai[k] * bt[k * n + j];
                }
                row[j] = dot;
            }
        }
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
};

#if defined(__x86_64__)

SVS_VALIDATE_BOOL_ENV(SVS_AVX512_F)
#if SVS_AVX512_F
template <> struct InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512> {
    SVS_NOINLINE static void
    build_packed(const float* a, size_t m, const PackedOperand& b, float* out);
    SVS_NOINLINE static void
    build(const float* a, size_t m, const float* b, size_t n, size_t d, float* out);
    SVS_NOINLINE static void
    column_max(const float* a, size_t m, const PackedOperand& b, float* out);
    SVS_NOINLINE static void
    column_max(const svs::Float16* a, size_t m, const PackedOperand& b, float* out);
};
#endif

SVS_VALIDATE_BOOL_ENV(SVS_AVX512_F)
SVS_VALIDATE_BOOL_ENV(SVS_AVX2)
#if !SVS_AVX512_F && SVS_AVX2
template <> struct InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2> {
    SVS_NOINLINE static void
    build_packed(const float* a, size_t m, const PackedOperand& b, float* out);
    SVS_NOINLINE static void
    build(const float* a, size_t m, const float* b, size_t n, size_t d, float* out);
    SVS_NOINLINE static void
    column_max(const float* a, size_t m, const PackedOperand& b, float* out);
    SVS_NOINLINE static void
    column_max(const svs::Float16* a, size_t m, const PackedOperand& b, float* out);
};
#endif

#endif // defined(__x86_64__)

#if defined(__x86_64__)
extern template struct InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>;
extern template struct InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2>;
#endif

/// Dense inner-product table between a large row-major matrix and a small row-major
/// matrix, both float32: out[i * n + j] = <a_i, b_j> for i in [0, m), j in [0, n). Picks
/// the ISA implementation once per call, outside every loop over rows or columns.
///
/// @param a Row-major "A" operand, m rows of d floats each; the large, streamed operand
///     (e.g. all centroids).
/// @param m Number of rows of `a`. 0 is valid and leaves `out` untouched.
/// @param b Row-major "B" operand, n rows of d floats each; the small operand (e.g. one
///     query's tokens). Packed once per call into a thread_local PackedOperand the
///     implementation owns, so `b` itself is read once and not retained.
/// @param n Number of rows of `b`. 0 is valid and leaves `out` untouched.
/// @param d Row width shared by `a` and `b`.
/// @param out Row-major output buffer, caller-owned, at least m * n floats: out[i * n + j]
///     holds <a_i, b_j>.
class InnerProductMatrix {
  public:
    static void
    compute(const float* a, size_t m, const float* b, size_t n, size_t d, float* out) {
        if (m == 0 || n == 0) {
            return;
        }
        switch (select_isa()) {
            case AVX_AVAILABILITY::AVX512:
                InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512>::build(a, m, b, n, d, out);
                return;
            case AVX_AVAILABILITY::AVX2:
                InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2>::build(a, m, b, n, d, out);
                return;
            case AVX_AVAILABILITY::NONE:
                InnerProductMatrixImpl<AVX_AVAILABILITY::NONE>::build(a, m, b, n, d, out);
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
