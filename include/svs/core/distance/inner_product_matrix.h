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
#include "svs/lib/preprocessor.h"

#include <cstddef>

namespace svs::distance {

// One accumulator per (A-row, B-lane-block), no horizontal reduction: measured on this
// machine, 8 independent accumulators is where row-blocking stops buying throughput,
// matching ~4-cycle FMA latency x 2 ports (see the PR description for the sweep).
inline constexpr size_t inner_product_matrix_row_block = 8;

/// Per-ISA kernel for the dense inner-product table of a large row-major A (m x d) against
/// a small row-major B (n x d), both float32: out[i * n + j] = <A_i, B_j>, out row-major by
/// A's rows. This primary template is the scalar, compiler-autovectorized generic
/// implementation: both the fallback when no SIMD ISA is available and the reference the
/// ISA-specific specializations below are checked against in tests. Those specializations
/// are declared here and defined in avx2.cpp / avx512.cpp so their intrinsics only compile
/// under their own -march flags, mirroring QChamferExactImpl above.
template <AVX_AVAILABILITY Avx> struct InnerProductMatrixImpl {
    SVS_NOINLINE static void
    build(const float* a, size_t m, const float* b, size_t n, size_t d, float* out) {
        for (size_t i = 0; i < m; ++i) {
            const float* ai = a + i * d;
            float* row = out + i * n;
            for (size_t j = 0; j < n; ++j) {
                const float* bj = b + j * d;
                float dot = 0.0F;
                for (size_t k = 0; k < d; ++k) {
                    dot += ai[k] * bj[k];
                }
                row[j] = dot;
            }
        }
    }
};

#if defined(__x86_64__)

SVS_VALIDATE_BOOL_ENV(SVS_AVX512_F)
#if SVS_AVX512_F
template <> struct InnerProductMatrixImpl<AVX_AVAILABILITY::AVX512> {
    SVS_NOINLINE static void
    build(const float* a, size_t m, const float* b, size_t n, size_t d, float* out);
};
#endif

SVS_VALIDATE_BOOL_ENV(SVS_AVX512_F)
SVS_VALIDATE_BOOL_ENV(SVS_AVX2)
#if !SVS_AVX512_F && SVS_AVX2
template <> struct InnerProductMatrixImpl<AVX_AVAILABILITY::AVX2> {
    SVS_NOINLINE static void
    build(const float* a, size_t m, const float* b, size_t n, size_t d, float* out);
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
///     query's tokens). Packed once per call into ISA-specific scratch the implementation
///     owns (thread_local, as the GEM query transpose above does), so `b` itself is read
///     once and not retained.
/// @param n Number of rows of `b`. 0 is valid and leaves `out` untouched.
/// @param d Row width shared by `a` and `b`.
/// @param out Row-major output buffer, caller-owned, at least m * n floats: out[i * n + j]
///     holds <a_i, b_j>.
class InnerProductMatrix {
  public:
    static void
    compute(const float* a, size_t m, const float* b, size_t n, size_t d, float* out) {
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

  private:
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
