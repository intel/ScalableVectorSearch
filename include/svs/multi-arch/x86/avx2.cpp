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
#include "svs/core/distance/inner_product.h"

// Define SIMD ops here with AVX2 optimizations
// Compiled with -march=haswell to generate optimized AVX2 instructions
namespace svs::distance {

///// Inner Product SIMD Ops /////

template <> struct IPFloatOp<8, AVX_AVAILABILITY::AVX2> : public svs::simd::ConvertToFloat<8> {
    using parent = svs::simd::ConvertToFloat<8>;
    using mask_t = typename parent::mask_t;
    static constexpr size_t simd_width = 8;

    static __m256 init() { return _mm256_setzero_ps(); }

    static __m256 accumulate(__m256 accumulator, __m256 a, __m256 b) {
        return _mm256_fmadd_ps(a, b, accumulator);
    }

    static __m256 accumulate(mask_t /*m*/, __m256 accumulator, __m256 a, __m256 b) {
        // For AVX2, masking is handled in the load operations
        return _mm256_fmadd_ps(a, b, accumulator);
    }

    static __m256 combine(__m256 x, __m256 y) { return _mm256_add_ps(x, y); }
    static float reduce(__m256 x) { return simd::_mm256_reduce_add_ps(x); }
};

///// L2 SIMD Ops /////

template <> struct L2FloatOp<8, AVX_AVAILABILITY::AVX2> : public svs::simd::ConvertToFloat<8> {
    using parent = svs::simd::ConvertToFloat<8>;
    using mask_t = typename parent::mask_t;
    static constexpr size_t simd_width = 8;

    static __m256 init() { return _mm256_setzero_ps(); }

    static __m256 accumulate(__m256 accumulator, __m256 a, __m256 b) {
        auto c = _mm256_sub_ps(a, b);
        return _mm256_fmadd_ps(c, c, accumulator);
    }

    static __m256 accumulate(mask_t /*m*/, __m256 accumulator, __m256 a, __m256 b) {
        // For AVX2, masking is handled in the load operations
        auto c = _mm256_sub_ps(a, b);
        return _mm256_fmadd_ps(c, c, accumulator);
    }

    static __m256 combine(__m256 x, __m256 y) { return _mm256_add_ps(x, y); }
    static float reduce(__m256 x) { return simd::_mm256_reduce_add_ps(x); }
};

///// Cosine Similarity SIMD Ops /////

template <> struct CosineFloatOp<8, AVX_AVAILABILITY::AVX2> : public svs::simd::ConvertToFloat<8> {
    using parent = svs::simd::ConvertToFloat<8>;
    using mask_t = typename parent::mask_t;
    static constexpr size_t simd_width = 8;

    // A lightweight struct to contain both the partial results for the inner product
    // of the left-hand and right-hand as well as partial results for computing the norm
    // of the right-hand.
    struct Pair {
        __m256 op;
        __m256 norm;
    };

    static Pair init() { return {_mm256_setzero_ps(), _mm256_setzero_ps()}; };

    static Pair accumulate(Pair accumulator, __m256 a, __m256 b) {
        return {
            _mm256_fmadd_ps(a, b, accumulator.op), _mm256_fmadd_ps(b, b, accumulator.norm)};
    }

    static Pair accumulate(mask_t /*m*/, Pair accumulator, __m256 a, __m256 b) {
        // For AVX2, masking is handled in the load operations
        return {
            _mm256_fmadd_ps(a, b, accumulator.op), _mm256_fmadd_ps(b, b, accumulator.norm)};
    }

    static Pair combine(Pair x, Pair y) {
        return {_mm256_add_ps(x.op, y.op), _mm256_add_ps(x.norm, y.norm)};
    }

    static std::pair<float, float> reduce(Pair x) {
        return std::make_pair(
            simd::_mm256_reduce_add_ps(x.op), simd::_mm256_reduce_add_ps(x.norm)
        );
    }
};

} // namespace svs::distance

#endif
