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
namespace svs::distance {

///// Inner Product SIMD Ops /////

template <> struct IPFloatOp<16, AVX_AVAILABILITY::AVX512> : public svs::simd::ConvertToFloat<16> {
    using parent = svs::simd::ConvertToFloat<16>;
    using mask_t = typename parent::mask_t;

    static __m512 init() { return _mm512_setzero_ps(); }

    static __m512 accumulate(__m512 accumulator, __m512 a, __m512 b) {
        return _mm512_fmadd_ps(a, b, accumulator);
    }

    static __m512 accumulate(mask_t m, __m512 accumulator, __m512 a, __m512 b) {
        return _mm512_mask3_fmadd_ps(a, b, accumulator, m);
    }

    static __m512 combine(__m512 x, __m512 y) { return _mm512_add_ps(x, y); }
    static float reduce(__m512 x) { return _mm512_reduce_add_ps(x); }
};

template <> struct IPVNNIOp<int16_t, 32, AVX_AVAILABILITY::AVX512> : public svs::simd::ConvertForVNNI<int16_t, 32> {
    using parent = svs::simd::ConvertForVNNI<int16_t, 32>;
    using reg_t = typename parent::reg_t;
    using mask_t = typename parent::mask_t;

    SVS_FORCE_INLINE static reg_t init() { return _mm512_setzero_si512(); }
    SVS_FORCE_INLINE static reg_t accumulate(__m512i accumulator, __m512i a, __m512i b) {
        return _mm512_dpwssd_epi32(accumulator, a, b);
    }

    SVS_FORCE_INLINE static reg_t
    accumulate(mask_t m, reg_t accumulator, reg_t a, reg_t b) {
        return _mm512_mask_dpwssd_epi32(accumulator, m, a, b);
    }

    SVS_FORCE_INLINE static reg_t combine(reg_t x, reg_t y) {
        return _mm512_add_epi32(x, y);
    }

    SVS_FORCE_INLINE static float reduce(reg_t x) {
        return lib::narrow_cast<float>(_mm512_reduce_add_epi32(x));
    }
};

///// L2 SIMD Ops /////

template <> struct L2FloatOp<16, AVX_AVAILABILITY::AVX512> : public svs::simd::ConvertToFloat<16> {
    using parent = svs::simd::ConvertToFloat<16>;
    using mask_t = typename parent::mask_t;

    static __m512 init() { return _mm512_setzero_ps(); }

    static __m512 accumulate(__m512 accumulator, __m512 a, __m512 b) {
        auto c = _mm512_sub_ps(a, b);
        return _mm512_fmadd_ps(c, c, accumulator);
    }

    static __m512 accumulate(mask_t m, __m512 accumulator, __m512 a, __m512 b) {
        auto c = _mm512_maskz_sub_ps(m, a, b);
        return _mm512_mask3_fmadd_ps(c, c, accumulator, m);
    }

    static __m512 combine(__m512 x, __m512 y) { return _mm512_add_ps(x, y); }
    static float reduce(__m512 x) { return _mm512_reduce_add_ps(x); }
};

template <> struct L2VNNIOp<int16_t, 32, AVX_AVAILABILITY::AVX512> : public svs::simd::ConvertForVNNI<int16_t, 32> {
    using parent = svs::simd::ConvertForVNNI<int16_t, 32>;
    using reg_t = typename parent::reg_t;
    using mask_t = typename parent::mask_t;

    SVS_FORCE_INLINE static reg_t init() { return _mm512_setzero_si512(); }
    SVS_FORCE_INLINE static reg_t accumulate(reg_t accumulator, reg_t a, reg_t b) {
        auto c = _mm512_sub_epi16(a, b);
        return _mm512_dpwssd_epi32(accumulator, c, c);
    }

    SVS_FORCE_INLINE static reg_t
    accumulate(mask_t m, reg_t accumulator, reg_t a, reg_t b) {
        auto c = _mm512_maskz_sub_epi16(m, a, b);
        // `c` already contains zeros, so no need to mask the accumulation operation.
        return _mm512_mask_dpwssd_epi32(accumulator, m, c, c);
    }

    SVS_FORCE_INLINE static reg_t combine(reg_t x, reg_t y) {
        return _mm512_add_epi32(x, y);
    }

    SVS_FORCE_INLINE static float reduce(reg_t x) {
        return lib::narrow_cast<float>(_mm512_reduce_add_epi32(x));
    }
};

///// Cosine Similarity SIMD Ops /////

template <> struct CosineFloatOp<16, AVX_AVAILABILITY::AVX512> : public svs::simd::ConvertToFloat<16> {
    using parent = svs::simd::ConvertToFloat<16>;
    using mask_t = typename parent::mask_t;

    // A lightweight struct to contain both the partial results for the inner product
    // of the left-hand and right-hand as well as partial results for computing the norm
    // of the right-hand.
    struct Pair {
        __m512 op;
        __m512 norm;
    };

    static Pair init() { return {_mm512_setzero_ps(), _mm512_setzero_ps()}; };

    static Pair accumulate(Pair accumulator, __m512 a, __m512 b) {
        return {
            _mm512_fmadd_ps(a, b, accumulator.op), _mm512_fmadd_ps(b, b, accumulator.norm)};
    }

    static Pair accumulate(mask_t m, Pair accumulator, __m512 a, __m512 b) {
        return {
            _mm512_mask3_fmadd_ps(a, b, accumulator.op, m),
            _mm512_mask3_fmadd_ps(b, b, accumulator.norm, m)};
    }

    static Pair combine(Pair x, Pair y) {
        return {_mm512_add_ps(x.op, y.op), _mm512_add_ps(x.norm, y.norm)};
    }

    static std::pair<float, float> reduce(Pair x) {
        return std::make_pair(_mm512_reduce_add_ps(x.op), _mm512_reduce_add_ps(x.norm));
    }
};

} // namespace svs::distance

#include "svs/core/distance/cosine.h"
#include "svs/core/distance/euclidean.h"
#include "svs/core/distance/inner_product.h"



