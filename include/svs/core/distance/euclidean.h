/*
 * Copyright 2023 Intel Corporation
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

// svs
#include "svs/core/distance/distance_core.h"
#include "svs/core/distance/simd_utils.h"
#include "svs/lib/float16.h"
#include "svs/lib/preprocessor.h"
#include "svs/lib/saveload.h"
#include "svs/lib/static.h"

// stl
#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <type_traits>
#include <x86intrin.h>

// Implementation Notes regarding Intel(R) AVX Extentions
// Top most entry in the bulleted list underneath each type pair <T,U> is the preferred
// implementation based on the available extension.
//
// The vector width used is added as well in case there is an apparent mismatch between
// Intel(R) AVX extension and the preferred vector width due to performance (sometimes, smaller
// vector widths are faster).
//
// Versions for older extensions are implemented as fallbacks.
//
// TODO: Implement testing for non-Intel(R) AVX-512 implementations.
// TODO: Alphabetize implementations.
// TODO: Refactor distance computation implementation to avoid the need to explicitly
// implement kernels for all type combinations.
//
// <float,float>
// - AVX512F, Width 16
// - Intel(R) AVX2, Width 8.
//
// <float,Float16>
// - Intel(R) AVX2, Vector Width 8 (smaller vector width is faster.
// - [[TODO]]: What is the correct check for the `_mm256_cvtph_ps` intrinsic?
//
// <int8_t,float>
// - AVX512F, Vector Width 16 (promotes to float32).
//
// <uint8_t,float>
// - AVX512F, Vector Width 16 (promotes to float32).
//
// <int8_t,int8_t>
// - AVX512VNNI, Width 32
// - AVX512F, Width 16 (promotes to float32).
//
// <uint8_t,uint8_t>
// - AVX512VNNI, Width 32
// - AVX512F, Width 16 (promotes to float32).

namespace svs::distance {
// Forward declare implementation to allow entry point to be near the top.
template <size_t N, typename Ea, typename Eb> struct L2Impl;

// Generic Entry Point
// Call as one of either:
// ```
// (1) L2::compute(a, b, length)
// (2) L2::compute<length>(a, b)
// ```
// Where (2) is when length is known at compile time and (1) is when length is not.
class L2 {
  public:
    template <typename Ea, typename Eb>
    static constexpr float compute(const Ea* a, const Eb* b, size_t N) {
        return L2Impl<Dynamic, Ea, Eb>::compute(a, b, lib::MaybeStatic(N));
    }

    template <size_t N, typename Ea, typename Eb>
    static constexpr float compute(const Ea* a, const Eb* b) {
        return L2Impl<N, Ea, Eb>::compute(a, b, lib::MaybeStatic<N>());
    }
};

///
/// @brief Functor for computing the square Euclidean distance.
///
/// This is the primary functor for implementing the square Euclidean distance between
/// two vectors in R^n. This functor uses the externally defined
/// \ref compute_distancel2 "compute" method and is thus capable of being extended
/// externally.
///
struct DistanceL2 {
    /// Vectors are more similar if their distance is smaller.
    using compare = std::less<>;

    ///
    /// This functor does not use any local scratch space to assist in computation and
    /// thus may be shared across threads and queries safely.
    ///
    static constexpr bool implicit_broadcast = true;

    // IO
    static constexpr std::string_view name = "squared_l2";
    static bool
    check_load_compatibility(std::string_view schema, svs::lib::Version version) {
        return DistanceSerialization::check_load_compatibility(schema, version);
    }

    lib::SaveTable save() const { return DistanceSerialization::save(name); }
    DistanceL2 static load(const lib::ContextFreeLoadTable& table) {
        // Throws if check fails.
        DistanceSerialization::check_load(table, name);
        return DistanceL2{};
    }
};

inline constexpr bool operator==(DistanceL2, DistanceL2) { return true; }

///
/// @ingroup distance_overload
/// @anchor compute_distancel2
/// @brief Compute the squared Euclidean distance between two vectors in R^n.
///
/// @tparam Ea The element type for each component of the left-hand argument.
/// @tparam Eb The element type for each component of the right-hand argument.
/// @tparam Da The compile-time length of left-hand argument. May be ``svs::Dynamic`` if
///     this is to be discovered during runtime.
/// @tparam Db The compile-time length of right-hand argument. May be ``svs::Dynamic`` if
///     this is to be discovered during runtime.
///
/// @param a The left-hand vector. Typically, this position is used for the query.
/// @param b The right-hand vector. Typically, this position is used for a dataset vector.
///
/// The base pointers for ``a`` and ``b`` need not be aligned. Mixed types for ``Ea`` and
/// ``Eb`` are supported.
///
/// *Performance Tips*
/// - Specifying the size parameters ``Da`` and ``Db`` can greatly improve performance.
/// - Compiling and executing on an Intel(R) AVX-512 system will improve performance.
///
template <Arithmetic Ea, Arithmetic Eb, size_t Da, size_t Db>
float compute(DistanceL2 /*unused*/, std::span<Ea, Da> a, std::span<Eb, Db> b) {
    assert(a.size() == b.size());
    constexpr size_t extent = lib::extract_extent(Da, Db);
    if constexpr (extent == Dynamic) {
        return L2::compute(a.data(), b.data(), a.size());
    } else {
        return L2::compute<extent>(a.data(), b.data());
    }
}

/////
///// Generic Implementation
/////

template <size_t N, typename Ea, typename Eb>
float generic_l2(
    const Ea* a, const Eb* b, lib::MaybeStatic<N> length = lib::MaybeStatic<N>()
) {
    float result = 0;
    for (size_t i = 0; i < length.size(); ++i) {
        auto temp = static_cast<float>(a[i]) - static_cast<float>(b[i]);
        result += temp * temp;
    }
    return result;
}

template <size_t N, typename Ea, typename Eb> struct L2Impl {
    static constexpr float
    compute(const Ea* a, const Eb* b, lib::MaybeStatic<N> length = lib::MaybeStatic<N>()) {
        return generic_l2(a, b, length);
    }
};

/////
///// Intel(R) AVX-512 Implementations
/////

// SIMD accelerated operations that convert both left and right hand arguments to
// ``float`` and perform arithmetic on those floating point operands.
template <size_t SIMDWidth> struct L2FloatOp;

// SIMD accelerated operations that convert both left and right hand arguments to
// ``To`` and perform arithmetic on those integer operands.
template <std::integral To, size_t SIMDWidth> struct L2VNNIOp;

SVS_VALIDATE_BOOL_ENV(SVS_AVX512_F)
#if SVS_AVX512_F

template <> struct L2FloatOp<16> : public svs::simd::ConvertToFloat<16> {
    using parent = svs::simd::ConvertToFloat<16>;
    using mask_t = typename parent::mask_t;

    // Here, we can fill-in the shared init, accumulate, combine, and reduce methods.
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

// Small Integers
SVS_VALIDATE_BOOL_ENV(SVS_AVX512_VNNI)
#if SVS_AVX512_VNNI

template <> struct L2VNNIOp<int16_t, 32> : public svs::simd::ConvertForVNNI<int16_t, 32> {
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

// VNNI Dispatching
template <size_t N> struct L2Impl<N, int8_t, int8_t> {
    SVS_NOINLINE static float
    compute(const int8_t* a, const int8_t* b, lib::MaybeStatic<N> length) {
        return simd::generic_simd_op(L2VNNIOp<int16_t, 32>(), a, b, length);
    }
};

template <size_t N> struct L2Impl<N, uint8_t, uint8_t> {
    SVS_NOINLINE static float
    compute(const uint8_t* a, const uint8_t* b, lib::MaybeStatic<N> length) {
        return simd::generic_simd_op(L2VNNIOp<int16_t, 32>(), a, b, length);
    }
};

#endif

// Floating and Mixed Types
template <size_t N> struct L2Impl<N, float, float> {
    SVS_NOINLINE static float
    compute(const float* a, const float* b, lib::MaybeStatic<N> length) {
        return simd::generic_simd_op(L2FloatOp<16>{}, a, b, length);
    }
};

template <size_t N> struct L2Impl<N, float, uint8_t> {
    SVS_NOINLINE static float
    compute(const float* a, const uint8_t* b, lib::MaybeStatic<N> length) {
        return simd::generic_simd_op(L2FloatOp<16>{}, a, b, length);
    };
};

template <size_t N> struct L2Impl<N, float, int8_t> {
    SVS_NOINLINE static float
    compute(const float* a, const int8_t* b, lib::MaybeStatic<N> length) {
        return simd::generic_simd_op(L2FloatOp<16>{}, a, b, length);
    };
};

template <size_t N> struct L2Impl<N, float, Float16> {
    SVS_NOINLINE static float
    compute(const float* a, const Float16* b, lib::MaybeStatic<N> length) {
        return simd::generic_simd_op(L2FloatOp<16>{}, a, b, length);
    }
};

template <size_t N> struct L2Impl<N, Float16, float> {
    SVS_NOINLINE static float
    compute(const Float16* a, const float* b, lib::MaybeStatic<N> length) {
        return simd::generic_simd_op(L2FloatOp<16>{}, a, b, length);
    }
};

template <size_t N> struct L2Impl<N, Float16, Float16> {
    SVS_NOINLINE static float
    compute(const Float16* a, const Float16* b, lib::MaybeStatic<N> length) {
        return simd::generic_simd_op(L2FloatOp<16>{}, a, b, length);
    };
};

#endif

/////
///// Intel(R) AVX2 Implementations
/////

SVS_VALIDATE_BOOL_ENV(SVS_AVX512_F)
SVS_VALIDATE_BOOL_ENV(SVS_AVX2)
#if !SVS_AVX512_F && SVS_AVX2
template <size_t N> struct L2Impl<N, float, float> {
    SVS_NOINLINE static float
    compute(const float* a, const float* b, lib::MaybeStatic<N> length) {
        constexpr size_t vector_size = 8;

        // Peel off the last iterations if the SIMD vector width does not evenly the total
        // vector width.
        size_t upper = lib::upper<vector_size>(length);
        auto rest = lib::rest<vector_size>(length);
        auto sum = _mm256_setzero_ps();
        for (size_t j = 0; j < upper; j += vector_size) {
            auto va = _mm256_loadu_ps(a + j);
            auto vb = _mm256_loadu_ps(b + j);
            auto tmp = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(tmp, tmp, sum);
        }
        return simd::_mm256_reduce_add_ps(sum) + generic_l2(a + upper, b + upper, rest);
    }
};

template <size_t N> struct L2Impl<N, Float16, Float16> {
    SVS_NOINLINE static float
    compute(const Float16* a, const Float16* b, lib::MaybeStatic<N> length) {
        constexpr size_t vector_size = 8;

        // Peel off the last iterations if the SIMD vector width does not evenly the total
        // vector width.
        size_t upper = lib::upper<vector_size>(length);
        auto rest = lib::rest<vector_size>(length);
        auto sum = _mm256_setzero_ps();
        for (size_t j = 0; j < upper; j += vector_size) {
            auto va =
                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j)));
            auto vb =
                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + j)));
            auto tmp = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(tmp, tmp, sum);
        }
        return simd::_mm256_reduce_add_ps(sum) + generic_l2(a + upper, b + upper, rest);
    }
};

template <size_t N> struct L2Impl<N, float, Float16> {
    SVS_NOINLINE static float
    compute(const float* a, const Float16* b, lib::MaybeStatic<N> length) {
        constexpr size_t vector_size = 8;

        // Peel off the last iterations if the SIMD vector width does not evenly the total
        // vector width.
        size_t upper = lib::upper<vector_size>(length);
        auto rest = lib::rest<vector_size>(length);
        auto sum = _mm256_setzero_ps();
        for (size_t j = 0; j < upper; j += vector_size) {
            auto va = _mm256_loadu_ps(a + j);
            auto vb =
                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + j)));
            auto tmp = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(tmp, tmp, sum);
        }
        return simd::_mm256_reduce_add_ps(sum) + generic_l2(a + upper, b + upper, rest);
    }
};

template <size_t N> struct L2Impl<N, float, int8_t> {
    SVS_NOINLINE static float
    compute(const float* a, const int8_t* b, lib::MaybeStatic<N> length) {
        constexpr size_t vector_size = 8;

        // Peel off the last iterations if the SIMD vector width does not evenly the total
        // vector width.
        size_t upper = lib::upper<vector_size>(length);
        auto rest = lib::rest<vector_size>(length);
        auto sum = _mm256_setzero_ps();
        for (size_t j = 0; j < upper; j += vector_size) {
            auto va = _mm256_castsi256_ps(
                _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(a + j))
            );
            auto vb = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_cvtsi64_si128(*(reinterpret_cast<const int64_t*>(b + j)))
            ));
            auto tmp = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(tmp, tmp, sum);
        }
        return simd::_mm256_reduce_add_ps(sum) + generic_l2(a + upper, b + upper, rest);
    }
};

template <size_t N> struct L2Impl<N, int8_t, int8_t> {
    SVS_NOINLINE static float
    compute(const int8_t* a, const int8_t* b, lib::MaybeStatic<N> length) {
        constexpr size_t vector_size = 8;

        size_t upper = lib::upper<vector_size>(length);
        auto rest = lib::rest<vector_size>(length);
        auto sum = _mm256_setzero_ps();
        for (size_t j = 0; j < upper; j += vector_size) {
            // * Strategy: Load 8 bytes as a 64-bit int.
            // * Use `_mm_cvtsi64_si128` to convert to a 128-bit vector.
            // * Use `mm256_evtepi8_epi32` to convert the 8-bytes to
            // 8 32-bit integers.
            // * Finally, convert to single precision floating point.
            auto va = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_cvtsi64_si128(*(reinterpret_cast<const int64_t*>(a + j)))
            ));
            auto vb = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_cvtsi64_si128(*(reinterpret_cast<const int64_t*>(b + j)))
            ));
            auto diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }
        return simd::_mm256_reduce_add_ps(sum) + generic_l2(a + upper, b + upper, rest);
    }
};

template <size_t N> struct L2Impl<N, uint8_t, uint8_t> {
    SVS_NOINLINE static float
    compute(const uint8_t* a, const uint8_t* b, lib::MaybeStatic<N> length) {
        constexpr size_t vector_size = 8;

        size_t upper = lib::upper<vector_size>(length);
        auto rest = lib::rest<vector_size>(length);
        auto sum = _mm256_setzero_ps();
        for (size_t j = 0; j < upper; j += vector_size) {
            // * Strategy: Load 8 bytes as a 64-bit int.
            // * Use `_mm_cvtsi64_si128` to convert to a 128-bit vector.
            // * Use `mm256_evtepi8_epi32` to convert the 8-bytes to
            // 8 32-bit integers.
            // * Finally, convert to single precision floating point.
            auto va = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                _mm_cvtsi64_si128(*(reinterpret_cast<const int64_t*>(a + j)))
            ));
            auto vb = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                _mm_cvtsi64_si128(*(reinterpret_cast<const int64_t*>(b + j)))
            ));
            auto diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }
        return simd::_mm256_reduce_add_ps(sum) + generic_l2(a + upper, b + upper, rest);
    }
};

#endif
} // namespace svs::distance
