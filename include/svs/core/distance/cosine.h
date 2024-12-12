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
#include "svs/lib/saveload.h"
#include "svs/lib/static.h"

// stl
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>
#include <x86intrin.h>

namespace svs::distance {
// Forward declare implementation to allow entry point to be near the top.
template <size_t N, typename Ea, typename Eb> struct CosineSimilarityImpl;

// Generic Entry Point
// Call as one of either:
// ```
// (1) CosineSimilarity::compute(a, b, length)
// (2) CosineSimilarity::compute<length>(a, b)
// ```
// Where (2) is when length is known at compile time and (1) is when length is not.
class CosineSimilarity {
  public:
    template <typename Ea, typename Eb>
    static constexpr float compute(const Ea* a, const Eb* b, float a_norm, size_t N) {
        return CosineSimilarityImpl<Dynamic, Ea, Eb>::compute(
            a, b, a_norm, lib::MaybeStatic(N)
        );
    }

    template <size_t N, typename Ea, typename Eb>
    static constexpr float compute(const Ea* a, const Eb* b, float a_norm) {
        return CosineSimilarityImpl<N, Ea, Eb>::compute(
            a, b, a_norm, lib::MaybeStatic<N>()
        );
    }
};

///
/// @brief Functor for computing Cosine Similarity.
///
/// This is the primary functor for implementing the Cosine similarity between
/// two vectors in R^n. This functor uses the externally defined
/// \ref compute_distancecosine "compute" method and is thus capable of being extended
/// externally.
///
struct DistanceCosineSimilarity {
  public:
    /// Vectors are more similar if their similarity is greater.
    using compare = std::greater<>;

    /// Fix-argument is required.
    static constexpr bool must_fix_argument = true;

    ///
    /// This functor uses ``fix_argument`` to compute the norm of the left-hand argument.
    /// As such, it is stateful and not implicitly broadcastable.
    ///
    static const bool implicit_broadcast = false;

    /// @brief Compute and store the norm of ``x``
    template <typename T, size_t Extent> void fix_argument(const std::span<T, Extent>& x) {
        norm_ = norm(x);
    }

    ///// Members
    // Norm of the current query.
    float norm_;

    // IO
    static constexpr std::string_view name = "cosine_similarity";

    lib::SaveTable save() const { return DistanceSerialization::save(name); }

    static bool
    check_load_compatibility(std::string_view schema, svs::lib::Version version) {
        return DistanceSerialization::check_load_compatibility(schema, version);
    }

    DistanceCosineSimilarity static load(const lib::ContextFreeLoadTable& table) {
        // Throws if check fails.
        DistanceSerialization::check_load(table, name);
        return DistanceCosineSimilarity();
    }
};

inline constexpr bool operator==(DistanceCosineSimilarity, DistanceCosineSimilarity) {
    return true;
}

///
/// @ingroup distance_overload
/// @anchor compute_distancecosine
/// @brief Compute the Cosine simmilarity between two vectors in R^n.
///
/// @tparam Ea The element type for each component of the left-hand argument.
/// @tparam Eb The element type for each component of the right-hand argument.
/// @tparam Da The compile-time length of left-hand argument. May be ``svs::Dynamic`` if
///     this is to be discovered during runtime.
/// @tparam Db The compile-time length of right-hand argument. May be ``svs::Dynamic`` if
///     this is to be discovered during runtime.
///
/// @param distance The cosine similarity distance functor. Must have had ``fix_argument``
///     called previously with left-hand argument ``a``.
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
float compute(DistanceCosineSimilarity distance, std::span<Ea, Da> a, std::span<Eb, Db> b) {
    assert(a.size() == b.size());
    constexpr size_t extent = lib::extract_extent(Da, Db);
    if constexpr (extent == Dynamic) {
        return CosineSimilarity::compute(a.data(), b.data(), distance.norm_, a.size());
    } else {
        return CosineSimilarity::compute<extent>(a.data(), b.data(), distance.norm_);
    }
}

/////
///// Generic Implementation
/////

template <size_t N, typename Ea, typename Eb>
float generic_cosine_similarity(
    const Ea* a,
    const Eb* b,
    float a_norm,
    lib::MaybeStatic<N> length = lib::MaybeStatic<N>()
) {
    float result = 0;
    float accum = 0;
    for (size_t i = 0; i < length.size(); ++i) {
        float bi_float = static_cast<float>(b[i]);
        accum += bi_float * bi_float;
        result += static_cast<float>(a[i]) * bi_float;
    }
    return result / (a_norm * std::sqrt(accum));
};

template <size_t N, typename Ea, typename Eb> struct CosineSimilarityImpl {
    static float compute(
        const Ea* a,
        const Eb* b,
        float a_norm,
        lib::MaybeStatic<N> length = lib::MaybeStatic<N>()
    ) {
        return generic_cosine_similarity(a, b, a_norm, length);
    }
};

/////
///// Intel(R) AVX-512 Implementations
/////

// Shared implementation among those that use floating-point arithmetic.
template <size_t SIMDWidth> struct CosineFloatOp;

SVS_VALIDATE_BOOL_ENV(SVS_AVX512_F)
#if SVS_AVX512_F

template <> struct CosineFloatOp<16> : public svs::simd::ConvertToFloat<16> {
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

// Small Integers
SVS_VALIDATE_BOOL_ENV(SVS_AVX512_VNNI)
#if SVS_AVX512_VNNI
template <size_t N> struct CosineSimilarityImpl<N, int8_t, int8_t> {
    SVS_NOINLINE static float
    compute(const int8_t* a, const int8_t* b, float a_norm, lib::MaybeStatic<N> length) {
        auto sum = _mm512_setzero_epi32();
        auto bnorm_accum = _mm512_setzero_epi32();
        auto mask = create_mask<32>(length);
        auto all = no_mask<32>();

        for (size_t j = 0; j < length.size(); j += 32) {
            auto temp_a =
                _mm256_maskz_loadu_epi8(islast<32>(length, j) ? mask : all, a + j);
            auto va = _mm512_cvtepi8_epi16(temp_a);

            auto temp_b =
                _mm256_maskz_loadu_epi8(islast<32>(length, j) ? mask : all, b + j);
            auto vb = _mm512_cvtepi8_epi16(temp_b);

            bnorm_accum = _mm512_dpwssd_epi32(bnorm_accum, vb, vb);
            sum = _mm512_dpwssd_epi32(sum, va, vb);
        }

        float b_norm = std::sqrt(static_cast<float>(_mm512_reduce_add_epi32(bnorm_accum)));
        return lib::narrow_cast<float>(_mm512_reduce_add_epi32(sum)) / (a_norm * b_norm);
    }
};

template <size_t N> struct CosineSimilarityImpl<N, uint8_t, uint8_t> {
    SVS_NOINLINE static float
    compute(const uint8_t* a, const uint8_t* b, float a_norm, lib::MaybeStatic<N> length) {
        auto sum = _mm512_setzero_epi32();
        auto bnorm_accum = _mm512_setzero_epi32();
        auto mask = create_mask<32>(length);
        auto all = no_mask<32>();

        for (size_t j = 0; j < length.size(); j += 32) {
            auto temp_a =
                _mm256_maskz_loadu_epi8(islast<32>(length, j) ? mask : all, a + j);
            auto va = _mm512_cvtepu8_epi16(temp_a);

            auto temp_b =
                _mm256_maskz_loadu_epi8(islast<32>(length, j) ? mask : all, b + j);
            auto vb = _mm512_cvtepu8_epi16(temp_b);

            bnorm_accum = _mm512_dpwssd_epi32(bnorm_accum, vb, vb);
            sum = _mm512_dpwssd_epi32(sum, va, vb);
        }
        float b_norm = std::sqrt(static_cast<float>(_mm512_reduce_add_epi32(bnorm_accum)));
        return lib::narrow_cast<float>(_mm512_reduce_add_epi32(sum)) / (a_norm * b_norm);
    }
};

#endif

// Floating and Mixed Types
template <size_t N> struct CosineSimilarityImpl<N, float, float> {
    SVS_NOINLINE static float
    compute(const float* a, const float* b, float a_norm, lib::MaybeStatic<N> length) {
        auto [sum, norm] = simd::generic_simd_op(CosineFloatOp<16>(), a, b, length);
        return sum / (std::sqrt(norm) * a_norm);
    }
};

template <size_t N> struct CosineSimilarityImpl<N, float, uint8_t> {
    SVS_NOINLINE static float
    compute(const float* a, const uint8_t* b, float a_norm, lib::MaybeStatic<N> length) {
        auto [sum, norm] = simd::generic_simd_op(CosineFloatOp<16>(), a, b, length);
        return sum / (std::sqrt(norm) * a_norm);
    };
};

template <size_t N> struct CosineSimilarityImpl<N, float, int8_t> {
    SVS_NOINLINE static float
    compute(const float* a, const int8_t* b, float a_norm, lib::MaybeStatic<N> length) {
        auto [sum, norm] = simd::generic_simd_op(CosineFloatOp<16>(), a, b, length);
        return sum / (std::sqrt(norm) * a_norm);
    };
};

template <size_t N> struct CosineSimilarityImpl<N, float, Float16> {
    SVS_NOINLINE static float
    compute(const float* a, const Float16* b, float a_norm, lib::MaybeStatic<N> length) {
        auto [sum, norm] = simd::generic_simd_op(CosineFloatOp<16>(), a, b, length);
        return sum / (std::sqrt(norm) * a_norm);
    }
};
#endif
} // namespace svs::distance
