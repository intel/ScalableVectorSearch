/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
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
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);

    lib::SaveTable save() const {
        return lib::SaveTable(save_version, {SVS_LIST_SAVE(name)});
    }

    DistanceCosineSimilarity static load(
        const toml::table& table, const lib::Version& version
    ) {
        // Version check
        if (version != save_version) {
            throw ANNEXCEPTION("Unhandled version!");
        }

        auto retrieved = lib::load_at<std::string>(table, "name");
        if (retrieved != name) {
            throw ANNEXCEPTION(
                "Loading error. Expected name {}. Instead, got {}!", name, retrieved
            );
        }
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
/// - Compiling and executing on an AVX512 system will improve performance.
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
///// AVX512 Implementations
/////

#if defined(__AVX512F__)

// Small Integers
#if defined(__AVX512VNNI__)
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
#elif defined(__AVX512BW__) && defined(__KNCNI__)
template <size_t N> struct CosineSimilarityImpl<N, int8_t, int8_t> {
    SVS_NOINLINE static float
    compute(const int8_t* a, const int8_t* b, float a_norm, lib::MaybeStatic<N> length) {
        auto sum = _mm512_setzero_epi32();
        auto bnorm_accum = _mm512_setzero_epi32();
        auto mask = create_mask<16>(length);
        auto all = no_mask<16>();

        for (size_t j = 0; j < length.size(); j += 16) {
            auto temp_a1 = _mm_maskz_loadu_epi8(islast<16>(length, j) ? mask : all, a + j);
            auto va = _mm512_cvtepi8_epi32(temp_a1);

            auto temp_b1 = _mm_maskz_loadu_epi8(islast<16>(length, j) ? mask : all, b + j);
            auto vb = _mm512_cvtepi8_epi32(temp_b1);

            bnorm_accum = _mm512_fmadd_epi32(vb, vb, bnorm_accum);
            sum = _mm512_fmadd_epi32(va, vb, sum);
        }
        float b_norm = std::sqrt(static_cast<float>(_mm512_reduce_add_epi32(bnorm_accum)));
        return _mm512_reduce_add_epi32(sum) / (a_norm * b_norm);
    }
};

template <size_t N> struct CosineSimilarityImpl<N, uint8_t, uint8_t> {
    SVS_NOINLINE static float
    compute(const uint8_t* a, const uint8_t* b, float a_norm, lib::MaybeStatic<N> length) {
        auto sum = _mm512_setzero_epi32();
        auto bnorm_accum = _mm512_setzero_epi32();
        auto mask = create_mask<16>(length);
        auto all = no_mask<16>();

        for (size_t j = 0; j < length.size(); j += 16) {
            auto temp_a1 = _mm_maskz_loadu_epi8(islast<16>(length, j) ? mask : all, a + j);
            auto va = _mm512_cvtepu8_epi32(temp_a1);

            auto temp_b1 = _mm_maskz_loadu_epi8(islast<16>(length, j) ? mask : all, b + j);
            auto vb = _mm512_cvtepu8_epi32(temp_b1);
            bnorm_accum = _mm512_fmadd_epi32(vb, vb, bnorm_accum);
            sum = _mm512_fmadd_epi32(va, vb, sum);
        }
        float b_norm = std::sqrt(static_cast<float>(_mm512_reduce_add_epi32(bnorm_accum)));
        return _mm512_reduce_add_epi32(sum) / (a_norm * b_norm);
    }
};
#endif

// Floating and Mixed Types
template <size_t N> struct CosineSimilarityImpl<N, float, float> {
    SVS_NOINLINE static float
    compute(const float* a, const float* b, float a_norm, lib::MaybeStatic<N> length) {
        auto sum = _mm512_setzero_ps();
        auto bnorm_accum = _mm512_setzero_ps();
        auto mask = create_mask<16>(length);
        auto all = no_mask<16>();
        for (size_t j = 0; j < length.size(); j += 16) {
            auto va = _mm512_maskz_loadu_ps(islast<16>(length, j) ? mask : all, a + j);
            auto vb = _mm512_maskz_loadu_ps(islast<16>(length, j) ? mask : all, b + j);
            bnorm_accum = _mm512_fmadd_ps(vb, vb, bnorm_accum);
            sum = _mm512_fmadd_ps(va, vb, sum);
        }
        float b_norm = std::sqrt(_mm512_reduce_add_ps(bnorm_accum));
        return _mm512_reduce_add_ps(sum) / (a_norm * b_norm);
    }
};

template <size_t N> struct CosineSimilarityImpl<N, float, uint8_t> {
    SVS_NOINLINE static float
    compute(const float* a, const uint8_t* b, float a_norm, lib::MaybeStatic<N> length) {
        auto sum = _mm512_setzero_ps();
        auto bnorm_accum = _mm512_setzero_ps();
        auto mask = create_mask<16>(length);
        auto all = no_mask<16>();
        for (size_t j = 0; j < length.size(); j += 16) {
            // Load and convert the integers to floating point.
            auto va = _mm512_maskz_loadu_ps(islast<16>(length, j) ? mask : all, a + j);
            auto tb = _mm_maskz_loadu_epi8(islast<16>(length, j) ? mask : all, b + j);
            auto vb = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(tb));
            bnorm_accum = _mm512_fmadd_ps(vb, vb, bnorm_accum);
            sum = _mm512_fmadd_ps(va, vb, sum);
        }
        float b_norm = std::sqrt(_mm512_reduce_add_ps(bnorm_accum));
        return _mm512_reduce_add_ps(sum) / (a_norm * b_norm);
    };
};

template <size_t N> struct CosineSimilarityImpl<N, float, int8_t> {
    SVS_NOINLINE static float
    compute(const float* a, const int8_t* b, float a_norm, lib::MaybeStatic<N> length) {
        auto sum = _mm512_setzero_ps();
        auto bnorm_accum = _mm512_setzero_ps();
        auto mask = create_mask<16>(length);
        auto all = no_mask<16>();
        for (size_t j = 0; j < length.size(); j += 16) {
            // Load and convert the integers to floating point.
            auto va = _mm512_maskz_loadu_ps(islast<16>(length, j) ? mask : all, a + j);
            auto tb = _mm_maskz_loadu_epi8(islast<16>(length, j) ? mask : all, b + j);
            auto vb = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(tb));
            bnorm_accum = _mm512_fmadd_ps(vb, vb, bnorm_accum);
            sum = _mm512_fmadd_ps(va, vb, sum);
        }
        float b_norm = std::sqrt(_mm512_reduce_add_ps(bnorm_accum));
        return _mm512_reduce_add_ps(sum) / (a_norm * b_norm);
    };
};

template <size_t N> struct CosineSimilarityImpl<N, float, Float16> {
    SVS_NOINLINE static float
    compute(const float* a, const Float16* b, float a_norm, lib::MaybeStatic<N> length) {
        auto sum = _mm512_setzero_ps();
        auto bnorm_accum = _mm512_setzero_ps();
        auto mask = create_mask<16>(length);
        auto all = no_mask<16>();

        for (size_t j = 0; j < length.size(); j += 16, a += 16, b += 16) {
            auto va = _mm512_maskz_loadu_ps(islast<16>(length, j) ? mask : all, a);
            auto vb_f16 = _mm256_maskz_loadu_epi16(islast<16>(length, j) ? mask : all, b);
            auto vb = _mm512_cvtph_ps(vb_f16);
            bnorm_accum = _mm512_fmadd_ps(vb, vb, bnorm_accum);
            sum = _mm512_fmadd_ps(va, vb, sum);
        }

        float b_norm = std::sqrt(_mm512_reduce_add_ps(bnorm_accum));
        return _mm512_reduce_add_ps(sum) / (a_norm * b_norm);
    }
};
#endif
} // namespace svs::distance
