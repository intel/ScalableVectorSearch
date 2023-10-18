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

// Implementation Notes regarding AVX Extentions
// Top most entry in the bulleted list underneath each type pair <T,U> is the preferred
// implementation based on the available extension.
//
// The vector width used is added as well in case there is an apparent mismatch between
// AVX extension and the preferred vector width due to performance (sometimes, smaller
// vector widths are faster).
//
// Versions for older extensions are implemented as fallbacks.
//
// TODO: Implement testing for non-AVX512 implementations.
// TODO: Alphabetize implementations.
// TODO: Refactor distance computation implementation to avoid the need to explicitly
// implement kernels for all type combinations.
//
// <float,float>
// - AVX512F, Width 16
// - AVX2, Width 8.
//
// <float,Float16>
// - AVX2, Vector Width 8 (smaller vector width is faster.
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
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);

    lib::SaveTable save() const {
        return lib::SaveTable(save_version, {SVS_LIST_SAVE(name)});
    }

    DistanceL2 static load(const toml::table& table, const lib::Version& version) {
        if (version != save_version) {
            throw ANNEXCEPTION("Unhandled version!");
        }

        auto retrieved = lib::load_at<std::string>(table, "name");
        if (retrieved != name) {
            throw ANNEXCEPTION(
                "Loading error. Expected name {}. Instead, got {}!", name, retrieved
            );
        }
        return DistanceL2();
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
/// - Compiling and executing on an AVX512 system will improve performance.
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
///// AVX512 Implementations
/////

#if defined(__AVX512F__)

// Small Integers
#if defined(__AVX512VNNI__)
template <size_t N> struct L2Impl<N, int8_t, int8_t> {
    SVS_NOINLINE static float
    compute(const int8_t* a, const int8_t* b, lib::MaybeStatic<N> length) {
        auto sum = _mm512_setzero_epi32();
        auto mask = create_mask<32>(length);
        auto all = no_mask<32>();

        for (size_t j = 0; j < length.size(); j += 32) {
            auto temp_a =
                _mm256_maskz_loadu_epi8(islast<32>(length, j) ? mask : all, a + j);
            auto va = _mm512_cvtepi8_epi16(temp_a);

            auto temp_b =
                _mm256_maskz_loadu_epi8(islast<32>(length, j) ? mask : all, b + j);
            auto vb = _mm512_cvtepi8_epi16(temp_b);

            auto diff = _mm512_sub_epi16(va, vb);
            sum = _mm512_dpwssd_epi32(sum, diff, diff);
        }
        return lib::narrow_cast<float>(_mm512_reduce_add_epi32(sum));
    }
};

template <size_t N> struct L2Impl<N, uint8_t, uint8_t> {
    SVS_NOINLINE static float
    compute(const uint8_t* a, const uint8_t* b, lib::MaybeStatic<N> length) {
        auto sum = _mm512_setzero_epi32();
        auto mask = create_mask<32>(length);
        auto all = no_mask<32>();

        for (size_t j = 0; j < length.size(); j += 32) {
            auto temp_a =
                _mm256_maskz_loadu_epi8(islast<32>(length, j) ? mask : all, a + j);
            auto va = _mm512_cvtepu8_epi16(temp_a);

            auto temp_b =
                _mm256_maskz_loadu_epi8(islast<32>(length, j) ? mask : all, b + j);
            auto vb = _mm512_cvtepu8_epi16(temp_b);

            auto diff = _mm512_sub_epi16(va, vb);
            sum = _mm512_dpwssd_epi32(sum, diff, diff);
        }
        return lib::narrow_cast<float>(_mm512_reduce_add_epi32(sum));
    }
};
#elif defined(__AVX512BW__) && defined(__KNCNI__)
template <size_t N> struct L2Impl<N, int8_t, int8_t> {
    SVS_NOINLINE static float
    compute(const int8_t* a, const int8_t* b, lib::MaybeStatic<N> length) {
        auto sum = _mm512_setzero_epi32();
        auto mask = create_mask<16>(length);
        auto all = no_mask<16>();

        for (size_t j = 0; j < length.size(); j += 16) {
            auto temp_a1 = _mm_maskz_loadu_epi8(islast<16>(length, j) ? mask : all, a + j);
            auto va = _mm512_cvtepi8_epi32(temp_a1);

            auto temp_b1 = _mm_maskz_loadu_epi8(islast<16>(length, j) ? mask : all, b + j);
            auto vb = _mm512_cvtepi8_epi32(temp_b1);

            auto diff = _mm512_sub_epi32(va, vb);
            sum = _mm512_fmadd_epi32(diff, diff, sum);
        }
        return _mm512_reduce_add_epi32(sum);
    }
};

template <size_t N> struct L2Impl<N, uint8_t, uint8_t> {
    SVS_NOINLINE static float
    compute(const uint8_t* a, const uint8_t* b, lib::MaybeStatic<N> length) {
        auto sum = _mm512_setzero_epi32();
        auto mask = create_mask<16>(length);
        auto all = no_mask<16>();

        for (size_t j = 0; j < length.size(); j += 16) {
            auto temp_a1 = _mm_maskz_loadu_epi8(islast<16>(length, j) ? mask : all, a + j);
            auto va = _mm512_cvtepu8_epi32(temp_a1);

            auto temp_b1 = _mm_maskz_loadu_epi8(islast<16>(length, j) ? mask : all, b + j);
            auto vb = _mm512_cvtepu8_epi32(temp_b1);

            auto diff = _mm512_sub_epi32(va, vb);
            sum = _mm512_fmadd_epi32(diff, diff, sum);
        }
        return _mm512_reduce_add_epi32(sum);
    }
};
#endif

// Floating and Mixed Types
template <size_t N> struct L2Impl<N, float, float> {
    SVS_NOINLINE static float
    compute(const float* a, const float* b, lib::MaybeStatic<N> length) {
        auto sum = _mm512_setzero_ps();
        auto mask = create_mask<16>(length);
        auto all = no_mask<16>();
        for (size_t j = 0; j < length.size(); j += 16) {
            auto va = _mm512_maskz_loadu_ps(islast<16>(length, j) ? mask : all, a + j);
            auto vb = _mm512_maskz_loadu_ps(islast<16>(length, j) ? mask : all, b + j);
            auto tmp = _mm512_sub_ps(va, vb);
            sum = _mm512_fmadd_ps(tmp, tmp, sum);
        }
        return _mm512_reduce_add_ps(sum);
    }
};

template <size_t N> struct L2Impl<N, float, uint8_t> {
    SVS_NOINLINE static float
    compute(const float* a, const uint8_t* b, lib::MaybeStatic<N> length) {
        auto sum = _mm512_setzero_ps();
        auto mask = create_mask<16>(length);
        auto all = no_mask<16>();
        for (size_t j = 0; j < length.size(); j += 16) {
            // Load and convert the integers to floating point.
            auto va = _mm512_maskz_loadu_ps(islast<16>(length, j) ? mask : all, a + j);
            auto tb = _mm_maskz_loadu_epi8(islast<16>(length, j) ? mask : all, b + j);
            auto vb = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(tb));
            auto tmp = _mm512_sub_ps(va, vb);
            sum = _mm512_fmadd_ps(tmp, tmp, sum);
        }
        return _mm512_reduce_add_ps(sum);
    };
};

template <size_t N> struct L2Impl<N, float, int8_t> {
    SVS_NOINLINE static float
    compute(const float* a, const int8_t* b, lib::MaybeStatic<N> length) {
        auto sum = _mm512_setzero_ps();
        auto mask = create_mask<16>(length);
        auto all = no_mask<16>();
        for (size_t j = 0; j < length.size(); j += 16) {
            // Load and convert the integers to floating point.
            auto va = _mm512_maskz_loadu_ps(islast<16>(length, j) ? mask : all, a + j);
            auto tb = _mm_maskz_loadu_epi8(islast<16>(length, j) ? mask : all, b + j);
            auto vb = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(tb));
            auto tmp = _mm512_sub_ps(va, vb);
            sum = _mm512_fmadd_ps(tmp, tmp, sum);
        }
        return _mm512_reduce_add_ps(sum);
    };
};

template <size_t N> struct L2Impl<N, float, Float16> {
    SVS_NOINLINE static float
    compute(const float* a, const Float16* b, lib::MaybeStatic<N> length) {
        auto sum = _mm512_setzero_ps();
        auto mask = create_mask<16>(length);
        auto all = no_mask<16>();

        for (size_t j = 0; j < length.size(); j += 16) {
            auto va = _mm512_maskz_loadu_ps(islast<16>(length, j) ? mask : all, a + j);
            auto vb_f16 =
                _mm256_maskz_loadu_epi16(islast<16>(length, j) ? mask : all, b + j);
            auto vb = _mm512_cvtph_ps(vb_f16);
            auto tmp = _mm512_sub_ps(va, vb);
            sum = _mm512_fmadd_ps(tmp, tmp, sum);
        }

        return _mm512_reduce_add_ps(sum);
    }
};

template <size_t N> struct L2Impl<N, Float16, Float16> {
    SVS_NOINLINE static float
    compute(const Float16* a, const Float16* b, lib::MaybeStatic<N> length) {
        auto sum = _mm512_setzero_ps();
        auto mask = create_mask<16>(length);
        auto all = no_mask<16>();

        for (size_t j = 0; j < length.size(); j += 16) {
            auto va_f16 =
                _mm256_maskz_loadu_epi16(islast<16>(length, j) ? mask : all, a + j);
            auto vb_f16 =
                _mm256_maskz_loadu_epi16(islast<16>(length, j) ? mask : all, b + j);
            auto va = _mm512_cvtph_ps(va_f16);
            auto vb = _mm512_cvtph_ps(vb_f16);
            auto tmp = _mm512_sub_ps(va, vb);
            sum = _mm512_fmadd_ps(tmp, tmp, sum);
        }

        return _mm512_reduce_add_ps(sum);
    };
};

// template <size_t N> struct L2Impl<N, Float16, Float16> {
//     SVS_NOINLINE static float
//     compute(const Float16* a, const Float16* b, lib::MaybeStatic<N> length) {
//         auto sum = _mm256_setzero_ps();
//         auto mask = create_mask<8>(length);
//         auto all = no_mask<8>();
//
//         for (size_t j = 0; j < length.size(); j += 8) {
//             auto va_f16 = _mm_maskz_loadu_epi16(islast<8>(length, j) ? mask : all, a +
//             j); auto vb_f16 = _mm_maskz_loadu_epi16(islast<8>(length, j) ? mask : all, b
//             + j); auto va = _mm256_cvtph_ps(va_f16); auto vb = _mm256_cvtph_ps(vb_f16);
//             auto tmp = _mm256_sub_ps(va, vb);
//             sum = _mm256_fmadd_ps(tmp, tmp, sum);
//         }
//
//         return simd::_mm256_reduce_add_ps(sum);
//     };
// };
#endif

/////
///// AVX 2 Implementations
/////

#if !defined(__AVX512F__) && defined(__AVX2__)
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
