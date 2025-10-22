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
#include "svs/lib/avx_detection.h"
#include "svs/lib/float16.h"
#include "svs/lib/preprocessor.h"
#include "svs/lib/saveload.h"
#include "svs/lib/static.h"

// stl
#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>

namespace svs::distance {
// Forward declare implementation to allow entry point to be near the top.
template <size_t N, typename Ea, typename Eb, AVX_AVAILABILITY Avx> struct IPImpl;

// Generic Entry Point
// Call as one of either:
// ```
// (1) IP::compute(a, b, length)
// (2) IP::compute<length>(a, b)
// ```
// Where (2) is when length is known at compile time and (1) is when length is not.
class IP {
  public:
    template <typename Ea, typename Eb>
    static constexpr float compute(const Ea* a, const Eb* b, size_t N) {
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx512f_supported(), 1)) {
            return IPImpl<Dynamic, Ea, Eb, AVX_AVAILABILITY::AVX512>::compute(
                a, b, lib::MaybeStatic(N)
            );
        }
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx2_supported(), 1)) {
            return IPImpl<Dynamic, Ea, Eb, AVX_AVAILABILITY::AVX2>::compute(
                a, b, lib::MaybeStatic(N)
            );
        }
        return IPImpl<Dynamic, Ea, Eb, AVX_AVAILABILITY::NONE>::compute(
            a, b, lib::MaybeStatic(N)
        );
    }

    template <size_t N, typename Ea, typename Eb>
    static constexpr float compute(const Ea* a, const Eb* b) {
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx512f_supported(), 1)) {
            if constexpr (is_dim_supported<N>()) {
                return IPImpl<N, Ea, Eb, AVX_AVAILABILITY::AVX512>::compute(
                    a, b, lib::MaybeStatic<N>()
                );
            } else {
                return IPImpl<Dynamic, Ea, Eb, AVX_AVAILABILITY::AVX512>::compute(
                    a, b, lib::MaybeStatic(N)
                );
            }
        }
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx2_supported(), 1)) {
            if constexpr (is_dim_supported<N>()) {
                return IPImpl<N, Ea, Eb, AVX_AVAILABILITY::AVX2>::compute(
                    a, b, lib::MaybeStatic<N>()
                );
            } else {
                return IPImpl<Dynamic, Ea, Eb, AVX_AVAILABILITY::AVX2>::compute(
                    a, b, lib::MaybeStatic(N)
                );
            }
        }
        return IPImpl<N, Ea, Eb, AVX_AVAILABILITY::NONE>::compute(
            a, b, lib::MaybeStatic<N>()
        );
    }
};

///
/// @brief Functor for computing the Inner Product.
///
/// This is the primary functor for implementing the Inner Product similarity between
/// two vectors in R^n. This functor uses the externally defined
/// \ref compute_distanceip "compute" method and is thus capable of being extended
/// externally.
///
struct DistanceIP {
    /// Vectors are more similar if their similarity is greater.
    using compare = std::greater<>;

    ///
    /// This functor does not use any local scratch space to assist in computation and
    /// thus may be shared across threads and queries safely.
    ///
    static constexpr bool implicit_broadcast = true;

    // IO
    static constexpr std::string_view name = "inner_product";

    static bool
    check_load_compatibility(std::string_view schema, svs::lib::Version version) {
        return DistanceSerialization::check_load_compatibility(schema, version);
    }

    lib::SaveTable save() const { return DistanceSerialization::save(name); }
    DistanceIP static load(const lib::ContextFreeLoadTable& table) {
        // Throws if retrieved name is incorrect.
        DistanceSerialization::check_load(table, name);
        return DistanceIP();
    }
};

inline constexpr bool operator==(DistanceIP, DistanceIP) { return true; }

///
/// @ingroup distance_overload
/// @anchor compute_distanceip
/// @brief Compute the Inner Product similarity between two vectors in R^n.
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
float compute(DistanceIP /*unused*/, std::span<Ea, Da> a, std::span<Eb, Db> b) {
    assert(a.size() == b.size());
    constexpr size_t extent = lib::extract_extent(Da, Db);
    if constexpr (extent == Dynamic) {
        return IP::compute(a.data(), b.data(), a.size());
    } else {
        return IP::compute<extent>(a.data(), b.data());
    }
}

/////
///// Generic Implementation
/////

template <size_t N, typename Ea, typename Eb>
float generic_ip(
    const Ea* a, const Eb* b, lib::MaybeStatic<N> length = lib::MaybeStatic<N>()
) {
    float result = 0;
    for (size_t i = 0; i < length.size(); ++i) {
        result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
    }
    return result;
}

template <size_t N, typename Ea, typename Eb, AVX_AVAILABILITY Avx> struct IPImpl {
    SVS_NOINLINE static float
    compute(const Ea* a, const Eb* b, lib::MaybeStatic<N> length = lib::MaybeStatic<N>()) {
        return generic_ip(a, b, length);
    }
};

/////
///// Intel(R) AVX-512 Implementations
/////

// Shared implementation among those that use floating-point arithmetic.
// Now parameterized by AVX_AVAILABILITY to enable explicit instantiation at SIMD op level
template <size_t SIMDWidth, AVX_AVAILABILITY Avx> struct IPFloatOp;

// SIMD accelerated operations that convert both left and right hand arguments to
// ``To`` and perform arithmetic on those integer operands.
template <std::integral To, size_t SIMDWidth, AVX_AVAILABILITY Avx> struct IPVNNIOp;

// Extern template declarations - definitions in avx512.cpp and avx2.cpp
extern template struct IPFloatOp<16, AVX_AVAILABILITY::AVX512>;
extern template struct IPVNNIOp<int16_t, 32, AVX_AVAILABILITY::AVX512>;
extern template struct IPFloatOp<8, AVX_AVAILABILITY::AVX2>;

SVS_VALIDATE_BOOL_ENV(SVS_AVX512_F)
#if SVS_AVX512_F

// VNNI Dispatching
template <size_t N> struct IPImpl<N, int8_t, int8_t, AVX_AVAILABILITY::AVX512> {
    SVS_NOINLINE static float
    compute(const int8_t* a, const int8_t* b, lib::MaybeStatic<N> length) {
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx512vnni_supported(), 1)) {
            return simd::generic_simd_op(IPVNNIOp<int16_t, 32, AVX_AVAILABILITY::AVX512>(), a, b, length);
        }
        // fallback to AVX512
        return svs::simd::generic_simd_op(IPFloatOp<16, AVX_AVAILABILITY::AVX512>{}, a, b, length);
    }
};

template <size_t N> struct IPImpl<N, uint8_t, uint8_t, AVX_AVAILABILITY::AVX512> {
    SVS_NOINLINE static float
    compute(const uint8_t* a, const uint8_t* b, lib::MaybeStatic<N> length) {
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx512vnni_supported(), 1)) {
            return simd::generic_simd_op(IPVNNIOp<int16_t, 32, AVX_AVAILABILITY::AVX512>(), a, b, length);
        }
        // fallback to AVX512
        return svs::simd::generic_simd_op(IPFloatOp<16, AVX_AVAILABILITY::AVX512>{}, a, b, length);
    }
};

#endif

// Floating and Mixed Types
template <size_t N> struct IPImpl<N, float, float, AVX_AVAILABILITY::AVX512> {
    SVS_NOINLINE static float
    compute(const float* a, const float* b, lib::MaybeStatic<N> length) {
        return svs::simd::generic_simd_op(IPFloatOp<16, AVX_AVAILABILITY::AVX512>{}, a, b, length);
    }
};

template <size_t N> struct IPImpl<N, float, uint8_t, AVX_AVAILABILITY::AVX512> {
    SVS_NOINLINE static float
    compute(const float* a, const uint8_t* b, lib::MaybeStatic<N> length) {
        return svs::simd::generic_simd_op(IPFloatOp<16, AVX_AVAILABILITY::AVX512>{}, a, b, length);
    };
};

template <size_t N> struct IPImpl<N, float, int8_t, AVX_AVAILABILITY::AVX512> {
    SVS_NOINLINE static float
    compute(const float* a, const int8_t* b, lib::MaybeStatic<N> length) {
        return svs::simd::generic_simd_op(IPFloatOp<16, AVX_AVAILABILITY::AVX512>{}, a, b, length);
    };
};

template <size_t N> struct IPImpl<N, float, Float16, AVX_AVAILABILITY::AVX512> {
    SVS_NOINLINE static float
    compute(const float* a, const Float16* b, lib::MaybeStatic<N> length) {
        return svs::simd::generic_simd_op(IPFloatOp<16, AVX_AVAILABILITY::AVX512>{}, a, b, length);
    }
};

template <size_t N> struct IPImpl<N, Float16, float, AVX_AVAILABILITY::AVX512> {
    SVS_NOINLINE static float
    compute(const Float16* a, const float* b, lib::MaybeStatic<N> length) {
        return svs::simd::generic_simd_op(IPFloatOp<16, AVX_AVAILABILITY::AVX512>{}, a, b, length);
    }
};

template <size_t N> struct IPImpl<N, Float16, Float16, AVX_AVAILABILITY::AVX512> {
    SVS_NOINLINE static float
    compute(const Float16* a, const Float16* b, lib::MaybeStatic<N> length) {
        return svs::simd::generic_simd_op(IPFloatOp<16, AVX_AVAILABILITY::AVX512>{}, a, b, length);
    }
};
#endif

/////
///// Intel(R) AVX2 Implementations
/////

// AVX2 implementations - always compiled, reference extern SIMD ops defined in avx2.cpp
template <size_t N> struct IPImpl<N, float, float, AVX_AVAILABILITY::AVX2> {
    SVS_NOINLINE static float
    compute(const float* a, const float* b, lib::MaybeStatic<N> length) {
        return svs::simd::generic_simd_op(IPFloatOp<8, AVX_AVAILABILITY::AVX2>{}, a, b, length);
    }
};

template <size_t N> struct IPImpl<N, Float16, Float16, AVX_AVAILABILITY::AVX2> {
    SVS_NOINLINE static float
    compute(const Float16* a, const Float16* b, lib::MaybeStatic<N> length) {
        return svs::simd::generic_simd_op(IPFloatOp<8, AVX_AVAILABILITY::AVX2>{}, a, b, length);
    }
};

template <size_t N> struct IPImpl<N, float, Float16, AVX_AVAILABILITY::AVX2> {
    SVS_NOINLINE static float
    compute(const float* a, const Float16* b, lib::MaybeStatic<N> length) {
        return svs::simd::generic_simd_op(IPFloatOp<8, AVX_AVAILABILITY::AVX2>{}, a, b, length);
    }
};

template <size_t N> struct IPImpl<N, float, int8_t, AVX_AVAILABILITY::AVX2> {
    SVS_NOINLINE static float
    compute(const float* a, const int8_t* b, lib::MaybeStatic<N> length) {
        return svs::simd::generic_simd_op(IPFloatOp<8, AVX_AVAILABILITY::AVX2>{}, a, b, length);
    }
};

template <size_t N> struct IPImpl<N, int8_t, int8_t, AVX_AVAILABILITY::AVX2> {
    SVS_NOINLINE static float
    compute(const int8_t* a, const int8_t* b, lib::MaybeStatic<N> length) {
        return svs::simd::generic_simd_op(IPFloatOp<8, AVX_AVAILABILITY::AVX2>{}, a, b, length);
    }
};

template <size_t N> struct IPImpl<N, uint8_t, uint8_t, AVX_AVAILABILITY::AVX2> {
    SVS_NOINLINE static float
    compute(const uint8_t* a, const uint8_t* b, lib::MaybeStatic<N> length) {
        return svs::simd::generic_simd_op(IPFloatOp<8, AVX_AVAILABILITY::AVX2>{}, a, b, length);
    }
};

} // namespace svs::distance
