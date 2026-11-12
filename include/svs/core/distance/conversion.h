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

// svs
#include "svs/core/distance/distance_core.h"
#include "svs/core/distance/simd_utils.h"
#include "svs/lib/avx_detection.h"
#include "svs/lib/float16.h"
#include "svs/lib/preprocessor.h"
#include "svs/lib/static.h"
#include "svs/lib/threads/threadlocal.h"

// stl
#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace svs::distance {

/// Alignment, in bytes, that ``Convert`` requires of its destination.
/// It equals one Intel(R) AVX-512 register, so every full-width store is aligned.
inline constexpr size_t conversion_alignment = 64;

// `PreconvertedArgument` obtains its alignment from `CacheAlignedAllocator`. If the two
// diverge, its buffers stop satisfying the precondition of `Convert`.
static_assert(conversion_alignment == threads::CACHE_LINE_BYTES);

// Forward declare implementation to allow entry point to be near the top.
template <size_t N, typename From, typename To, AVX_AVAILABILITY Avx> struct ConvertImpl;

///
/// @brief Convert ``length`` elements of ``src`` into ``dst``.
///
/// Separates the element conversion that distance kernels otherwise repeat on every call
/// from the distance computation itself: a vector converted once can be passed to
/// ``distance::compute`` any number of times without being converted again.
///
class Convert {
  public:
    ///
    /// @param src Source elements. Need not be aligned.
    /// @param dst Destination for ``length`` elements. Must be aligned to
    ///     ``conversion_alignment`` bytes and must not overlap ``src``.
    /// @param length The number of elements to convert.
    ///
    template <typename From, typename To>
    static void apply(const From* src, To* dst, size_t length) {
        assert(reinterpret_cast<std::uintptr_t>(dst) % conversion_alignment == 0);
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx512f_supported(), 1)) {
            return ConvertImpl<Dynamic, From, To, AVX_AVAILABILITY::AVX512>::apply(
                src, dst, lib::MaybeStatic(length)
            );
        }
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx2_supported(), 1)) {
            return ConvertImpl<Dynamic, From, To, AVX_AVAILABILITY::AVX2>::apply(
                src, dst, lib::MaybeStatic(length)
            );
        }
        return ConvertImpl<Dynamic, From, To, AVX_AVAILABILITY::NONE>::apply(
            src, dst, lib::MaybeStatic(length)
        );
    }
};

/////
///// Generic Implementation
/////

template <size_t N, typename From, typename To, AVX_AVAILABILITY Avx> struct ConvertImpl {
    SVS_NOINLINE static void apply(const From* src, To* dst, lib::MaybeStatic<N> length) {
        std::copy(src, src + length.size(), dst);
    }
};

/////
///// Intel(R) AVX-512 Implementations
/////

SVS_VALIDATE_BOOL_ENV(SVS_AVX512_F)
#if SVS_AVX512_F

template <size_t N> struct ConvertImpl<N, Float16, float, AVX_AVAILABILITY::AVX512> {
    SVS_NOINLINE static void
    apply(const Float16* src, float* dst, lib::MaybeStatic<N> length) {
        using op = simd::ConvertToFloat<16>;
        size_t i = 0;
        for (; i + op::simd_width <= length.size(); i += op::simd_width) {
            _mm512_store_ps(dst + i, op::load(src + i));
        }
        if (i < length.size()) {
            auto mask = create_mask<op::simd_width>(length);
            _mm512_mask_store_ps(dst + i, mask, op::load(mask, src + i));
        }
    }
};

#endif

/////
///// Intel(R) AVX2 Implementations
/////

SVS_VALIDATE_BOOL_ENV(SVS_AVX512_F)
SVS_VALIDATE_BOOL_ENV(SVS_AVX2)
#if !SVS_AVX512_F && SVS_AVX2

template <size_t N> struct ConvertImpl<N, Float16, float, AVX_AVAILABILITY::AVX2> {
    SVS_NOINLINE static void
    apply(const Float16* src, float* dst, lib::MaybeStatic<N> length) {
        using op = simd::ConvertToFloat<8>;
        size_t i = 0;
        for (; i + op::simd_width <= length.size(); i += op::simd_width) {
            _mm256_store_ps(dst + i, op::load(src + i));
        }
        if (i < length.size()) {
            // A full-width store here would write past the last converted element.
            auto mask = create_mask<op::simd_width>(length);
            _mm256_maskstore_ps(
                dst + i,
                _mm256_castps_si256(simd::create_blend_mask_avx2(mask)),
                op::load(mask, src + i)
            );
        }
    }
};

#endif

///
/// @brief Pre-allocated storage for one distance argument in the working type of ``Dist``.
///
/// @tparam Dist The distance functor the converted argument is passed to.
/// @tparam Argument The argument type as stored. ``distance_argument_t<Dist, Argument>``
///     names the type it is converted to.
///
/// Construct once with the dimensionality of the vectors it will hold, then call
/// ``operator()`` for each vector. Converting a vector once lets every later
/// ``distance::compute`` call that uses it, in either argument position, skip the
/// conversion.
///
/// The span returned by ``operator()`` stays valid until the next ``operator()`` call on
/// the same object. When a conversion takes place, its ``data()`` is aligned to
/// ``conversion_alignment`` bytes. An object is not safe to share between threads: use one
/// per thread. Copies own separate storage.
///
template <typename Dist, typename Argument> class PreconvertedArgument;

///
/// @brief No conversion: holds no storage and forwards each argument unchanged.
///
template <typename Dist, typename Argument>
    requires std::same_as<distance_argument_t<Dist, Argument>, Argument>
class PreconvertedArgument<Dist, Argument> {
  public:
    ///
    /// @brief Construct the storage.
    ///
    /// @param dimensions Ignored: nothing is converted, so nothing is stored.
    ///
    explicit PreconvertedArgument(size_t SVS_UNUSED(dimensions)) {}

    ///
    /// @brief Return ``argument`` unchanged.
    ///
    /// A prvalue argument is returned by value and an lvalue by reference, so binding the
    /// result to ``const auto&`` behaves exactly as binding ``argument`` itself would.
    ///
    template <typename A> A operator()(A&& argument) const {
        return std::forward<A>(argument);
    }
};

///
/// @brief Converts each arithmetic span into one buffer, allocated at construction.
///
template <typename Dist, typename T, size_t Extent>
    requires(!std::same_as<
             distance_argument_t<Dist, std::span<const T, Extent>>,
             std::span<const T, Extent>>)
class PreconvertedArgument<Dist, std::span<const T, Extent>> {
  public:
    /// The type ``operator()`` returns.
    using converted_type = distance_argument_t<Dist, std::span<const T, Extent>>;

  private:
    using value_type = std::remove_const_t<typename converted_type::element_type>;

  public:
    ///
    /// @brief Allocate the storage.
    ///
    /// @param dimensions The number of elements of every vector later passed to
    ///     ``operator()``.
    ///
    explicit PreconvertedArgument(size_t dimensions)
        : buffer_(dimensions) {}

    ///
    /// @brief Convert ``argument`` into the storage and return a view of the result.
    ///
    /// @param argument The vector to convert. Its size must equal the ``dimensions``
    ///     given at construction.
    ///
    converted_type operator()(std::span<const T, Extent> argument) {
        assert(argument.size() == buffer_.size());
        Convert::apply(argument.data(), buffer_.data(), argument.size());
        return converted_type(buffer_.data(), buffer_.size());
    }

  private:
    std::vector<value_type, threads::CacheAlignedAllocator<value_type>> buffer_;
};

#if defined(__x86_64__)

#include "svs/multi-arch/x86/preprocessor.h"

DISTANCE_CONVERT_EXTERN_TEMPLATE(AVX_AVAILABILITY::AVX512);
DISTANCE_CONVERT_EXTERN_TEMPLATE(AVX_AVAILABILITY::AVX2);

#endif

} // namespace svs::distance
