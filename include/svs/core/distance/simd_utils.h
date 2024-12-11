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

#include <array>
#include <limits>
#include <type_traits>

#include "x86intrin.h"

#include "svs/lib/float16.h"
#include "svs/lib/static.h"

namespace svs {
namespace simd {

inline float _mm256_reduce_add_ps(__m256 x) {
    const float* base = reinterpret_cast<float*>(&x);
    float sum{0};
    for (size_t i = 0; i < 8; ++i) {
        sum += base[i];
    }
    return sum;
}
} // namespace simd

namespace detail {
template <size_t N> struct MaskRepr {};
template <> struct MaskRepr<2> {
    using type = uint8_t;
};
template <> struct MaskRepr<4> {
    using type = uint8_t;
};
template <> struct MaskRepr<8> {
    using type = uint8_t;
};
template <> struct MaskRepr<16> {
    using type = uint16_t;
};
template <> struct MaskRepr<32> {
    using type = uint32_t;
};
template <> struct MaskRepr<64> {
    using type = uint64_t;
};

template <typename T> struct MaskIntrinsic {};
template <> struct MaskIntrinsic<uint8_t> {
    using mask_type = __mmask8;
};
template <> struct MaskIntrinsic<uint16_t> {
    using mask_type = __mmask16;
};
template <> struct MaskIntrinsic<uint32_t> {
    using mask_type = __mmask32;
};
template <> struct MaskIntrinsic<uint64_t> {
    using mask_type = __mmask64;
};
} // namespace detail

// Given a length `N`, obtain an appropriate integer type used as a mask for `N` lanes in
// an Intel(R) AVX vector operation.
template <size_t N> using mask_repr_t = typename detail::MaskRepr<N>::type;

// Given an unsigned integer type `T`, obtain the appropriate mask intrinsic type.
template <typename T> using mask_intrinsic_t = typename detail::MaskIntrinsic<T>::mask_type;

// Given a length `N`, obtain an appropriate mask intrinsic type.
template <size_t N> using mask_intrinsic_from_length_t = mask_intrinsic_t<mask_repr_t<N>>;

// Construct masks for the passed lengths.
template <size_t VecLength, size_t N>
constexpr mask_intrinsic_from_length_t<VecLength> create_mask(lib::MaybeStatic<N> length) {
    using MaskType = mask_repr_t<VecLength>;
    constexpr MaskType one{0x1};
    MaskType shift = length.size() % VecLength;
    MaskType mask_raw =
        shift == 0 ? std::numeric_limits<MaskType>::max() : (one << shift) - one;
    return mask_raw;
}

template <size_t VecLength> constexpr mask_intrinsic_from_length_t<VecLength> no_mask() {
    return std::numeric_limits<mask_repr_t<VecLength>>::max();
}

/////
///// A generic SIMD op.
/////

namespace simd {

/// @brief A common generic routine for SIMD distance kernels.
///
/// SIMD accelerated distance kernels typically share the same pattern consisting of:
///
/// 1. An initialized accumulator
/// 2. Load and conversion of the left-hand argument.
/// 3. Load and conversion of the right-hand argument.
/// 4. An accumulation step.
///
/// When using SIMD, this accumulator is generally a SIMD register (for cosine-similarity,
/// it might be a pair of registers if the norm of the right-hand argument is being computed
/// as well). The use of SIMD register requires a reduction over the final state of the
/// accumulator to retrieve a scalar.
///
/// There are some other considerations:
/// 1. To exploit ILP, it can be helpful to both unroll the loop (which also slightly
///    reduces loop overhead).
///
///    For static dimensions, the compiler may be capable of doing this. In general, clang
///    is *much* more willing to do this than GCC.
///
///    Unrolling really only seems to matter for small dimensionalities. Presumably, once
///    the dimensionality is sufficiently high, we end up waiting for prefetched vectors
///    to come from L2/L3 anyways, so nothing we do on the computation front seems to
///    matter.
///
/// 2. Using multiple accumulators can really help in some situations. Floating point
///    arithmetic is not associative, so generally the compiler must strictly obey program
///    semantics when optimizing. This means that if a single accumulator register is used,
///    we introduce a long chain dependency in the instruction stream. Intel(R) AVX functional
///    units are generally pipelined and so have a relatively high latency (4 cycles is common)
///    but with a high throughput.
///
///    For example: Cascadelake and greater servers have two execution port that offer the
///    bulk of Intel(R) AVX-512 functionality. When fully utilized, SIMD instructions can obtain
///    a throughput of 2 ops per cycle (separate from loads, which can sustain another 2 ops
///    (3 ops on Sapphire Rapids) per cycle.
///
///    A long dependence on a single accumulation register basically throws all that
///    horse-power away.
///
///    Again, when vector data is far from the CPU (high in the cache or in memory), this
///    doesn't seem to make a meaningful difference. However, for lower dimensional data
///    or for data that is hot in the cache, this can make a big difference.
///
/// 3. When handling remainder elements (for example, where the dimensionality is odd),
///    we want to use predicated SIMD instead of falling back to scalar ops.
///
/// All of this is tedious to implement over and over for slightly different distance
/// kernels. This generic SIMD operation tries to take all of this into consideration and
/// distill it into an approach that is easier to use.
///
/// @param op A (usually empty) class implementing the operation interface (described below)
/// @param a The left-hand argument to the operation.
/// @param b The right-hand argument to the operation.
/// @param count The number of elements in memory pointed to by `a` and `b` (must be the
///     same for both arguments.
///
/// The functor `op` must implement the following interface:
/// @code{cpp}
/// struct Op {
/// // The width of the SIMD elements returned from the load instructions.
/// // This will be used to generate predicates.
/// static constexpr size_t simd_width;
///
/// // Initialize and return the accumulator variable.
/// // The type `Accumulator` not needed internally but used for exposition purposes
/// // for the rest of the interface.
/// Accumulator init() const;
///
/// // Load a value from the left-hand pointer (optionally with a predicate).
/// AValue load_a(const Ea*) const;
/// AValue load_a(svs::mask_repr_t<simd_width>, const Ea*) const;
///
/// // Load a value from the right-hand pointer (optionally with a predicate).
/// BValue load_b(const Eb*) const;
/// BValue load_b(svs::mask_repr_t<simd_width>, const Eb*) const;
///
/// // Perform an accumulation.
/// Accumulator accumulate(Accumulator, AValue, BValue) const;
/// Accumulator accumulate(
///     svs::mask_repr_t<simd_width>, Accumulator, AValue, BValue
/// ) const;
///
/// // Combine two accumulators.
/// Accumulator combine(Accumulator, Accumulator) const;
///
/// // Perform a final reduction.
/// // Any type is allowed since higher-level routines may still need to perform
/// // post-processing on the result.
/// Any reduce(Accumulator) const;
/// @endcode
///
template <typename Op, typename Ea, typename Eb, size_t N>
SVS_FORCE_INLINE auto
generic_simd_op(Op op, const Ea* a, const Eb* b, lib::MaybeStatic<N> count) {
    // Generically unroll by 4.
    constexpr size_t simd_width = Op::simd_width;
    constexpr size_t unroll = 4;
    constexpr size_t main = unroll * simd_width;

    auto s0 = op.init();
    size_t i = 0;

    // Main sequence - process 64 elements per loop iteration.
    if (i + main <= count.size()) {
        auto s1 = op.init();
        auto s2 = op.init();
        auto s3 = op.init();

        for (; i + main <= count.size(); i += main) {
            auto a0 = op.load_a(a + i + 0 * simd_width);
            auto a1 = op.load_a(a + i + 1 * simd_width);
            auto a2 = op.load_a(a + i + 2 * simd_width);
            auto a3 = op.load_a(a + i + 3 * simd_width);

            auto b0 = op.load_b(b + i + 0 * simd_width);
            auto b1 = op.load_b(b + i + 1 * simd_width);
            auto b2 = op.load_b(b + i + 2 * simd_width);
            auto b3 = op.load_b(b + i + 3 * simd_width);

            s0 = op.accumulate(s0, a0, b0);
            s1 = op.accumulate(s1, a1, b1);
            s2 = op.accumulate(s2, a2, b2);
            s3 = op.accumulate(s3, a3, b3);
        }

        s0 = op.combine(op.combine(s0, s1), op.combine(s2, s3));
    }

    // Full-width epilogue.
    for (; i + simd_width <= count.size(); i += simd_width) {
        s0 = op.accumulate(s0, op.load_a(a + i), op.load_b(b + i));
    }

    // Ragged epilogue.
    if (i < count.size()) {
        auto mask = create_mask<simd_width>(count);
        s0 = op.accumulate(mask, s0, op.load_a(mask, a + i), op.load_b(mask, b + i));
    }
    return op.reduce(s0);
}

// A utility base class for converting simd-converting to floating point.
template <size_t SIMDWidth> struct ConvertToFloat;

SVS_VALIDATE_BOOL_ENV(SVS_AVX512_F)
#if SVS_AVX512_F

// Common implementations for converting arguments to floats.
// Partially satisfies the requirements for a `generic_simd_op` operation.
template <> struct ConvertToFloat<16> {
    static constexpr size_t simd_width = 16;
    using mask_t = svs::mask_repr_t<simd_width>;

    // from float
    static __m512 load(const float* ptr) { return _mm512_loadu_ps(ptr); }
    static __m512 load(mask_t m, const float* ptr) { return _mm512_maskz_loadu_ps(m, ptr); }

    // from float16
    static __m512 load(const Float16* ptr) {
        return _mm512_cvtph_ps(_mm256_loadu_epi16(ptr));
    }

    static __m512 load(mask_t m, const Float16* ptr) {
        return _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(m, ptr));
    }

    // from int8
    static __m512 load(const uint8_t* ptr) {
        return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_epi8(ptr)));
    }
    static __m512 load(mask_t m, const uint8_t* ptr) {
        return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(m, ptr)));
    }

    // from uint8
    static __m512 load(const int8_t* ptr) {
        return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_epi8(ptr)));
    }
    static __m512 load(mask_t m, const int8_t* ptr) {
        return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(m, ptr)));
    }

    // We do not need to treat the left or right-hand differently.
    // Simple call the overloaded `load` methods.
    template <typename A> static __m512 load_a(const A* a) { return load(a); }
    template <typename A> static __m512 load_a(mask_t m, const A* a) { return load(m, a); }
    template <typename B> static __m512 load_b(const B* b) { return load(b); }
    template <typename B> static __m512 load_b(mask_t m, const B* b) { return load(m, b); }
};

#endif

// A base class used for customizing generic SIMD operations using VNNI instructions.
//
// Converts intermediate data into ``SINDWidth`` wide SIMD registers containing values of
// type ``To``.
//
// Expected to implement the following interface:
// @code{cpp}
// template <std::integral To, size_t SIMDWidth> struct ConvertForVNNI {
//     static constexpr size_t simd_width = SIMDWidth;
//     using reg_t = /*width_defined*/
//     using mask_t = svs::mask_repr_t<simd_width>;
//
//     template<typename A> reg_t load_a(const A* a);
//     template<typename A> reg_t load_a(mask_t m, const A* a);
//
//     template<typename B> reg_t load_b(const B* b);
//     template<typename B> reg_t load_b(mask_t m, const B* b);
// };
// @endcode
//
// The behavior of ``load_a`` and ``load_b`` is identical.
template <std::integral To, size_t SIMDWidth> struct ConvertForVNNI;

SVS_VALIDATE_BOOL_ENV(SVS_AVX512_VNNI)
#if SVS_AVX512_VNNI

template <> struct ConvertForVNNI<int16_t, 32> {
    static constexpr size_t simd_width = 32;
    using reg_t = __m512i;
    using mask_t = svs::mask_repr_t<simd_width>;

    // uint8
    SVS_FORCE_INLINE static reg_t load(const uint8_t* ptr) {
        return _mm512_cvtepu8_epi16(_mm256_loadu_epi8(ptr));
    }

    SVS_FORCE_INLINE static reg_t load(mask_t m, const uint8_t* ptr) {
        return _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(m, ptr));
    }

    // int8
    SVS_FORCE_INLINE static reg_t load(const int8_t* ptr) {
        return _mm512_cvtepi8_epi16(_mm256_loadu_epi8(ptr));
    }

    SVS_FORCE_INLINE static reg_t load(mask_t m, const int8_t* ptr) {
        return _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(m, ptr));
    }

    template <typename A> SVS_FORCE_INLINE static reg_t load_a(const A* a) {
        return load(a);
    }

    template <typename A> SVS_FORCE_INLINE static reg_t load_a(mask_t m, const A* a) {
        return load(m, a);
    }

    template <typename B> SVS_FORCE_INLINE static reg_t load_b(const B* b) {
        return load(b);
    }

    template <typename B> SVS_FORCE_INLINE static reg_t load_b(mask_t m, const B* b) {
        return load(m, b);
    }
};

#endif

} // namespace simd
} // namespace svs
