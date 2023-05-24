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
// an AVX vector operation.
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
} // namespace svs
