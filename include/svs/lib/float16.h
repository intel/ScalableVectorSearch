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

#include "svs/lib/narrow.h"
#include "svs/lib/type_traits.h"
#include "svs/third-party/fmt.h"

#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>
#if defined(__F16C__)
#include <x86intrin.h>
#endif

namespace svs {
namespace float16 {
namespace detail {

// TODO: Update to `bitcast` if its available in the standard library.
inline uint32_t bitcast_float_to_uint32(const float x) {
    static_assert(sizeof(float) == sizeof(uint32_t));
    uint32_t u;
    memcpy(&u, &x, sizeof(x));
    return u;
}

inline float bitcast_uint32_to_float(const uint32_t x) {
    static_assert(sizeof(float) == sizeof(uint32_t));
    float f;
    memcpy(&f, &x, sizeof(x));
    return f;
}

// reference:
// https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
inline float float16_to_float_untyped_slow(const uint16_t x) {
    // without norm/denorm (it makes code slow)
    const uint32_t e = (x & 0x7C00) >> 10; // exponent
    const uint32_t m = (x & 0x03FF) << 13; // mantissa
    return bitcast_uint32_to_float(
        (x & 0x8000) << 16 | static_cast<uint32_t>(e != 0) * ((e + 112) << 23 | m)
    );
}

inline uint16_t float_to_float16_untyped_slow(const float x) {
    // round-to-nearest-even: add last bit after truncated mantissa
    const uint32_t b = bitcast_float_to_uint32(x) + 0x00001000;
    const uint32_t e = (b & 0x7F800000) >> 23; // exponent
    const uint32_t m = b & 0x007FFFFF;         // mantissa
    return (b & 0x80000000) >> 16 |
           static_cast<uint32_t>(e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
           static_cast<uint32_t>((e < 113) && (e > 101)) *
               ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
           static_cast<uint32_t>(e > 143) *
               0x7FFF; // sign : normalized : denormalized : saturate
}

// If the processor is new enough, we can use hardware intrinsics to perform the conversion
// without using bit-level manipulation.
//
// Here, we check if the `F16C` set is enabled and if so, we define the intrinsic based
// conversion functions.
//
// The entry point for users of the conversion is `*_to_*_untyped`, which will dispatch
// to either the slow or fast version, depending on the architecture.
#if defined(__F16C__)
inline float float16_to_float_untyped_fast(const uint16_t x) {
    auto converted = _mm_cvtph_ps(_mm_set1_epi16(std::bit_cast<int16_t>(x)));
    return _mm_cvtss_f32(converted);
}
inline uint16_t float_to_float16_untyped_fast(const float x) {
    auto converted = _mm_cvtps_ph(__m128{x}, _MM_FROUND_NO_EXC);
    return _mm_extract_epi16(converted, 0);
}

inline float float16_to_float_untyped(const uint16_t x) {
    return float16_to_float_untyped_fast(x);
}
inline uint16_t float_to_float16_untyped(const float x) {
    return float_to_float16_untyped_fast(x);
}
#else
inline float float16_to_float_untyped(const uint16_t x) {
    return float16_to_float_untyped_slow(x);
}
inline uint16_t float_to_float16_untyped(const float x) {
    return float_to_float16_untyped_slow(x);
}
#endif
} // namespace detail

// On GCC - we need to add this attribute so that Float16 members can appear inside
// packed structs.
class __attribute__((packed)) Float16 {
  public:
    Float16() = default;

    // converting constructors
    explicit Float16(float x)
        : value_{detail::float_to_float16_untyped(x)} {}
    explicit Float16(double x)
        : Float16(lib::narrow_cast<float>(x)) {}
    explicit Float16(size_t x)
        : Float16(lib::narrow<float>(x)) {}
    explicit Float16(int x)
        : Float16(lib::narrow<float>(x)) {}

    // conversion functions
    operator float() const { return detail::float16_to_float_untyped(value_); }
    Float16& operator=(float x) {
        value_ = detail::float_to_float16_untyped(x);
        return *this;
    }

    // Allow users to set and expect the contents of the class as a uint16_t using an
    // explicit API.
    static Float16 from_raw(uint16_t value) { return Float16{value, FromRawTag{}}; }
    uint16_t raw() const { return value_; }

  private:
    // Use a tag to construct from a raw value in order to still allow a constructor
    // for a lone `uint16_t`.
    struct FromRawTag {};
    explicit Float16(uint16_t value, FromRawTag /*unused*/)
        : value_{value} {}
    uint16_t value_;
};
static_assert(std::is_trivial_v<Float16>);
static_assert(std::is_standard_layout_v<Float16>);

/////
///// Operators
/////

// For equality, still use `float` rather than the underlying bit pattern to handle cases
// like signed zeros.
inline bool operator==(Float16 x, Float16 y) { return float{x} == float{y}; }

/////
///// Pretty Printing
/////

} // namespace float16

using Float16 = float16::Float16;

// SVS local arithmetric trait.
template <> inline constexpr bool is_arithmetic_v<Float16> = true;
template <> inline constexpr bool is_signed_v<Float16> = true;
template <> inline constexpr bool allow_lossy_conversion<float, Float16> = true;

} // namespace svs

// Apply hashing to `Float16`
namespace std {
template <> struct hash<svs::Float16> {
    inline std::size_t operator()(const svs::Float16& x) const noexcept {
        return std::hash<float>()(x);
    }
};
} // namespace std

// Formatting and Printing
template <> struct fmt::formatter<svs::Float16> : svs::format_empty {
    auto format(svs::Float16 x, auto& ctx) const {
        return fmt::format_to(ctx.out(), "{}f16", float{x});
    }
};

inline std::ostream& operator<<(std::ostream& stream, svs::Float16 x) {
    return stream << fmt::format("{}", x);
}
