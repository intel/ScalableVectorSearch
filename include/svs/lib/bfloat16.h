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

#pragma once

#include "svs/lib/narrow.h"
#include "svs/lib/type_traits.h"
#include "svs/third-party/fmt.h"

#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>
#if defined(__AVX512_BF16__)
#include <x86intrin.h>
#endif

namespace svs {
namespace bfloat16 {
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

inline float bfloat16_to_float_untyped(const uint16_t x) {
    const uint32_t u = x << 16;
    return bitcast_uint32_to_float(u);
}

inline uint16_t float_to_bfloat16_untyped(const float x) {
    const uint32_t u = bitcast_float_to_uint32(x);
    return lib::narrow<uint16_t>(u >> 16);
}

} // namespace detail

// On GCC, we need to add this attribute so that BFloat16 members can appear inside
// packed structs.
class __attribute__((packed)) BFloat16 {
  public:
    BFloat16() = default;

    // converting constructors
    explicit BFloat16(float x)
        : value_{detail::float_to_bfloat16_untyped(x)} {}
    explicit BFloat16(double x)
        : BFloat16(lib::narrow_cast<float>(x)) {}
    explicit BFloat16(size_t x)
        : BFloat16(lib::narrow<float>(x)) {}
    explicit BFloat16(int x)
        : BFloat16(lib::narrow<float>(x)) {}

    // conversion functions
    operator float() const { return detail::bfloat16_to_float_untyped(value_); }
    BFloat16& operator=(float x) {
        value_ = detail::float_to_bfloat16_untyped(x);
        return *this;
    }

    // Allow users to set and expect the contents of the class as a uint16_t using an
    // explicit API.
    static BFloat16 from_raw(uint16_t value) { return BFloat16{value, FromRawTag{}}; }
    uint16_t raw() const { return value_; }

  private:
    // Use a tag to construct from a raw value in order to still allow a constructor
    // for a lone `uint16_t`.
    struct FromRawTag {};
    explicit BFloat16(uint16_t value, FromRawTag /*unused*/)
        : value_{value} {}
    uint16_t value_;
};
static_assert(std::is_trivial_v<BFloat16>);
static_assert(std::is_standard_layout_v<BFloat16>);

/////
///// Operators
/////

// For equality, still use `float` rather than the underlying bit pattern to handle cases
// like signed zeros.
inline bool operator==(BFloat16 x, BFloat16 y) { return float{x} == float{y}; }

/////
///// Pretty Printing
/////

} // namespace bfloat16

using BFloat16 = bfloat16::BFloat16;

// SVS local arithmetric trait.
template <> inline constexpr bool is_arithmetic_v<BFloat16> = true;
template <> inline constexpr bool is_signed_v<BFloat16> = true;
template <> inline constexpr bool allow_lossy_conversion<float, BFloat16> = true;

} // namespace svs

// Apply hashing to `BFloat16`
namespace std {
template <> struct hash<svs::BFloat16> {
    inline std::size_t operator()(const svs::BFloat16& x) const noexcept {
        return std::hash<float>()(x);
    }
};
} // namespace std

// Formatting and Printing
template <> struct fmt::formatter<svs::BFloat16> : svs::format_empty {
    auto format(svs::BFloat16 x, auto& ctx) const {
        return fmt::format_to(ctx.out(), "{}f16", float{x});
    }
};

inline std::ostream& operator<<(std::ostream& stream, svs::BFloat16 x) {
    return stream << fmt::format("{}", x);
}
