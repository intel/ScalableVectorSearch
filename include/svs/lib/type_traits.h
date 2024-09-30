/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
 */
#pragma once

#include <concepts>
#include <functional>
#include <limits>
#include <span>
#include <type_traits>

namespace svs {
///
/// @brief Open-ended equivalent to ``std::is_arithmetic_v``.
///
/// Unlike the standard library variable, specialization of this construct is allowed.
///
template <typename T> inline constexpr bool is_arithmetic_v = std::is_arithmetic_v<T>;

///
/// @brief Open-ended equivalent to ``std::is_signed_v``.
///
/// Unlike the standard library variable, specialization of this construct is allowed.
///
template <typename T> inline constexpr bool is_signed_v = std::is_signed_v<T>;

template <typename T>
concept Arithmetic = is_arithmetic_v<std::remove_const_t<T>>;

///
/// @brief Open-ended trait to enable lossy conversions.
///
/// Specialization List:
///     double -> float
///     float -> svs::Float16
///
template <typename From, typename To> inline constexpr bool allow_lossy_conversion = false;
template <> inline constexpr bool allow_lossy_conversion<double, float> = true;

namespace type_traits {
///
/// Construct a sentinel element for type T with respect to the comparison function `Cmp`.
/// The sentinal value ``s`` should satisfy
/// ```
/// cmp(x, s) == true
/// ```
/// For all valid (non-NaN) values of `x`.
///
template <typename T, typename Cmp> struct Sentinel;

template <typename T, typename Cmp> constexpr T sentinel_v = Sentinel<T, Cmp>::value;

// Provide overloads for integers and floating point numbers.
template <std::integral I> struct Sentinel<I, std::less<>> {
    static constexpr I value = std::numeric_limits<I>::max();
};

template <std::integral I> struct Sentinel<I, std::greater<>> {
    static constexpr I value = std::numeric_limits<I>::lowest();
};

template <std::floating_point F> struct Sentinel<F, std::less<>> {
    static constexpr F value = std::numeric_limits<F>::max();
};

template <std::floating_point F> struct Sentinel<F, std::greater<>> {
    static constexpr F value = std::numeric_limits<F>::lowest();
};

///
/// A tombstone element for the type ``T`` with respect to the comparison function ``Cmp``.
/// The tombstone value ``s`` should satisfy
/// ```
/// cmp(x, s) == false
/// ```
/// For all valid (non-NaN) values of ``x``.
///
template <typename T, typename Cmp> struct TombStone;

template <typename T, typename Cmp> constexpr T tombstone_v = TombStone<T, Cmp>::value;

template <std::integral I> struct TombStone<I, std::less<>> {
    static constexpr I value = std::numeric_limits<I>::lowest();
};

template <std::integral I> struct TombStone<I, std::greater<>> {
    static constexpr I value = std::numeric_limits<I>::max();
};

template <std::floating_point F> struct TombStone<F, std::less<>> {
    static constexpr F value = std::numeric_limits<F>::lowest();
};

template <std::floating_point F> struct TombStone<F, std::greater<>> {
    static constexpr F value = std::numeric_limits<F>::max();
};

} // namespace type_traits
} // namespace svs
