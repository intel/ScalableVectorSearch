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

namespace type_traits {
template <typename T> struct MakeConst {
    using type = const T;
};

template <typename T> using make_const_t = typename MakeConst<T>::type;

// Specialize `std::span`
template <typename T, size_t Extent> struct MakeConst<std::span<T, Extent>> {
    using type = std::span<const T, Extent>;
};

// clang-format off
template <typename T>
concept DatabaseElement = requires {
    // Need to be able to make it constant.
    typename make_const_t<T>;
};
// clang-format on

///
/// Construct a sentinel element for type T with respect to the comparison function `Cmp`.
/// The sentinal value should `s` should satisfy
/// ```
/// cmp(x, s) == true
/// ```
/// For all valid (reasonable) values of `x`.
///
template <typename T, typename Cmp> struct Sentinel {};

template <typename T, typename Cmp> constexpr T sentinel_v = Sentinel<T, Cmp>::value;

// Provide overloads for integers and floating point numbers.
template <std::integral I> struct Sentinel<I, std::less<>> {
    static constexpr I value = std::numeric_limits<I>::max();
};

template <std::integral I> struct Sentinel<I, std::greater<>> {
    static constexpr I value = std::numeric_limits<I>::min();
};

template <std::floating_point F> struct Sentinel<F, std::less<>> {
    static constexpr F value = std::numeric_limits<F>::max();
};

template <std::floating_point F> struct Sentinel<F, std::greater<>> {
    static constexpr F value = std::numeric_limits<F>::min();
};

} // namespace type_traits
} // namespace svs
