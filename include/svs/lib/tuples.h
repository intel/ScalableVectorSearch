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

#include "svs/lib/preprocessor.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <tuple>
#include <utility>

namespace svs {
namespace lib {

// There's lots of forwarding going on in this file.
// Use the `static_cast` trick to maybe (probably not) reduce compile-times somewhat
// by instantiating fewer templates.
#define SVS_FORWARD(T, x) static_cast<T&&>(x)
#define SVS_SIZE_CHECK(Ts, Is) static_assert(sizeof...(Ts) == sizeof...(Is))

// The normal tuple constructors don't seem to work like we want:
// * `std::make_tuple`: Calls `std::decay_t` on its arguments and will make copies of
//   objects returned by reference, which isn't necessarily what we want.
// * `std::tie`: Always creates references to its arguments, which isn't what we want if
//   something is returned by value.
// * `std::forward_as_tuple` likewise wants to gather things as references.
//
// What we want (I think) is this:
// * If something is returned by value, keep it by value.
// * If something is returned by reference, keep it as a reference.
// * If something is returned by const reference, keep it as const reference.
// The solution is variadic universal references!
template <typename... Ts> constexpr auto SVS_FORCE_INLINE astuple(Ts&&... xs) {
    return std::tuple<Ts...>(SVS_FORWARD(Ts, xs)...);
}

namespace detail {
template <typename F, typename... Ts, size_t... Is>
constexpr auto SVS_FORCE_INLINE
map_impl(F&& f, std::tuple<Ts...>& xs, std::index_sequence<Is...> /*unused*/) {
    SVS_SIZE_CHECK(Ts, Is);
    return astuple(f(std::get<Is>(xs))...);
}

template <typename F, typename... Ts, size_t... Is>
constexpr auto SVS_FORCE_INLINE map_impl(
    F&& f, const std::tuple<Ts...>& xs, std::index_sequence<Is...> /*unused*/
) {
    SVS_SIZE_CHECK(Ts, Is);
    return astuple(f(std::get<Is>(xs))...);
}

// Simpler version of `map` that simply calls the provided function on each element
// of a tuple.
//
// Does not aggregate and return the results of each call.
template <typename F, typename... Ts, size_t... Is>
constexpr void SVS_FORCE_INLINE
foreach_impl(F&& f, std::tuple<Ts...>& xs, std::index_sequence<Is...> /*unused*/) {
    SVS_SIZE_CHECK(Ts, Is);
    (f(std::get<Is>(xs)), ...);
}

template <typename F, typename... Ts, size_t... Is>
constexpr void SVS_FORCE_INLINE
foreach_impl(F&& f, const std::tuple<Ts...>& xs, std::index_sequence<Is...> /*unused*/) {
    SVS_SIZE_CHECK(Ts, Is);
    (f(std::get<Is>(xs)), ...);
}

template <typename F, typename... Ts, size_t... Is>
constexpr void SVS_FORCE_INLINE
foreach_r_impl(F&& f, std::tuple<Ts...>& xs, std::index_sequence<Is...> /*unused*/) {
    SVS_SIZE_CHECK(Ts, Is);
    constexpr size_t N = sizeof...(Ts) - 1;
    (f(std::get<N - Is>(xs)), ...);
}

template <typename F, typename... Ts, size_t... Is>
constexpr void SVS_FORCE_INLINE
foreach_r_impl(F&& f, const std::tuple<Ts...>& xs, std::index_sequence<Is...> /*unused*/) {
    SVS_SIZE_CHECK(Ts, Is);
    constexpr size_t N = sizeof...(Ts) - 1;
    (f(std::get<N - Is>(xs)), ...);
}
} // namespace detail

// Entry points for the above implementations.
template <typename F, typename... Ts>
constexpr auto SVS_FORCE_INLINE map(std::tuple<Ts...>& xs, F&& f) {
    return detail::map_impl(
        SVS_FORWARD(F, f), xs, std::make_index_sequence<sizeof...(Ts)>{}
    );
}

template <typename F, typename... Ts>
constexpr auto SVS_FORCE_INLINE map(const std::tuple<Ts...>& xs, F&& f) {
    return detail::map_impl(
        SVS_FORWARD(F, f), xs, std::make_index_sequence<sizeof...(Ts)>{}
    );
}

// `foreach`.
// Support `const` and `nonconst` versions I guess.
template <typename F, typename... Ts>
constexpr void SVS_FORCE_INLINE foreach (std::tuple<Ts...>& xs, F && f) {
    detail::foreach_impl(SVS_FORWARD(F, f), xs, std::make_index_sequence<sizeof...(Ts)>());
}

template <typename F, typename... Ts>
constexpr void SVS_FORCE_INLINE foreach (const std::tuple<Ts...>& xs, F && f) {
    detail::foreach_impl(SVS_FORWARD(F, f), xs, std::make_index_sequence<sizeof...(Ts)>());
}

// `foreach_r`
// Like `foreach`, but iterates over element in the tuple in reverse order.
template <typename F, typename... Ts>
constexpr void SVS_FORCE_INLINE foreach_r(std::tuple<Ts...>& xs, F&& f) {
    detail::foreach_r_impl(
        SVS_FORWARD(F, f), xs, std::make_index_sequence<sizeof...(Ts)>()
    );
}

template <typename F, typename... Ts>
constexpr void SVS_FORCE_INLINE foreach_r(const std::tuple<Ts...>& xs, F&& f) {
    detail::foreach_r_impl(
        SVS_FORWARD(F, f), xs, std::make_index_sequence<sizeof...(Ts)>()
    );
}

#undef SVS_FORWARD
#undef SVS_SIZE_CHECK

/////
///// Tuple Hash
/////

// Code from boost
// Reciprocal of the golden ratio helps spread entropy
//     and handles duplicates.
// See Mike Seymour in magic-numbers-in-boosthash-combine:
//     http://stackoverflow.com/questions/4948780
namespace detail {
template <typename T> inline size_t combine(size_t seed, const T& x) noexcept {
    return seed ^ (std::hash<T>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}
} // namespace detail

///
/// @brief Hashing functor for ``std::tuple``.
///
/// The tuple hash works by combining the hashes of each element of the tuple.
///
struct TupleHash {
    template <typename... Ts>
    size_t operator()(const std::tuple<Ts...>& xs) const noexcept {
        size_t seed = 0xc0ffee;
        foreach (xs, [&seed](auto&& x) { seed = detail::combine(seed, x); })
            ;
        return seed;
    }
};

} // namespace lib
} // namespace svs
