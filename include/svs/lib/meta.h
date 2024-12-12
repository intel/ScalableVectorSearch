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

///
/// @ingroup lib_public
/// @defgroup lib_public_dimensions Static and Dynamic Dimensionality.
///

///
/// @ingroup lib_public
/// @defgroup lib_public_types Types and type lists.
///

#include "svs/lib/datatype.h"
#include "svs/lib/tuples.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace svs {

///
/// @ingroup lib_public_dimensions
/// @brief Special value representing run-time dimensionality.
///
/// Throughout the code base, static size information for various vector types can be
/// passed to potentially improve the quality of generated code.
///
/// While this can be beneficial at runtime, it does come at the cost of increased
/// compilation time and the need to dispatch to specialized implementations if they exist.
///
/// When run-time dimensionality is desired instead, the use of the sentinel value
/// ``Dynamic`` can be used.
///
const size_t Dynamic = std::dynamic_extent;

namespace lib {

///
/// @ingroup lib_public_types
/// @brief Empty struct for reasoning about types.
///
/// @tparam T - The C++ type this struct refers to.
///
template <typename T> struct Type {
    /// The type this struct is representing.
    using type = T;

    /// Implicit conversion for special types.
    constexpr operator DataType()
        requires has_datatype_v<T>
    {
        return datatype_v<T>;
    }
};

///
/// @ingroup lib_public_types
/// @brief A list of types.
///
template <typename... Ts> struct Types {
    /// The number of types in the list.
    static constexpr size_t size = sizeof...(Ts);
    /// Return an array of the type enums.
    static constexpr std::array<svs::DataType, sizeof...(Ts)> data_types()
        requires(has_datatype_v<Ts> && ...)
    {
        return std::array<svs::DataType, sizeof...(Ts)>{datatype_v<Ts>...};
    }
};

namespace detail {
template <typename T> inline constexpr bool is_types_v = false;
template <typename... Ts> inline constexpr bool is_types_v<svs::lib::Types<Ts...>> = true;
} // namespace detail

template <typename T>
concept TypeList = detail::is_types_v<T>;

///
/// @ingroup lib_public_types
/// @brief Return whether the requested type is in the type list.
///
template <typename T, typename... Ts>
constexpr bool in(Types<Ts...> SVS_UNUSED(typelist) = {}) {
    return (std::is_same_v<T, Ts> || ...);
}

///
/// @ingroup lib_public_types
/// @brief Return whether the requested runtime type is in the compiletime type list.
///
/// @param datatype The runtime data type.
/// @param typelist The list of accepted types.
///
/// Requires that elements in the type list have a know corresponding data type.
///
template <typename... Ts>
constexpr bool in(DataType datatype, Types<Ts...> SVS_UNUSED(typelist) = {}) {
    static_assert((has_datatype_v<Ts> && ...));
    return ((datatype_v<Ts> == datatype) || ...);
}

///
/// @ingroup lib_public_types
/// @brief Generator helper for each type.
///
/// @param types - The list of types to pass to the functor `f`.
/// @param f - Functor accepting a single argument `Type<T>` called once for each `T` in
///     `types`.
///
template <typename F, typename... Ts>
constexpr void for_each_type(Types<Ts...> SVS_UNUSED(types), F&& f) {
    (f(Type<Ts>()), ...);
}

template <typename T, typename F, typename... Ts>
std::vector<T> make_vec(Types<Ts...> SVS_UNUSED(types), F&& f) {
    return std::vector<T>{{f(Type<Ts>())...}};
}

/////
///// Match
/////

namespace detail {
struct ThrowMismatchError {
    void operator()(svs::DataType type) const {
        throw ANNEXCEPTION("Type {} is not supported for this operation!", type);
    }
};
} // namespace detail

template <
    typename F,
    typename T,
    typename... Ts,
    typename OnError = detail::ThrowMismatchError>
auto match(lib::Types<T, Ts...> /*unused*/, DataType type, F&& f, OnError&& on_error = {}) {
    if (type == datatype_v<T>) {
        return f(lib::Type<T>{});
    }

    // At the end of recursion, throw an exception.
    if constexpr (sizeof...(Ts) == 0) {
        if constexpr (std::is_void_v<std::invoke_result_t<OnError, DataType>>) {
            on_error(type);
            throw ANNEXCEPTION("The type {} is not allowed by this operation!", type);
        } else {
            return on_error(type);
        }
    } else {
        return match(lib::Types<Ts...>{}, type, SVS_FWD(f), SVS_FWD(on_error));
    }
}

/////
///// Compile time `size_t`.
/////

///
/// @ingroup lib_public_dimensions
/// @brief Compile-time dimensionality.
///
/// @tparam N - The static number of dimensions.
///
/// If dynamic (i.e. runtime) dimensionality is required, set `N = svs::Dynamic`.
///
template <size_t N> class Val {
  public:
    /// Construct a new Val instance.
    constexpr Val() = default;
    /// Return the value parameter `N`.
    static constexpr size_t value = N;
};

template <size_t N, size_t M>
constexpr bool operator==(Val<N> /*unused*/, Val<M> /*unused*/) {
    return false;
}

template <size_t N> constexpr bool operator==(Val<N> /*unused*/, Val<N> /*unused*/) {
    return true;
}

// Operators
template <size_t N> constexpr size_t operator/(Val<N> /*unused*/, size_t y) {
    return N / y;
}
template <size_t N> constexpr size_t operator/(size_t x, Val<N> /*unused*/) {
    return x / N;
}
template <size_t N, size_t M>
constexpr Val<N / M> operator/(Val<N> /*unused*/, Val<M> /*unused*/) {
    return Val<N / M>{};
}

template <size_t N> constexpr auto forward_extent(size_t x) {
    if (x != N) {
        throw ANNEXCEPTION(
            "Tring to forward a compile time value of {} with a runtime value of {}!", N, x
        );
    }
    return Val<N>{};
}
template <> inline constexpr auto forward_extent<Dynamic>(size_t x) { return x; }

// The result type of extent forwarding.
// The arguemnt "0" is simply used as a place-holder to get the correct type.
template <size_t N> using forward_extent_t = decltype(forward_extent<Val<N>>(0));

template <typename T> inline constexpr bool is_val_type_v = false;
template <auto N> inline constexpr bool is_val_type_v<Val<N>> = true;

template <typename T>
concept ValType = is_val_type_v<T>;

// Concept for accepting accepting either a `Val` or something convertible to an integer.
template <typename T>
concept IntegerLike = std::convertible_to<T, size_t> || is_val_type_v<T>;

template <std::integral I> constexpr I as_integral(I x) { return x; }
template <ValType T> constexpr size_t as_integral() { return T::value; }
template <size_t N> constexpr size_t as_integral(Val<N> SVS_UNUSED(x) = {}) { return N; }

// A template dance to determine whether or not the first type of a parameter pack is
// a `LoadContext` or not.
template <typename T, typename... Args> inline constexpr bool first_is() {
    if constexpr (sizeof...(Args) == 0) {
        return false;
    } else {
        using FirstT = std::decay_t<std::tuple_element_t<0, std::tuple<Args...>>>;
        return std::is_same_v<T, FirstT>;
    }
}

/////
///// Typename
/////

namespace detail {

constexpr std::pair<size_t, size_t> typename_prefix_and_suffix() {
#if defined(__clang__)
    constexpr auto prefix =
        std::string_view("auto svs::lib::generate_typename() [T = ").size();
    constexpr auto suffix = std::string_view("]").size();
#elif defined(__GNUC__)
    constexpr auto prefix =
        std::string_view("constexpr auto svs::lib::generate_typename() [with T = ").size();
    constexpr auto suffix = std::string_view("]").size();
#endif
    return std::make_pair(prefix, suffix);
}

} // namespace detail

template <typename T> constexpr auto generate_typename() {
    constexpr auto pretty = std::string_view(__PRETTY_FUNCTION__);
    constexpr auto prefix_and_suffix = detail::typename_prefix_and_suffix();
    assert(pretty.size() > prefix_and_suffix.first + prefix_and_suffix.second);
    auto pretty_mutable = pretty;

    pretty_mutable.remove_prefix(prefix_and_suffix.first);
    pretty_mutable.remove_suffix(prefix_and_suffix.second);
    constexpr size_t size =
        pretty.size() - prefix_and_suffix.first - prefix_and_suffix.second;
    auto make = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::array<const char, sizeof...(Is) + 1>{pretty_mutable[Is]...};
    };
    return make(std::make_index_sequence<size>());
}

} // namespace lib

} // namespace svs
