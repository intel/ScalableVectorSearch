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
namespace meta {

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
};

///
/// @ingroup lib_public_types
/// @brief Return whether the requested type is in the type list.
///
template <typename T, typename... Ts> constexpr bool in(Types<Ts...> /*unused*/) {
    return false || (std::is_same_v<T, Ts> || ...);
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
constexpr bool in(DataType datatype, Types<Ts...> SVS_UNUSED(typelist)) {
    static_assert(true && (has_datatype_v<Ts> && ...));
    return false || ((datatype_v<Ts> == datatype) || ...);
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
std::vector<T> make_vec(Types<Ts...> types, F&& f) {
    auto result = std::vector<T>();
    for_each_type(types, [&](auto type) { result.push_back(f(type)); });
    return result;
}

/////
///// Unwrapping
/////

template <typename T> struct Unwrapper;

// Specializations
template <std::integral I> struct Unwrapper<I> {
    using type = size_t;
    static constexpr size_t unwrap(size_t x) { return x; }
};

template <typename T> using unwrapped_t = typename Unwrapper<std::remove_cvref_t<T>>::type;

///
/// Perform type-domain to runtime-value conversion.
///
template <typename T> unwrapped_t<std::remove_cvref_t<T>> constexpr unwrap(T&& x) {
    return Unwrapper<std::remove_cvref_t<T>>::unwrap(std::forward<T>(x));
}

// Recursively define `Unwrapper` for tuples.
template <typename... Ts> struct Unwrapper<std::tuple<Ts...>> {
    using type = std::tuple<unwrapped_t<Ts>...>;
    static constexpr type unwrap(const std::tuple<Ts...>& t) {
        return map(t, [](auto&& x) { return meta::unwrap(x); });
    }
};

// Map `Type`s to DataType values.
template <typename T> struct Unwrapper<Type<T>> {
    using type = DataType;
    static constexpr type unwrap(Type<T> /*unused*/) { return datatype_v<T>; }
};

template <typename... Ts> constexpr std::tuple<unwrapped_t<Ts>...> make_key(Ts&&... ts) {
    return std::tuple<unwrapped_t<Ts>...>(meta::unwrap(ts)...);
}

/////
///// Match
/////

template <typename F, typename T, typename... Ts>
auto match(meta::Types<T, Ts...> /*unused*/, DataType type, F&& f) {
    if (type == datatype_v<T>) {
        return f(meta::Type<T>{});
    }

    // At the end of recursion, throw an exception.
    if constexpr (sizeof...(Ts) == 0) {
        throw ANNEXCEPTION("Type {} is not supported for this operation!", type);
    } else {
        return match(meta::Types<Ts...>{}, type, std::forward<F>(f));
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

template <size_t N> struct Unwrapper<Val<N>> {
    using type = size_t;
    static constexpr type unwrap(Val<N> /*unused*/) { return N; }
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

// Concept for accepting accepting either a `Val` or something convertible to an integer.
template <typename T>
concept IntegerLike = std::convertible_to<T, size_t> || is_val_type_v<T>;

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

} // namespace meta
} // namespace lib

// Namespace Alias.
namespace meta = lib::meta;

} // namespace svs
