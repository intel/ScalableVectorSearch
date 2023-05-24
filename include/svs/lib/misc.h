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

#include <bit>
#include <span>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "svs/lib/exception.h"
#include "svs/lib/meta.h"

namespace svs::lib {

///
/// @brief Tag type to disambuate that zero initialization is desired.
///
/// For some data types (e.g., ``svs::lib::UUID``), default construction may have
/// semantics other than zero initialization. In such cases, ``svs::lib::ZeroInitializer``
/// may be used to indicate that the corresponding memory for the object should be zeroed.
///
struct ZeroInitializer {};

///
/// @brief Tag for priority dispatch.
///
/// Helpful when we may need to inject shims somewhere in a dispatch pipeline.
/// See https://quuxplusone.github.io/blog/2021/07/09/priority-tag/ for a more in-depth
/// explanation.
///
template <size_t N> struct PriorityTag : PriorityTag<N - 1> {
    static constexpr PriorityTag<N - 1> next() { return PriorityTag<N - 1>(); }
};

template <> struct PriorityTag<0> {};

namespace detail {
template <typename T> struct SpanLikeImpl {
    static constexpr bool value = false;
};

template <typename T, size_t N> struct SpanLikeImpl<std::span<T, N>> {
    static constexpr bool value = true;
};
} // namespace detail
template <typename T> inline constexpr bool is_spanlike_v = detail::SpanLikeImpl<T>::value;

/// @brief Concept encompasing any span type of span.
template <typename T>
concept AnySpanLike = requires { is_spanlike_v<T>; };

///
/// Perform the operation `ceiling(x / y)`.
///
inline constexpr size_t div_round_up(size_t x, size_t y) {
    return (x / y) + static_cast<size_t>((x % y) != 0);
}

inline constexpr size_t round_up_to_multiple_of(size_t x, size_t multiple_of) {
    return multiple_of * (div_round_up(x, multiple_of));
}

///
/// Construct a span for a vector.
///
template <typename T, typename Alloc> std::span<T> as_span(std::vector<T, Alloc>& v) {
    return std::span<T>(v.data(), v.size());
}

template <typename T, typename Alloc>
std::span<const T> as_const_span(const std::vector<T, Alloc>& v) {
    return std::span<const T>(v.data(), v.size());
}

template <typename T, typename Alloc>
std::span<const T> as_span(const std::vector<T, Alloc>& v) {
    return as_const_span(v);
}

namespace detail {
inline void bounds_check(size_t got, size_t expected) {
    if (got != expected) {
        throw ANNEXCEPTION(fmt::format("Size mismatch. Got {}, expected {}!", got, expected)
        );
    }
}
} // namespace detail

template <size_t N, typename T, typename Alloc>
std::span<T, N> as_span(std::vector<T, Alloc>& v) {
    if constexpr (N == Dynamic) {
        return as_span(v);
    } else {
        detail::bounds_check(v.size(), N);
        return std::span<T, N>(v.data(), v.size());
    }
}

template <size_t N, typename T, typename Alloc>
std::span<const T, N> as_const_span(const std::vector<T, Alloc>& v) {
    if constexpr (N == Dynamic) {
        return as_const_span(v);
    } else {
        detail::bounds_check(v.size(), N);
        return std::span<const T, N>(v.data(), v.size());
    }
}

template <size_t N, typename T, typename Alloc>
std::span<const T, N> as_span(const std::vector<T, Alloc>& v) {
    return as_const_span<N>(v);
}

///
/// Compose two operators together.
///
template <typename Outer, typename Inner> class Compose {
  private:
    Inner inner_;
    Outer outer_;

  public:
    Compose(Outer outer, Inner inner)
        : inner_{std::move(inner)}
        , outer_{std::move(outer)} {}

    template <typename... Args> auto operator()(Args&&... args) {
        return outer_(inner_(std::forward<Args>(args)...));
    }
};

/**
 * Returns whether the vector size is a multiple of 32 or 64 bytes. Used by aligned_alloc
 * @param vec_size vector size
 * @return 64 or 32 if he vector size is a multiple of 64 or 32, respectively. Otherwise,
 * returns 0.
 */
inline size_t compute_alignment(size_t vec_size) {
    std::array<size_t, 2> alignment_vec{64, 32};
    for (auto alignment : alignment_vec) {
        if (vec_size % alignment == 0) {
            return alignment;
        }
    }
    return 0;
}

template <typename InputIterator1, typename InputIterator2>
size_t count_intersect(
    InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2
) {
    const auto set1 = std::unordered_set(first1, last1);
    const auto set2 = std::unordered_set(first2, last2);
    return std::count_if(set1.begin(), set1.end(), [&](const auto& x) {
        return set2.find(x) != set2.end();
    });
}

// Return the size of the intersection between the ranges `a` and `b`.
// The time complexity is `O(a.size() * b.size())`, so don't use on ranges that are
// too large.
//
// Multiplicity only counts once.
template <typename T, typename U> size_t count_intersect(T&& a, U&& b) {
    return count_intersect(a.begin(), a.end(), b.begin(), b.end());
};

///
/// Empty type used in optional returns that still need a type.
///
struct Empty {};

///
/// Identity functor.
///
struct identity {
    template <typename T> constexpr T operator()(T x) const { return x; }

    template <typename A, typename B, typename... Args>
    constexpr std::tuple<A, B, Args...> operator()(A&& a, B&& b, Args&&... args) const {
        return {std::forward<A>(a), std::forward<B>(b), std::forward<Args>(args)...};
    }
};

struct donothing {
    template <typename... Args> constexpr void operator()(Args&&... /*args*/) const {}
};

///
/// A constant with value `V`.
///
template <auto V> struct Const {
    using type = decltype(V);
};

// Define `Unwrap` for `Const`.
namespace meta {
template <auto V> struct Unwrapper<Const<V>> {
    using type = typename Const<V>::type;
    static constexpr size_t unwrap(Const<V> /*unused*/ = Const<V>()) { return V; }
};
} // namespace meta

///
/// Functor that returns its stored results from `operator()` regardless of the arguemnts
/// passed.
///
template <typename T> class Returns {
  public:
    constexpr explicit Returns(T val)
        : val_{std::move(val)} {}

    template <typename... Args> constexpr T operator()(Args&&... /*unused*/) const {
        return val_;
    }

  private:
    T val_;
};

///
/// Specialization of `Returns` for compile-time constant values.
///
template <auto V> class Returns<Const<V>> {
  public:
    using return_type = typename Const<V>::type;
    constexpr explicit Returns() = default;
    constexpr explicit Returns(Const<V> /*unused*/) {}

    template <typename... Args>
    constexpr return_type operator()(Args&&... /*unused*/) const {
        return V;
    }
};

// Common predicates.
using ReturnsTrueType = Returns<Const<true>>;

///
/// Representation of a positive integer power of 2.
///
class PowerOfTwo {
  public:
    // Constructors
    PowerOfTwo() = default;
    explicit PowerOfTwo(size_t value)
        : value_{value} {}

    // Accessors
    [[nodiscard]] constexpr size_t raw() const { return value_; }
    [[nodiscard]] constexpr size_t value() const { return size_t{1} << value_; }
    [[nodiscard]] constexpr size_t mod_mask() const { return value() - size_t{1}; }

  private:
    /// Members
    size_t value_;
};

///
/// Compute the rounded-down division of the `numberator` by the `denominator`.
/// Faster than normal division since the denominator is guarenteed to be a power of two.
///
inline constexpr size_t operator/(size_t numerator, PowerOfTwo denominator) {
    return numerator >> denominator.raw();
}

///
/// Compute the modulus of `numberator` and `denominator`.
/// Faster than normal division since the denominator is guarenteed to be a power of two.
///
inline constexpr size_t operator%(size_t numerator, PowerOfTwo denominator) {
    return numerator & denominator.mod_mask();
}

///
/// Compute the product `x` * `y`;
/// Faster the normal multiplication because `y` is guarenteed to be a power of two.
///
inline constexpr size_t operator*(size_t x, PowerOfTwo y) { return x << y.raw(); }

///
/// Compute the product `x` * `y`;
/// Faster the normal multiplication because `x` is guarenteed to be a power of two.
///
inline constexpr size_t operator*(PowerOfTwo x, size_t y) { return y << x.raw(); }

inline PowerOfTwo prevpow2(size_t value) {
    if (value == 0) {
        throw ANNEXCEPTION("0 has no previous power of two!");
    }
    size_t leading_zeros = std::countl_zero(value);
    // Maximum possible number of leading zeros.
    // Subtract 1 because we have already ruled out the posibilty that the input
    // value is zero.
    const size_t max_leading_zeros = 8 * sizeof(size_t) - 1;
    return PowerOfTwo{max_leading_zeros - leading_zeros};
};
} // namespace svs::lib
