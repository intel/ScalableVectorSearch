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
/// @defgroup distance_public Distance Helpers
/// @brief Distance Helpers
///

#include "svs/lib/exception.h"

#include <concepts>
#include <cstddef>
#include <utility>
#include <vector>

namespace svs::distance {

// Usually, we detect when calls to `fix_argument` are required.
// However, we would like to fail in cases where such a call *should* be made, but isn't due
// to SFINAE silently dropping it.
//
// Defining ``static constexpr bool must_fix_argument = true`` will ensure that calls to
// ``distance::maybe_fix_argument`` will always be made (and fail if an appropriate
// overload cannot be found).
template <typename T> constexpr bool fix_argument_mandated() { return false; }

template <typename T>
    requires requires { T::must_fix_argument; } && T::must_fix_argument
constexpr bool fix_argument_mandated() {
    return true;
}

// clang-format off
template <typename F, typename A>
concept ShouldFix = fix_argument_mandated<F>() || requires(F f, A a) {
    // Has a member function that can fix arguments of type `const A&`.
    f.fix_argument(a);
};

template <typename F, typename A>
concept HasOneArgCompute = requires(F& f, A a) {
    f.compute(a);
};

template <typename F, typename A, typename B>
concept HasTwoArgCompute = requires(F& f, A a, B b) {
    f.compute(a, b);
};
// clang-format on

///
/// @ingroup distance_public
///
/// Perform an argument fixing of query `a` for the distance functor `f`.
/// This provides a hook mechanism allowing distance functors to preprocess the query
/// before performing any distance computations if desired.
///
/// This should be called before calling ``svs::distance::compute`` where `b` is
/// allowed to vary but `a` is not.
///
/// Modifications to `a` require a new call to `distance::maybe_fix_argument`.
///
/// If distance functors wish to opt-in to argument fixing, the must define a member
/// function `fix_argument(A)` which will be automatically called by
/// `distance::maybe_fix_argument`.
///
/// If distance functors do not define such a member function, then
/// `distance::maybe_fix_argument` becomes a no-op unless ``f`` defines the static member
/// variable
/// @code{cpp}
/// static constexpr bool must_fix_argument = true;
/// @endcode
/// in which case, missing overloads to ``f.fix_argument(A)`` become compiler errors.
///
template <typename F, typename A> void maybe_fix_argument(F& f, A a) {
    if constexpr (ShouldFix<F, A>) {
        f.fix_argument(a);
    }
}

///
/// @ingroup distance_public
///
/// Perform a distance computation with the distance functor `f`, query `a` and
/// element `b`.
///
/// Functors `f` can opt-in to this method in one of three ways:
/// * Direct overloading of `distance::compute`. This can be used to extend the behavior
///   of existing functors without modification.
/// * If `f` does not implement argument fixing for this combination of types, then `f`
///   may implement a member function `f.compute(a, b)` which will be called
///   automatically.
/// * if `f` **does** implement argument fixing for this type combination, then `f`
///   may implement a single argument member function `f.compute(b)` which will be called.
///
/// Argument fixing is implied if `f.fix_argument(a)` is valid.
/// In order to apply to all distance functors, ``distance::maybe_fix_argument``
/// should be called each time the value of ``a`` changes.
///
#if defined(__DOXYGEN__)
template <typename F, typename A, typename B> float compute(F& f, A a, B b);
#endif

// Generic two-argument `compute` implementation.
// Tries to fall-back to a member function.
//
// However, implementations are free to specialize the `free` function in order
// to provide an open interface to be extended by other types.
template <typename A, typename B, HasOneArgCompute<B> F> float compute(F& f, A /*a*/, B b) {
    return f.compute(b);
}

template <typename A, typename B, HasTwoArgCompute<A, B> F> float compute(F& f, A a, B b) {
    return f.compute(a, b);
}

// clang-format off

template <typename F, typename A, typename B>
concept FixDistance = requires(F& f, A a, B b) {
    requires ShouldFix<F, A>;

    // Compute distance against a fixed argument
    { compute(f, a, b) } -> std::convertible_to<float>;

    // Can compare distances
    typename F::compare;
};

template <typename F, typename A, typename B>
concept UnfixDistance = requires(F& f, A a, B b) {
    // Two argument distance computation.
    { compute(f, a, b) } -> std::convertible_to<float>;

    // Can compare distances
    typename F::compare;
};
// clang-format on

// Algorithms accepting abstract `Distance` types should support either distance types
// requiring fixing or those that don't require fixing.
template <typename F, typename A, typename B>
concept Distance = FixDistance<F, A, B> || UnfixDistance<F, A, B>;

// Here, we would like to add a method to create comparison operators from instances of
// functions.
//
// We build some machinery to do that such that implementations are free to provide a
// `.comparator()` method if they wish.
namespace detail {
// clang-format off
template <typename T>
concept HasComparator = requires(const T& x) {
    typename T::compare;
    { x.comparator() } -> std::same_as<typename T::compare>;
};
// clang-format on
} // namespace detail

///
/// @ingroup distance_public
///
/// Obtain the type of the comparator used for the distance functor `T`.
///
template <typename T> using compare_t = typename T::compare;

///
/// @ingroup distance_public
///
/// Return a comparison functgor for distance functors `T`.
/// For example, distance functors that wish to minimize distances may return
/// `std::less<>` while those those wishing to maximize may return `std::greater<>`.
///
/// @param x The distance functor to obtain the comparison functor for.
/// @returns A comparison functor `f` with an overloaded `operator()` such that
///
///     f(float, float) -> bool
///
/// is valid.
///
/// Custom types can provide support for `distance::comparator` in one of three ways:
/// * Direct overloading of `distance::comparator`.
/// * If the desired comparison functor is default constructible, providing a type
///   alias `T::compare`.
/// * Providing a member function `x.compare()` returning the functor.
///
/// When `T` defines both a member function and a type alias, the member function takes
/// precedence.
///
template <typename T> compare_t<T> comparator(const T& x);

// -- comparator implementations
// member function
template <detail::HasComparator T> auto comparator(const T& x) -> compare_t<T> {
    return x.comparator();
}

// type alias
template <typename T> auto comparator(const T& /*x*/) -> compare_t<T> {
    return typename T::compare{};
}

/////
///// Traits
/////

///
/// @ingroup distance_public
///
/// Return whether distance functors of type `T` are implicitly broadcastable.
///
/// To be implicitly broadcastable, must not require argument fixing. In other words,
/// it must be safe to call `distance::compute` with varying left and right-hand arguments
/// without needing to call `maybe_fix_argument`.
///
/// To opt into implicit broadcasting, classes should implement a constexpr static boolean
/// member `implicit_broadcast = true`.
///
template <typename T> constexpr bool implicitly_broadcastable() { return false; }

// Return `true` if type `T` defines a static member `implicit_broadcast` which constant
// evaluates to `true`.
//
// The motivation here is that is a distance type classifies itself as `implicit_broadcast`,
// then we can avoid creating vectors of such types when creating multiple copies when
// batching distance operations.
template <typename T>
    requires requires { T::implicit_broadcast; } && T::implicit_broadcast
constexpr bool implicitly_broadcastable() {
    return true;
}

///
/// @ingroup distance_public
///
/// Efficiently create copies of the distance functor `f` of type `T` to allow batch
/// distance computations involving multiple queries.
///
/// Be default, this is done by invoking the copy constructor the necessary number of times.
/// However, if explicit copies of the functor `f` are not needed for correctness
/// (i.e., `distance::implicitly_broadcastable<T>() == true`, than these copies will not be
/// made resulting in a slightly more efficient implementation.
///
/// Invariants:
///
/// * `size()` cannot be zero. This is because distance functions requiring broadcasting
///   are often stateful and may have some shared (usually read-only) state.
///
///   If the underlying storage for this class ever becomes empty, we lose that storage.
///
template <typename T> class BroadcastDistance {
  public:
    ///
    /// Construct `ncopies` of the distance functor `distance`.
    ///
    BroadcastDistance(T distance, size_t ncopies)
        : distances_{ncopies, distance} {}

    ///
    /// Retrieve the `i`th distance functor.
    ///
    T& operator[](size_t i) { return distances_.at(i); }
    const T& operator[](size_t i) const { return distances_.at(i); }

    ///
    /// Return the number of distance functors.
    ///
    size_t size() const { return distances_.size(); }

    ///
    /// Resize the number of stored distance functors.
    ///
    void resize(size_t new_size) {
        if (new_size == 0) {
            throw ANNEXCEPTION("Cannot resize stateful distances to zero!");
        }
        distances_.resize(new_size, distances_[0]);
    }

  private:
    std::vector<T> distances_;
};

// Broadcasted distance functions.
// If the distance function is stateless - will only maintain a single copy.
// Otherwise, will generate copies.
template <typename T>
    requires(implicitly_broadcastable<T>())
class BroadcastDistance<T> {
  public:
    BroadcastDistance(T distance, size_t size)
        : size_{size}
        , distance_{std::move(distance)} {}

    T& operator[](size_t /*unused*/) { return distance_; }
    const T& operator[](size_t /*unused*/) const { return distance_; }

    size_t size() const { return size_; }

    void resize(size_t newsize) { size_ = newsize; }

  private:
    size_t size_;
    [[no_unique_address]] T distance_;
};

} // namespace svs::distance
