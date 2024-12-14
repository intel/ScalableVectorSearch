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

// svs
#include "svs/lib/preprocessor.h"

// stl
#include <cassert>
#include <type_traits>

namespace svs::lib {

/////
///// ScopeGuard
/////

namespace detail {

// A dummy type used for non-cancelable scope guards providing a similar interface to a
// boolean.
//
// This type should be empty to avoid increasing the memory footprint of non-cancelable
// guards.
struct AlwaysFalse {
    constexpr AlwaysFalse() = default;
    constexpr AlwaysFalse([[maybe_unused]] bool init) {
        // `AlwaysFalse` should never be initialized to `true`.
        assert(init == false);
    }

    constexpr AlwaysFalse& operator=([[maybe_unused]] bool value) {
        // `AlwaysFalse` should never be assigned to `true`.
        assert(value == false);
        return *this;
    }
    constexpr operator bool() { return false; }
};

} // namespace detail

///
/// @brief Scope guard that invokes its callback's call operator upon destruction.
///
/// This is a class that provides a RAII style callback/cleanup mechanism at the end of
/// a scoped block.
///
/// When this object's desctructor is run, it will invoke
/// ```c++
/// std::remove_reference_t<T>::operator()()
/// ```
/// of its contained callback.
///
/// Expects the callback operator to be ``noexcept`` and have a ``void`` return type.
///
/// This class is non-copyable and non-moveable.
///
/// The utility function ``svs::lib::make_scope_guard`` and
/// ``svs::lib::make_dismissable_scope_guard`` can be used for type deduction when
/// creating a ScopeGuard. In the latter case, the created ``ScopeGuard`` can be dismissed
/// using the ``.dismiss()`` method. Dismissed ``ScopeGuard``s will not run the deferred
/// callable.
///
/// ```c++
/// size_t count = 0;
/// {
///     auto guard = svs::lib::make_scope_guard([&count]() noexcept {
///         ++count;
///     });
/// }
/// assert(count == 1);
///
/// {
///     auto guard = svs::lib::make_dismissable_scope_guard([&count]() noexcept {
///         ++count;
///     });
/// }
/// assert(count == 2);
///
/// {
///     auto guard = svs::lib::make_dismissable_scope_guard([&count]() noexcept {
///         ++count;
///     });
///     guard.dismiss();
/// }
/// // Captured function was never run.
/// assert(count == 2);
/// ```
///
/// @tparam T The type of the callback operator. May be a reference type if the ScopeGuard
///     was constructed from an l-value refences to the provided callback.
/// @tparam Dismissable Boolean value parameter indicating whether or not this ScopeGuard
///     is dismissable.
///
/// @sa ``svs::lib::make_scope_guard``, ``svs::lib::make_dismissable_scope_guard``.
///
template <typename T, bool Dismissable> class ScopeGuard {
  public:
    /// Definition: T
    using callback_type = T;
    static constexpr bool is_reference = std::is_reference_v<std::remove_const_t<T>>;
    using dismissed_type = std::conditional_t<Dismissable, bool, detail::AlwaysFalse>;

    static_assert(
        std::is_nothrow_invocable_v<callback_type>, "ScopeGuard callable must be noexcept."
    );

    static_assert(std::is_void_v<std::invoke_result_t<callback_type>>);

    /// ScopeGuard is not default constructible.
    ScopeGuard() = delete;

    ///
    /// @brief Create a ScopeGuard for the given callback.
    ///
    /// Implementation notes:
    ///
    /// If the callback is a reference type, then C++ reference collapsing yields
    /// ```
    /// callback_type&& -> T& && -> T&
    /// ```
    /// which turns the argument `f` to a reference.
    ///
    /// If the callback is a non-reference type, then we have an r-value reference as
    /// expected.
    ///
    explicit ScopeGuard(callback_type&& f)
        : f_{SVS_FWD(f)} {}

    /// @brief Dismiss the ScopeGuard so the callable is not invoked.
    ///
    /// Requires the ScopeGuard to be dismissable.
    void dismiss() noexcept
        requires Dismissable
    {
        dismissed_ = true;
    }

    // Disable the special member functions.
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
    ScopeGuard(ScopeGuard&&) = delete;
    ScopeGuard& operator=(ScopeGuard&&) = delete;

    // Invoke the callback on destruction.
    ~ScopeGuard() {
        if (!dismissed_) {
            f_();
        }
    }

    ///// Members
  private:
    T f_;
    [[no_unique_address]] dismissed_type dismissed_{false};
};

namespace detail {
template <typename T>
using deduce_scopeguard_parameter_t =
    std::conditional_t<std::is_rvalue_reference_v<T>, std::decay_t<T>, T>;
}

///
/// @brief Construct an ScopeGuard wrapped around the argument ``f``.
///
/// Constructs ScopeGuard around the argument without invoking the object's copy
/// constructor.
///
/// * If ``f`` is given as an l-value reference, the constructed ScopeGuard will reference
///   ``f``.
/// * If `f` is given as an r-value reference, then the constructed ScopeGuard will take
///   ownership of `f` via `f`'s move constructor.
///
/// This latter requires ``f`` to be noexcept move constructible.
///
template <typename F>
ScopeGuard<detail::deduce_scopeguard_parameter_t<F&&>, false> make_scope_guard(F&& f) {
    return ScopeGuard<detail::deduce_scopeguard_parameter_t<F&&>, false>(SVS_FWD(f));
}

///
/// @brief Construct an active but dismissable ScopeGuard wrapped around the argument ``f``.
///
/// Constructs ScopeGuard around the argument without invoking the object's copy
/// constructor.
///
/// * If ``f`` is given as an l-value reference, the constructed ScopeGuard will reference
///   ``f``.
/// * If `f` is given as an r-value reference, then the constructed ScopeGuard will take
///   ownership of `f` via `f`'s move constructor.
///
/// This latter requires ``f`` to be noexcept move constructible.
///
template <typename F>
ScopeGuard<detail::deduce_scopeguard_parameter_t<F&&>, true>
make_dismissable_scope_guard(F&& f) {
    return ScopeGuard<detail::deduce_scopeguard_parameter_t<F&&>, true>(SVS_FWD(f));
}

} // namespace svs::lib
