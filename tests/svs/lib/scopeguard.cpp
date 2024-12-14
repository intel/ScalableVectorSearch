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

// svs
#include "svs/lib/scopeguard.h"
#include "svs/third-party/fmt.h"

// catch2
#include "catch2/catch_template_test_macros.hpp"
#include "catch2/catch_test_macros.hpp"

// stl
#include <utility>

///// Test Setup
namespace {

struct ScopeGuardCallback {
  public:
    ScopeGuardCallback() = default;
    ScopeGuardCallback(size_t* external_calls)
        : external_calls_{external_calls} {}

    // Disable copy constructors.
    ScopeGuardCallback(const ScopeGuardCallback&) = delete;
    ScopeGuardCallback& operator=(const ScopeGuardCallback&) = delete;

    ScopeGuardCallback(ScopeGuardCallback&&) = default;
    ScopeGuardCallback& operator=(ScopeGuardCallback&&) = default;
    ~ScopeGuardCallback() = default;

    void operator()() noexcept {
        ++local_calls_;
        if (external_calls_ != nullptr) {
            ++(*external_calls_);
        }
    }

    // The const version only increments the global count (if one exists)
    void operator()() const noexcept {
        if (external_calls_ != nullptr) {
            ++(*external_calls_);
        }
    }

    ///// Members
    size_t local_calls_ = 0;
    size_t* external_calls_ = nullptr;
};

} // namespace

CATCH_TEST_CASE("ScopeGuard static ssserts", "[lib][scopeguard]") {
    CATCH_STATIC_REQUIRE(std::is_same_v<
                         svs::lib::detail::deduce_scopeguard_parameter_t<size_t&&>,
                         size_t>);
    CATCH_STATIC_REQUIRE(std::is_same_v<
                         svs::lib::detail::deduce_scopeguard_parameter_t<size_t&>,
                         size_t&>);
    CATCH_STATIC_REQUIRE(std::is_same_v<
                         svs::lib::detail::deduce_scopeguard_parameter_t<const size_t&>,
                         const size_t&>);
}

CATCH_TEMPLATE_TEST_CASE_SIG(
    "ScopeGuard", "[lib][scopeguard]", ((bool Dismissable), Dismissable), false, true
) {
    auto make_guard = [](auto&& x) {
        if constexpr (Dismissable) {
            return svs::lib::make_dismissable_scope_guard(SVS_FWD(x));
        } else {
            return svs::lib::make_scope_guard(SVS_FWD(x));
        }
    };

    CATCH_SECTION("Static Checks") {
        using T = svs::lib::ScopeGuard<ScopeGuardCallback, Dismissable>;
        static_assert(
            !std::is_copy_constructible_v<T>, "Expected copy constructor to be deleted!"
        );
        static_assert(
            !std::is_move_constructible_v<T>, "Expected move constructor to be deleted!"
        );
        static_assert(
            !std::is_copy_assignable_v<T>,
            "Expected copy assignment operator to be deleted!"
        );
        static_assert(
            !std::is_move_assignable_v<T>,
            "Expected move assignment operator to be deleted!"
        );

        // Trivial constructor should be deleted.
        static_assert(
            !std::is_default_constructible_v<T>,
            "ScopeGuard should not be default constructible!"
        );

        // Cannot construct a non-reference scope-guard from an lvalue reference.
        static_assert(
            !std::is_constructible_v<T, ScopeGuardCallback&>,
            "ScopeGuard should not attempt to copy its arguments!"
        );
        static_assert(
            !std::is_constructible_v<T, const ScopeGuardCallback&>,
            "ScopeGuard should not attempt to copy its arguments!"
        );
    }

    CATCH_SECTION("Basic Behavior") {
        size_t global = 0;
        auto x = ScopeGuardCallback(&global);
        CATCH_REQUIRE(x.local_calls_ == 0);
        CATCH_REQUIRE(global == 0);
        { auto g = svs::lib::ScopeGuard<ScopeGuardCallback&, Dismissable>(x); }
        CATCH_REQUIRE(x.local_calls_ == 1);
        CATCH_REQUIRE(global == 1);

        if constexpr (Dismissable) {
            // Dismissing the scope guard should have no effect.
            {
                auto g = svs::lib::ScopeGuard<ScopeGuardCallback&, Dismissable>(x);
                g.dismiss();
            }
            CATCH_REQUIRE(x.local_calls_ == 1);
            CATCH_REQUIRE(global == 1);
        }

        // Mutable Reference
        {
            auto g = make_guard(x);
            using gT = decltype(g);
            static_assert(std::is_same_v<typename gT::callback_type, ScopeGuardCallback&>);
            static_assert(gT::is_reference);
            // Make sure we don't have a space overhead for non-Dismissable scope guards
            if constexpr (!Dismissable) {
                static_assert(sizeof(gT) == sizeof(void*));
            } else {
                static_assert(sizeof(gT) > sizeof(void*));
            }
        }
        CATCH_REQUIRE(x.local_calls_ == 2);
        CATCH_REQUIRE(global == 2);

        // If dismissable - ensure that dismissing works.
        if constexpr (Dismissable) {
            {
                auto g = make_guard(x);
                g.dismiss();
            }
            CATCH_REQUIRE(x.local_calls_ == 2);
            CATCH_REQUIRE(global == 2);
        }

        // Const Reference
        {
            auto g = make_guard(std::as_const(x));
            using gT = decltype(g);
            static_assert(std::is_same_v<
                          typename gT::callback_type,
                          const ScopeGuardCallback&>);
            static_assert(gT::is_reference);
            if constexpr (!Dismissable) {
                static_assert(sizeof(gT) == sizeof(void*));
            } else {
                static_assert(sizeof(gT) > sizeof(void*));
            }
        }
        CATCH_REQUIRE(x.local_calls_ == 2);
        CATCH_REQUIRE(global == 3);

        // If dismissable - ensure that dismissing works.
        if constexpr (Dismissable) {
            {
                auto g = make_guard(std::as_const(x));
                g.dismiss();
            }
            CATCH_REQUIRE(x.local_calls_ == 2);
            CATCH_REQUIRE(global == 3);
        }

        // Rvalue Reference
        {
            auto g = make_guard(ScopeGuardCallback(&global));
            using gT = decltype(g);
            static_assert(std::is_same_v<typename gT::callback_type, ScopeGuardCallback>);
            static_assert(!gT::is_reference);
            if constexpr (!Dismissable) {
                static_assert(sizeof(gT) == sizeof(ScopeGuardCallback));
            } else {
                static_assert(sizeof(gT) > sizeof(ScopeGuardCallback));
            }
        }
        CATCH_REQUIRE(global == 4);

        if constexpr (Dismissable) {
            {
                auto g = make_guard(ScopeGuardCallback(&global));
                g.dismiss();
            }
            CATCH_REQUIRE(global == 4);
        }
    }
}
