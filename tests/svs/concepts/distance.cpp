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
#include "svs/concepts/distance.h"

// svs-test
#include "tests/utils/require_error.h"

// catch2
#include "catch2/catch_test_macros.hpp"

namespace {
// Comparison operators.
struct CustomComparator : std::less<> {
    CustomComparator(int v)
        : std::less<>{}
        , value{v} {}
    int value;
};

struct HasComparator {
    using compare = CustomComparator;
    compare comparator() const { return CustomComparator(10); }
};

struct NoComparator {
    using compare = std::less<>;
};

// Implicit Broadcasting.
struct ImplicitBroadcast {
    static constexpr bool implicit_broadcast = true;
};

struct NotImplicitBroadcast_Field {
    static constexpr bool implicit_broadcast = false;
};

struct NotImplicitBroadcast_NoField {};

// Mandating fix argument.
struct FixNotRequired {};
struct FixRequiredButNotImplemented {
    static constexpr bool must_fix_argument = true;
};

} // namespace

CATCH_TEST_CASE("Distance Concepts", "[core][distance]") {
    // Static checks
    CATCH_SECTION("Static Checks") {
        CATCH_STATIC_REQUIRE(
            svs::distance::fix_argument_mandated<FixRequiredButNotImplemented>()
        );

        CATCH_STATIC_REQUIRE_FALSE(svs::distance::fix_argument_mandated<FixNotRequired>());
        CATCH_STATIC_REQUIRE(svs::distance::ShouldFix<FixRequiredButNotImplemented, int>);
        CATCH_STATIC_REQUIRE_FALSE(svs::distance::ShouldFix<FixNotRequired, int>);
    }

    CATCH_SECTION("Comparator") {
        static_assert(svs::distance::detail::HasComparator<HasComparator>);
        HasComparator a{};
        CustomComparator cmp_a = svs::distance::comparator(a);
        CATCH_REQUIRE(cmp_a.value == 10);

        NoComparator b{};
        auto cmp_b = svs::distance::comparator(b);
        CATCH_REQUIRE(std::is_same_v<decltype(cmp_b), std::less<>>);
    }

    CATCH_SECTION("Implicit Broadcast") {
        CATCH_REQUIRE(svs::distance::implicitly_broadcastable<ImplicitBroadcast>() == true);
        CATCH_REQUIRE(
            svs::distance::implicitly_broadcastable<NotImplicitBroadcast_Field>() == false
        );
        CATCH_REQUIRE(
            svs::distance::implicitly_broadcastable<NotImplicitBroadcast_NoField>() == false
        );
    }

    CATCH_SECTION("Broadcast Distance") {
        // The types marked as not implicitly broadcastable should have unique addresses
        // for each entry.
        CATCH_SECTION("Not Implicit") {
            auto bcast =
                svs::distance::BroadcastDistance(NotImplicitBroadcast_NoField{}, 2);
            CATCH_REQUIRE(std::addressof(bcast[0]) != std::addressof(bcast[1]));
        }

        CATCH_SECTION("Implicit") {
            auto bcast = svs::distance::BroadcastDistance(ImplicitBroadcast{}, 2);
            CATCH_REQUIRE(std::addressof(bcast[0]) == std::addressof(bcast[1]));
        }
    }
}
