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
