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

// svs
#include "svs/concepts/distance.h"

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

} // namespace

CATCH_TEST_CASE("Distance Concepts", "[core][distance]") {
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
