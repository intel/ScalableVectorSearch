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

// stdlib
#include <type_traits>
#include <vector>

// svs
#include "svs/lib/meta.h"

// catch2
#include "catch2/catch_test_macros.hpp"

CATCH_TEST_CASE("Val", "[core]") {
    CATCH_SECTION("Value Extracting") {
        auto a = svs::meta::Val<1>{};
        CATCH_REQUIRE(a.value == 1);
        CATCH_REQUIRE(svs::meta::Val<1>::value == 1);
    }

    // Unwrapping.
    CATCH_SECTION("Unwrapping") {
        namespace meta = svs::meta;
        using DataType = svs::DataType;

        CATCH_REQUIRE(meta::unwrap(meta::Val<10>{}) == 10);
        auto d = meta::Val<100>{};
        CATCH_REQUIRE(meta::unwrap(d) == 100);

        // Equality
        CATCH_REQUIRE(meta::Val<100>{} == meta::Val<100>{});
        CATCH_REQUIRE(meta::Val<100>{} != meta::Val<101>{});

        // Integers.
        size_t x = 10;
        CATCH_REQUIRE(meta::unwrap(x) == 10);
        int y = 100;
        CATCH_REQUIRE(meta::unwrap(y) == 100);

        // Data Types
        {
            CATCH_REQUIRE(meta::unwrap(meta::Type<svs::Float16>()) == DataType::float16);
            CATCH_REQUIRE(meta::unwrap(meta::Type<float>()) == DataType::float32);
            CATCH_REQUIRE(meta::unwrap(meta::Type<double>()) == DataType::float64);
        }

        // Tuples
        {
            auto t = std::make_tuple(100, meta::Val<5>());
            CATCH_REQUIRE(meta::unwrap(t) == std::make_tuple(100, 5));

            auto u = std::make_tuple(meta::Type<float>(), meta::Val<4>(), 5);
            CATCH_REQUIRE(meta::unwrap(u) == std::make_tuple(DataType::float32, 4, 5));
        }
    }

    // Types
    CATCH_SECTION("Type checking") {
        constexpr auto types = svs::meta::Types<float, uint8_t, int64_t>();
        constexpr bool a = svs::meta::in<float>(types);
        CATCH_REQUIRE(a);
        constexpr bool b = svs::meta::in<uint8_t>(types);
        CATCH_REQUIRE(b);
        constexpr bool c = svs::meta::in<int64_t>(types);
        CATCH_REQUIRE(c);
        constexpr bool d = svs::meta::in<double>(types);
        CATCH_REQUIRE(!d);
    }

    CATCH_SECTION("Dynamic Type Checking") {
        using DataType = svs::DataType;
        constexpr auto types = svs::meta::Types<float, uint8_t, int64_t>();
        constexpr bool a = svs::meta::in(DataType::float32, types);
        CATCH_REQUIRE(a);
        constexpr bool b = svs::meta::in(DataType::uint8, types);
        CATCH_REQUIRE(b);
        constexpr bool c = svs::meta::in(DataType::int64, types);
        CATCH_REQUIRE(c);
        constexpr bool d = svs::meta::in(DataType::float64, types);
        CATCH_REQUIRE(!d);
    }

    // Extent Forwarding
    CATCH_SECTION("Extent Forwarding") {
        CATCH_REQUIRE_THROWS_AS(svs::meta::forward_extent<0>(100), svs::ANNException);
        CATCH_REQUIRE(svs::meta::forward_extent<100>(100) == svs::meta::Val<100>{});

        CATCH_REQUIRE(svs::meta::forward_extent<svs::Dynamic>(0) == 0);
        CATCH_REQUIRE(svs::meta::forward_extent<svs::Dynamic>(10) == 10);
    }

    // Is Val
    CATCH_SECTION("Is Val") {
        CATCH_REQUIRE(svs::meta::is_val_type_v<size_t> == false);
        CATCH_REQUIRE(svs::meta::is_val_type_v<svs::meta::Val<2>> == true);
        CATCH_REQUIRE(svs::meta::is_val_type_v<svs::meta::Val<100>> == true);
    }
}
