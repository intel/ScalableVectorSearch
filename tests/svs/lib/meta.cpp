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
        auto a = svs::lib::Val<1>{};
        CATCH_REQUIRE(a.value == 1);
        CATCH_REQUIRE(svs::lib::Val<1>::value == 1);
    }

    // Types
    CATCH_SECTION("Type checking") {
        constexpr auto types = svs::lib::Types<float, uint8_t, int64_t>();
        constexpr bool a = svs::lib::in<float>(types);
        CATCH_REQUIRE(a);
        constexpr bool b = svs::lib::in<uint8_t>(types);
        CATCH_REQUIRE(b);
        constexpr bool c = svs::lib::in<int64_t>(types);
        CATCH_REQUIRE(c);
        constexpr bool d = svs::lib::in<double>(types);
        CATCH_REQUIRE(!d);
    }

    CATCH_SECTION("Dynamic Type Checking") {
        using DataType = svs::DataType;
        constexpr auto types = svs::lib::Types<float, uint8_t, int64_t>();
        constexpr bool a = svs::lib::in(DataType::float32, types);
        CATCH_REQUIRE(a);
        constexpr bool b = svs::lib::in(DataType::uint8, types);
        CATCH_REQUIRE(b);
        constexpr bool c = svs::lib::in(DataType::int64, types);
        CATCH_REQUIRE(c);
        constexpr bool d = svs::lib::in(DataType::float64, types);
        CATCH_REQUIRE(!d);
    }

    // Extent Forwarding
    CATCH_SECTION("Extent Forwarding") {
        CATCH_REQUIRE_THROWS_AS(svs::lib::forward_extent<0>(100), svs::ANNException);
        CATCH_REQUIRE(svs::lib::forward_extent<100>(100) == svs::lib::Val<100>{});

        CATCH_REQUIRE(svs::lib::forward_extent<svs::Dynamic>(0) == 0);
        CATCH_REQUIRE(svs::lib::forward_extent<svs::Dynamic>(10) == 10);
    }

    // Is Val
    CATCH_SECTION("Is Val") {
        CATCH_REQUIRE(svs::lib::is_val_type_v<size_t> == false);
        CATCH_REQUIRE(svs::lib::is_val_type_v<svs::lib::Val<2>> == true);
        CATCH_REQUIRE(svs::lib::is_val_type_v<svs::lib::Val<100>> == true);
    }

    CATCH_SECTION("As Integral") {
        CATCH_STATIC_REQUIRE(svs::lib::as_integral<10>() == 10);
        CATCH_STATIC_REQUIRE(svs::lib::as_integral<svs::lib::Val<20>>() == 20);

        CATCH_REQUIRE(svs::lib::as_integral(10) == 10);
        CATCH_REQUIRE(svs::lib::as_integral(svs::lib::Val<10>()) == 10);
    }

    // generate_typename
    CATCH_SECTION("Generate Typename") {
        constexpr auto name = svs::lib::generate_typename<int64_t>();
        // Check for null-temination.
        CATCH_STATIC_REQUIRE(name.back() == '\0');
        auto v = std::string_view(name.data());
        auto u = std::string_view(name.begin(), name.end() - 1);
        CATCH_REQUIRE(v == u);
        CATCH_REQUIRE(v.find("long") != std::string_view::npos);
    }
}
