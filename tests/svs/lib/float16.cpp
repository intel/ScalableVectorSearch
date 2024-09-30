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
#include <type_traits>

#include "svs/lib/float16.h"
#include "svs/lib/narrow.h"

#include "catch2/catch_test_macros.hpp"

CATCH_TEST_CASE("Testing Float16", "[core][float16]") {
    CATCH_SECTION("Implicit Conversion") {
        svs::Float16 x{1.0f};
        float y = x;
        CATCH_REQUIRE(y == 1.0f);

        x = svs::Float16{-1};
        CATCH_REQUIRE(float{x} == -1.0f);

        // Construct from `size_t`
        x = svs::Float16{size_t{100}};
        CATCH_REQUIRE(float{x} == 100.0f);

        // Default Construction.
        CATCH_REQUIRE(svs::Float16{} == svs::Float16(float{0}));
    }

    CATCH_SECTION("Arithmetic") {
        CATCH_REQUIRE(svs::is_arithmetic_v<svs::Float16>);
        CATCH_REQUIRE(svs::is_signed_v<svs::Float16>);

        auto x = svs::Float16{1};
        auto y = svs::Float16{2};
        CATCH_REQUIRE(x + y == 3);
        CATCH_REQUIRE(x != y);
        CATCH_REQUIRE(x < y);
        CATCH_REQUIRE(!(y < x));
        CATCH_REQUIRE(y - x == svs::Float16(1));
    }

    CATCH_SECTION("Narrow") {
        float x_good{1.0f};
        auto y_good = svs::lib::narrow<svs::Float16>(x_good);
        CATCH_REQUIRE(float{y_good} == x_good);

        // Use low precision outside the bounds of `Float16`.
        float x_bad{0.000012f};
        CATCH_REQUIRE_THROWS_AS(
            svs::lib::narrow<svs::Float16>(x_bad), svs::lib::narrowing_error
        );

        // Fail when constructing from typemax integers
        CATCH_REQUIRE_THROWS_AS(
            svs::Float16(std::numeric_limits<size_t>::max() - 1), svs::lib::narrowing_error
        );

        CATCH_REQUIRE_THROWS_AS(
            svs::Float16(std::numeric_limits<int>::max() - int{1}),
            svs::lib::narrowing_error
        );
        CATCH_REQUIRE_THROWS_AS(
            svs::Float16(std::numeric_limits<int>::min() + int{1}),
            svs::lib::narrowing_error
        );
    }
}
