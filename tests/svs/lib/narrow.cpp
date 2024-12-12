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

// stdlib
#include <limits>

// svs
#include "svs/lib/narrow.h"

// catch2
#include "catch2/catch_test_macros.hpp"

CATCH_TEST_CASE("Narrow", "[testing_utilities]") {
    // Test int32_t to int8_t conversion.
    CATCH_SECTION("Int32 to Int8") {
        for (int32_t i = -128; i < 128; ++i) {
            CATCH_REQUIRE(svs::lib::narrow<int8_t>(i) == int8_t(i));
        }
    }

    // Test uint32_t to uint8_t conversion.
    CATCH_SECTION("UInt32 to UInt8") {
        for (uint32_t i = 0; i < 256; ++i) {
            CATCH_REQUIRE(svs::lib::narrow<uint8_t>(i) == uint8_t(i));
        }
    }

    // Make sure we actually get failures.
    CATCH_REQUIRE_THROWS_AS(
        svs::lib::narrow<int8_t>(int32_t(-129)), svs::lib::narrowing_error
    );

    CATCH_REQUIRE_THROWS_AS(
        svs::lib::narrow<int8_t>(int32_t(128)), svs::lib::narrowing_error
    );

    CATCH_REQUIRE_THROWS_AS(
        svs::lib::narrow<uint8_t>(uint32_t(256)), svs::lib::narrowing_error
    );

    ///// Floating Point conversions.
    CATCH_REQUIRE(svs::lib::narrow<float>(1.0) == 1.0f);

    // The following fails on Clang with optimizations turned on unless `narrow` is
    // marked as `[[gnu::noinline]]` or otherwise disables compile-time propagation of
    // values.
    //
    // This is because technically, the underlying `static_cast` is undefined, which
    // clang uses to optimize out the throwing branch of `narrow`.
    CATCH_REQUIRE_THROWS_AS(
        svs::lib::narrow<float>(std::numeric_limits<size_t>::max() - 1),
        svs::lib::narrowing_error
    );
}
