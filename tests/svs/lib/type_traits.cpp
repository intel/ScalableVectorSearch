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
// header under test
#include "svs/lib/type_traits.h"

// catch2
#include "catch2/catch_test_macros.hpp"

CATCH_TEST_CASE("Type Traits", "[lib][type_traits]") {
    // Work around the inability to express the minimum signed 64-bit integer as a literal
    // by using some constexpr arithmetic.
    constexpr int64_t smallest_int64 = 2 * static_cast<int64_t>(-4611686018427387904);

    CATCH_SECTION("Sentinel Less") {
        using LT = std::less<>;
        // Floating Point
        CATCH_STATIC_REQUIRE(svs::type_traits::sentinel_v<float, LT> == 3.4028235E38F);
        CATCH_STATIC_REQUIRE(
            svs::type_traits::sentinel_v<double, LT> == 1.7976931348623157e308
        );
        // Signed Integer
        CATCH_STATIC_REQUIRE(svs::type_traits::sentinel_v<int8_t, LT> == 127);
        CATCH_STATIC_REQUIRE(svs::type_traits::sentinel_v<int16_t, LT> == 32767);
        CATCH_STATIC_REQUIRE(svs::type_traits::sentinel_v<int32_t, LT> == 2147483647);
        CATCH_STATIC_REQUIRE(
            svs::type_traits::sentinel_v<int64_t, LT> == 9223372036854775807
        );
        // Unsigned Integer
        CATCH_STATIC_REQUIRE(svs::type_traits::sentinel_v<uint8_t, LT> == 255);
        CATCH_STATIC_REQUIRE(svs::type_traits::sentinel_v<uint16_t, LT> == 0xFFFF);
        CATCH_STATIC_REQUIRE(svs::type_traits::sentinel_v<uint32_t, LT> == 0xFFFF'FFFF);
        CATCH_STATIC_REQUIRE(
            svs::type_traits::sentinel_v<uint64_t, LT> == 0xFFFF'FFFF'FFFF'FFFF
        );
    }

    CATCH_SECTION("Sentinel Greater") {
        using GT = std::greater<>;
        // Floating Point
        CATCH_STATIC_REQUIRE(svs::type_traits::sentinel_v<float, GT> == -3.4028235E38F);
        CATCH_STATIC_REQUIRE(
            svs::type_traits::sentinel_v<double, GT> == -1.7976931348623157e308
        );
        // Signed Integer
        CATCH_STATIC_REQUIRE(svs::type_traits::sentinel_v<int8_t, GT> == -128);
        CATCH_STATIC_REQUIRE(svs::type_traits::sentinel_v<int16_t, GT> == -32768);
        CATCH_STATIC_REQUIRE(svs::type_traits::sentinel_v<int32_t, GT> == -2147483648);
        CATCH_STATIC_REQUIRE(svs::type_traits::sentinel_v<int64_t, GT> == smallest_int64);
        // Unsigned Integer
        CATCH_STATIC_REQUIRE(svs::type_traits::sentinel_v<uint8_t, GT> == 0);
        CATCH_STATIC_REQUIRE(svs::type_traits::sentinel_v<uint16_t, GT> == 0);
        CATCH_STATIC_REQUIRE(svs::type_traits::sentinel_v<uint32_t, GT> == 0);
        CATCH_STATIC_REQUIRE(svs::type_traits::sentinel_v<uint64_t, GT> == 0);
    }

    CATCH_SECTION("Tombstone Less") {
        using LT = std::less<>;
        // Floating Point
        CATCH_STATIC_REQUIRE(svs::type_traits::tombstone_v<float, LT> == -3.4028235E38F);
        CATCH_STATIC_REQUIRE(
            svs::type_traits::tombstone_v<double, LT> == -1.7976931348623157e308
        );
        // Signed Integer
        CATCH_STATIC_REQUIRE(svs::type_traits::tombstone_v<int8_t, LT> == -128);
        CATCH_STATIC_REQUIRE(svs::type_traits::tombstone_v<int16_t, LT> == -32768);
        CATCH_STATIC_REQUIRE(svs::type_traits::tombstone_v<int32_t, LT> == -2147483648);
        CATCH_STATIC_REQUIRE(svs::type_traits::tombstone_v<int64_t, LT> == smallest_int64);
        // Unsigned Integer
        CATCH_STATIC_REQUIRE(svs::type_traits::tombstone_v<uint8_t, LT> == 0);
        CATCH_STATIC_REQUIRE(svs::type_traits::tombstone_v<uint16_t, LT> == 0);
        CATCH_STATIC_REQUIRE(svs::type_traits::tombstone_v<uint32_t, LT> == 0);
        CATCH_STATIC_REQUIRE(svs::type_traits::tombstone_v<uint64_t, LT> == 0);
    }

    CATCH_SECTION("Tombstone Greater") {
        using GT = std::greater<>;
        // Floating Point
        CATCH_STATIC_REQUIRE(svs::type_traits::tombstone_v<float, GT> == 3.4028235E38F);
        CATCH_STATIC_REQUIRE(
            svs::type_traits::tombstone_v<double, GT> == 1.7976931348623157e308
        );
        // Signed Integer
        CATCH_STATIC_REQUIRE(svs::type_traits::tombstone_v<int8_t, GT> == 127);
        CATCH_STATIC_REQUIRE(svs::type_traits::tombstone_v<int16_t, GT> == 32767);
        CATCH_STATIC_REQUIRE(svs::type_traits::tombstone_v<int32_t, GT> == 2147483647);
        CATCH_STATIC_REQUIRE(
            svs::type_traits::tombstone_v<int64_t, GT> == 9223372036854775807
        );
        // Unsigned Integer
        CATCH_STATIC_REQUIRE(svs::type_traits::tombstone_v<uint8_t, GT> == 255);
        CATCH_STATIC_REQUIRE(svs::type_traits::tombstone_v<uint16_t, GT> == 0xFFFF);
        CATCH_STATIC_REQUIRE(svs::type_traits::tombstone_v<uint32_t, GT> == 0xFFFF'FFFF);
        CATCH_STATIC_REQUIRE(
            svs::type_traits::tombstone_v<uint64_t, GT> == 0xFFFF'FFFF'FFFF'FFFF
        );
    }
}
