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
#include "svs/lib/version.h"

// catch2
#include "catch2/catch_test_macros.hpp"

CATCH_TEST_CASE("Version Numbers", "[lib][versions]") {
    namespace lib = svs::lib;
    constexpr auto v = lib::Version(0, 2, 4);
    auto x = lib::Version(0, 2, 4);
    CATCH_REQUIRE(x == v);
    CATCH_REQUIRE(!(x < v));

    auto str = v.str();
    CATCH_REQUIRE(str == "v0.2.4");
    auto u = lib::Version(str);
    CATCH_REQUIRE(u == v);
    CATCH_REQUIRE(lib::Version("v10.20.355534") == lib::Version(10, 20, 355534));

    // Comparison.
    auto cmp = [](const lib::Version& left, const lib::Version& right) {
        CATCH_REQUIRE(left < right);
        CATCH_REQUIRE(!(right < left));
    };

    cmp(lib::Version(10, 20, 30), lib::Version(11, 20, 30));
    cmp(lib::Version(10, 20, 30), lib::Version(10, 21, 30));
    cmp(lib::Version(10, 20, 30), lib::Version(10, 20, 31));
}

// Keep in-sync with CMakeLists.txt
CATCH_TEST_CASE("Global Version", "[lib][versions]") {
    static_assert(svs::lib::svs_version == svs::lib::Version(0, 0, 4), "Version mismatch!");
    CATCH_REQUIRE(svs::lib::svs_version == svs::lib::Version(0, 0, 4));
}
