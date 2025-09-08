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
    static_assert(svs::lib::svs_version == svs::lib::Version(0, 0, 10), "Version mismatch!");
    CATCH_REQUIRE(svs::lib::svs_version == svs::lib::Version(0, 0, 10));
}
