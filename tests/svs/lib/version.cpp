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
    static_assert(svs::lib::svs_version == svs::lib::Version(0, 0, 1), "Version mismatch!");
    CATCH_REQUIRE(svs::lib::svs_version == svs::lib::Version(0, 0, 1));
}
