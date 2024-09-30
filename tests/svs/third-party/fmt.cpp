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
#include "svs/third-party/fmt.h"

// catch2
#include "catch2/catch_test_macros.hpp"

struct CustomPoint {
    int x;
    int y;
};

template <> struct fmt::formatter<CustomPoint> : svs::format_empty {
    auto format(const auto& x, auto& ctx) const {
        return fmt::format_to(ctx.out(), "CustomPoint({}, {})", x.x, x.y);
    }
};

CATCH_TEST_CASE("fmtlib", "[fmt]") {
    CATCH_SECTION("Printing Macros") {
        auto i = 10;
        auto j_ = 20;
        CATCH_REQUIRE(SVS_SHOW_STRING(i) == "i: 10");
        CATCH_REQUIRE(SVS_SHOW_STRING_(j) == "j: 20");
    }

    CATCH_SECTION("Empty Formatting") {
        auto pt = CustomPoint{1, 2};
        auto str = fmt::format("{}", pt);
        CATCH_REQUIRE(str == "CustomPoint(1, 2)");

        // Use a runtime string to ensure that an error is thrown for a non-empty format
        // string.
        CATCH_REQUIRE_THROWS_AS(fmt::format(fmt::runtime("{:p}"), pt), fmt::format_error);
    }
}
