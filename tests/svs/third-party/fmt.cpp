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
