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

CATCH_TEST_CASE("Empty Parsing", "[fmt]") {
    auto pt = CustomPoint{1, 2};
    auto str = fmt::format("{}", pt);
    CATCH_REQUIRE(str == "CustomPoint(1, 2)");

    // Use a runtime string to ensure that an error is thrown for a non-empty format string.
    CATCH_REQUIRE_THROWS_AS(fmt::format(fmt::runtime("{:p}"), pt), fmt::format_error);
}
