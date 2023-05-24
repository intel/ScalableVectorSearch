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
#include "svs/lib/algorithms.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <vector>

CATCH_TEST_CASE("Algorithms", "[lib][lib-algorithms]") {
    CATCH_SECTION("All unique") {
        auto x = std::vector<int>{0, 2, 5, 100, 4, 99};
        CATCH_REQUIRE(svs::lib::all_unique(x.begin(), x.end()));
        x.push_back(2);
        CATCH_REQUIRE(!svs::lib::all_unique(x.begin(), x.end()));
    }
}
