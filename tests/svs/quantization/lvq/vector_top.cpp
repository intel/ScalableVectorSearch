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

// Header under test
#include "svs/quantization/lvq/lvq.h"

// Catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <limits>

namespace lvq = svs::quantization::lvq;

CATCH_TEST_CASE("Vector Quantization Top", "[quantization][lvq]") {
    CATCH_SECTION("GlobalMinMax") {
        auto extrema = lvq::GlobalMinMax();

        // Ensure correct default construction.
        CATCH_REQUIRE(extrema.min() == std::numeric_limits<float>::max());
        CATCH_REQUIRE(extrema.max() == std::numeric_limits<float>::lowest());

        extrema.update(0);
        CATCH_REQUIRE(extrema.min() == 0);
        CATCH_REQUIRE(extrema.max() == 0);

        extrema.update(10);
        CATCH_REQUIRE(extrema.min() == 0);
        CATCH_REQUIRE(extrema.max() == 10);

        extrema.update(-10);
        CATCH_REQUIRE(extrema.min() == -10);
        CATCH_REQUIRE(extrema.max() == 10);

        extrema.update(5);
        CATCH_REQUIRE(extrema.min() == -10);
        CATCH_REQUIRE(extrema.max() == 10);
    }
}
