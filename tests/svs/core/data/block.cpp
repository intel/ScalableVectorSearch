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

// stdlib
#include <span>

// svs
#include "svs/core/data/block.h"

// catch2
#include "catch2/catch_test_macros.hpp"

namespace {
template <size_t Extent = svs::Dynamic> void test_blocked() {
    // Use a small block size so we can test the block bridging logic.
    size_t blocksize_bytes = 4096;
    size_t num_elements = 2000;
    size_t dimensions = 5;

    // Sanity check to prevent future changes from messing up this test.
    if constexpr (Extent != svs::Dynamic) {
        CATCH_REQUIRE(Extent == dimensions);
    }

    size_t expected_blocksize = 128;

    auto data =
        svs::data::BlockedData<float, Extent>(num_elements, dimensions, blocksize_bytes);
    CATCH_REQUIRE(data.dimensions() == 5);
    CATCH_REQUIRE(data.blocksize_bytes().value() == blocksize_bytes);
    CATCH_REQUIRE(data.blocksize().value() == expected_blocksize);
    CATCH_REQUIRE(data.size() == num_elements);

    std::vector<float> values(dimensions);
    CATCH_REQUIRE(values.size() == dimensions);
    auto set_contents = [&]() {
        for (size_t i = 0; i < data.size(); ++i) {
            std::fill(values.begin(), values.end(), i);
            data.set_datum(i, std::span<float, Extent>{values.data(), values.size()});
        }
    };

    auto check_contents = [&](const auto& this_data) {
        for (size_t i : this_data.eachindex()) {
            // Make sure prefetching at least works.
            data.prefetch(i);

            // Make sure that our data assignment was propagated correctly.
            auto datum = data.get_datum(i);
            CATCH_REQUIRE(datum.size() == dimensions);
            CATCH_REQUIRE(std::all_of(datum.begin(), datum.end(), [&](float v) {
                return v == i;
            }));
        }
    };

    set_contents();
    check_contents(data);
    check_contents(data.copy());

    ///// Resizing
    CATCH_REQUIRE(data.num_blocks() == 16);

    // Increase in size
    data.resize(4000);
    CATCH_REQUIRE(data.capacity() > 4000);
    CATCH_REQUIRE(data.num_blocks() == 32);

    set_contents();
    check_contents(data);
    check_contents(data.copy());

    // Decrease in size
    data.resize(2000);
    CATCH_REQUIRE(data.capacity() < 4000);
    CATCH_REQUIRE(data.num_blocks() == 16);
    check_contents(data);
    check_contents(data.copy());
}
} // namespace

CATCH_TEST_CASE("Testing Blocked Data", "[core][data][blocked]") {
    CATCH_SECTION("Basic Functionality") {
        test_blocked();
        test_blocked<5>();
    }
}
