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
#include "svs/core/data/simple.h"

// svs
#include "svs/core/data.h"

// test utilities
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stdlib
#include <span>

namespace {

template <typename T> bool is_blocked(const T&) { return false; }
template <typename T, size_t N> bool is_blocked(const svs::data::BlockedData<T, N>&) {
    return true;
}

template <typename Left, typename Right>
bool data_equal(const Left& left, const Right& right) {
    auto lsize = left.size();
    auto ldims = left.dimensions();

    auto rsize = right.size();
    auto rdims = right.dimensions();

    if (lsize != rsize || ldims != rdims) {
        return false;
    }

    for (size_t i = 0; i < lsize; ++i) {
        auto l = left.get_datum(i);
        auto r = right.get_datum(i);
        if (!std::equal(l.begin(), l.end(), r.begin())) {
            return false;
        }
    }
    return true;
}

template <size_t Extent = svs::Dynamic> void test_blocked() {
    // Use a small block size so we can test the block bridging logic.
    size_t blocksize_bytes = 4096;
    size_t num_elements = 2000;
    size_t dimensions = 5;

    // Sanity check to prevent future changes from messing up this test.
    if constexpr (Extent != svs::Dynamic) {
        CATCH_REQUIRE(Extent == dimensions);
    }
    CATCH_REQUIRE(!is_blocked(10));

    size_t expected_blocksize = 128;

    auto parameters = svs::data::BlockingParameters{
        .blocksize_bytes = svs::lib::prevpow2(blocksize_bytes)};
    auto allocator = svs::data::Blocked<svs::lib::Allocator<float>>(parameters);
    auto data = svs::data::BlockedData<float, Extent>(num_elements, dimensions, allocator);
    CATCH_REQUIRE(is_blocked(data));
    CATCH_REQUIRE(data.dimensions() == 5);
    CATCH_REQUIRE(data.blocksize_bytes().value() == blocksize_bytes);
    CATCH_REQUIRE(data.blocksize().value() == expected_blocksize);
    CATCH_REQUIRE(data.size() == num_elements);

    auto set_contents = [dimensions](auto& data) {
        std::vector<float> values(dimensions);
        for (size_t i = 0; i < data.size(); ++i) {
            std::fill(values.begin(), values.end(), i);
            data.set_datum(i, std::span<float, Extent>{values.data(), values.size()});
        }
    };

    auto check_contents = [dimensions](const auto& this_data) {
        for (size_t i : this_data.eachindex()) {
            // Make sure prefetching at least works.
            this_data.prefetch(i);

            // Make sure that our data assignment was propagated correctly.
            auto datum = this_data.get_datum(i);
            CATCH_REQUIRE(datum.size() == dimensions);
            CATCH_REQUIRE(std::all_of(datum.begin(), datum.end(), [&](float v) {
                return v == i;
            }));
        }
    };

    set_contents(data);
    check_contents(data);
    auto copy = data.copy();
    check_contents(data.copy());
    CATCH_REQUIRE(is_blocked(copy));
    CATCH_REQUIRE(data_equal(data, data.copy()));

    ///// Resizing
    CATCH_REQUIRE(data.num_blocks() == 16);

    // Increase in size
    data.resize(4000);
    CATCH_REQUIRE(data.capacity() > 4000);
    CATCH_REQUIRE(data.num_blocks() == 32);

    set_contents(data);
    check_contents(data);
    check_contents(data.copy());

    // Decrease in size
    data.resize(2000);
    CATCH_REQUIRE(data.capacity() < 4000);
    CATCH_REQUIRE(data.num_blocks() == 16);
    check_contents(data);
    check_contents(data.copy());

    ///// Saving and Loading.
    svs_test::prepare_temp_directory();
    auto temp = svs_test::temp_directory();
    svs::lib::save_to_disk(data, temp);
    auto simple_data = svs::VectorDataLoader<float>(temp).load();
    check_contents(simple_data);
    CATCH_REQUIRE(!is_blocked(simple_data));
    CATCH_REQUIRE(data_equal(simple_data, data));

    // Reload as a blocked dataset.
    auto reloaded = svs::lib::load_from_disk<svs::data::BlockedData<float>>(temp);
    check_contents(reloaded);
    CATCH_REQUIRE(is_blocked(reloaded));
    CATCH_REQUIRE(data_equal(reloaded, data));
}
} // namespace

CATCH_TEST_CASE("Testing Blocked Data", "[core][data][blocked]") {
    CATCH_SECTION("BlockingParameters") {
        using T = svs::data::BlockingParameters;
        auto p = T{};
        CATCH_REQUIRE(p.blocksize_bytes == T::default_blocksize_bytes);

        p = T{.blocksize_bytes = svs::lib::PowerOfTwo(10)};
        CATCH_REQUIRE(p.blocksize_bytes == svs::lib::PowerOfTwo(10));
    }

    CATCH_SECTION("Blocked Allocator") {
        // Use an integer for the "allocator" to test value propagation.
        // Since the `Blocked` class doesn't actually use the allocator, this is okay
        // for functionality testing.
        using T = svs::data::Blocked<int>;
        using P = svs::data::BlockingParameters;
        auto x = T();
        CATCH_REQUIRE(x.get_allocator() == 0); // Default constructed integer.
        CATCH_REQUIRE(x.parameters() == P{});

        x = T(10);
        CATCH_REQUIRE(x.get_allocator() == 10);
        CATCH_REQUIRE(x.parameters() == P{});

        auto p = P{.blocksize_bytes = svs::lib::PowerOfTwo(10)};
        x = T(p);
        CATCH_REQUIRE(x.get_allocator() == 0);
        CATCH_REQUIRE(x.parameters() == p);

        x = T(p, 10);
        CATCH_REQUIRE(x.get_allocator() == 10);
        CATCH_REQUIRE(x.parameters() == p);
    }

    CATCH_SECTION("Basic Functionality") {
        test_blocked();
        test_blocked<5>();
    }
}
