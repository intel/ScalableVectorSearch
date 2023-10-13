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

// Header under test.
#include "svs/core/compact.h"

// Utility headers
#include "svs/core/data/simple.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <vector>

namespace {
template <typename Data> void sequential_fill(Data& data) {
    using T = typename Data::element_type;
    for (size_t i = 0, imax = data.size(); i < imax; ++i) {
        auto datum = data.get_datum(i);
        std::fill(datum.begin(), datum.end(), svs::lib::narrow_cast<T>(i));
    }
}

template <typename T, typename U> bool check_line(std::span<T> data, U x) {
    return std::all_of(data.begin(), data.end(), [&](const auto& y) { return y == x; });
}

template <typename Data> bool check_sequential(const Data& data) {
    for (size_t i = 0, imax = data.size(); i < imax; ++i) {
        if (!check_line(data.get_datum(i), i)) {
            return false;
        }
    }
    return true;
}
} // namespace

CATCH_TEST_CASE("Simple Data Compaction", "[core][compaction]") {
    CATCH_SECTION("Simple Threaded") {
        auto data = svs::data::SimpleData<uint32_t>(10, 20);
        auto buffer = svs::data::SimpleData<uint32_t>(2, 20);

        sequential_fill(data);
        CATCH_REQUIRE(check_sequential(data));
        // Make sure the contents of "data" was initialized correctly.
        for (size_t i = 0, imax = data.size(); i < imax; ++i) {
            CATCH_REQUIRE(check_line(data.get_datum(i), i));
        }

        // Initially test with a sequential thread pool.
        auto pool = svs::threads::SequentialThreadPool();
        auto new_to_old = std::vector<size_t>({0, 2, 4, 5, 8, 9});
        svs::compact_data(data, buffer, new_to_old, pool);
        for (size_t i = 0, imax = new_to_old.size(); i < imax; ++i) {
            auto val = new_to_old.at(i);
            CATCH_REQUIRE(check_line(data.get_datum(i), val));
        }

        // Reset and go again, this time with two threads.
        sequential_fill(data);
        CATCH_REQUIRE(check_sequential(data));
        auto tpool = svs::threads::NativeThreadPool(2);
        svs::compact_data(data, buffer, new_to_old, tpool);
        for (size_t i = 0, imax = new_to_old.size(); i < imax; ++i) {
            auto val = new_to_old.at(i);
            CATCH_REQUIRE(check_line(data.get_datum(i), val));
        }

        // Make sure we get an error if we use the wrong-sized buffer.
        buffer = svs::data::SimpleData<uint32_t>(4, 100);
        CATCH_REQUIRE_THROWS_AS(
            svs::compact_data(data, buffer, new_to_old, tpool), svs::ANNException
        );
    }

    CATCH_SECTION("Blocked Data") {
        auto data = svs::data::BlockedData<float>(100, 20);
        sequential_fill(data);
        CATCH_REQUIRE(check_sequential(data));

        auto new_to_old = std::vector<uint32_t>{};
        for (size_t i = 0, imax = data.size() / 3; i < imax; ++i) {
            new_to_old.push_back(3 * i);
        }

        // Single-threaded version
        data.compact(new_to_old, 20);
        for (size_t i = 0, imax = new_to_old.size(); i < imax; ++i) {
            auto val = new_to_old.at(i);
            CATCH_REQUIRE(check_line(data.get_datum(i), val));
        }

        // Multi-threaded version.
        sequential_fill(data);
        CATCH_REQUIRE(check_sequential(data));
        auto tpool = svs::threads::NativeThreadPool(2);
        data.compact(new_to_old, tpool, 20);
        for (size_t i = 0, imax = new_to_old.size(); i < imax; ++i) {
            auto val = new_to_old.at(i);
            CATCH_REQUIRE(check_line(data.get_datum(i), val));
        }
    }
}
