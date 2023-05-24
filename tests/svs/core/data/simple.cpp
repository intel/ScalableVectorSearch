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
#include <type_traits>

// svs
#include "svs/core/allocator.h"
#include "svs/core/data/simple.h"

// catch2
#include "catch2/catch_test_macros.hpp"

namespace {
template <typename T> void set_sequential(T& x) {
    size_t count = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        for (auto& j : x.get_datum(i)) {
            j = count;
            ++count;
        }
    }
}

template <typename T> bool is_sequential(const T& x) {
    size_t count = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        for (auto& j : x.get_datum(i)) {
            if (j != count) {
                return false;
            }
            ++count;
        }
    }
    return true;
}

template <typename T, size_t Extent>
bool wants_const_view(svs::data::ConstSimpleDataView<T, Extent> data) {
    return is_sequential(data);
}
} // namespace

CATCH_TEST_CASE("Testing Simple Data", "[core][data]") {
    // Default SimpleData
    CATCH_SECTION("Default SimpleData") {
        auto x = svs::data::SimpleData<float>(100, 10);
        CATCH_REQUIRE(x.size() == 100);
        CATCH_REQUIRE(x.dimensions() == 10);
        CATCH_REQUIRE(x.get_datum(0).size() == 10);
        CATCH_REQUIRE(x.get_datum(0).extent == svs::Dynamic);
        set_sequential(x);
        CATCH_REQUIRE(is_sequential(x));

        // Make sure `is_sequential` can fail.
        x.get_datum(0)[0] = 100;
        CATCH_REQUIRE(is_sequential(x) == false);
        set_sequential(x);

        // Construct a view.
        auto y = svs::data::SimpleDataView(x);
        CATCH_REQUIRE(y.size() == x.size());
        CATCH_REQUIRE(y.dimensions() == x.dimensions());
        CATCH_REQUIRE(is_sequential(y));

        // Try implicit conversion.
        y = x;
        CATCH_REQUIRE(is_sequential(y));

        const auto z = svs::data::ConstSimpleDataView(x);
        CATCH_REQUIRE(is_sequential(z));
        CATCH_REQUIRE(wants_const_view(x.cview()));
    }

    CATCH_SECTION("Static Simple") {
        auto x = svs::data::SimpleData<float, 4>(100, 4);
        CATCH_REQUIRE(x.size() == 100);
        CATCH_REQUIRE(x.dimensions() == 4);
        CATCH_REQUIRE(x.get_datum(0).size() == 4);
        CATCH_REQUIRE(x.get_datum(0).extent == 4);

        set_sequential(x);
        CATCH_REQUIRE(is_sequential(x));
        auto y = svs::data::SimpleDataView(x);
        CATCH_REQUIRE(y.size() == x.size());
        CATCH_REQUIRE(y.dimensions() == x.dimensions());
        CATCH_REQUIRE(is_sequential(y));
        CATCH_REQUIRE(y.get_datum(0).extent == 4);
    }

    // Now, try different allocators.
    CATCH_SECTION("Polymorphic Data") {
        auto allocator = svs::lib::VectorAllocator{};
        auto x = svs::data::SimplePolymorphicData<size_t>(allocator, 10, 20);
        CATCH_REQUIRE(x.size() == 10);
        CATCH_REQUIRE(x.dimensions() == 20);
        CATCH_REQUIRE(x.get_datum(2).extent == svs::Dynamic);
        set_sequential(x);
        CATCH_REQUIRE(is_sequential(x));

        // Move assign a new object into `x`.
        x = svs::data::SimplePolymorphicData<size_t>(
            svs::lib::UniquePtrAllocator{}, 10, 20
        );
        CATCH_REQUIRE(!is_sequential(x));
        set_sequential(x);
        CATCH_REQUIRE(is_sequential(x));
    }
}
