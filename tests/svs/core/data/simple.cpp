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

// svs
#include "svs/core/data/simple.h"
#include "svs/core/allocator.h"
#include "svs/core/data/view.h"

#include "tests/svs/core/data/data.h"

// stdlib
#include <span>
#include <type_traits>

// catch2
#include "catch2/catch_test_macros.hpp"

namespace {

template <typename T, size_t Extent>
bool wants_const_view(svs::data::ConstSimpleDataView<T, Extent> data) {
    return svs_test::data::is_sequential(data);
}

template <typename T> void fill_lines(T& x) {
    for (size_t i = 0; i < x.size(); ++i) {
        for (auto& j : x.get_datum(i)) {
            j = i;
        }
    }
}

template <typename T> bool check_fill_lines(T& x, size_t offset = 0) {
    for (size_t i = 0; i < x.size(); ++i) {
        for (auto j : x.get_datum(i)) {
            if (j != i + offset) {
                return false;
            }
        }
    }
    return true;
}
} // namespace

CATCH_TEST_CASE("Testing Simple Data", "[core][data]") {
    // Default SimpleData
    CATCH_SECTION("Default SimpleData") {
        auto x = svs::data::SimpleData<float>(100, 10);
        CATCH_REQUIRE(x.size() == 100);
        CATCH_REQUIRE(x.dimensions() == 10);
        CATCH_REQUIRE(x.capacity() == 100);

        CATCH_REQUIRE(x.get_datum(0).size() == 10);
        CATCH_REQUIRE(x.get_datum(0).extent == svs::Dynamic);
        svs_test::data::set_sequential(x);
        CATCH_REQUIRE(svs_test::data::is_sequential(x));

        // Make sure `is_sequential` can fail.
        x.get_datum(0)[0] = 100;
        CATCH_REQUIRE(svs_test::data::is_sequential(x) == false);
        svs_test::data::set_sequential(x);

        // Construct a view.
        {
            auto y = x.view();
            CATCH_REQUIRE(y.size() == x.size());
            CATCH_REQUIRE(y.dimensions() == x.dimensions());
            CATCH_REQUIRE(y.data() == x.data());
            CATCH_REQUIRE(svs_test::data::is_sequential(y));
        }

        // Const view.
        {
            const auto z = x.cview();
            CATCH_REQUIRE(svs_test::data::is_sequential(z));
            CATCH_REQUIRE(wants_const_view(x.cview()));
        }

        // Resizing - ensure that the first elements are untouched.
        auto* original_ptr = x.data();
        x.resize(200);
        CATCH_REQUIRE(x.size() == 200);
        CATCH_REQUIRE(x.dimensions() == 10);
        CATCH_REQUIRE(x.capacity() == 200);

        // Ensure reallocation took place.
        CATCH_REQUIRE(original_ptr != x.data());
        original_ptr = x.data();
        {
            // Manually construct a smaller view to ensure the first elements remain
            // unchanged after the resizing.
            auto v = svs::data::ConstSimpleDataView<float>(x.data(), 100, x.dimensions());
            CATCH_REQUIRE(svs_test::data::is_sequential(v));
        }

        // Ensure mutation of the larger data still works.
        svs_test::data::set_sequential(x);
        CATCH_REQUIRE(svs_test::data::is_sequential(x));

        // Resize down - ensure no reallocation takes place.
        x.resize(100);
        CATCH_REQUIRE(x.size() == 100);
        CATCH_REQUIRE(x.capacity() == 200);
        CATCH_REQUIRE(svs_test::data::is_sequential(x));
        CATCH_REQUIRE(x.data() == original_ptr);

        // Resizing back should not trigger a reallocation.
        x.resize(200);
        CATCH_REQUIRE(x.data() == original_ptr);
        CATCH_REQUIRE(x.size() == 200);
        CATCH_REQUIRE(x.capacity() == 200);

        // Finally, drop back down in size and invoke `shrink_to_fit`.
        x.resize(100);
        x.shrink_to_fit();
        CATCH_REQUIRE(x.data() != original_ptr);
        CATCH_REQUIRE(x.size() == 100);
        CATCH_REQUIRE(x.capacity() == 100);
        CATCH_REQUIRE(svs_test::data::is_sequential(x));
    }

    CATCH_SECTION("Views") {
        auto x = svs::data::SimpleData<float, 4>(100, 4);
        fill_lines(x);
        CATCH_REQUIRE(check_fill_lines(x));
        CATCH_REQUIRE(!check_fill_lines(x, 1));

        auto y = svs::data::make_view(x, svs::threads::UnitRange(0, 5));
        CATCH_REQUIRE(y.size() == 5);
        CATCH_REQUIRE(y.dimensions() == x.dimensions());
        CATCH_REQUIRE(check_fill_lines(y));
        CATCH_REQUIRE(!check_fill_lines(y, 10));
        CATCH_REQUIRE(y.parent_indices() == svs::threads::UnitRange(0, 5));

        y = svs::data::make_view(x, svs::threads::UnitRange(10, 20));
        CATCH_REQUIRE(y.size() == 10);
        CATCH_REQUIRE(y.dimensions() == x.dimensions());
        CATCH_REQUIRE(!check_fill_lines(y));
        CATCH_REQUIRE(check_fill_lines(y, 10));

        // Make sure we get an error if we go out of bounds.
        CATCH_REQUIRE_THROWS_AS(
            svs::data::make_view(x, svs::threads::UnitRange(10, 110)), svs::ANNException
        );

        CATCH_REQUIRE_THROWS_AS(
            svs::data::make_view(x, svs::threads::UnitRange(-10, 10)), svs::ANNException
        );

        // Const paths
        auto z = svs::data::make_const_view(x, svs::threads::UnitRange(90, 95));
        CATCH_REQUIRE(z.size() == 5);
        CATCH_REQUIRE(z.dimensions() == x.dimensions());
        CATCH_REQUIRE(!check_fill_lines(z));
        CATCH_REQUIRE(check_fill_lines(z, 90));
    }

    CATCH_SECTION("Static Simple") {
        auto x = svs::data::SimpleData<float, 4>(100, 4);
        CATCH_REQUIRE(x.size() == 100);
        CATCH_REQUIRE(x.dimensions() == 4);
        CATCH_REQUIRE(x.get_datum(0).size() == 4);
        CATCH_REQUIRE(x.get_datum(0).extent == 4);

        svs_test::data::set_sequential(x);
        CATCH_REQUIRE(svs_test::data::is_sequential(x));
        auto y = x.view();
        CATCH_REQUIRE(y.data() == x.data());
        CATCH_REQUIRE(y.size() == x.size());
        CATCH_REQUIRE(y.dimensions() == x.dimensions());
        CATCH_REQUIRE(svs_test::data::is_sequential(y));
        CATCH_REQUIRE(y.get_datum(0).extent == 4);
    }

    // Conversion to and from AnonymousData.
    CATCH_SECTION("AnonymousData interop") {
        auto x = svs::data::SimpleData<float, 10>(100, 10);
        auto y = svs::AnonymousArray<2>(x);
        CATCH_REQUIRE(get<float>(y) == x.data());
        CATCH_REQUIRE(y.dims() == std::array<size_t, 2>{100, 10});

        // Conversion to dynamically sizxed.
        auto a = svs::data::ConstSimpleDataView<float>(y);
        CATCH_REQUIRE(a.data() == x.data());
        CATCH_REQUIRE(a.size() == x.size());
        CATCH_REQUIRE(a.dimensions() == x.dimensions());

        // Conversion to correct static size.
        auto b = svs::data::ConstSimpleDataView<float, 10>(y);
        CATCH_REQUIRE(b.data() == x.data());
        CATCH_REQUIRE(b.size() == x.size());
        CATCH_REQUIRE(b.dimensions() == x.dimensions());

        // Incorrect static size should fail.
        CATCH_REQUIRE_THROWS_AS(
            (svs::data::ConstSimpleDataView<float, 20>(y)), svs::ANNException
        );

        // Incorrect type.
        CATCH_REQUIRE_THROWS_AS(
            svs::data::ConstSimpleDataView<double>(y), svs::ANNException
        );
    }
}
