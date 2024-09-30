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
#include "svs/core/data/view.h"

// test utilities
#include "tests/svs/core/data/data.h"

// other misc SVS stuff
#include "svs/lib/threads/types.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <span>
#include <vector>

using namespace svs_test::data;

CATCH_TEST_CASE("Data view", "[core][data][view]") {
    CATCH_SECTION("Check indices") {
        using svs::data::detail::check_indices;
        CATCH_SECTION("Unit Range") {
            auto range = svs::threads::UnitRange(0, 10);
            // The following two should succeed.
            check_indices(range, 10);
            check_indices(range, 11);

            // These should throw.
            CATCH_REQUIRE_THROWS_AS(check_indices(range, 9), svs::ANNException);
            CATCH_REQUIRE_THROWS_AS(
                check_indices(svs::threads::UnitRange(-1, 2), 10), svs::ANNException
            );
        }

        CATCH_SECTION("Vector and span") {
            auto v = std::vector<size_t>{3, 2, 1};
            check_indices(v, 4);
            check_indices(svs::lib::as_const_span(v), 4);
            check_indices(v, 5);
            check_indices(svs::lib::as_const_span(v), 5);

            CATCH_REQUIRE_THROWS_AS(check_indices(v, 3), svs::ANNException);
            CATCH_REQUIRE_THROWS_AS(
                check_indices(svs::lib::as_const_span(v), 3), svs::ANNException
            );
        }
    }

    CATCH_SECTION("Views") {
        using svs::data::make_const_view;
        using svs::data::make_view;
        using svs::threads::UnitRange;

        // Base data should looks like this:
        // 0, 10, 20, 30 ...
        auto base = MockDataset(MockDataset::Iota{0, 10, 100});

        auto check = [](const auto& d, size_t start, size_t step, size_t length) {
            using T = typename std::decay_t<decltype(d)>::value_type;
            CATCH_REQUIRE(d.size() <= length);
            for (size_t i = 0; i < length; ++i) {
                if (d.get_datum(i) != svs::lib::narrow<T>(start + step * i)) {
                    return false;
                }
            }
            return true;
        };

        auto check_init = [](const auto& d, std::initializer_list<size_t> expected) {
            using T = typename std::decay_t<decltype(d)>::value_type;
            CATCH_REQUIRE(d.size() == expected.size());
            auto begin = std::begin(expected);
            for (size_t i = 0; i < d.size(); ++i) {
                if (d.get_datum(i) != svs::lib::narrow<T>(*(begin + i))) {
                    return false;
                }
            }
            return true;
        };

        CATCH_SECTION("Unit Range") {
            auto v = make_const_view(base, UnitRange(10, 20));
            CATCH_REQUIRE(v.size() == 10);
            CATCH_REQUIRE(v.dimensions() == base.dimensions());
            for (size_t i = 0; i < v.size(); ++i) {
                CATCH_REQUIRE(v.parent_id(i) == 10 + i);
            }
            CATCH_REQUIRE(v.parent_indices() == UnitRange<int>(10, 20));
            CATCH_REQUIRE(v.parent_indices() != UnitRange<int>(20, 30));
            CATCH_REQUIRE(v.eachindex() == UnitRange<size_t>(0, 10));
            CATCH_REQUIRE(&v.parent() == &base);

            CATCH_REQUIRE(std::is_same_v<
                          typename decltype(v)::value_type,
                          MockDataset::value_type>);
            CATCH_REQUIRE(std::is_same_v<
                          typename decltype(v)::const_value_type,
                          MockDataset::const_value_type>);
            CATCH_REQUIRE(check(v, 100, 10, 10));

            // Make a new view
            CATCH_REQUIRE(check(make_const_view(v, UnitRange(0, v.size())), 100, 10, 10));
            CATCH_REQUIRE(check(make_const_view(v, UnitRange(0, 5)), 100, 10, 5));
            CATCH_REQUIRE(check(make_const_view(v, UnitRange(5, 10)), 150, 10, 5));

            auto vmut = make_view(base, UnitRange(10, 20));
            CATCH_REQUIRE(check(make_view(vmut, UnitRange(0, vmut.size())), 100, 10, 10));
            CATCH_REQUIRE(check(make_view(vmut, UnitRange(0, 5)), 100, 10, 5));
            CATCH_REQUIRE(check(make_view(vmut, UnitRange(5, 10)), 150, 10, 5));

            // Test out mutation.
            vmut.set_datum(0, 10);
            CATCH_REQUIRE(vmut.get_datum(0) == 10);
            CATCH_REQUIRE(base.get_datum(10) == 10);
        }

        CATCH_SECTION("Vector") {
            auto ids = std::vector<size_t>({10, 20, 30, 40});
            auto v = make_const_view(base, ids);
            CATCH_REQUIRE(v.dimensions() == base.dimensions());
            CATCH_REQUIRE(v.size() == 4);
            CATCH_REQUIRE(v.get_datum(0) == base.get_datum(10));
            CATCH_REQUIRE(v.get_datum(1) == base.get_datum(20));
            CATCH_REQUIRE(v.get_datum(2) == base.get_datum(30));
            CATCH_REQUIRE(v.get_datum(3) == base.get_datum(40));

            // Parent ID
            CATCH_REQUIRE(v.parent_id(0) == 10);
            CATCH_REQUIRE(v.parent_id(1) == 20);
            CATCH_REQUIRE(v.parent_id(2) == 30);
            CATCH_REQUIRE(v.parent_id(3) == 40);
            // Make sure the parent IDs are equal, but that they aren't pointer equal
            // (i.e., a copy was made)>
            CATCH_REQUIRE(v.parent_indices() == ids);
            CATCH_REQUIRE(ids.data() != v.parent_indices().data());
            CATCH_REQUIRE(&v.parent() == &base);

            // Views of view.
            CATCH_REQUIRE(
                check_init(make_const_view(v, UnitRange(0, 4)), {100, 200, 300, 400})
            );
            CATCH_REQUIRE(check_init(make_const_view(v, UnitRange(0, 2)), {100, 200}));
            CATCH_REQUIRE(check_init(make_const_view(v, UnitRange(2, 4)), {300, 400}));

            auto vmut = make_view(base, std::vector<size_t>({10, 20, 30, 40}));
            CATCH_REQUIRE(vmut.size() == 4);
            CATCH_REQUIRE(vmut.get_datum(0) == base.get_datum(10));
            CATCH_REQUIRE(vmut.get_datum(1) == base.get_datum(20));
            CATCH_REQUIRE(vmut.get_datum(2) == base.get_datum(30));
            CATCH_REQUIRE(vmut.get_datum(3) == base.get_datum(40));

            // View of view.
            CATCH_REQUIRE(check_init(make_view(vmut, UnitRange(0, 4)), {100, 200, 300, 400})
            );
            CATCH_REQUIRE(check_init(make_view(vmut, UnitRange(0, 2)), {100, 200}));
            CATCH_REQUIRE(check_init(make_view(vmut, UnitRange(2, 4)), {300, 400}));

            vmut.set_datum(0, 0);
            CATCH_REQUIRE(vmut.get_datum(0) == 0);
            CATCH_REQUIRE(base.get_datum(10) == 0);
        }
    }
}
