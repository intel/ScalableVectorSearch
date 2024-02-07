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
#include "svs/index/vamana/filter.h"
#include "svs/lib/threads/types.h"

// catch2
#include "catch2/catch_test_macros.hpp"

namespace {

template <typename I, size_t N>
void assert_is_reset(const svs::index::vamana::VisitedFilter<I, N>& filter) {
    using filter_type = svs::index::vamana::VisitedFilter<I, N>;
    for (size_t i = 0; i < filter.capacity(); ++i) {
        CATCH_REQUIRE(filter.at(i) == filter_type::sentinel);
    }
}

template <typename I, size_t N> void test_filter() {
    using filter_type = svs::index::vamana::VisitedFilter<I, N>;
    auto filter = filter_type{};
    // Make sure all entries are set to the sentinel value.
    assert_is_reset(filter);
    CATCH_STATIC_REQUIRE(filter_type::filter_capacity == size_t(1) << N);

    auto cap = filter.capacity();
    auto first_group = svs::threads::UnitRange<size_t>(0, cap);
    auto second_group = svs::threads::UnitRange<size_t>(cap, 2 * cap);

    for (auto i : first_group) {
        CATCH_REQUIRE(!filter.contains(i));
    }

    // Run through the filter, adding elements.
    // `emplace` should return false because only sentinel values are stored.
    for (auto i : first_group) {
        CATCH_REQUIRE(!filter.emplace(i));
    }

    // Now, the filter should contain the entries we just added.
    for (auto i : first_group) {
        CATCH_REQUIRE(filter.contains(i));
    }

    // Adding them again should return true.
    for (auto i : first_group) {
        CATCH_REQUIRE(filter.emplace(i));
    }

    // Now, if we add the next bucket group, they shouldn't be in the container.
    for (auto i : second_group) {
        CATCH_REQUIRE(!filter.emplace(i));
    }

    // The first group should be over written.
    for (auto i : first_group) {
        CATCH_REQUIRE(!filter.contains(i));
    }

    // But the next group should exist.
    for (auto i : second_group) {
        CATCH_REQUIRE(filter.contains(i));
    }

    // Emplacing the first group again should overwrite the second group.
    for (auto i : first_group) {
        CATCH_REQUIRE(!filter.emplace(i));
    }

    for (auto i : second_group) {
        CATCH_REQUIRE(!filter.contains(i));
    }
    for (auto i : first_group) {
        CATCH_REQUIRE(filter.contains(i));
    }
    filter.reset();
    assert_is_reset(filter);
}

} // namespace

CATCH_TEST_CASE("Visited Filter", "[vamana][visited_filter]") {
    test_filter<uint32_t, 14>();
    test_filter<uint32_t, 15>();
    test_filter<uint32_t, 16>();
    test_filter<uint32_t, 17>();
    test_filter<uint32_t, 18>();
}
