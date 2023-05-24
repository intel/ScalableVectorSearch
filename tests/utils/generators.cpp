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

#include <algorithm>
#include <iostream>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "catch2/generators/catch_generators_random.hpp"

#include "tests/utils/generators.h"

template <typename T> void test_generator(T lo, T hi, size_t test_length) {
    // std::vector
    {
        std::vector<T> x(test_length);
        auto generator = svs_test::make_generator<T>(lo, hi);
        CATCH_REQUIRE(std::all_of(x.begin(), x.end(), [](const T& i) { return i == T(0); })
        );
        svs_test::populate(x, generator, x.size());
        CATCH_REQUIRE(x.size() == test_length);
        auto first = *x.begin();
        CATCH_REQUIRE(std::any_of(x.begin(), x.end(), [first](const T& i) {
            return i != first;
        }));

        auto lo_converted = svs_test::convert_to<svs_test::catch_generator_type_t<T>>(lo);
        auto hi_converted = svs_test::convert_to<svs_test::catch_generator_type_t<T>>(hi);

        CATCH_REQUIRE(std::all_of(x.begin(), x.end(), [=](const T& i) {
            auto i_converted = svs_test::convert_to<svs_test::catch_generator_type_t<T>>(i);
            return lo_converted <= i_converted && i_converted <= hi_converted;
        }));
    }

    // std::unordered_set
    {
        std::unordered_set<T> x{};
        auto generator = svs_test::make_generator<T>(lo, hi);
        CATCH_REQUIRE(x.size() == 0);
        svs_test::populate(x, generator, test_length);
        // Could be less than due to repeats.
        CATCH_REQUIRE(x.size() <= test_length);

        // Make sure all elements are in-bounds.
        auto lo_converted = svs_test::convert_to<svs_test::catch_generator_type_t<T>>(lo);
        auto hi_converted = svs_test::convert_to<svs_test::catch_generator_type_t<T>>(hi);

        CATCH_REQUIRE(std::all_of(x.begin(), x.end(), [=](const T& i) {
            auto i_converted = svs_test::convert_to<svs_test::catch_generator_type_t<T>>(i);
            return lo_converted <= i_converted && i_converted <= hi_converted;
        }));
    }
}

CATCH_TEST_CASE("Generator Conversion", "[testing_utilities]") {
    CATCH_REQUIRE(
        std::is_same_v<float, svs_test::catch_generator_type_t<svs::Float16>> == true
    );
    CATCH_REQUIRE(
        std::is_same_v<uint32_t, svs_test::catch_generator_type_t<uint8_t>> == true
    );
    CATCH_REQUIRE(
        std::is_same_v<int32_t, svs_test::catch_generator_type_t<int8_t>> == true
    );
}

CATCH_TEST_CASE("Generator Behavior", "[testing_utilities]") {
    constexpr size_t test_length = 100;
    CATCH_SECTION("float") {
        auto generator = svs_test::make_generator<float>(-10.0f, 10.f);
        CATCH_REQUIRE(
            std::is_same_v<std::remove_cvref_t<decltype(generator.get())>, float> == true
        );
        test_generator<float>(-10.0f, 10.0f, test_length);
    }

    CATCH_SECTION("float16") {
        svs::Float16 lo = svs_test::convert_to<svs::Float16>(-10.0f);
        svs::Float16 hi = svs_test::convert_to<svs::Float16>(10.0f);
        CATCH_REQUIRE(svs_test::convert_to<float>(lo) == -10.0f);
        CATCH_REQUIRE(svs_test::convert_to<float>(hi) == 10.0f);

        // Make sure we get the correct type.
        auto generator = svs_test::make_generator<svs::Float16>(-10.0f, 10.0f);
        CATCH_REQUIRE(
            std::is_same_v<std::remove_cvref_t<decltype(generator.get())>, float> == true
        );
        test_generator<svs::Float16>(lo, hi, test_length);
    }

    CATCH_SECTION("uint8_t") {
        auto generator = svs_test::make_generator<uint8_t>(0, 255);
        CATCH_REQUIRE(
            std::is_same_v<std::remove_cvref_t<decltype(generator.get())>, uint32_t> == true
        );
        test_generator<uint8_t>(0, 255, test_length);
    }

    CATCH_SECTION("int8_t") {
        auto generator = svs_test::make_generator<int8_t>(-128, 127);
        CATCH_REQUIRE(
            std::is_same_v<std::remove_cvref_t<decltype(generator.get())>, int32_t> == true
        );
        test_generator<int8_t>(-128, 127, test_length);
    }

    // Test the `std::unordered_set` implementation.
    CATCH_SECTION("Unordered Set Population") {}
}
