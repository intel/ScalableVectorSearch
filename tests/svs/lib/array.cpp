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

#include <array>
#include <cstdint>
#include <span>
#include <tuple>
#include <vector>

#include "svs/lib/array.h"
#include "svs/lib/static.h"
#include "svs/lib/tuples.h"

#include "catch2/catch_test_macros.hpp"

namespace {

///
/// Test that 1-dimensional indexing works correctly for static and dynamic
/// dimensions.
///
/// The argument can either be an integer or a `svs::meta::Val` type.
///
template <typename T> void test_offset_1d(T bound1) {
    // The internal representation of dimensions is as a heterogeneous tuple.
    // Here, that tuple is constructed.
    auto dims = std::make_tuple(bound1);

    // Convert the argument (whether it's an integer or a `Val`) to a `size_t`.
    size_t b1 = svs::meta::unwrap(bound1);
    size_t expected = 0;
    for (size_t i = 0; i < b1; ++i) {
        CATCH_REQUIRE(
            svs::detail::offset(dims, svs::detail::unchecked_make_array(i)) == expected
        );
        ++expected;
    }
    CATCH_REQUIRE(expected == b1);
}

template <typename T, typename U> void test_offset_2d(T bound1, U bound2) {
    auto dims = std::make_tuple(bound1, bound2);
    size_t b1 = svs::meta::unwrap(bound1);
    size_t b2 = svs::meta::unwrap(bound2);
    size_t upper = b1 * b2;

    size_t expected = 0;
    for (size_t j = 0; j < b1; ++j) {
        for (size_t i = 0; i < b2; ++i) {
            CATCH_REQUIRE(
                svs::detail::offset(dims, svs::detail::unchecked_make_array(j, i)) ==
                expected
            );
            ++expected;
        }
    }
    CATCH_REQUIRE(expected == upper);
}

template <typename T, typename U, typename V>
void test_offset_3d(T bound1, U bound2, V bound3) {
    auto dims = std::make_tuple(bound1, bound2, bound3);
    size_t b1 = svs::meta::unwrap(bound1);
    size_t b2 = svs::meta::unwrap(bound2);
    size_t b3 = svs::meta::unwrap(bound3);
    size_t upper = b1 * b2 * b3;

    size_t expected = 0;
    for (size_t k = 0; k < b1; ++k) {
        for (size_t j = 0; j < b2; ++j) {
            for (size_t i = 0; i < b3; ++i) {
                CATCH_REQUIRE(
                    svs::detail::offset(dims, svs::detail::unchecked_make_array(k, j, i)) ==
                    expected
                );
                ++expected;
            }
        }
    }
    CATCH_REQUIRE(expected == upper);
}
} // namespace

CATCH_TEST_CASE("Array Utilities", "[core][array]") {
    CATCH_SECTION("Make Array") {
        uint16_t a = 1;
        uint32_t b = 1000;
        size_t c = 1234;
        uint8_t d = 5;
        CATCH_REQUIRE(svs::detail::unchecked_make_array(a) == std::to_array<size_t>({a}));
        CATCH_REQUIRE(
            svs::detail::unchecked_make_array(a, b) == std::to_array<size_t>({a, b})
        );
        CATCH_REQUIRE(
            svs::detail::unchecked_make_array(a, b, c) == std::to_array<size_t>({a, b, c})
        );
        CATCH_REQUIRE(
            svs::detail::unchecked_make_array(a, b, c, d) ==
            std::to_array<size_t>({a, b, c, d})
        );
    }

    CATCH_SECTION("Default Strides", "[core][array]") {
        namespace dt = svs::detail;
        namespace meta = svs::meta;
        // Dimension 1
        {
            auto a = std::tuple<size_t>(100);
            auto b = std::make_tuple(meta::Val<1>{});
            CATCH_REQUIRE(dt::default_strides(a) == std::to_array<size_t>({1}));
            CATCH_REQUIRE(dt::default_strides(b) == std::to_array<size_t>({1}));
        }
        // Dimension 2
        {
            auto a = std::make_tuple(size_t{10}, size_t{100});
            auto b = std::make_tuple(meta::Val<10>{}, size_t{100});
            auto c = std::make_tuple(size_t{10}, meta::Val<100>{});
            auto d = std::make_tuple(meta::Val<10>{}, meta::Val<100>{});

            auto expected = std::to_array<size_t>({100, 1});
            CATCH_REQUIRE(dt::default_strides(a) == expected);
            CATCH_REQUIRE(dt::default_strides(b) == expected);
            CATCH_REQUIRE(dt::default_strides(c) == expected);
            CATCH_REQUIRE(dt::default_strides(d) == expected);
        }
    }

    CATCH_SECTION("Offset", "[core][array]") {
        namespace dt = svs::detail;
        namespace meta = svs::meta;
        // Dimension 1
        {
            test_offset_1d(size_t{128});
            test_offset_1d(meta::Val<128>{});
        }
        // Dimension 2
        {
            test_offset_2d(size_t{128}, size_t{10});
            test_offset_2d(meta::Val<128>{}, size_t{10});
            test_offset_2d(meta::Val<128>{}, meta::Val<10>{});
        }
        // Dimension 3
        {
            test_offset_3d(size_t{5}, size_t{3}, size_t{4});
            test_offset_3d(meta::Val<5>{}, size_t{3}, size_t{4});
            test_offset_3d(size_t{5}, meta::Val<3>{}, size_t{4});
            test_offset_3d(meta::Val<5>{}, meta::Val<3>{}, size_t{4});
            test_offset_3d(size_t{5}, size_t{3}, meta::Val<4>{});
            test_offset_3d(meta::Val<5>{}, size_t{3}, meta::Val<4>{});
            test_offset_3d(size_t{5}, meta::Val<3>{}, meta::Val<4>{});
            test_offset_3d(meta::Val<5>{}, meta::Val<3>{}, meta::Val<4>{});
        }
    }

    CATCH_SECTION("Extent Propagaion", "[core][array]") {
        namespace dt = svs::detail;
        namespace meta = svs::meta;
        CATCH_REQUIRE(dt::get_extent_impl<size_t> == svs::Dynamic);
        CATCH_REQUIRE(dt::get_extent_impl<meta::Val<10>> == 10);
        CATCH_REQUIRE(dt::get_extent_impl<meta::Val<128>> == 128);

        // 1D case
        CATCH_REQUIRE(dt::getextent<std::tuple<size_t>> == svs::Dynamic);
        CATCH_REQUIRE(dt::getextent<std::tuple<meta::Val<23>>> == 23);

        // 2D case
        CATCH_REQUIRE(dt::getextent<std::tuple<size_t, size_t>> == svs::Dynamic);
        CATCH_REQUIRE(dt::getextent<std::tuple<meta::Val<23>, size_t>> == svs::Dynamic);
        CATCH_REQUIRE(dt::getextent<std::tuple<size_t, meta::Val<23>>> == 23);
    }
}

CATCH_TEST_CASE("Array", "[core][array]") {
    std::vector<int> input{};
    for (size_t i = 0; i < 100; ++i) {
        input.push_back(i);
    }
    CATCH_SECTION("Square Dynamic") {
        // Does this work?
        auto x = svs::make_dense_array(input.data(), 10, 10);
        CATCH_REQUIRE(std::is_same_v<
                      decltype(x),
                      svs::DenseArray<int, std::tuple<size_t, size_t>, int*>>);

        CATCH_REQUIRE(x.ndims() == 2);
        CATCH_REQUIRE(x.dims() == std::to_array<size_t>({10, 10}));
        CATCH_REQUIRE(x.getsize<0>() == 10);
        CATCH_REQUIRE(x.getsize<1>() == 10);
        size_t count = 0;
        for (size_t j = 0; j < x.getsize<0>(); ++j) {
            auto slice = x.slice(j);
            CATCH_REQUIRE(
                std::equal(slice.begin(), slice.end(), input.begin() + j * x.getsize<1>())
            );

            for (size_t i = 0; i < x.getsize<1>(); ++i) {
                CATCH_REQUIRE(x.at(j, i) == input[count]);
                ++count;
                x.at(j, i) = 0;
            }
        }
        CATCH_REQUIRE(std::all_of(input.begin(), input.end(), [](auto i) { return i == 0; })
        );

        CATCH_REQUIRE(x.extent() == svs::Dynamic);
    }

    CATCH_SECTION("Rectangular Dynamic") {
        auto x = svs::make_dense_array(input.data(), 4, 25);
        CATCH_REQUIRE(std::is_same_v<
                      decltype(x),
                      svs::DenseArray<int, std::tuple<size_t, size_t>, int*>>);

        CATCH_REQUIRE(x.dims() == std::to_array<size_t>({4, 25}));
        CATCH_REQUIRE(x.getsize<0>() == 4);
        CATCH_REQUIRE(x.getsize<1>() == 25);
        size_t count = 0;
        for (size_t j = 0; j < x.getsize<0>(); ++j) {
            auto slice = x.slice(j);
            CATCH_REQUIRE(slice.size() == x.getsize<1>());
            CATCH_REQUIRE(
                std::equal(slice.begin(), slice.end(), input.begin() + j * x.getsize<1>())
            );

            for (size_t i = 0; i < x.getsize<1>(); ++i) {
                CATCH_REQUIRE(x.at(j, i) == input[count]);
                ++count;
                x.at(j, i) = 0;
            }
        }
        CATCH_REQUIRE(std::all_of(input.begin(), input.end(), [](auto i) { return i == 0; })
        );

        CATCH_REQUIRE(x.extent() == svs::Dynamic);
    }

    CATCH_SECTION("Rectangular Mixed") {
        constexpr size_t extent = 4;
        auto x = svs::make_dense_array(input.data(), 25, svs::meta::Val<extent>{});
        CATCH_REQUIRE(std::is_same_v<
                      decltype(x),
                      svs::DenseArray<
                          int,
                          std::tuple<size_t, svs::meta::Val<extent>>,
                          int*>>);

        CATCH_REQUIRE(x.dims() == std::to_array<size_t>({25, extent}));
        CATCH_REQUIRE(x.getsize<1>() == extent);
        CATCH_REQUIRE(x.getsize<0>() == 25);
        size_t count = 0;
        for (size_t j = 0; j < x.getsize<0>(); ++j) {
            auto slice = x.slice(j);
            CATCH_REQUIRE(slice.extent == extent);
            CATCH_REQUIRE(std::equal(
                input.begin() + j * x.getsize<1>(),
                input.begin() + (j + 1) * x.getsize<1>(),
                slice.begin()
            ));

            for (size_t i = 0; i < x.getsize<1>(); ++i) {
                CATCH_REQUIRE(x.at(j, i) == input[count]);
                ++count;
                x.at(j, i) = 0;
            }
        }
        CATCH_REQUIRE(std::all_of(input.begin(), input.end(), [](auto i) { return i == 0; })
        );

        CATCH_REQUIRE(x.extent() == extent);
    }

    CATCH_SECTION("Vector Initialization") {
        auto x = svs::make_dense_array<float>(svs::lib::VectorAllocator{}, 5, 20);
        CATCH_REQUIRE(x.size() == 100);
        CATCH_REQUIRE(x.bytes() == 400);

        const std::vector<float>& base = x.getbase();
        CATCH_REQUIRE(base.size() == 100);
        CATCH_REQUIRE(base.capacity() >= 100);

        CATCH_REQUIRE(x.ndims() == 2);
        CATCH_REQUIRE(x.dims() == std::to_array<size_t>({5, 20}));

        size_t count = 0;
        for (auto& i : x) {
            i = count;
            count++;
        }

        count = 0;
        for (size_t j = 0; j < x.getsize<0>(); ++j) {
            auto slice = x.slice(j);
            for (auto i : slice) {
                CATCH_REQUIRE(i == count);
                ++count;
            }
        }
    }

    /////
    ///// Copying
    /////

    CATCH_SECTION("Copying and Views") {
        auto x = svs::make_dense_array<int>(100, 100);
        int i = 0;
        for (auto& j : x) {
            j = i;
            i++;
        }

        auto view = x.view();
        CATCH_REQUIRE(x.size() == view.size());
        CATCH_REQUIRE(x.dims() == view.dims());
        // Pointers are the same
        CATCH_REQUIRE(x.data() == view.data());

        // Copy construct.
        // Contents should be the same but pointers should not.
        auto y = x;
        i = 0;
        for (const auto& j : y) {
            CATCH_REQUIRE(j == i);
            i++;
        }
        CATCH_REQUIRE(y.data() != x.data());
        // Copy assign views.
        auto yview = y.view();
        view = yview;
        CATCH_REQUIRE(view.data() == y.data());

        // Copy Assignment.
        auto z = svs::make_dense_array<int>(10, 10);
        for (auto& j : z) {
            j = 0;
        }
        x = z;
        CATCH_REQUIRE(x.size() == z.size());
        CATCH_REQUIRE(x.data() != z.data());
        CATCH_REQUIRE(std::equal(x.begin(), x.end(), z.begin()));
    }
}
