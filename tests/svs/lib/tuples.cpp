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
#include <iostream>
#include <type_traits>
#include <vector>

#include "svs/lib/tuples.h"

#include "catch2/catch_test_macros.hpp"

template <typename T> struct VectorWrapperProto {
    VectorWrapperProto(std::vector<T>&& buffer)
        : buffer_{std::move(buffer)} {}
    std::vector<T>& data() { return buffer_; }
    std::vector<T> buffer_;
};

template <typename T> struct ValueWrapperProto {
    ValueWrapperProto(T value)
        : value_{value} {}
    const T& data() { return value_; }
    T value_;
};

CATCH_TEST_CASE("Testing `map`", "[core][core_utils]") {
    CATCH_SECTION("astuple") {
        // Capture by value
        auto a1 = svs::lib::astuple(1);
        CATCH_REQUIRE(std::is_same_v<std::tuple<int>, decltype(a1)>);

        // rvalue reference should be come "by value"
        auto a2 = svs::lib::astuple(std::vector{1, 2, 3});
        CATCH_REQUIRE(std::is_same_v<std::tuple<std::vector<int>>, decltype(a2)>);

        // lvalue non-const reference.
        auto v1 = std::vector<int>{1, 2, 3, 4};
        auto a3 = svs::lib::astuple(v1);
        CATCH_REQUIRE(std::is_same_v<std::tuple<std::vector<int>&>, decltype(a3)>);

        // rvalue reference via `move`.
        auto v2 = std::vector<int>{10, 20, 30, 40};
        auto a4 = svs::lib::astuple(std::move(v2));
        CATCH_REQUIRE(std::is_same_v<std::tuple<std::vector<int>>, decltype(a4)>);

        // const lvalue reference
        const auto v3 = std::vector<size_t>{0, 100, 200};
        auto a5 = svs::lib::astuple(v3);
        CATCH_REQUIRE(std::is_same_v<std::tuple<const std::vector<size_t>&>, decltype(a5)>);
    }

    CATCH_SECTION("Basic Functionality") {
        auto x = std::make_tuple(1, 2.0, 3.0f);
        CATCH_REQUIRE(std::is_same_v<std::tuple<int, double, float>, decltype(x)>);
        auto y = svs::lib::map(x, [](auto i) { return 2 * i; });
        CATCH_REQUIRE(std::is_same_v<std::tuple<int, double, float>, decltype(y)>);
        CATCH_REQUIRE(y == std::make_tuple(2, 4.0, 6.0f));
    }

    CATCH_SECTION("Forwards References") {
        auto a = VectorWrapperProto(std::vector<int>{1, 2, 3, 4});
        auto b = ValueWrapperProto(std::string{"hello world"});

        // N.B.: Use `std::tie` to preserve references to the original `a` and `b`.
        // `std::make_tuple` will copy the arguments.
        auto tup = std::tie(a, b);
        auto result = svs::lib::map(
            tup,
            // Use the `-> decltype(auto)` trailing return type to preserve references
            // returned by the body of the lambba.
            [](auto& i) -> decltype(auto) { return i.data(); }
        );

        CATCH_REQUIRE(std::addressof(a.data()) == std::addressof(a.buffer_));
        CATCH_REQUIRE(std::addressof(b.data()) == std::addressof(b.value_));

        CATCH_REQUIRE(std::is_same_v<std::vector<int>&, decltype(std::get<0>(result))>);
        CATCH_REQUIRE(std::is_same_v<const std::string&, decltype(std::get<1>(result))>);

        using expected_type = typename std::tuple<std::vector<int>&, const std::string&>;
        CATCH_REQUIRE(std::is_same_v<expected_type, decltype(result)>);

        // Ensure that the objects were successfully propagated through all the
        // concatenation.
        CATCH_REQUIRE(std::addressof(std::get<0>(result)) == std::addressof(a.buffer_));
        CATCH_REQUIRE(std::addressof(std::get<1>(result)) == std::addressof(b.value_));
    }
}

// Introduce two different type to ensure heterogeneity.
struct CallCounter1 {
    CallCounter1()
        : count_{0} {}
    void called(size_t count) { count_ += count; }
    size_t count_;
};

struct CallCounter2 {
    CallCounter2()
        : count_{0} {}
    void called(size_t count) { count_ += count; }
    size_t count_;
};

CATCH_TEST_CASE("Testing `foreach`", "[core][core_utils]") {
    CATCH_SECTION("Forward `foreach`") {
        auto x = CallCounter1{};
        auto y = CallCounter2{};

        auto tup = std::tie(x, y, x);
        svs::lib::foreach (tup, [](auto& i) { i.called(5); });
        // argument `y` should have been visited once.
        CATCH_REQUIRE(y.count_ == 5);

        // argument `x` should have been visited twice because we put it into the tuple
        // twice.
        CATCH_REQUIRE(x.count_ == 10);

        // Next, try changing the elements in the tuple.
        // Use `std::decasy_t<decltype(i)>` to get the base type of the argument.
        // * `decltype(i)` gets the type, but can be a reference.
        // * `std::decay_t` removes the reference.
        // This allows us to get the base type and default construct it.
        svs::lib::foreach (tup, [](auto& i) { i = std::decay_t<decltype(i)>{}; });
        CATCH_REQUIRE(std::get<0>(tup).count_ == 0);
        CATCH_REQUIRE(std::get<1>(tup).count_ == 0);
        CATCH_REQUIRE(std::get<2>(tup).count_ == 0);
    }

    CATCH_SECTION("Forward `foreach const`") {
        auto x = CallCounter1{};
        auto y = CallCounter2{};

        const auto tup = std::tie(x, y, x);
        svs::lib::foreach (tup, [](auto& i) { i.called(5); });
        // argument `y` should have been visited once.
        CATCH_REQUIRE(y.count_ == 5);

        // argument `x` should have been visited twice because we put it into the tuple
        // twice.
        CATCH_REQUIRE(x.count_ == 10);
    }

    CATCH_SECTION("Forward `foreach` with capture") {
        std::array<size_t, 3> dest{};
        auto x = std::make_tuple(30, 20, 10);

        size_t prod = 1;
        size_t count = 0;
        svs::lib::foreach (x, [&prod, &count, &dest](auto i) {
            dest[count] = prod;
            prod *= i;
            ++count;
        });
        CATCH_REQUIRE(prod == 30 * 20 * 10);
        CATCH_REQUIRE(count == 3);
        CATCH_REQUIRE(dest == std::array<size_t, 3>({1, 30, 600}));
    }

    CATCH_SECTION("Reverse `const foreach_r` with capture") {
        std::array<size_t, 3> dest{};
        const auto x = std::make_tuple(30, 20, 10);

        size_t prod = 1;
        size_t count = dest.size() - 1;
        svs::lib::foreach_r(x, [&prod, &count, &dest](const auto& i) {
            dest[count] = prod;
            prod *= i;
            --count;
        });
        CATCH_REQUIRE(prod == 30 * 20 * 10);
        CATCH_REQUIRE(count == std::numeric_limits<size_t>::max());
        CATCH_REQUIRE(dest == std::array<size_t, 3>({200, 10, 1}));
    }

    CATCH_SECTION("Reverse `foreach`") {
        std::array<size_t, 3> dest{};
        auto x = std::make_tuple(30, 20, 10);

        size_t prod = 1;
        size_t count = dest.size() - 1;
        svs::lib::foreach_r(x, [&prod, &count, &dest](auto& i) {
            dest[count] = prod;
            prod *= i;
            --count;
            i = 0;
        });
        CATCH_REQUIRE(prod == 30 * 20 * 10);
        CATCH_REQUIRE(count == std::numeric_limits<size_t>::max());
        CATCH_REQUIRE(dest == std::array<size_t, 3>({200, 10, 1}));
        CATCH_REQUIRE(std::get<0>(x) == 0);
        CATCH_REQUIRE(std::get<1>(x) == 0);
        CATCH_REQUIRE(std::get<2>(x) == 0);
    }

    CATCH_SECTION("Hash") {
        svs::lib::TupleHash hash{};
        auto x = std::make_tuple(30, 20, 10);
        auto y = std::make_tuple(30, 20, 9);
        CATCH_REQUIRE(std::is_same_v<decltype(hash(x)), size_t>);
        CATCH_REQUIRE(hash(x) != hash(y));
        CATCH_REQUIRE(hash(x) == hash(x));

        // Unordered Map.
        auto map =
            std::unordered_map<std::tuple<int, int, int>, int, svs::lib::TupleHash>{};

        map[x] = 10;
        map[y] = 100;

        CATCH_REQUIRE(map.contains(x));
        CATCH_REQUIRE(map.contains(y));
        CATCH_REQUIRE(!map.contains(std::make_tuple(5, 5, 5)));

        CATCH_REQUIRE(map[x] == 10);
        CATCH_REQUIRE(map[y] == 100);
    }
}
