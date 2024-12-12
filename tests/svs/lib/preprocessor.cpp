/*
 * Copyright 2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Header under test.
#include "svs/lib/preprocessor.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <string>
#include <string_view>

namespace detail = svs::preprocessor::detail;

namespace {

struct TestClass {
    int a_;
    double b_;
    std::string c_;
    std::string d_;

    friend bool operator==(TestClass, TestClass) = default;

    SVS_CHAIN_SETTER_(TestClass, a);
    SVS_CHAIN_SETTER_(TestClass, b);
    SVS_CHAIN_SETTER_(TestClass, c);
    SVS_CHAIN_SETTER_TYPED_(TestClass, std::string_view, d);
};

} // namespace

CATCH_TEST_CASE("Preprocessor", "[lib][preprocessor]") {
    CATCH_SECTION("consteval functions") {
        // strlen
        CATCH_STATIC_REQUIRE(detail::strlen("") == 0);
        CATCH_STATIC_REQUIRE(detail::strlen("0") == 1);
        CATCH_STATIC_REQUIRE(detail::strlen("1") == 1);
        CATCH_STATIC_REQUIRE(detail::strlen("hello world") == 11);

        // is_valid
        CATCH_STATIC_REQUIRE(!detail::is_valid(nullptr));
        CATCH_STATIC_REQUIRE(!detail::is_valid(""));
        CATCH_STATIC_REQUIRE(detail::is_valid("0"));
        CATCH_STATIC_REQUIRE(detail::is_valid("1"));
        CATCH_STATIC_REQUIRE(!detail::is_valid("01"));

        // is_one_or_zero
        CATCH_STATIC_REQUIRE(!detail::is_one_or_zero(nullptr));
        CATCH_STATIC_REQUIRE(!detail::is_one_or_zero(""));
        CATCH_STATIC_REQUIRE(detail::is_one_or_zero("0"));
        CATCH_STATIC_REQUIRE(detail::is_one_or_zero("1"));
        CATCH_STATIC_REQUIRE(!detail::is_one_or_zero("01"));
    }

    CATCH_SECTION("chain setters") {
        // Rvalue reference.
        constexpr std::string_view hi = "hello world";
        auto x = TestClass().b(20.0).a(4).d(hi);
        CATCH_REQUIRE(x == TestClass{.a_ = 4, .b_ = 20.0, .c_ = "", .d_ = std::string(hi)});

        // lvalue reference.
        x.c("foo").a(-1);
        CATCH_REQUIRE(
            x == TestClass{.a_ = -1, .b_ = 20.0, .c_ = "foo", .d_ = std::string(hi)}
        );

        x.d("bar");
        CATCH_REQUIRE(x == TestClass{.a_ = -1, .b_ = 20.0, .c_ = "foo", .d_ = "bar"});
    }
}
