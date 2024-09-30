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
