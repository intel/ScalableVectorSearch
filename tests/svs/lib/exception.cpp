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
#include <iostream>
#include <string>

#include "svs/lib/exception.h"

#include "catch2/catch_test_macros.hpp"

bool startswith(const std::string& haystack, const std::string& needle) {
    return haystack.rfind(needle, 0) == 0;
}

bool endswith(const std::string& haystack, const std::string& needle) {
    size_t last_possible_position = haystack.size() - needle.size();
    return haystack.find(needle, last_possible_position) == last_possible_position;
}

bool throws() {
    throw svs::ANNException("Something went wrong!", SVS_LINEINFO);
    return true;
}

CATCH_TEST_CASE("ANNException", "[core]") {
    CATCH_SECTION("Constructors") {
        // rvalue ref
        auto a = svs::ANNException("rvalue string");
        CATCH_REQUIRE(std::string{a.what()} == "rvalue string");
        // const lvalue ref;
        auto lvalue_string = "lvalue string";
        auto b = svs::ANNException(lvalue_string);
        CATCH_REQUIRE(std::string{b.what()} == "lvalue string");
        // variadic constructor
        // NOTE: The checked error message is not very pretty.
        auto c =
            svs::ANNException("{}{}{}{}", "rvalue string", lvalue_string, 10, SVS_LINEINFO);
        CATCH_REQUIRE(startswith(std::string{c.what()}, "rvalue stringlvalue string10(line")
        );
    }

    CATCH_SECTION("Throwing") { CATCH_REQUIRE_THROWS_AS(throws(), svs::ANNException); }
}
