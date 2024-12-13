/*
 * Copyright 2023 Intel Corporation
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
