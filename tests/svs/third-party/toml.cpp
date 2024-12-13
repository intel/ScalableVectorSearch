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

// header under test
#include "svs/third-party/toml.h"

// catch2
#include "catch2/catch_test_macros.hpp"

namespace {

const std::string_view test_toml = R"(
string = "hello world"
array = [10, 20]
integer = 10
integer_signed = -1
float = 1.0

[[array_of_dict]]
a = 10
[[array_of_dict]]
a = 20

[table]
b = 200
a = 100
)";

}

CATCH_TEST_CASE("Toml Handling", "[third-party]") {
    namespace toml_helper = svs::toml_helper;
    auto table = toml::parse(test_toml);

    // Happy paths
    CATCH_SECTION("String") {
        auto& x = toml_helper::get_as<toml::value<std::string>>(table, "string");
        CATCH_REQUIRE(x.get() == "hello world");

        // Bad cast
        CATCH_REQUIRE_THROWS_AS(
            toml_helper::get_as<toml::value<int64_t>>(table, "string"), svs::ANNException
        );

        // Bad name
        CATCH_REQUIRE_THROWS_AS(
            toml_helper::get_as<toml::value<int64_t>>(table, "stirng"), svs::ANNException
        );
        CATCH_REQUIRE(toml_helper::get_as<std::string>(table, "string") == "hello world");
    }

    CATCH_SECTION("Integers") {
        auto x = toml_helper::get_as<toml::value<int64_t>>(table, "integer");
        CATCH_REQUIRE(x.get() == 10);
        CATCH_REQUIRE(toml_helper::get_as<int64_t>(table, "integer") == 10);
        CATCH_REQUIRE(toml_helper::get_as<uint64_t>(table, "integer") == 10);

        x = toml_helper::get_as<toml::value<int64_t>>(table, "integer_signed");
        CATCH_REQUIRE(x.get() == -1);
        CATCH_REQUIRE(toml_helper::get_as<int64_t>(table, "integer_signed") == -1);
        CATCH_REQUIRE_THROWS_AS(
            toml_helper::get_as<uint64_t>(table, "integer_signed"),
            svs::lib::narrowing_error
        );
    }

    CATCH_SECTION("Floating Point") {
        auto x = toml_helper::get_as<toml::value<double>>(table, "float");
        CATCH_REQUIRE(x.get() == 1);
        CATCH_REQUIRE(toml_helper::get_as<double>(table, "float") == 1);
        CATCH_REQUIRE(toml_helper::get_as<float>(table, "float") == 1);
    }

    CATCH_SECTION("Arrays") {
        auto x = toml_helper::get_as<toml::array>(table, "array");
        CATCH_REQUIRE(x.size() == 2);
        CATCH_REQUIRE(toml_helper::get_as<int64_t>(x.at(0)) == 10);
        CATCH_REQUIRE(toml_helper::get_as<int64_t>(x.at(1)) == 20);
    }

    CATCH_SECTION("Table") {
        auto x = toml_helper::get_as<toml::table>(table, "table");
        CATCH_REQUIRE(toml_helper::get_as<int64_t>(x, "a") == 100);
        CATCH_REQUIRE(toml_helper::get_as<int64_t>(x, "b") == 200);
    }
}
