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
