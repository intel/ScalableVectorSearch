/**
 *    Copyright (C) 2023, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

// Code under test.
#include "svs-benchmark/benchmark.h"

// svstest
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_string.hpp"

CATCH_TEST_CASE("SaveDirectoryChecker", "[benchmark]") {
    auto temp = svs_test::prepare_temp_directory_v2();
    auto checker = svsbenchmark::SaveDirectoryChecker();

    auto table = toml::table(
        {{"empty", ""},
         {"key1", svs::lib::save(temp / "test")},
         {"key2", svs::lib::save(temp / "test2")},
         {"dne", svs::lib::save(temp / "deoes_not" / "exist")}}
    );

    CATCH_SECTION("Check Uniqueness") {
        // Empty strings should become empty optionals.
        CATCH_REQUIRE(!checker.extract(table, "empty").has_value());

        auto path = checker.extract(table, "key1");
        CATCH_REQUIRE(path.has_value());
        CATCH_REQUIRE(path.value() == temp / "test");

        path = checker.extract(table, "key2");
        CATCH_REQUIRE(path.has_value());
        CATCH_REQUIRE(path.value() == temp / "test2");

        // We should get an error if we add a path that has already been seen.
        CATCH_REQUIRE_THROWS_MATCHES(
            checker.extract(table, "key1"),
            svs::ANNException,
            svs_test::ExceptionMatcher(
                Catch::Matchers::ContainsSubstring("found multiple times")
            )
        );

        // Should also get an exception for non-existent parents"
        CATCH_REQUIRE_THROWS_MATCHES(
            checker.extract(table, "dne"),
            svs::ANNException,
            svs_test::ExceptionMatcher(Catch::Matchers::ContainsSubstring("does not exist"))
        );
    }
}
