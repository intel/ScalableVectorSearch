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
