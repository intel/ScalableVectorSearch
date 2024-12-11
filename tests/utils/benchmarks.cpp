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

#include <sstream>
#include <string>
#include <tuple>

#include "catch2/catch_test_macros.hpp"

#include "tests/utils/benchmarks.h"

// Tests for utilties
CATCH_TEST_CASE("Testing Name Generation", "[testing_utilities]") {
    std::ostringstream stream{};
    // type_names
    CATCH_SECTION("type_names: One Argument") {
        svs_test::type_names<float>(stream);
        CATCH_REQUIRE(stream.str() == "float32");
    }

    CATCH_SECTION("type_names: Two Arguments") {
        svs_test::type_names<float, uint8_t>(stream);
        CATCH_REQUIRE(stream.str() == "float32_uint8");
    }

    CATCH_SECTION("type_names: Three Arguments") {
        svs_test::type_names<float, svs_test::Val<100>, int64_t>(stream);
        CATCH_REQUIRE(stream.str() == "float32_100_int64");
    }

    // TypeNameGenerator
    CATCH_SECTION("TypeNameGenerator: One Argument") {
        svs_test::TypeNameGenerator<std::tuple<svs_test::Val<0>>>::generate(stream);
        CATCH_REQUIRE(stream.str() == "_<0>");
    }

    CATCH_SECTION("TypeNameGenerator: Two Arguments") {
        svs_test::TypeNameGenerator<std::tuple<svs_test::Val<0>, double>>::generate(stream);
        CATCH_REQUIRE(stream.str() == "_<0_float64>");
    }

    CATCH_SECTION("TypeNameGenerator: Three Arguments") {
        svs_test::TypeNameGenerator<
            std::tuple<svs_test::Val<0>, double, svs_test::Val<-1>>>::generate(stream);
        CATCH_REQUIRE(stream.str() == "_<0_float64_-1>");
    }

    // value_names
    CATCH_SECTION("value_names: One Argument") {
        svs_test::value_names(stream, 1);
        CATCH_REQUIRE(stream.str() == "_1");
    }

    CATCH_SECTION("value_names: Two Arguments") {
        svs_test::value_names(stream, 1, 1.0);
        CATCH_REQUIRE(stream.str() == "_1_1");
    }

    CATCH_SECTION("value_names: Three Arguments") {
        svs_test::value_names(stream, 1, 1.0, "string");
        CATCH_REQUIRE(stream.str() == "_1_1_string");
    }
}

CATCH_TEST_CASE("Testing Benchmark Naming", "[testing_utilities]") {
    auto direct = svs_test::benchmark_name<std::tuple<float, svs_test::Val<8>, int64_t>>(
        "MyPrefix", 1, "hello"
    );
    CATCH_REQUIRE(direct == "MyPrefix_<float32_8_int64>_1_hello");
    auto macro =
        BENCHMARK_NAME_TEMPLATE("MyPrefix", (float, svs_test::Val<8>, int64_t), 1, "hello");
    CATCH_REQUIRE(direct == macro);

    // Now, test various combinations of parameters
    CATCH_REQUIRE(BENCHMARK_NAME_TEMPLATE("ABC", ()) == "ABC");
    CATCH_REQUIRE(BENCHMARK_NAME_TEMPLATE("ABC", (), 1) == "ABC_1");
    CATCH_REQUIRE(BENCHMARK_NAME_TEMPLATE("ABC", (), 1, 2) == "ABC_1_2");
    CATCH_REQUIRE(BENCHMARK_NAME_TEMPLATE("ABC", (svs_test::Val<2>)) == "ABC_<2>");
    CATCH_REQUIRE(
        BENCHMARK_NAME_TEMPLATE("ABC", (svs_test::Val<2>, float)) == "ABC_<2_float32>"
    );
    CATCH_REQUIRE(
        BENCHMARK_NAME_TEMPLATE("ABC", (svs_test::Val<2>, float), 10) ==
        "ABC_<2_float32>_10"
    );
}
