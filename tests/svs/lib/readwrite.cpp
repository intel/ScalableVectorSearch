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

// stdlib
#include <algorithm>
#include <cstdint>
#include <span>
#include <sstream>
#include <vector>

// svs
#include "svs/lib/readwrite.h"

// catch2
#include "catch2/catch_test_macros.hpp"

struct TestHeader {
    TestHeader() = default;

    size_t a;
    double b;
    int64_t c;
    char __reserved__[10];
};

CATCH_TEST_CASE("Testing Read/Write", "[core][core_utils]") {
    // Use a `stringstream` since it acts as an in-memory version of `ofstream` and
    // `ifstream`.
    auto stream = std::stringstream{};
    CATCH_SECTION("Native Types") {
        CATCH_REQUIRE(svs::lib::write_binary(stream, int{10}) == sizeof(int));
        CATCH_REQUIRE(svs::lib::write_binary(stream, double{-100.452}) == sizeof(double));

        // Read the values back out.
        stream.seekg(0, std::stringstream::beg);
        auto i = svs::lib::read_binary<int>(stream);
        CATCH_REQUIRE(i == 10);
        auto d = svs::lib::read_binary<double>(stream);
        CATCH_REQUIRE(d == -100.452);
    }

    CATCH_SECTION("Custom Types") {
        auto header = TestHeader();
        header.a = 1234;
        header.b = -1000;
        header.c = -2304987;
        CATCH_REQUIRE(svs::lib::write_binary(stream, header) == sizeof(header));
        stream.seekg(0, std::stringstream::beg);
        auto read = svs::lib::read_binary<TestHeader>(stream);
        CATCH_REQUIRE(header.a == read.a);
        CATCH_REQUIRE(header.b == read.b);
        CATCH_REQUIRE(header.c == read.c);
        CATCH_REQUIRE(
            std::equal(header.__reserved__, header.__reserved__ + 10, read.__reserved__)
        );
    }

    CATCH_SECTION("Vectors and Spans") {
        auto a = std::vector<size_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        auto b = std::vector<size_t>(a.size());
        // Write `a` directly.
        CATCH_REQUIRE(svs::lib::write_binary(stream, a) == sizeof(size_t) * a.size());

        stream.seekg(0, std::stringstream::beg);
        svs::lib::read_binary(stream, b);
        CATCH_REQUIRE(std::equal(a.begin(), a.end(), b.begin()));

        // Write as a span
        auto aspan = std::span(a.data(), a.size());
        CATCH_REQUIRE(
            svs::lib::write_binary(stream, aspan) == sizeof(size_t) * aspan.size()
        );

        stream.seekg(0, std::stringstream::beg);
        svs::lib::read_binary(stream, b);
        CATCH_REQUIRE(std::equal(a.begin(), a.end(), b.begin()));
    }
}
