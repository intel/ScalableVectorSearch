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

#include <cassert>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>

#include "svs/lib/float16.h"

#include "catch2/catch_test_macros.hpp"

#include "tests/utils/utils.h"

bool svs_test::compare_files(const std::string& a, const std::string& b) {
    auto x = std::ifstream(a, std::ifstream::binary | std::ifstream::ate);
    auto y = std::ifstream(b, std::ifstream::binary | std::ifstream::ate);
    if (x.fail() || x.fail()) {
        std::ostringstream message{};
        message << "File: " << (x.fail() ? a : b) << " could not be found!";
        throw std::runtime_error(message.str());
    }

    // Check file positions
    if (x.tellg() != y.tellg()) {
        return false;
    }

    // Seek back to the start and compare byte by byte.
    x.seekg(0, std::ifstream::beg);
    y.seekg(0, std::ifstream::beg);

    using char_type = std::ifstream::char_type;
    return std::equal(
        std::istreambuf_iterator<char_type>(x),
        std::istreambuf_iterator<char_type>(),
        std::istreambuf_iterator<char_type>(y)
    );
}

CATCH_TEST_CASE("Testing type_name", "[testing_utilities]") {
    // type_name
    CATCH_REQUIRE(svs_test::type_name<uint8_t>() == "uint8");
    CATCH_REQUIRE(svs_test::type_name<uint16_t>() == "uint16");
    CATCH_REQUIRE(svs_test::type_name<svs::Float16>() == "float16");
    CATCH_REQUIRE(svs_test::type_name<uint32_t>() == "uint32");
    CATCH_REQUIRE(svs_test::type_name<uint64_t>() == "uint64");

    CATCH_REQUIRE(svs_test::type_name<int8_t>() == "int8");
    CATCH_REQUIRE(svs_test::type_name<int16_t>() == "int16");
    CATCH_REQUIRE(svs_test::type_name<int32_t>() == "int32");
    CATCH_REQUIRE(svs_test::type_name<int64_t>() == "int64");

    CATCH_REQUIRE(svs_test::type_name<float>() == "float32");
    CATCH_REQUIRE(svs_test::type_name<double>() == "float64");

    CATCH_REQUIRE(svs_test::type_name<svs_test::Val<0>>() == "0");
    CATCH_REQUIRE(svs_test::type_name<svs_test::Val<100>>() == "100");
}
