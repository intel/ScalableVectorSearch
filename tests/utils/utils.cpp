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

#include "svs/lib/float16.h"

#include "catch2/catch_test_macros.hpp"

#include "tests/utils/utils.h"

#include <cassert>
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

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

std::vector<uint64_t> svs_test::permute_indices(size_t max_id) {
    // Construct a scrambled version of the ids to retrieve.
    auto ids = std::vector<uint64_t>(max_id);
    for (size_t i = 0; i < max_id; ++i) {
        ids.at(i) = i;
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(ids.begin(), ids.end(), g);
    return ids;
}

void svs_test::Lens::apply(toml::table* table, bool expect_exists) const {
    // Recurse until we get to the last key.
    size_t num_keys = key_chain_.size();
    for (size_t i = 0, imax = num_keys - 1; i < imax; ++i) {
        auto v = table->operator[](key_chain_[i]);
        if (!v) {
            throw ANNEXCEPTION(
                "Error accessing key {} of {}!", i, fmt::join(key_chain_, ", ")
            );
        }
        table = v.as<toml::table>();
        if (table == nullptr) {
            throw ANNEXCEPTION(
                "Cannot interpret key {} of {} as a table!", i, fmt::join(key_chain_, ", ")
            );
        }
    }

    const auto& last_key = key_chain_.back();
    auto [_, inserted] = table->insert_or_assign(last_key, *value_);
    if (inserted && expect_exists) {
        throw ANNEXCEPTION("Expected the last key {} to exist!", last_key);
    }
}

void svs_test::mutate_table(
    const std::filesystem::path& src,
    const std::filesystem::path& dst,
    std::initializer_list<svs_test::Lens> lenses
) {
    auto table = toml::parse_file(src.native());
    for (const auto& lens : lenses) {
        lens.apply(std::ref(table));
    }
    auto io = svs::lib::open_write(dst);
    io << table << "\n";
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
