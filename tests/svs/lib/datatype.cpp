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

// svs
#include "svs/lib/datatype.h"
#include "svs/lib/saveload.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <string_view>
#include <type_traits>

template <typename CPP, svs::DataType SVS> void test(std::string_view name) {
    CATCH_REQUIRE(std::is_same_v<CPP, svs::cpp_type_t<SVS>>);
    CATCH_REQUIRE(svs::datatype_v<CPP> == SVS);

    // Make sure "name" is constexpr
    constexpr auto constexpr_name = svs::name<SVS>();

    // TODO: Catch doesn't like comparing string_view's?
    auto name_string = std::string(name);
    CATCH_REQUIRE(std::string(constexpr_name) == name_string);
    CATCH_REQUIRE(std::string(svs::name(SVS)) == name_string);

    // Make sure parsing works.
    CATCH_REQUIRE(svs::parse_datatype(constexpr_name) == SVS);

    // saving and loading.
    CATCH_REQUIRE(svs::lib::load<svs::DataType>(svs::lib::save(SVS)) == SVS);
}

CATCH_TEST_CASE("Data Type", "[core][datatype]") {
    using DataType = svs::DataType;
    CATCH_SECTION("Type Conversion") {
        test<uint8_t, DataType::uint8>("uint8");
        test<uint16_t, DataType::uint16>("uint16");
        test<uint32_t, DataType::uint32>("uint32");
        test<uint64_t, DataType::uint64>("uint64");
        CATCH_REQUIRE(svs::parse_datatype("uint128") == DataType::undef);

        test<int8_t, DataType::int8>("int8");
        test<int16_t, DataType::int16>("int16");
        test<int32_t, DataType::int32>("int32");
        test<int64_t, DataType::int64>("int64");
        CATCH_REQUIRE(svs::parse_datatype("int128") == DataType::undef);

        test<svs::Float16, DataType::float16>("float16");
        test<float, DataType::float32>("float32");
        test<double, DataType::float64>("float64");
        CATCH_REQUIRE(svs::parse_datatype("float128") == DataType::undef);

        test<std::byte, DataType::byte>("byte");

        CATCH_REQUIRE(svs::datatype_v<std::string> == DataType::undef);
        CATCH_REQUIRE(svs::parse_datatype("undef") == DataType::undef);
    }

    CATCH_SECTION("Hash") {
        CATCH_REQUIRE(DataType::uint8 == DataType::uint8);
        CATCH_REQUIRE(DataType::uint8 != DataType::uint16);

        std::hash<DataType> hash{};
        CATCH_REQUIRE(hash(DataType::int16) == hash(DataType::int16));
        CATCH_REQUIRE(hash(DataType::int16) != hash(DataType::float32));

        // Use in a hash table.
        std::unordered_map<DataType, int> table{};
        table[DataType::float16] = 5;
        table[DataType::float32] = 10;
        CATCH_REQUIRE(!table.contains(DataType::int8));
        CATCH_REQUIRE(table.contains(DataType::float16));
        CATCH_REQUIRE(table.contains(DataType::float32));

        CATCH_REQUIRE(table[DataType::float16] == 5);
        CATCH_REQUIRE(table[DataType::float32] == 10);
    }

    CATCH_SECTION("Formatting") {
        CATCH_REQUIRE(svs::lib::format({DataType::float32}) == "float32");
        CATCH_REQUIRE(
            svs::lib::format({DataType::uint8, DataType::uint16}) == "uint8 and uint16"
        );
        CATCH_REQUIRE(
            svs::lib::format({DataType::uint8, DataType::uint16, DataType::float32}) ==
            "uint8, uint16, and float32"
        );
    }

    CATCH_SECTION("Pointer Erasure") {
        std::vector<int32_t> v(100);
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = i;
        }

        svs::ConstErasedPointer ptr{v.data()};
        CATCH_REQUIRE(ptr.type() == DataType::int32);

        const auto* derived = svs::get<int32_t>(ptr);
        for (int32_t i = 0, imax = svs::lib::narrow<int32_t>(v.size()); i < imax; ++i) {
            CATCH_REQUIRE((derived[i] == i));
        }

        // Incorrect conversion throws an exception.
        CATCH_REQUIRE_THROWS_AS(svs::get<float>(ptr), svs::ANNException);
    }
}
