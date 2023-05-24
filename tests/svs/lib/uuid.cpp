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

// Header under test
#include "svs/lib/uuid.h"
#include "svs/lib/readwrite.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>

namespace lib = svs::lib;

void validate_uuid(std::string_view uuid) {
    // 32 octets plus 4 hyphens
    CATCH_REQUIRE(uuid.size() == 36);
    // Check that the hyphens are in the right spot.
    CATCH_REQUIRE(uuid[8] == '-');
    CATCH_REQUIRE(uuid[13] == '-');
    CATCH_REQUIRE(uuid[18] == '-');
    CATCH_REQUIRE(uuid[23] == '-');
    // Check version 4
    CATCH_REQUIRE(uuid[14] == '4');
    // Check variant 1
    auto var = lib::ascii_hex_to_byte(uuid[19]);
    CATCH_REQUIRE(var >= 8);
}

CATCH_TEST_CASE("UUID", "[lib][uuid]") {
    CATCH_SECTION("Hex and Nibbles") {
        std::string numbers = "0123456789abcdefABCDEF";
        auto expected = std::vector<uint8_t>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                             11, 12, 13, 14, 15, 10, 11, 12, 13, 14, 15};

        // ASCII Hex to Byte
        CATCH_REQUIRE(numbers.size() == expected.size());
        for (size_t i = 0; i < numbers.size(); ++i) {
            char ch = numbers[i];
            uint8_t ex = expected[i];
            CATCH_REQUIRE(lib::ascii_hex_to_byte(ch) == ex);
        }

        // Byte to ASCII hex
        for (size_t i = 0; i < 16; ++i) {
            uint8_t byte = lib::narrow<uint8_t>(i);
            CATCH_REQUIRE(lib::nibble_to_ascii_hex(byte) == numbers[i]);
        }

        // Make sure errors are thrown for all non-ascii characters.
        auto valid = std::unordered_map<char, uint8_t>();
        for (size_t i = 0; i < numbers.size(); ++i) {
            valid[numbers[i]] = expected[i];
        }

        size_t seen = 0; // number of valid characters seen.
        int istart = std::numeric_limits<char>::min();
        int iend = std::numeric_limits<char>::max();

        for (int i = istart; i < iend; ++i) {
            char ch = lib::narrow<char>(i);
            // If this character is valid.
            if (auto it = valid.find(ch); it != valid.end()) {
                CATCH_REQUIRE(it->first == ch);
                CATCH_REQUIRE(lib::ascii_hex_to_byte(it->first) == it->second);
                ++seen;
            } else {
                CATCH_REQUIRE_THROWS_AS(lib::ascii_hex_to_byte(ch), svs::ANNException);
            }
        }
        CATCH_REQUIRE(seen == numbers.size());
    }

    CATCH_SECTION("Hex and Bytes") {
        for (size_t i = 0; i < 256; ++i) {
            auto ss = std::ostringstream();
            ss << std::hex << std::setw(2) << std::setfill('0') << i;
            std::string expected = ss.str();
            CATCH_REQUIRE(expected.size() == 2);

            uint8_t byte = lib::narrow<uint8_t>(i);
            auto chars = lib::byte_to_ascii_hex(byte);
            CATCH_REQUIRE(expected == std::string({chars.first, chars.second}));
            CATCH_REQUIRE(i == lib::ascii_octet_to_byte(expected[0], expected[1]));
        }
    }

    CATCH_SECTION("Generation") {
        for (size_t i = 0; i < 1000; ++i) {
            auto id = lib::UUID();
            std::string str = id.str();
            validate_uuid(str);

            auto other = lib::UUID(str);
            CATCH_REQUIRE(id == other);
        }
    }

    CATCH_SECTION("Zero Initialization") {
        auto id = lib::UUID{lib::ZeroInitializer()};
        const auto& raw = id.raw();

        auto pred = [](const uint8_t v) { return v == 0; };
        CATCH_REQUIRE(std::all_of(raw.begin(), raw.end(), pred));
    }

    CATCH_SECTION("Error handling") {
        CATCH_REQUIRE(lib::UUID() != lib::UUID());
        // Round-trip UUID strings
        auto uuid_string = std::string("ac4C2b21-E7b7-446A-983a-90ed1e79D7e2");
        auto uuid_string_lower = std::string("ac4c2b21-e7b7-446a-983a-90ed1e79d7e2");
        CATCH_REQUIRE(lib::UUID(uuid_string).str() == uuid_string_lower);

        // String is too short
        auto uuid_string_short = std::string("ac492b21-e7b7-446a-983a-90ed1e7907e");
        CATCH_REQUIRE_THROWS_AS(lib::UUID(uuid_string_short), svs::ANNException);

        // Can't decode correctly as ascii hex numbers
        auto uuid_string_invalid = std::string("ac492b21-e7bx-446a-983a-90ed1e7907e2");
        CATCH_REQUIRE_THROWS_AS(lib::UUID(uuid_string_invalid), svs::ANNException);

        // Hyphen is missing
        auto uuid_string_badhyphen = std::string("ac492b21-e7b7?446a-983a-90ed1e7907e2");
        CATCH_REQUIRE_THROWS_AS(lib::UUID(uuid_string_badhyphen), svs::ANNException);
    }

    CATCH_SECTION("Serialization") {
        auto stream = std::stringstream{};
        auto uuid = lib::UUID();
        lib::write_binary(stream, uuid);

        auto deserialized = lib::UUID(lib::ZeroInitializer());
        lib::read_binary(stream, deserialized);
        CATCH_REQUIRE(uuid == deserialized);
    }

    CATCH_SECTION("Constexpr") {
        constexpr auto uuid_constexpr = lib::UUID("f5bbbc26-e3bf-41bb-96f5-66fea1b55bd1");
        auto uuid_notconstexpr = lib::UUID("f5bbbc26-e3bf-41bb-96f5-66fea1b55bd1");
        CATCH_REQUIRE(uuid_constexpr == uuid_notconstexpr);
    }
}
