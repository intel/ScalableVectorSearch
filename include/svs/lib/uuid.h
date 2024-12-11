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

#pragma once

// svs
#include "svs/lib/exception.h"
#include "svs/lib/misc.h"
#include "svs/lib/narrow.h"

// stl
#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <string_view>

namespace svs::lib {

///
/// @brief Convert an ASCII encoded hexadecimal alphanumeric.
///
/// @param ch The character to convert. See preconditions below.
///
/// @returns The machine equivalend to `ch`.
///
/// Parameter `ch` must be one of the following values: "0123456789abcdefABCDEF".
/// If not, an `ANNException` will be thrown.
///
inline constexpr uint8_t ascii_hex_to_byte(char ch) {
    // Check if the number is in 0 to 9
    if (ch >= '0' && ch <= '9') {
        return lib::narrow_cast<uint8_t>(ch - '0');
    } else if (ch >= 'a' && ch <= 'f') {
        return lib::narrow_cast<uint8_t>((ch - 'a') + uint8_t(10));
    } else if (ch >= 'A' && ch <= 'F') {
        return lib::narrow_cast<uint8_t>((ch - 'A') + uint8_t(10));
    }
    throw ANNEXCEPTION("Character \"{}\" is not a hexadecimal digit!", ch);
}

///
/// @brief Convert the argument to an alphanumeric ASCII hexadecimal digit.
///
/// @param byte The value to convert. Must be between 0 and 15 inclusive. If not, the
///        behavior is undefined.
///
/// @returns The ASCII encoding for `byte`
///
inline constexpr char nibble_to_ascii_hex(uint8_t byte) {
    assert(byte <= 15);
    return (byte < 10) ? ('0' + byte) : ('a' + (byte - uint8_t(10)));
}

inline constexpr uint8_t ascii_octet_to_byte(char hi, char lo) {
    return (ascii_hex_to_byte(hi) << 4) | ascii_hex_to_byte(lo);
}

inline constexpr std::pair<char, char> byte_to_ascii_hex(uint8_t byte) {
    auto mask = uint8_t(0xf);
    return std::make_pair(
        nibble_to_ascii_hex((byte >> 4) & mask), nibble_to_ascii_hex(byte & mask)
    );
}

///// Universally unique identifier: Version 4 - variant 1
///// https://en.wikipedia.org/wiki/Universally_unique_identifier
class UUID {
  public:
    /// Number of characters in a formatted string.
    static const size_t num_formatted_chars = 36;
    /// The number of bytes used to encode a UUID.
    static const size_t num_bytes = 16;

    static const unsigned version = 4;
    static const unsigned variant = 1;

    /// @brief Construct a zero-initialized UUID struct.
    constexpr UUID(ZeroInitializer SVS_UNUSED(tag)) {
        std::fill(uuid_.begin(), uuid_.end(), uint8_t(0));
    }

    /// @brief Construct a UUID directly.
    constexpr UUID(std::array<uint8_t, num_bytes> data)
        : uuid_{std::move(data)} {}

    ///
    /// @brief Construct a randomly generated UUID.
    ///
    /// The generated UUID will be compliant with Version 4 (randomly generated), Variant 1.
    /// See https://en.wikipedia.org/wiki/Universally_unique_identifier for details.
    ///
    UUID() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distribution(0, 256);
        // The formatted string is always 36 characters long.
        for (size_t i = 0; i < num_bytes; i++) {
            uuid_[i] = distribution(gen);
        }

        // Special positions
        // Version
        {
            uint8_t& digit = uuid_[flip(6)];
            auto mask = uint8_t(0x0f);
            digit = (digit & mask) | uint8_t(0x40);
        }

        // Variant
        {
            uint8_t& digit = uuid_[flip(8)];
            auto mask = uint8_t(0x3f);
            digit = (digit & mask) | uint8_t(0x80);
        }
    }

    ///
    /// @brief Parse a UUID from a string.
    ///
    /// @param str The string to parse.
    ///
    /// The string should be exactly of the form: "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"
    /// where
    ///
    /// * `X` is an ASCII hexadecimal alphanumeric. Valid values include 0-9, a-f, and A-F.
    /// * `-` is a literal hyphen.
    ///
    /// Throws exceptions in the following cases:
    ///
    /// * `str.size() != UUID::num_formatted_chars`: UUID string cannot possibly be
    ///   formatted correctly.
    /// * Hyphens to not occur at character positions 8, 13, 18, and 23.
    /// * Non-hexadecimal alphanumeric characters occur at any position that should not
    ///   belong to a hyphen.
    ///
    constexpr UUID(std::string_view str) {
        const size_t nchars = str.size();
        const size_t expected = num_formatted_chars;
        if (nchars != expected) {
            throw ANNEXCEPTION(
                "UUID string does not contain {} characters! Instead, it has {}!",
                expected,
                nchars
            );
        }

        auto hyphens = hyphen_locations();
        auto next_hyphen = hyphens.begin();

        auto is = str.begin();
        auto end = str.end();
        size_t i = 0;
        while (is != end) {
            assert(is != end);
            auto hi = char{*is};
            ++is;
            assert(is != end);
            auto lo = char{*is};
            ++is;
            uuid_[flip(i)] = ascii_octet_to_byte(hi, lo);

            if (i == *next_hyphen) {
                assert(is != end);
                auto hyphen = char{*is};
                ++is;
                if (hyphen != '-') {
                    throw ANNEXCEPTION("Malformed UUID string!");
                }
                ++next_hyphen;
            }
            ++i;
        }
        assert(is == end);
    }

    std::string str() const {
        auto s = std::string(36, char(0));
        auto is = s.begin();

        auto hyphens = hyphen_locations();
        auto next_hyphen = hyphens.begin();
        for (size_t i = 0; i < num_bytes; ++i) {
            auto [hi, lo] = byte_to_ascii_hex(uuid_[flip(i)]);
            assert(is != s.end());
            *is = hi;
            ++is;
            assert(is != s.end());
            *is = lo;
            ++is;

            if (i == *next_hyphen) {
                assert(is != s.end());
                *is = '-';
                ++is;
                ++next_hyphen;
            }
        }
        assert(is == s.end());
        return s;
    }

    constexpr const std::array<uint8_t, num_bytes>& raw() const { return uuid_; }

  private:
    static constexpr std::array<size_t, 5> hyphen_locations() {
        return std::array<size_t, 5>{3, 5, 7, 9, num_bytes};
    }

    static constexpr size_t flip(size_t i) { return num_bytes - i - 1; }

    // Members
    std::array<uint8_t, num_bytes> uuid_;
};

constexpr bool operator==(const UUID& a, const UUID& b) { return a.raw() == b.raw(); }

// Static checks.
static_assert(sizeof(UUID) == 16, "UUID must be 16 bytes!");
static_assert(std::is_trivially_copyable_v<UUID>, "UUID must be trivially copyable!");

inline constexpr UUID ZeroUUID = UUID(ZeroInitializer());

} // namespace svs::lib
