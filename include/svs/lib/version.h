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

// Version class and global library version.
#pragma once

// svs
#include "svs/lib/exception.h"

// third-party
#include "svs/third-party/fmt.h"

// stl
#include <charconv>
#include <string>
#include <string_view>

// Forward Declaration.
namespace svs::lib {
struct Version;
}

namespace fmt {
template <> struct formatter<svs::lib::Version> : public svs::format_empty {
    auto format(const auto& x, auto& ctx) const {
        return fmt::format_to(ctx.out(), "v{}.{}.{}", x.major, x.minor, x.patch);
    }
};
} // namespace fmt

namespace svs::lib {

///
/// @brief Parse the provided view as a base-10 integer.
///
/// @tparam T The integer type to parse.
///
/// @param view The object to parse.
///
/// @returns The parsed integer.
///
/// @throws svs::ANNException if anything goes wrong during parsing.
///
template <std::integral T> [[nodiscard]] T parse_int(std::string_view view) {
    auto value = T{0};
    auto result = std::from_chars(view.begin(), view.end(), value);
    if (result.ec != std::errc()) {
        throw ANNEXCEPTION("Something went wrong with number parsing!");
    }
    return value;
}

///
/// @brief A representation of the typical three-numbered version identifier.
///
/// The version numbers are expected to roughly follow semantic versioning.
///
/// * MAJOR versions are incremented when incompatible API changes are made.
/// * MINOR versions are incremented when functionality is added in a backward compatible
///     manner.
/// * PATCH versions are for backwards compatible bug fixes.
///
/// In general, no guarentees are made with a version number "v0.0.x".
/// Such items are experimental and should not be relied upon.
///
/// Version numbers "v0.x.y" represent actively changing APIs and should be used with care.
///
struct Version {
    /// @brief Return the formatted version as "vMAJOR.MINOR.PATCH".
    std::string str() const { return fmt::format("{}", *this); }

    /// @brief Construct a new Version class.
    constexpr explicit Version(size_t major, size_t minor, size_t patch)
        : major{major}
        , minor{minor}
        , patch{patch} {}

    ///
    /// @brief Construct a new Version class by parsing a formatted string.
    ///
    /// @throws svs::ANNException if the string is malformed.
    ///
    /// The string to be formatted should be *exactly* in the form "vMAJOR.MINOR.PATCH"
    /// where each of MAJOR, MINOR, and PATCH is a positive base-10 integer.
    ///
    explicit Version(std::string_view v) {
        auto npos = v.npos;
        if (!v.starts_with('v')) {
            throw ANNEXCEPTION("Formatted version string doesn't begin with a 'v'!");
        }

        auto mallformed = []() { throw ANNEXCEPTION("Malformed version!"); };

        size_t start = 1;
        size_t stop = v.find('.', start);
        if (stop == npos) {
            mallformed();
        }
        major = parse_int<size_t>(v.substr(start, stop - start));

        // Parse minor
        start = stop + 1;
        stop = v.find('.', start);
        if (stop == npos) {
            mallformed();
        }
        minor = parse_int<size_t>(v.substr(start, stop - start));
        // parse to the end of the string.
        patch = parse_int<size_t>(v.substr(stop + 1));
    }

    ///// Members
    size_t major;
    size_t minor;
    size_t patch;
};

///
/// @brief Compare two versions for equality.
///
/// Two versions are equal if all fields compare equal.
///
inline constexpr bool operator==(const Version& x, const Version& y) {
    return (x.major == y.major) && (x.minor == y.minor) && (x.patch == y.patch);
}

/// @brief Compare two versions for a "less than" relationship.
inline constexpr bool operator<(const Version& x, const Version& y) {
    // Compare major.
    if (x.major < y.major) {
        return true;
    } else if (x.major > y.major) {
        return false;
    }

    // Major is equal -- compare minor.
    if (x.minor < y.minor) {
        return true;
    } else if (x.minor > y.minor) {
        return false;
    }

    // Minor is equal -- compare patch.
    return x.patch < y.patch;
}

inline constexpr bool operator>(const Version& x, const Version& y) { return y < x; }

inline constexpr bool operator<=(const Version& x, const Version& y) {
    return x == y || x < y;
}
inline constexpr bool operator>=(const Version& x, const Version& y) {
    return x == y || x > y;
}

///
/// Global Library Version
/// Macro-defines are established in the top level CMakeLists.txt.
///
inline constexpr Version svs_version =
    Version(SVS_VERSION_MAJOR, SVS_VERSION_MINOR, SVS_VERSION_PATCH);

} // namespace svs::lib
