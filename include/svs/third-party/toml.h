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

#pragma once

// local
#include "svs/lib/datatype.h"
#include "svs/lib/exception.h"
#include "svs/lib/narrow.h"

// third-party
#include "toml++/toml.h"

// stl
#include <concepts>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace svs {

inline bool maybe_config_file(const std::filesystem::path& path) {
    return path.extension() == ".toml";
}

template <typename T>
const T& get_checked(const std::optional<T>& x, std::string_view key) {
    if (!x.has_value()) {
        if constexpr (has_datatype_v<T>) {
            throw ANNEXCEPTION(
                "Table has a key ",
                key,
                " but it's not the correct type (",
                name<datatype_v<T>>(),
                '!'
            );
        } else {
            throw ANNEXCEPTION("Table has a key ", key, " but it's not the correct type!");
        }
    }
    return *x;
}

namespace detail {
inline void
throw_if_empty(const toml::node_view<const toml::node>& view, std::string_view key) {
    if (!view) {
        throw ANNEXCEPTION("Table does not have an entry at position ", key, '!');
    }
}
} // namespace detail

template <typename T> T get_checked(const toml::table& table, std::string_view key) {
    auto view = table[key];
    detail::throw_if_empty(view, key);
    return get_checked(view.value<T>(), key);
}

/////
///// Prepare objects for TOML writing.
/////

///
/// @brief Convert integer types to a form for saving.
///
/// Either converts losslessly to a 64-bit signed integer or fails with an exception.
///
template <std::integral I> int64_t prepare(I x) { return lib::narrow<int64_t>(x); }

///
/// @brief Convert floating point types to a form for saving.
///
/// Either converts losslessly to a 64-bit float or fails with an exception.
///
template <std::floating_point F> double prepare(F x) { return lib::narrow<double>(x); }

template <typename T> toml::array prepare(const std::vector<T>& v) {
    auto array = toml::array{};
    for (const auto& x : v) {
        array.push_back(x);
    }
    return array;
}

inline std::string prepare(const std::filesystem::path& path) { return path; }

/////
///// Reading
/////

template <typename T>
T get(const toml::table& table, std::string_view key, T default_value) {
    return lib::narrow<T>(table[key].value_or(default_value));
}

///
/// @brief Get the value stored in the table corresponding to the given key.
///
/// @param table The table to extract from.
/// @param key The key to access.
///
/// Throws ANNException in the following cases:
///     * `key` does not exist in `table`.
///     * `key` does exist but is not convertible to `T`.
///     * `key` does exist but conversion to `T` is lossy.
///
template <typename T> T get(const toml::table& table, std::string_view key) {
    return lib::narrow<T>(get_checked<T>(table, key));
}

///
/// @brief Get the string value at the given path.
///
/// @param table The table to extract from.
/// @param key The key to access.
///
/// @returns The result of the table access if it exists and is a string. Otherwise, an
///          empty optiona.
///
inline std::optional<std::string> get(const toml::table& table, std::string_view key) {
    return table[key].value<std::string>();
}

inline std::string
get(const toml::table& table, std::string_view key, std::string_view default_value) {
    return std::string(table[key].value_or(default_value));
}

inline const toml::table& subtable(const toml::table& table, std::string_view key) {
    auto* sub = table[key].as_table();
    if (sub == nullptr) {
        throw ANNEXCEPTION("Tried to access non-existent subtable at key ", key, '!');
    }
    return *sub;
}

template <typename T>
std::vector<T> get_vector(const toml::table& table, std::string_view key) {
    // First, we need make sure what we have is actually an array.
    if (auto* array = table[key].as_array()) {
        auto v = std::vector<T>();
        for (const auto& item : *array) {
            v.push_back(lib::narrow<T>(item.template value<T>().value()));
        }
        return v;
    }
    throw ANNEXCEPTION("Key ", key, " does not point to an array!");
}

/////
///// Writing
/////

template <std::integral T> void emplace(toml::table& table, std::string_view key, T value) {
    table.emplace(key, lib::narrow<int64_t>(value));
}

template <std::floating_point T>
void emplace(toml::table& table, std::string_view key, T value) {
    table.emplace(key, lib::narrow_cast<double>(value));
}

inline void emplace(toml::table& table, std::string_view key, std::string_view value) {
    table.emplace(key, value);
}

} // namespace svs
