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

// local
// #include "svs/lib/datatype.h"
#include "svs/lib/exception.h"
#include "svs/lib/meta.h"
#include "svs/lib/narrow.h"
#include "svs/lib/timing.h"

// third-party
#include "fmt/ostream.h"
#include "toml++/toml.h"

// stl
#include <chrono>
#include <concepts>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace svs {

namespace toml_helper {

namespace detail {
template <typename T> struct IsTomlType {
    static constexpr bool value = false;
};

template <> struct IsTomlType<toml::node> {
    static constexpr bool value = true;
};

template <> struct IsTomlType<toml::table> {
    static constexpr bool value = true;
};

template <> struct IsTomlType<toml::array> {
    static constexpr bool value = true;
};

template <typename T> struct IsTomlType<toml::value<T>> {
    static constexpr bool value = true;
};

template <> struct IsTomlType<toml::date_time> {
    static constexpr bool value = true;
};

template <typename T> struct TypeMapping;

template <typename T> inline constexpr bool is_toml_value_v = false;
template <typename T> inline constexpr bool is_toml_value_v<toml::value<T>> = true;

// Built-in type mapping.
template <std::integral I> struct TypeMapping<I> {
    using type = int64_t;
};
template <> struct TypeMapping<bool> {
    using type = bool;
};
template <std::floating_point F> struct TypeMapping<F> {
    using type = double;
};
template <> struct TypeMapping<std::string> {
    using type = std::string;
};
// For better error messages.
template <typename T> struct NameMap;
template <> struct NameMap<int64_t> {
    static constexpr std::string_view name() { return "int64"; }
};
template <> struct NameMap<bool> {
    static constexpr std::string_view name() { return "bool"; }
};
template <> struct NameMap<double> {
    static constexpr std::string_view name() { return "float64"; }
};
template <> struct NameMap<std::string> {
    static constexpr std::string_view name() { return "string"; }
};

template <> struct NameMap<toml::table> {
    static constexpr std::string_view name() { return "toml-table"; }
};
template <> struct NameMap<toml::array> {
    static constexpr std::string_view name() { return "toml-array"; }
};
template <typename T> struct NameMap<toml::value<T>> {
    static constexpr std::string_view name() { return NameMap<T>::name(); }
};

} // namespace detail

template <typename T> inline constexpr bool is_toml_type_v = detail::IsTomlType<T>::value;

template <typename T>
concept TomlType = is_toml_type_v<T>;

template <typename T>
concept TomlValue = detail::is_toml_value_v<T>;

template <typename T> using type_mapping_t = typename detail::TypeMapping<T>::type;

template <typename T>
concept HasTypeMapping = requires { typename detail::TypeMapping<T>::type; };

// Try to safely convert the node reference to a remore refined type.
template <typename T>
    requires is_toml_type_v<T>
const T& get_as(const toml::node& node) {
    const auto* p = node.as<T>();
    if (p == nullptr) {
        throw ANNEXCEPTION(
            "Bad node cast at {} to type {}!",
            fmt::streamed(node.source()),
            detail::NameMap<T>::name()
        );
    }
    return *p;
}

// It is useful to hook into the path-checking at a higher level, but defer the actual
// refinement until later.
//
// This overload makes returning a node-reference more efficient.
template <> inline const toml::node& get_as<toml::node>(const toml::node& node) {
    return node;
}

template <HasTypeMapping T> T get_as(const toml::node& node) {
    return lib::narrow<T>(get_as<toml::value<type_mapping_t<T>>>(node).get());
}

template <typename T>
auto get_as(const toml::table& table, std::string_view key) -> decltype(auto) {
    auto view = table[key];
    if (!view) {
        throw ANNEXCEPTION(
            "Bad access to key {} in table at {}.", key, fmt::streamed(table.source())
        );
    }
    return get_as<T>(*view.node());
}

} // namespace toml_helper

///
/// Construct a date_time
///

inline toml::date_time date_time() {
    auto now = std::chrono::system_clock::now();
    auto today = std::chrono::floor<std::chrono::days>(now);
    auto ymd = std::chrono::year_month_day(today);
    auto date = toml::date(
        static_cast<int>(ymd.year()),
        static_cast<unsigned>(ymd.month()),
        static_cast<unsigned>(ymd.day())
    );
    auto hh_mm_ss = std::chrono::hh_mm_ss(now - today);
    auto time = toml::time{
        lib::narrow_cast<uint8_t>(hh_mm_ss.hours().count()),
        lib::narrow_cast<uint8_t>(hh_mm_ss.minutes().count()),
        lib::narrow_cast<uint8_t>(hh_mm_ss.seconds().count()),
    };
    return toml::date_time(date, time);
}

} // namespace svs
