/*
 * Copyright 2024 Intel Corporation
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
#include "svs/lib/expected.h"
#include "svs/lib/version.h"

// third-party
#include "svs/third-party/toml.h"

// stl
#include <string>
#include <string_view>

namespace svs {

inline bool config_file_by_extension(const std::filesystem::path& path) {
    return path.extension() == ".toml";
}

namespace lib {

// Reserved keyword for version strings in toml tables.
inline constexpr std::string_view config_version_key = "__version__";
inline constexpr std::string_view config_schema_key = "__schema__";
inline constexpr std::string_view config_file_name = "svs_config.toml";
inline constexpr std::string_view config_object_key = "object";

inline lib::Version get_version(const toml::table& table, std::string_view key) {
    return lib::Version(toml_helper::get_as<toml::value<std::string>>(table, key).get());
}

inline lib::Version get_version(const toml::table& table) {
    return lib::get_version(table, config_version_key);
}

inline std::string get_schema(const toml::table& table) {
    return toml_helper::get_as<toml::value<std::string>>(table, config_schema_key).get();
}

enum class TryLoadFailureReason {
    MissingFile,
    CouldNotResolveFile,
    MissingKey,
    InvalidSchema,
    InvalidVersion,
    Other
};

template <typename T> using TryLoadResult = lib::Expected<T, TryLoadFailureReason>;

} // namespace lib
} // namespace svs
