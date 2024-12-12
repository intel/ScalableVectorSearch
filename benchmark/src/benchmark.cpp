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

// svs-benchmark
#include "svs-benchmark/benchmark.h"

// svs
#include "svs/third-party/toml.h"

// third-party
#include "fmt/ostream.h"
#include "fmt/std.h"

// stl
#include <filesystem>
#include <optional>

namespace svsbenchmark {

void atomic_save(const toml::table& table, const std::filesystem::path& path) {
    auto temp_filename = path.stem();
    temp_filename += "_temp";
    temp_filename += path.extension();
    auto temppath = path.parent_path() / temp_filename;
    // Write the contents of the table to a temporary file.
    // Once the write is complete, use an atomic move to replace the actual destination
    // file.
    {
        auto ostream = svs::lib::open_write(temppath);
        ostream << table << '\n';
    }
    std::filesystem::rename(temppath, path);
}

void append_or_create(toml::table& table, const toml::table& data, std::string_view key) {
    auto itr = table.find(key);
    if (itr != table.end()) {
        // Get the node at the key.
        // It *should* be an array. If it isn't, abort!
        // Using a reference here is valid because we've checked that the iterator is valid.
        auto& node = itr->second;
        if (auto* array = node.as_array(); array) {
            array->push_back(data);
        } else {
            throw ANNEXCEPTION("Expected TOML node at path {} to be an array!", key);
        }
    } else {
        // Create and insert a new array.
        table.insert(key, toml::array(data));
    }
}

std::filesystem::path extract_filename(
    const svs::lib::ContextFreeLoadTable& table,
    std::string_view key,
    const std::optional<std::filesystem::path>& root
) {
    auto filename = svs::lib::load_at<std::filesystem::path>(table, key);
    bool prepend_root = root && filename.is_relative();
    if (prepend_root) {
        filename = *root / filename;
    }
    if (!std::filesystem::exists(filename)) {
        throw ANNEXCEPTION(
            "Could not find {}file {} (parsed from {})!",
            prepend_root ? "qualified " : "",
            filename,
            fmt::streamed(table.source_for(key))
        );
    }
    return filename;
}

/////
///// SaveDirectoryChecker
/////

std::optional<std::filesystem::path>
SaveDirectoryChecker::extract(const toml::table& table, std::string_view key) {
    const auto& node = svs::toml_helper::get_as<toml::node>(table, key);
    auto path = svs::lib::load<std::filesystem::path>(svs::lib::node_view(node));
    if (path.empty()) {
        return std::nullopt;
    }

    // Check for uniqueness.
    auto itr = directories_.find(path);
    if (itr != directories_.end()) {
        throw ANNEXCEPTION(
            "Save directory {} found multiple times ({}).",
            path,
            fmt::streamed(node.source())
        );
    }

    // Check if all directories up to the last exist.
    auto parent = path.parent_path();
    if (!std::filesystem::is_directory(parent)) {
        throw ANNEXCEPTION(
            "Parent for save directory {} does not exist! ({})",
            path,
            fmt::streamed(node.source())
        );
    }
    directories_.insert(itr, path);
    return std::optional<std::filesystem::path>(std::move(path));
}

} // namespace svsbenchmark
