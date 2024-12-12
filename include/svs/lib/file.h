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

// stl
#include <filesystem>
#include <fstream>

namespace svs::lib {

inline bool directory_safe_to_create_or_write(const std::filesystem::path& dir) {
    namespace fs = std::filesystem;

    // If this directory is a top level directory - we can't use it.
    if (!dir.has_parent_path()) {
        return false;
    }

    // Check that the parent exists.
    auto parent = dir.parent_path();
    if (!fs::is_directory(parent)) {
        return false;
    }

    // Make sure that the given directory is either already a directory, or doesn't exist.
    if (!fs::is_directory(dir) && fs::exists(dir)) {
        return false;
    }

    // TODO: File permissions ...
    return true;
}

inline bool check_file(const std::filesystem::path& path, std::ios_base::openmode mode) {
    namespace fs = std::filesystem;

    // If we are opening a file for only reading and it does not exist - throw an error.
    if ((mode & std::ios_base::in) && !(mode & std::ios_base::out)) {
        auto status = fs::status(path);
        bool exists = fs::exists(path);
        if (!exists || !fs::status_known(status) || fs::is_directory(status)) {
            throw ANNEXCEPTION("Trying to open non-existent file {} for reading!", path);
        }

        if (fs::is_empty(path)) {
            throw ANNEXCEPTION("Trying to open empty file {} for reading!", path);
        }
    }

    if (mode & std::ios_base::out) {
        // Check that the directory exists.
        if (path.has_parent_path() && !fs::is_directory(path.parent_path())) {
            throw ANNEXCEPTION(
                "Trying to open a file {} for writing in a non-existent directory!", path
            );
        }
    }
    return true;
}

namespace detail {
template <typename T> void check_stream(const T& /*x*/) {}
} // namespace detail

namespace file_flags {
static constexpr std::ios_base::openmode open =
    std::ios_base::in | std::ios_base::out | std::ios_base::binary;

static constexpr std::ios_base::openmode open_write =
    std::ios_base::out | std::ios_base::binary;

static constexpr std::ios_base::openmode open_read =
    std::ios_base::in | std::ios_base::binary;
} // namespace file_flags

inline std::fstream
open(const std::filesystem::path& path, std::ios_base::openmode mode = file_flags::open) {
    check_file(path, mode);
    return std::fstream(path, mode);
}

inline std::ofstream open_write(
    const std::filesystem::path& path, std::ios_base::openmode mode = file_flags::open_write
) {
    check_file(path, mode);
    return std::ofstream(path, mode);
}

inline std::ifstream open_read(
    const std::filesystem::path& path, std::ios_base::openmode mode = file_flags::open_read
) {
    check_file(path, mode);
    return std::ifstream(path, mode);
}

} // namespace svs::lib
