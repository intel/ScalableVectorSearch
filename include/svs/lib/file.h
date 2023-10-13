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

// svs
#include "svs/lib/exception.h"

// stl
#include <filesystem>
#include <fstream>

namespace svs::lib {

inline bool check_file(const std::filesystem::path& path, std::ios_base::openmode mode) {
    namespace fs = std::filesystem;
    // If we're reading and the file doesn't exist, thrown an exception
    if (mode & std::ios_base::in) {
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
