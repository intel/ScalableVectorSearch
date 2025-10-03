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
#include "svs/lib/uuid.h"

// stl
#include <filesystem>
#include <fstream>
#include <span>

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

inline std::filesystem::path unique_temp_directory_path(const std::string& prefix) {
    namespace fs = std::filesystem;
    auto temp_dir = fs::temp_directory_path();
    // Try up to 10 times to create a unique directory.
    for (int i = 0; i < 10; ++i) {
        auto dir = temp_dir / (prefix + "-" + svs::lib::UUID().str());
        if (!fs::exists(dir)) {
            return dir;
        }
        return dir;
    }
    throw ANNEXCEPTION("Could not create a unique temporary directory!");
}

// RAII helper to create and delete a temporary directory.
struct UniqueTempDirectory {
    std::filesystem::path path;

    UniqueTempDirectory(const std::string& prefix)
        : path{unique_temp_directory_path(prefix)} {
        std::filesystem::create_directories(path);
    }

    ~UniqueTempDirectory() {
        try {
            std::filesystem::remove_all(path);
        } catch (...) {
            // Ignore errors.
        }
    }

    std::filesystem::path get() const { return path; }
    operator const std::filesystem::path&() const { return path; }
};

// Simple directory archiver to pack/unpack a directory to/from a stream.
// Uses a simple custom binary format.
// Not meant to be super efficient, just a simple way to serialize a directory
// structure to a stream.
struct DirectoryArchiver {
    using size_type = uint64_t;

    // TODO: Define CACHELINE_BYTES in a common place
    // rather than duplicating it here and in prefetch.h
    static constexpr auto CACHELINE_BYTES = 64;
    static constexpr size_type magic_number = 0x5e2d58d9f3b4a6c1;

    static size_type write_size(std::ostream& os, size_type size) {
        os.write(reinterpret_cast<const char*>(&size), sizeof(size));
        if (!os) {
            throw ANNEXCEPTION("Error writing to stream!");
        }
        return sizeof(size);
    }

    static size_type read_size(std::istream& is, size_type& size) {
        is.read(reinterpret_cast<char*>(&size), sizeof(size));
        if (!is) {
            throw ANNEXCEPTION("Error reading from stream!");
        }
        return sizeof(size);
    }

    static size_type write_name(std::ostream& os, const std::string& name) {
        auto bytes = write_size(os, name.size());
        os.write(name.data(), name.size());
        if (!os) {
            throw ANNEXCEPTION("Error writing to stream!");
        }
        return bytes + name.size();
    }

    static size_type read_name(std::istream& is, std::string& name) {
        size_type size = 0;
        auto bytes = read_size(is, size);
        name.resize(size);
        is.read(name.data(), size);
        if (!is) {
            throw ANNEXCEPTION("Error reading from stream!");
        }
        return bytes + size;
    }

    static size_type write_file(
        std::ostream& stream,
        const std::filesystem::path& path,
        const std::filesystem::path& root
    ) {
        namespace fs = std::filesystem;
        check_file(path, std::ios_base::in | std::ios_base::binary);

        // Write the filename as a string.
        std::string filename = fs::relative(path, root).string();
        auto header_bytes = write_name(stream, filename);
        if (!stream) {
            throw ANNEXCEPTION("Error writing to stream!");
        }

        // Write the size of the file.
        size_type filesize = fs::file_size(path);
        header_bytes += write_size(stream, filesize);
        if (!stream) {
            throw ANNEXCEPTION("Error writing to stream!");
        }

        // Now write the actual file contents.
        std::ifstream in(path, std::ios_base::in | std::ios_base::binary);
        if (!in) {
            throw ANNEXCEPTION("Error opening file {} for reading!", path);
        }
        stream << in.rdbuf();
        if (!stream) {
            throw ANNEXCEPTION("Error writing to stream!");
        }

        return header_bytes + filesize;
    }

    static size_type read_file(std::istream& stream, const std::filesystem::path& root) {
        namespace fs = std::filesystem;

        // Read the filename as a string.
        std::string filename;
        auto header_bytes = read_name(stream, filename);
        if (!stream) {
            throw ANNEXCEPTION("Error reading from stream!");
        }

        auto path = root / filename;
        auto parent_dir = path.parent_path();
        if (!fs::exists(parent_dir)) {
            fs::create_directories(parent_dir);
        } else if (!fs::is_directory(parent_dir)) {
            throw ANNEXCEPTION("Path {} exists and is not a directory!", root);
        }
        check_file(path, std::ios_base::out | std::ios_base::binary);

        // Read the size of the file.
        std::uint64_t filesize = 0;
        header_bytes += read_size(stream, filesize);
        if (!stream) {
            throw ANNEXCEPTION("Error reading from stream!");
        }

        // Now write the actual file contents.
        std::ofstream out(path, std::ios_base::out | std::ios_base::binary);
        if (!out) {
            throw ANNEXCEPTION("Error opening file {} for writing!", path);
        }

        // Copy the data in chunks.
        constexpr size_t buffer_size = 1 << 13; // 8KB buffer
        alignas(CACHELINE_BYTES) char buffer[buffer_size];

        size_t bytes_remaining = filesize;
        while (bytes_remaining > 0) {
            size_t to_read = std::min(buffer_size, bytes_remaining);
            stream.read(buffer, to_read);
            if (!stream) {
                throw ANNEXCEPTION("Error reading from stream!");
            }
            out.write(buffer, to_read);
            if (!out) {
                throw ANNEXCEPTION("Error writing to file {}!", path);
            }
            bytes_remaining -= to_read;
        }

        return header_bytes + filesize;
    }

    static size_t pack(const std::filesystem::path& dir, std::ostream& stream) {
        namespace fs = std::filesystem;
        if (!fs::is_directory(dir)) {
            throw ANNEXCEPTION("Path {} is not a directory!", dir);
        }

        auto total_bytes = write_size(stream, magic_number);

        // Calculate the number of files in the directory.
        uint64_t filesnum = std::count_if(
            fs::recursive_directory_iterator{dir},
            fs::recursive_directory_iterator{},
            [&](const auto& entry) { return entry.is_regular_file(); }
        );
        total_bytes += write_size(stream, filesnum);

        // Now serialize each file in the directory recursively.
        for (const auto& entry : fs::recursive_directory_iterator{dir}) {
            if (entry.is_regular_file()) {
                total_bytes += write_file(stream, entry.path(), dir);
            }
            // Ignore other types of entries.
        }

        return total_bytes;
    }

    static size_t unpack(std::istream& stream, const std::filesystem::path& root) {
        namespace fs = std::filesystem;

        // Read and verify the magic number.
        size_type magic = 0;
        auto total_bytes = read_size(stream, magic);
        if (magic != magic_number) {
            throw ANNEXCEPTION("Invalid magic number in directory unpacking!");
        }

        size_type num_files = 0;
        total_bytes += read_size(stream, num_files);
        if (!stream) {
            throw ANNEXCEPTION("Error reading from stream!");
        }

        if (!fs::exists(root)) {
            fs::create_directories(root);
        } else if (!fs::is_directory(root)) {
            throw ANNEXCEPTION("Path {} exists and is not a directory!", root);
        }

        // Now deserialize each file in the directory.
        for (size_type i = 0; i < num_files; ++i) {
            total_bytes += read_file(stream, root);
        }

        return total_bytes;
    }
};
} // namespace svs::lib
