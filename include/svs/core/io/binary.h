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

#include <cassert>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <tuple>
#include <type_traits>

#include "svs/lib/exception.h"
#include "svs/lib/file.h"
#include "svs/lib/file_iterator.h"
#include "svs/lib/readwrite.h"

namespace svs {
namespace io {

// Utilities to read/write the DiskANN's binary file format.
namespace binary {

struct Header {
    uint32_t num_vectors;
    uint32_t vector_dim;
};

inline std::pair<size_t, size_t>
get_dims(std::ifstream& stream, std::string_view source = {}, size_t elsize_hint = 0) {
    using namespace std::literals; // for string_view literal

    stream.seekg(0, std::ifstream::beg);
    auto header = lib::read_binary<Header>(stream);

    stream.seekg(0, std::ifstream::end);
    size_t filesize = stream.tellg();
    stream.seekg(0, std::ifstream::beg);

    size_t filesize_minus_header = filesize - sizeof(header);
    size_t n_vec_elements =
        lib::narrow<size_t>(header.num_vectors) * lib::narrow<size_t>(header.vector_dim);
    size_t remainder = filesize_minus_header % n_vec_elements;
    size_t deduced_elsize = filesize_minus_header / n_vec_elements;

    bool elsize_match = elsize_hint == 0 ? true : elsize_hint == deduced_elsize;

    if (remainder != 0 || !elsize_match) {
        throw ANNEXCEPTION(
            "Cannot read elements of size {} from Binary file {}.",
            elsize_hint == 0 ? deduced_elsize : elsize_hint,
            source.empty() ? "(unknown)"sv : source
        );
    }
    return std::make_pair(header.num_vectors, header.vector_dim);
}

inline std::pair<size_t, size_t>
get_dims(const std::filesystem::path& path, size_t elsize_hint = 0) {
    auto stream = lib::open_read(path, std::ifstream::in | std::ifstream::binary);
    return get_dims(stream, std::string_view(path.native()), elsize_hint);
}

/////
///// Reading
/////

template <typename T> class BinaryReader {
  public:
    using value_type = T;
    explicit BinaryReader(const std::string& filename, size_t max_lines = Dynamic)
        : stream_{lib::open_read(filename, std::ifstream::in | std::ifstream::binary)}
        , max_lines_{0}
        , vectors_in_file_{0} {
        auto dims = get_dims(stream_, std::string_view(filename), sizeof(T));
        vectors_in_file_ = dims.first;
        dimensions_per_vector_ = dims.second;
        max_lines_ = std::min(max_lines, vectors_in_file_);
    }

    // Size parameters
    [[nodiscard]] size_t ndims() { return dimensions_per_vector_; }
    [[nodiscard]] size_t nvectors() { return vectors_in_file_; }
    [[nodiscard]] size_t vectors_to_read() { return max_lines_; }

    void resize(size_t max_lines = Dynamic) {
        max_lines_ = std::min(max_lines, nvectors());
    }

    auto begin() {
        // Seek to the start of the file.
        stream_.clear();
        stream_.seekg(sizeof(Header), std::ifstream::beg);

        // The binary data layout is very similar to SVS layout outside of the size of
        // the header.
        //
        // Thus, the iterators setups are very similar.
        auto vector_reader = lib::VectorReader<T>(ndims());
        return lib::heterogeneous_iterator(
            stream_, vectors_to_read(), std::move(vector_reader)
        );
    }

    lib::HeterogeneousFileEnd end() { return {}; }

  private:
    std::ifstream stream_;
    size_t max_lines_ = Dynamic;
    size_t vectors_in_file_;
    size_t dimensions_per_vector_;
};

/////
///// Writing
/////

template <typename T = void> class BinaryWriter {
  public:
    BinaryWriter(std::ofstream&& stream, size_t n_vecs, size_t dimension)
        : header_{lib::narrow<uint32_t>(n_vecs), lib::narrow<uint32_t>(dimension)}
        , stream_{std::move(stream)} {
        stream_.seekp(0, std::ofstream::beg);
        lib::write_binary(stream_, header_);
    }

    BinaryWriter(const std::string& path, size_t n_vecs, size_t dimension)
        : BinaryWriter(
              lib::open_write(path, std::ofstream::out | std::ofstream::binary),
              n_vecs,
              dimension
          ) {}

    template <typename U> BinaryWriter& append(U&& v) {
        for (auto i : v) {
            lib::write_binary(stream_, lib::io_convert<T>(i));
        }
        return *this;
    }

    template <typename U> BinaryWriter& operator<<(U&& v) { return append(v); }

    void flush() { stream_.flush(); }

  private:
    Header header_;
    std::ofstream stream_;
};

///
/// @brief Reference to a file encoded using DiskANN's binary format.
///
class BinaryFile {
  public:
    static constexpr bool is_memory_map_compatible = false;
    template <typename T> using reader_type = BinaryReader<T>;

    BinaryFile() = default;

    /// Construct a file reference for the given path.
    explicit BinaryFile(std::filesystem::path path)
        : path_{std::move(path)} {}

    ///
    /// @brief Open the file for reading and return an interactive reader for the file.
    ///
    template <typename T>
    BinaryReader<T>
    reader(lib::Type<T> SVS_UNUSED(type), size_t max_lines = Dynamic) const {
        return BinaryReader<T>(path_, max_lines);
    }

    ///
    /// @brief Open the file for writing and return an interactive writer for the file.
    ///
    /// @param n_vectors  The number of vectors to be written in the file.
    /// @param dimensions The number of dimensions in each vector being written.
    ///
    template <typename T>
    BinaryWriter<T> writer(size_t n_vectors, size_t dimensions) const {
        return BinaryWriter<T>(path_, n_vectors, dimensions);
    }

    ///
    /// @brief Return the dimensions of the binary encoded dataset.
    ///
    /// Throws an ANNException if the file does not exist.
    ///
    /// @param elsize_hint (optional)  Element size in bytes
    ///
    std::pair<size_t, size_t> get_dims(size_t elsize_hint = 0) const {
        return binary::get_dims(path_, elsize_hint);
    }

  private:
    std::filesystem::path path_;
};

} // namespace binary
} // namespace io
} // namespace svs
