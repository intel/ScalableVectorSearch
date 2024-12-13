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
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

#include "svs/lib/exception.h"
#include "svs/lib/file.h"
#include "svs/lib/file_iterator.h"
#include "svs/lib/readwrite.h"

namespace svs {
namespace io {

// Parameters regarding the layout of traditional `vecs` files.
namespace vecs {

/// The encoding of the length type in the vecs file.
using length_t = uint32_t;

namespace detail {
// Parameters used to deduce element size in the vecs file.
const std::array<size_t, 4> ALLOWED_ELEMENT_SIZES = {1, 2, 4, 8};
const size_t MAX_VECTORS_TO_INSPECT = 10;

// Deduce element size from a vecs file.
inline size_t deduce_element_size(std::ifstream& stream, size_t dims, size_t filesize) {
    auto is_last_vec = [&](const size_t& offset_bytes) {
        return offset_bytes + sizeof(length_t) > filesize;
    };

    auto next_dims = [&](const size_t& offset_bytes) {
        stream.seekg(offset_bytes, std::ifstream::beg);
        return lib::read_binary<length_t>(stream);
    };

    // First, find the element size that gives the correct dims of second vector,
    // and then, confirm that we get the same dimension for a few more vectors.
    for (size_t elsize : ALLOWED_ELEMENT_SIZES) {
        size_t offset = sizeof(length_t) + elsize * dims;
        if (is_last_vec(offset)) {
            return offset == filesize ? elsize : 0;
        }

        if (dims == next_dims(offset)) {
            size_t line_size = sizeof(length_t) + elsize * dims;
            bool is_check_valid = true;

            // Check that the dims of a few initial vectors is same
            for (size_t n_vec = 0; n_vec < MAX_VECTORS_TO_INSPECT; n_vec++) {
                offset = line_size * n_vec;
                if (is_last_vec(offset)) {
                    if (offset == filesize) {
                        return elsize;
                    } else {
                        is_check_valid = false;
                        break;
                    }
                }
                if (dims != next_dims(offset)) {
                    is_check_valid = false;
                    break;
                }
            }

            if (is_check_valid) {
                return elsize;
            }
        }
    }

    return 0;
}
} // namespace detail

inline std::pair<size_t, size_t>
get_dims(std::ifstream& stream, std::string_view source = {}, size_t elsize_hint = 0) {
    using namespace std::literals; // for string_view literal

    stream.seekg(0, std::ifstream::beg);
    size_t dims = lib::read_binary<length_t>(stream);

    stream.seekg(0, std::ifstream::end);
    size_t filesize = stream.tellg();
    stream.seekg(0, std::ifstream::beg);

    // Deduce element size if not provided
    size_t deduced_elsize = elsize_hint == 0
                                ? detail::deduce_element_size(stream, dims, filesize)
                                : elsize_hint;

    size_t total_line_size = deduced_elsize * dims + sizeof(length_t);
    size_t vectors_in_file = filesize / total_line_size;
    size_t remainder = filesize % total_line_size;
    if (deduced_elsize == 0 || remainder != 0) {
        throw ANNEXCEPTION(
            "Cannot read elements of size {} from Vecs file {}.",
            deduced_elsize,
            source.empty() ? "(unknown)"sv : source
        );
    }
    return std::make_pair(vectors_in_file, dims);
}

inline std::pair<size_t, size_t>
get_dims(const std::filesystem::path& path, size_t elsize_hint = 0) {
    auto stream = lib::open_read(path, std::ifstream::in | std::ifstream::binary);
    return get_dims(stream, std::string_view(path.native()), elsize_hint);
}

/////
///// Reading
/////

template <typename T> class VecsReader {
  public:
    using value_type = T;
    explicit VecsReader(const std::string& filename, size_t max_lines = Dynamic)
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
        stream_.seekg(0, std::ifstream::beg);

        // Create two readers:
        // (1) A value reader to read the length information (to make sure it's the same
        //     as the originally read dimension each time).
        // (2) A vector reader to read up one data vector at a time.
        //
        // When returning, we just want to return the vector data read.
        auto value_reader = lib::ValueReader<length_t>();
        auto vector_reader = lib::VectorReader<T>(ndims());
        return lib::heterogeneous_iterator(
            [](auto&& /*unused*/, auto&& span) { return span; },
            stream_,
            vectors_to_read(),
            value_reader,
            std::move(vector_reader)
        );
    }

    lib::HeterogeneousFileEnd end() { return {}; }

  private:
    // Members
    std::ifstream stream_;
    size_t max_lines_ = Dynamic;
    size_t vectors_in_file_;
    size_t dimensions_per_vector_;
};

/////
///// Writing
/////

template <typename T = void> class VecsWriter {
  public:
    // Initialize `writes_since_last_dimension_` to `dimension` so we write
    // the vector dimension the first time anything is written to the write.
    VecsWriter(std::ofstream&& stream, size_t dimension)
        : dimension_{dimension}
        , stream_{std::move(stream)} {}

    VecsWriter(const std::string& path, size_t dimension)
        : VecsWriter(
              lib::open_write(path, std::ofstream::out | std::ofstream::binary), dimension
          ) {}

    template <typename U> VecsWriter& append(U&& v) {
        lib::write_binary(stream_, static_cast<length_t>(v.size()));
        for (auto i : v) {
            lib::write_binary(stream_, lib::io_convert<T>(i));
        }
        return *this;
    }

    template <typename U> VecsWriter& operator<<(U&& v) { return append(v); }

    void flush() { stream_.flush(); }

  private:
    size_t dimension_;
    std::ofstream stream_;
};

///
/// @brief Reference to a file encoded using the Vecs format.
///
class VecsFile {
  public:
    static constexpr bool is_memory_map_compatible = false;
    template <typename T> using reader_type = VecsReader<T>;

    VecsFile() = default;

    /// Construct a file reference for the given path.
    explicit VecsFile(std::filesystem::path path)
        : path_{std::move(path)} {}

    ///
    /// @brief Open the file for reading and return an interactive reader for the file.
    ///
    template <typename T>
    VecsReader<T> reader(lib::Type<T> SVS_UNUSED(type), size_t max_lines = Dynamic) const {
        return VecsReader<T>(path_, max_lines);
    }

    ///
    /// @brief Open the file for writing and return an interactive writer for the file.
    ///
    /// @param dimensions The number of dimensions in each vector being written.
    ///
    template <typename T> VecsWriter<T> writer(size_t dimensions) const {
        return VecsWriter<T>(path_, dimensions);
    }

    ///
    /// @brief Return the dimensions of the vecs encoded dataset.
    ///
    /// Throws an ANNException if the file does not exist.
    ///
    /// @param elsize_hint (optional)  Element size in bytes
    ///
    std::pair<size_t, size_t> get_dims(size_t elsize_hint = 0) const {
        return vecs::get_dims(path_, elsize_hint);
    }

  private:
    std::filesystem::path path_;
};

} // namespace vecs
} // namespace io
} // namespace svs
