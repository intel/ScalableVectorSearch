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
inline std::pair<size_t, size_t> get_dims(std::ifstream& stream, size_t elsize) {
    stream.seekg(0, std::ifstream::beg);
    size_t dimensions_per_vector = lib::read_binary<length_t>(stream);

    // Compute the total number of vectors in the dataset.
    stream.seekg(0, std::ifstream::end);
    size_t filesize = stream.tellg();
    stream.seekg(0, std::ifstream::beg);

    size_t total_line_size = elsize * dimensions_per_vector + sizeof(length_t);
    size_t vectors_in_file = filesize / total_line_size;
    size_t remainder = filesize % total_line_size;
    if (remainder != 0) {
        throw ANNEXCEPTION(
            "Vecs file is the incorrect length! Expected {}, got {}.",
            total_line_size,
            filesize
        );
    }
    return std::make_pair(vectors_in_file, dimensions_per_vector);
}

inline std::pair<size_t, size_t>
get_dims(const std::filesystem::path& path, size_t elsize) {
    auto stream = lib::open_read(path, std::ifstream::in | std::ifstream::binary);
    return get_dims(stream, elsize);
}
} // namespace detail

template <typename T>
std::pair<size_t, size_t> get_dims(const std::filesystem::path& path) {
    return detail::get_dims(path, sizeof(T));
}

template <typename T> std::pair<size_t, size_t> get_dims(std::ifstream& stream) {
    return detail::get_dims(stream, sizeof(T));
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
        auto dims = get_dims<T>(stream_);
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
template <typename T = void> class VecsFile {
  public:
    static constexpr bool is_memory_map_compatible = false;

    VecsFile() = default;

    /// Construct a file reference for the given path.
    explicit VecsFile(std::filesystem::path path)
        : path_{std::move(path)} {}

    ///
    /// @brief Open the file for reading and return an interactive reader for the file.
    ///
    VecsReader<T>
    reader(lib::meta::Type<T> SVS_UNUSED(type), size_t max_lines = Dynamic) const
        requires(!std::is_same_v<T, void>)
    {
        return VecsReader<T>(path_, max_lines);
    }

    ///
    /// @brief Open the file for writing and return an interactive writer for the file.
    ///
    /// @param dimensions The number of dimensions in each vector being written.
    ///
    VecsWriter<T> writer(size_t dimensions) const {
        return VecsWriter<T>(path_, dimensions);
    }

    ///
    /// @brief Return the dimensions of the vecs encoded dataset.
    ///
    /// Throws an ANNException of the file does not exist.
    ///
    std::pair<size_t, size_t> get_dims() const { return vecs::get_dims<T>(path_); }

  private:
    std::filesystem::path path_;
};

} // namespace vecs
} // namespace io
} // namespace svs
