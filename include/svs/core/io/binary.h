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

// Parameters regarding the layout of traditional `vecs` files.
namespace binary {
struct Header {
    uint32_t num_vectors;
    uint32_t vector_dim;
};
const size_t MAX_LINES_DEFAULT = std::numeric_limits<size_t>::max();

inline std::pair<size_t, size_t> get_dims(std::ifstream& stream) {
    stream.seekg(0, std::ifstream::beg);
    auto header = lib::read_binary<Header>(stream);
    stream.seekg(0, std::ifstream::beg);
    return std::make_pair(header.num_vectors, header.vector_dim);
}

inline std::pair<size_t, size_t> get_dims(const std::filesystem::path& path) {
    auto stream = lib::open_read(path, std::ifstream::in | std::ifstream::binary);
    return get_dims(stream);
}

/////
///// Reading
/////

template <typename T> class BinaryReader {
  public:
    using value_type = T;
    explicit BinaryReader(const std::string& filename, size_t max_lines = MAX_LINES_DEFAULT)
        : stream_{lib::open_read(filename, std::ifstream::in | std::ifstream::binary)}
        , max_lines_{0}
        , vectors_in_file_{0} {
        auto dims = get_dims(stream_);
        vectors_in_file_ = dims.first;
        dimensions_per_vector_ = dims.second;
        max_lines_ = std::min(max_lines, vectors_in_file_);
    }

    // Size parameters
    [[nodiscard]] size_t ndims() { return dimensions_per_vector_; }
    [[nodiscard]] size_t nvectors() { return vectors_in_file_; }
    [[nodiscard]] size_t vectors_to_read() { return max_lines_; }

    void resize(size_t max_lines = MAX_LINES_DEFAULT) {
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
    size_t max_lines_ = MAX_LINES_DEFAULT;
    size_t vectors_in_file_;
    size_t dimensions_per_vector_;
};

///
/// @brief Reference to a file encoded using DiskANN's binary format.
///
class BinaryFile {
  public:
    static constexpr bool is_memory_map_compatible = false;

    BinaryFile() = default;

    explicit BinaryFile(std::filesystem::path path)
        : path_{std::move(path)} {}

    template <typename T>
    BinaryReader<T> reader(
        lib::meta::Type<T> SVS_UNUSED(type), size_t max_lines = MAX_LINES_DEFAULT
    ) const {
        return BinaryReader<T>(path_, max_lines);
    }

    std::pair<size_t, size_t> get_dims() const { return binary::get_dims(path_); }

  private:
    std::filesystem::path path_;
};

} // namespace binary
} // namespace io
} // namespace svs
