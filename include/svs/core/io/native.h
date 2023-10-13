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
#include "svs/core/allocator.h"
#include "svs/lib/file.h"
#include "svs/lib/file_iterator.h"
#include "svs/lib/uuid.h"

// Support the the SVS "native" format.
#include <array>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>

namespace svs {
namespace io {

///
/// @brief SVS implements an open-ended file encoding schema.
///
/// This allows for future backwards-compatible expansion of file formats.
///
enum class FileSchema : uint32_t { Vtest, V1 };

///
/// @brief The default file schema to use for new allocations.
///
static constexpr FileSchema DefaultSchema = FileSchema::V1;

///
/// @brief Return a unique name for the given schema.
///
/// This provide a method for obtaining the schema from a string.
///
template <FileSchema Schema> std::string name();

template <> inline std::string name<FileSchema::Vtest>() { return "Vtest"; }
template <> inline std::string name<FileSchema::V1>() { return "V1"; }

///
/// @brief Convert a FileSchema into a unique name.
///
/// @sa parse_schema
///
inline std::string name(FileSchema schema) {
    switch (schema) {
        case FileSchema::Vtest: {
            return name<FileSchema::Vtest>();
        }
        case FileSchema::V1: {
            return name<FileSchema::V1>();
        }
    }
    throw ANNEXCEPTION("Unreachable!");
}

inline std::ostream& operator<<(std::ostream& stream, FileSchema schema) {
    return stream << name(schema);
}

///
/// @brief Parse a string into a FileSchema (inverse of ``name``).
///
/// Throws an ``ANNException`` if the string cannot be parsed.
///
/// @sa name(FileSchema)
///
inline FileSchema parse_schema(const std::string& repr) {
    auto map = std::unordered_map<std::string, FileSchema>{
        {name<FileSchema::Vtest>(), FileSchema::Vtest},
        {name<FileSchema::V1>(), FileSchema::V1}};

    auto itr = map.find(repr);
    if (itr == map.end()) {
        throw ANNEXCEPTION("Unknown schema \"{}\"!", repr);
    }
    return itr->second;
}

namespace detail {
/// Read the given header from a file.
template <typename T> inline T read_header(const std::filesystem::path& path) {
    auto stream = lib::open_read(path, std::ifstream::in | std::ifstream::binary);
    return lib::read_binary<T>(stream);
}

template <typename T> inline std::pair<size_t, size_t> get_dims(std::ifstream& stream) {
    stream.seekg(0, std::ifstream::beg);
    auto header = lib::read_binary<T>(stream);
    stream.seekg(0, std::ifstream::beg);
    return std::make_pair(header.num_vectors_, header.dimensions_per_vector_);
}

template <typename T>
inline std::pair<size_t, size_t> get_dims(const std::filesystem::path& path) {
    auto stream = lib::open_read(path, std::ifstream::in | std::ifstream::binary);
    return get_dims<T>(stream);
}
} // namespace detail

///
/// @brief Smart pointer for memory mapping files that only contain a header plus data.
///
template <typename T, typename Header> class HeaderMappedPtr {
  public:
    static constexpr FileSchema schema = Header::schema;
    using metadata = Header;

    explicit HeaderMappedPtr(MMapPtr<T> ptr)
        : ptr_{std::move(ptr)} {
        ptr_.setoffset(sizeof(metadata));
    }

    // Save the current state to disk.
    void save(const metadata& header) {
        // For saving, we simply memory copy the header to the
        std::memcpy(ptr_.base(), &header, sizeof(metadata));
    }

    lib::UUID uuid() const {
        Header header;
        std::memcpy(&header, ptr_.base(), sizeof(Header));
        return header.uuid_;
    }

    T* data() { return ptr_.data(); }
    const T* data() const { return ptr_.data(); }

  private:
    // The actual pointer for the memory map.
    MMapPtr<T> ptr_;
};

///
/// @brief A reader class that can read vectors from a native file containing a header.
///
template <typename T> class Reader {
  public:
    using value_type = T;
    explicit Reader(std::ifstream stream, size_t nvectors, size_t ndims, size_t offset = 0)
        : stream_{std::move(stream)}
        , nvectors_{nvectors}
        , ndims_{ndims}
        , offset_{offset} {}

    explicit Reader(
        const std::string& filename, size_t nvectors, size_t ndims, size_t offset = 0
    )
        : Reader(
              lib::open_read(filename, std::ifstream::in | std::ifstream::binary),
              nvectors,
              ndims,
              offset
          ) {}

    [[nodiscard]] size_t ndims() { return ndims_; }
    [[nodiscard]] size_t nvectors() { return nvectors_; }

    void resize(size_t nvectors) { nvectors_ = nvectors; }

    auto begin() {
        // Clear any `eof` bits.
        stream_.clear();
        stream_.seekg(offset_, std::ifstream::beg);

        // If there's no metadata, then construct an iterator that simply yields
        // vectors. Otherwise, we need to bundle up the metadata as well.
        auto reader = lib::VectorReader<T>(ndims());
        return lib::heterogeneous_iterator(stream_, nvectors(), std::move(reader));
    }

    lib::HeterogeneousFileEnd end() { return {}; }

  private:
    std::ifstream stream_;
    size_t nvectors_;
    size_t ndims_;
    size_t offset_;
};

namespace vtest {

// The `Vtest` binary layout is very simple.
// There is a 64-byte header in the file.
//
// * 8-bytes magic number
// * 8-bytes of the header encode the number of elements bundled contained in the file.
// * 8-bytes give the size of each vector.
// * 16-bytes UUID.
//
// The rest is padding.
static constexpr size_t HEADER_SIZE = 64; // bytes
static constexpr size_t HEADER_PADDING =
    HEADER_SIZE - 3 * sizeof(size_t) - sizeof(lib::UUID);

/// @brief Describe whether this file schema supports memory mapping.
static constexpr bool is_memory_map_compatible = true;

/// @brief The magic number for V1 encoded files.
static constexpr uint64_t magic_number = 0xf83ad4901d434f04;

struct Header {
    explicit Header(
        size_t num_vectors = 0,
        size_t dimensions_per_vector_ = 0,
        lib::UUID uuid = lib::UUID(lib::ZeroInitializer())
    )
        : num_vectors_{num_vectors}
        , dimensions_per_vector_{dimensions_per_vector_}
        , uuid_{uuid} {
        for (auto& i : padding) {
            i = std::byte{0};
        }
    }

    // Members
    uint64_t magic_ = magic_number;
    uint64_t num_vectors_;
    uint64_t dimensions_per_vector_;
    lib::UUID uuid_;
    std::array<std::byte, HEADER_PADDING> padding{};
};

// Verify header dimensions
static_assert(sizeof(Header) == HEADER_SIZE, "Mismatch in header sizes!");
static_assert(std::is_trivially_copyable_v<Header>, "Header must be trivially copyable!");

// Don't implement "reader" and "writer" for the test file.
// The test file is more about handling memory mapping correctly.
class NativeFile {
  public:
    // Constructors
    explicit NativeFile(std::filesystem::path path)
        : path_{std::move(path)} {}

    // Type Aliases
    using metadata = Header;
    template <typename T> using pointer = HeaderMappedPtr<T, Header>;

    // Methods
    Header header() const { return detail::read_header<Header>(path_); }
    std::pair<size_t, size_t> get_dims() const { return detail::get_dims<Header>(path_); }
    lib::UUID uuid() const { return detail::read_header<Header>(path_).uuid_; }

    template <typename T>
    Reader<T>
    reader(lib::meta::Type<T> SVS_UNUSED(type), size_t max_lines = Dynamic) const {
        std::ifstream stream = lib::open_read(path_);
        auto dims = detail::get_dims<Header>(stream);
        auto nvectors = std::min(dims.first, max_lines);
        auto ndims = dims.second;
        return Reader<T>(std::move(stream), nvectors, ndims, sizeof(Header));
    }

    template <typename T>
    pointer<T> mmap(
        lib::meta::Type<T> SVS_UNUSED(type), lib::Bytes bytes, const MemoryMapper& mapper
    ) const {
        return pointer<T>{MMapPtr<T>{mapper.mmap(path_, bytes + sizeof(Header))}};
    }

    const std::filesystem::path& get_path() const { return path_; }

    // Members
  private:
    std::filesystem::path path_;
};
} // namespace vtest

namespace v1 {

// The `v1` binary layout is very simple.
// There is a 64-byte header in the file.
// - The first 8-bytes of the header encode the number of elements bundled contained
//   in the file.
// - The next 8-bytes give the size of each vector.
// The rest is padding.
static constexpr size_t header_size = 1024; // bytes
static constexpr size_t header_padding =
    header_size - 3 * sizeof(uint64_t) - sizeof(lib::UUID);

/// @brief The magic number for V1 encoded files.
static constexpr uint64_t magic_number = 0xcad4a6b2579980fe;

struct Header {
    explicit Header(
        size_t num_vectors = 0,
        size_t dimensions_per_vector_ = 0,
        lib::UUID uuid = lib::UUID(lib::ZeroInitializer())
    )
        : uuid_{uuid}
        , num_vectors_{num_vectors}
        , dimensions_per_vector_{dimensions_per_vector_} {
        init_padding();
    }
    // Members
    uint64_t magic_ = magic_number;
    lib::UUID uuid_;
    uint64_t num_vectors_;
    uint64_t dimensions_per_vector_;
    std::array<std::byte, header_padding> padding{};

  private:
    void init_padding() { padding.fill(std::byte{0}); }
};

static_assert(sizeof(Header) == header_size, "Mismatch in Native io::v1 header sizes!");
static_assert(std::is_trivially_copyable_v<Header>, "Header must be trivially copyable!");

template <typename T = void> class Writer {
  public:
    Writer(
        const std::string& path,
        size_t dimension,
        lib::UUID uuid = lib::UUID(lib::ZeroInitializer())
    )
        : dimension_{dimension}
        , uuid_{uuid}
        , stream_{lib::open_write(path, std::ofstream::out | std::ofstream::binary)} {
        // Write a temporary header.
        stream_.seekp(0, std::ofstream::beg);
        lib::write_binary(stream_, Header());
    }

    size_t dimensions() const { return dimension_; }
    void overwrite_num_vectors(size_t num_vectors) { vectors_written_ = num_vectors; }

    // TODO: Error checking to make sure the length is correct.
    template <typename U> Writer& append(U&& v) {
        for (const auto& i : v) {
            lib::write_binary(stream_, lib::io_convert<T>(i));
        }
        ++vectors_written_;
        return *this;
    }

    template <typename... Ts>
        requires std::is_same_v<T, void>
    Writer& append(std::tuple<Ts...>&& v) {
        lib::foreach (v, [&](const auto& x) { lib::write_binary(stream_, x); });
        ++vectors_written_;
        return *this;
    }

    template <typename U> Writer& operator<<(U&& v) { return append(std::forward<U>(v)); }

    void flush() { stream_.flush(); }

    void writeheader(bool resume = true) {
        auto position = stream_.tellp();
        // Write to the header the number of vectors actually written.
        stream_.seekp(0);
        assert(stream_.good());
        lib::write_binary(stream_, Header(vectors_written_, dimension_, uuid_));
        if (resume) {
            stream_.seekp(position, std::ofstream::beg);
        }
    }

    // We want a custom destructor in order to write the header with the correct
    // number of vectors.
    //
    // Because we're implementing a custom destructor, we also need to define the
    // special member functions.
    //
    // We delete the copy constructor and copy assignment operators because
    // `std::ofstream` isn't copyable anyways.
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;
    Writer(Writer&&) = delete;
    Writer& operator=(Writer&&) = delete;

    // Write the header for the file.
    ~Writer() noexcept { writeheader(); }

  private:
    size_t dimension_;
    lib::UUID uuid_;
    std::ofstream stream_;
    size_t writes_this_vector_ = 0;
    size_t vectors_written_ = 0;
};

///
/// @brief Version 1 of the SVS native file format.
///
/// TODO (MH): Remove template parameter from file.
///
class NativeFile {
  public:
    // Type aliases
    using metadata = Header;
    template <typename T> using pointer = HeaderMappedPtr<T, Header>;

    // Constructors
    explicit NativeFile(std::filesystem::path path)
        : path_{std::move(path)} {}

    // Methods
    Header header() const { return detail::read_header<Header>(path_); }

    lib::UUID uuid() const { return detail::read_header<Header>(path_).uuid_; }

    template <typename T>
    Reader<T>
    reader(lib::meta::Type<T> SVS_UNUSED(type), size_t max_lines = Dynamic) const {
        std::ifstream stream = lib::open_read(path_);
        auto dims = detail::get_dims<Header>(stream);
        auto nvectors = std::min(dims.first, max_lines);
        auto ndims = dims.second;
        return Reader<T>(std::move(stream), nvectors, ndims, sizeof(Header));
    }

    template <typename T>
    Writer<T> writer(
        lib::meta::Type<T> SVS_UNUSED(type),
        size_t dimension,
        lib::UUID uuid = lib::ZeroUUID
    ) const {
        return Writer<T>(path_, dimension, uuid);
    }

    Writer<> writer(size_t dimensions, lib::UUID uuid = lib::ZeroUUID) const {
        return writer(lib::meta::Type<void>(), dimensions, uuid);
    }

    std::pair<size_t, size_t> get_dims() const { return detail::get_dims<Header>(path_); }

    // Memory Map
    template <typename T>
    pointer<T> mmap(
        lib::meta::Type<T> SVS_UNUSED(type), lib::Bytes bytes, const MemoryMapper& mapper
    ) const {
        return pointer<T>{MMapPtr<T>{mapper.mmap(path_, bytes + sizeof(Header))}};
    }

    const std::filesystem::path& get_path() const { return path_; }

    // Members
  private:
    std::filesystem::path path_;
};
} // namespace v1

/// @brief Variant of metadata for all memory-mappable files.
using FileMetadata = std::variant<vtest::Header, v1::Header>;

const static std::unordered_map<uint64_t, FileSchema> MAGIC_TO_SCHEMA{
    {vtest::magic_number, FileSchema::Vtest}, {v1::magic_number, FileSchema::V1}};

///
/// @brief Get the 64-bit magic number from an opened ``std::ifstream``.
///
/// **Preconditions**
///
/// The stream must have recently been opened and pointing to the beginning of a file.
/// This function will not attempt to adjust the read position of the stream.
///
inline uint64_t get_magic_number(std::ifstream& stream) {
    return lib::read_binary<uint64_t>(stream);
}

///
/// @brief Get the 64-bit magic number from a file.
///
/// @param path The file to read.
///
/// Throws an ``ANNException`` if ``path`` does not point to a valid file.
///
inline uint64_t get_magic_number(const std::filesystem::path& path) {
    auto stream = lib::open_read(path);
    return get_magic_number(stream);
}

///
/// @brief Convert a file magic number to a file schema.
///
/// @param magic The magic number to look up.
///
/// @returns A FileSchema enum if the magic number is recognized. Otherwise, returns an
///          empty optional.
///
inline std::optional<FileSchema> from_magic_number(uint64_t magic) {
    auto itr = MAGIC_TO_SCHEMA.find(magic);
    if (itr == MAGIC_TO_SCHEMA.end()) {
        return std::optional<FileSchema>();
    }
    return std::optional<FileSchema>(itr->second);
}

///
/// @brief Return of file schema of the file pointed to by `path`.
///
/// @returns The schema of the file if it is recognized. An empty optional otherwise.
///
inline std::optional<FileSchema> classify(const std::filesystem::path& path) {
    return from_magic_number(get_magic_number(path));
}

namespace detail {
// Empty entry-point for file-type identification based on schema.
template <FileSchema Schema> struct FileType;
template <> struct FileType<FileSchema::Vtest> {
    using type = io::vtest::NativeFile;
};
template <> struct FileType<FileSchema::V1> {
    using type = io::v1::NativeFile;
};
} // namespace detail

/// @brief Convert a schema enum to a file type class.
template <FileSchema Schema> using file_type_t = typename detail::FileType<Schema>::type;

template <typename F>
auto visit_file_type(FileSchema schema, const std::filesystem::path& path, F&& f) {
    switch (schema) {
        case FileSchema::Vtest: {
            return f(file_type_t<FileSchema::Vtest>{path});
        }
        case FileSchema::V1: {
            return f(file_type_t<FileSchema::V1>{path});
        }
    }
    throw ANNException("Unreachable!");
}

///
/// @brief Return the UUID for the binary blob in the file pointed to by ``path``.
///
/// @param path The file path to inspect.
///
/// @returns The UUID of the file. If the file type is not recognized, the optional will
///          be empty.
///
/// Throws an exception if ``path`` cannot be opened for reading.
///
inline std::optional<lib::UUID> get_uuid(const std::filesystem::path& path) {
    // Step 1: Get the schema for the provided file.
    auto schema = classify(path);
    if (!schema) {
        return std::optional<lib::UUID>();
    }

    // Step 2: Use the schema to dispatch somewhere that knows how to get the UUID.
    return visit_file_type(schema.value(), path, [](const auto& file) {
        return file.uuid();
    });
}

///
/// @brief Find a file with the given UUID in a directory.
///
/// @param dir The directory to search.
/// @param uuid The UUID to find.
///
/// @returns The full filepath to the file corresponding to the requested UUID. Otherwise,
///          returns an empty optional.
///
///
///
inline std::optional<std::filesystem::path>
find_uuid(const std::filesystem::path& dir, const lib::UUID& uuid) {
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        auto maybe_uuid = get_uuid(entry);
        if (maybe_uuid == uuid) {
            return entry.path();
        }
    }
    return std::optional<std::filesystem::path>();
}

///
/// @brief A generic NativeFile that can use runtime dispatch for detection.
///
class NativeFile {
  public:
    template <typename T> using Writer = v1::Writer<T>;

    explicit NativeFile(std::filesystem::path path)
        : path_{std::move(path)} {}

    template <typename F> auto resolve(F&& f) const {
        auto schema = classify(path_);
        if (!schema) {
            throw ANNEXCEPTION("Could not resolve {} for native file loading!", path_);
        }

        return visit_file_type(schema.value(), path_, std::forward<F>(f));
    }

    ///
    /// @brief Get the dimensions of the resulting file.
    ///
    std::pair<size_t, size_t> get_dims() const {
        return resolve([](const auto& resolved) { return resolved.get_dims(); });
    }

    ///
    /// @brief Generate a default writer for native files.
    ///
    template <typename T>
    Writer<T> writer(
        lib::meta::Type<T> type, size_t dimensions, lib::UUID uuid = lib::ZeroUUID
    ) const {
        return v1::NativeFile(path_).writer(type, dimensions, uuid);
    }

    Writer<void> writer(size_t dimensions, lib::UUID uuid = lib::ZeroUUID) const {
        return writer(lib::meta::Type<void>(), dimensions, uuid);
    }

  private:
    std::filesystem::path path_;
};

} // namespace io
} // namespace svs
