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

#include <span>
#include <type_traits>
#include <vector>

namespace svs {
namespace lib {

/////
///// Reading Binary Objects
/////

// Partial specialization entry points.
template <typename Stream, typename T> struct BinaryReader {
    static void call(Stream& stream, T& x) {
        static_assert(std::is_trivially_copyable_v<T>);
        using char_type = typename Stream::char_type;
        stream.read(reinterpret_cast<char_type*>(&x), sizeof(T));
    }
};

template <typename Stream, typename T> struct BinaryReader<Stream, std::vector<T>> {
    static void call(Stream& stream, std::vector<T>& x) {
        static_assert(std::is_trivially_copyable_v<T>);
        using char_type = typename Stream::char_type;
        stream.read(reinterpret_cast<char_type*>(x.data()), sizeof(T) * x.size());
    }
};

/// @defgroup read_binary_group

///
/// @ingroup read_binary_group
/// @brief Read the canonical binary representation of `T` and store the results in `x`.
///
/// @param stream A ``std::istream`` compatible object.
/// @param x The destination to store the result.
///
/// Accepted types ``T``:
///
/// * Any type satisfying ``std::is_trivially_copyable``.
///   Parameter ``x`` will the be populated as if by ``std::memcpy``.
/// * A ``std::vector`` of such types.
///   Each element of the vector will be populated as if by recursively calling
///   ``svs::lib::read_binary`` on each element beginning from the beginning.
///
template <typename T, typename Stream> void read_binary(Stream& stream, T& x) {
    BinaryReader<Stream, T>::call(stream, x);
}

///
/// @ingroup read_binary_group
/// @brief Read the canonical binary representation of `T`.
///
/// @tparam T The type to read from the stream.
///
/// @param stream A ``std::istream`` compatible object.
///
/// Read the canonical binary representation of `T` as if by ``std::memcpy``.
/// Requires that ``T`` satisfies ``std::is_trivially_copyable``.
///
template <typename T, typename Stream> T read_binary(Stream& stream) {
    static_assert(std::is_trivially_copyable_v<T>);
    auto x = T{};
    read_binary(stream, x);
    return x;
}

/////
///// Binary Writing
/////

// We use a similar strategy for writing that we used for reading.
template <typename Stream, typename T> struct BinaryWriter {
    static size_t call(Stream& stream, const T& x) {
        static_assert(std::is_trivially_copyable_v<T>);
        using char_type = typename Stream::char_type;
        stream.write(reinterpret_cast<const char_type*>(&x), sizeof(T));
        return sizeof(T);
    }
};

template <typename Stream, typename T, size_t Extent>
struct BinaryWriter<Stream, std::span<T, Extent>> {
    static size_t call(Stream& stream, const std::span<T> span) {
        static_assert(std::is_trivially_copyable_v<T>);
        using char_type = typename Stream::char_type;
        size_t bytes = sizeof(T) * span.size();
        stream.write(reinterpret_cast<const char_type*>(span.data()), bytes);
        return bytes;
    }
};

template <typename Stream, typename T, typename Allocator>
struct BinaryWriter<Stream, std::vector<T, Allocator>> {
    static size_t call(Stream& stream, const std::vector<T, Allocator>& vec) {
        static_assert(std::is_trivially_copyable_v<T>);
        using char_type = typename Stream::char_type;
        size_t bytes = sizeof(T) * vec.size();
        stream.write(reinterpret_cast<const char_type*>(vec.data()), bytes);
        return bytes;
    }
};

///
/// @brief Write the canonical binary representation of ``val`` to the output stream.
///
/// @param stream A ``std::basic_ostream`` compatible object.j:wa
/// @param val The value to write.
///
/// @returns The number of bytes written to the stream.
///
/// Writing will occur sequentially on each byte beginning from the least significant
/// byte.
///
/// Accepted types:
///
/// * Any ``std::is_trivially_copyable`` type.
/// * A ``std::vector`` or ``std::span`` of such types.
///
template <typename T, typename Stream> size_t write_binary(Stream& stream, const T& val) {
    return BinaryWriter<Stream, T>::call(stream, val);
}
} // namespace lib
} // namespace svs
