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

///
/// @ingroup lib_public
/// @defgroup lib_public_datatype Primitive Data Types.
///

// local deps
#include "svs/lib/exception.h"
#include "svs/lib/float16.h"
#include "svs/third-party/fmt.h"

// stdlib
#include <cassert>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <type_traits>

namespace svs {

///
/// @ingroup lib_public_datatype
/// @brief Enum aliases for dense vector element types.
///
enum class DataType {
    uint8,
    uint16,
    uint32,
    uint64,
    int8,
    int16,
    int32,
    int64,
    float16,
    float32,
    float64,
    byte,
    undef
};

///
/// @ingroup lib_public_datatype
/// @brief Return a descriptive name for the corresponding datatype.
///
template <DataType Type> constexpr std::string_view name() { return "undef"; }
template <> inline constexpr std::string_view name<DataType::uint8>() { return "uint8"; }
template <> inline constexpr std::string_view name<DataType::uint16>() { return "uint16"; }
template <> inline constexpr std::string_view name<DataType::uint32>() { return "uint32"; }
template <> inline constexpr std::string_view name<DataType::uint64>() { return "uint64"; }

template <> inline constexpr std::string_view name<DataType::int8>() { return "int8"; }
template <> inline constexpr std::string_view name<DataType::int16>() { return "int16"; }
template <> inline constexpr std::string_view name<DataType::int32>() { return "int32"; }
template <> inline constexpr std::string_view name<DataType::int64>() { return "int64"; }

template <> inline constexpr std::string_view name<DataType::float16>() {
    return "float16";
}
template <> inline constexpr std::string_view name<DataType::float32>() {
    return "float32";
}
template <> inline constexpr std::string_view name<DataType::float64>() {
    return "float64";
}

template <> inline constexpr std::string_view name<DataType::byte>() { return "byte"; }

///
/// @ingroup lib_public_datatype
/// @brief Return a descriptive string for the corresponding datatype.
///
inline constexpr std::string_view name(DataType type) {
    // clang-format off
    switch (type) {
        case DataType::uint8: { return name<DataType::uint8>(); }
        case DataType::uint16: { return name<DataType::uint16>(); }
        case DataType::uint32: { return name<DataType::uint32>(); }
        case DataType::uint64: { return name<DataType::uint64>(); }

        case DataType::int8: { return name<DataType::int8>(); }
        case DataType::int16: { return name<DataType::int16>(); }
        case DataType::int32: { return name<DataType::int32>(); }
        case DataType::int64: { return name<DataType::int64>(); }

        case DataType::float16: { return name<DataType::float16>(); }
        case DataType::float32: { return name<DataType::float32>(); }
        case DataType::float64: { return name<DataType::float64>(); }

        case DataType::byte: { return name<DataType::byte>(); }

        default: { return name<DataType::undef>(); }
    }
    // clang-format on
}

inline constexpr DataType parse_datatype_floating(std::string_view name) {
    if (name == "float16") {
        return DataType::float16;
    } else if (name == "float32") {
        return DataType::float32;
    } else if (name == "float64") {
        return DataType::float64;
    }
    return DataType::undef;
}

inline constexpr DataType parse_datatype_unsigned(std::string_view name) {
    if (name == "uint8") {
        return DataType::uint8;
    } else if (name == "uint16") {
        return DataType::uint16;
    } else if (name == "uint32") {
        return DataType::uint32;
    } else if (name == "uint64") {
        return DataType::uint64;
    }
    return DataType::undef;
}

inline constexpr DataType parse_datatype_signed(std::string_view name) {
    if (name == "int8") {
        return DataType::int8;
    } else if (name == "int16") {
        return DataType::int16;
    } else if (name == "int32") {
        return DataType::int32;
    } else if (name == "int64") {
        return DataType::int64;
    }
    return DataType::undef;
}

inline constexpr DataType parse_datatype(std::string_view name) {
    // Handle outliers.
    if (name == "undef") {
        return DataType::undef;
    }
    if (name == "byte") {
        return DataType::byte;
    }

    // Floating point.
    if (name.starts_with("float")) {
        return parse_datatype_floating(name);
    }
    if (name.starts_with("uint")) {
        return parse_datatype_unsigned(name);
    }
    if (name.starts_with("int")) {
        return parse_datatype_signed(name);
    }
    return DataType::undef;
}

inline std::ostream& operator<<(std::ostream& stream, svs::DataType type) {
    return stream << name(type);
}

namespace lib {

// Formatting
inline std::string format_internal(
    const std::vector<DataType>& types, const char* delim, const char* last_delim
) {
    auto stream = std::ostringstream{};
    const auto end = types.end();
    const auto begin = types.begin();
    auto itr = types.begin();
    while (itr != end) {
        // Apply delimiter
        if (itr != begin) {
            if (std::next(itr) == end) {
                stream << last_delim;
            } else {
                stream << delim;
            }
        }
        stream << *itr;
        ++itr;
    }
    return stream.str();
}

///
/// @ingroup lib_public_datatype
/// @brief Create a formatted string of all data types present.
///
/// @param types Collection of data types to create a string for.
///
inline std::string format(const std::vector<DataType>& types) {
    const char* delim = ", ";
    // Don't put a comma between elements if there are only two of them.
    const char* last_delim = (types.size() == 2) ? " and " : ", and ";
    return format_internal(types, delim, last_delim);
}
} // namespace lib

namespace detail {
// clang-format off

// Map from enum to data type.
template <DataType Type> struct CppType {};

template <> struct CppType<DataType::uint8> { using type = uint8_t; };
template <> struct CppType<DataType::uint16> { using type = uint16_t; };
template <> struct CppType<DataType::uint32> { using type = uint32_t; };
template <> struct CppType<DataType::uint64> { using type = uint64_t; };

template <> struct CppType<DataType::int8> { using type = int8_t; };
template <> struct CppType<DataType::int16> { using type = int16_t; };
template <> struct CppType<DataType::int32> { using type = int32_t; };
template <> struct CppType<DataType::int64> { using type = int64_t; };

template <> struct CppType<DataType::float16> { using type = Float16; };
template <> struct CppType<DataType::float32> { using type = float; };
template <> struct CppType<DataType::float64> { using type = double; };

template <> struct CppType<DataType::byte> { using type = std::byte; };

// Map from data type to enum
template<typename T> inline constexpr DataType datatype_v = DataType::undef;

template<> inline constexpr DataType datatype_v<uint8_t> = DataType::uint8;
template<> inline constexpr DataType datatype_v<uint16_t> = DataType::uint16;
template<> inline constexpr DataType datatype_v<uint32_t> = DataType::uint32;
template<> inline constexpr DataType datatype_v<uint64_t> = DataType::uint64;

template<> inline constexpr DataType datatype_v<int8_t> = DataType::int8;
template<> inline constexpr DataType datatype_v<int16_t> = DataType::int16;
template<> inline constexpr DataType datatype_v<int32_t> = DataType::int32;
template<> inline constexpr DataType datatype_v<int64_t> = DataType::int64;

template<> inline constexpr DataType datatype_v<Float16> = DataType::float16;
template<> inline constexpr DataType datatype_v<float> = DataType::float32;
template<> inline constexpr DataType datatype_v<double> = DataType::float64;

template<> inline constexpr DataType datatype_v<std::byte> = DataType::byte;
// clang-format on
} // namespace detail

///
/// @ingroup lib_public_datatype
/// @brief Convert an DataType to its corresponding C++ type.
///
template <DataType Type> using cpp_type_t = typename detail::CppType<Type>::type;

///
/// @ingroup lib_public_datatype
/// @brief Convert a C++ type into its corresponding DataType.
///
template <typename T>
inline constexpr DataType datatype_v = detail::datatype_v<std::decay_t<T>>;

///
/// @ingroup lib_public_datatype
/// @brief Return whether the type `T` has a corresponding ``svs::DataType``.
///
template <typename T>
inline constexpr bool has_datatype_v = (datatype_v<T> != DataType::undef);

///
/// Erased Pointer
///

class ConstErasedPointer {
  public:
    ///
    /// @brief Construct a type-tagged erased pointer.
    ///
    template <typename T>
    explicit ConstErasedPointer(const T* data)
        : data_{static_cast<const void*>(data)}
        , type_{datatype_v<T>} {}

    ///
    /// @brief Return the underlying type.
    ///
    DataType type() const { return type_; }

    ///
    /// @brief Safely extract the wrapped pointer.
    ///
    /// @tparam T The desired type to cast the pointer as.
    ///
    /// If the requested parameter ``T`` does not match the actual type of the pointer,
    /// an ``ANNException`` is thrown.
    ///
    template <typename T> const T* get() const {
        if (datatype_v<T> == type_) {
            return static_cast<const T*>(data_);
        }
        throw ANNEXCEPTION("Bad type cast!");
    }

    ///
    /// @brief Unsafely extract the wrapped pointer.
    ///
    /// @tparam T The desired type to cast the pointer to.
    ///
    /// The requested type ``T`` must match the actual type of the pointer. If it doesn't,
    /// the behavior of the function is undefined.
    ///
    template <typename T> const T* get_unchecked() const {
        assert(datatype_v<T> == type_);
        // NOTE: Don't define an external helper function to call this member.
        // RATIONAL: Requiring `data.template get_unchecked<type>()` stands out visually
        // and can serve to highlight that something potentially unsafe is going on.
        return static_cast<const T*>(data_);
    }

  private:
    const void* data_;
    DataType type_;
};

///
/// Safely get the underlying pointer.
/// Throws `ANNException` if the converted type is incorrect.
///
template <typename T> const T* get(ConstErasedPointer ptr) { return ptr.template get<T>(); }

} // namespace svs

///// Formatting
template <> struct fmt::formatter<svs::DataType> : svs::format_empty {
    auto format(auto x, auto& ctx) const {
        return fmt::format_to(ctx.out(), "{}", svs::name(x));
    }
};
