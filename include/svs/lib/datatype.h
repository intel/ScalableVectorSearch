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

        case DataType::undef: { return name<DataType::undef>(); }
    }
    // clang-format on
    throw ANNEXCEPTION("Unhandled type!");
}

inline constexpr size_t element_size(DataType type) {
    // clang-format off
    switch (type) {
        case DataType::uint8: { return sizeof(uint8_t); }
        case DataType::uint16: { return sizeof(uint16_t); }
        case DataType::uint32: { return sizeof(uint32_t); }
        case DataType::uint64: { return sizeof(uint64_t); }

        case DataType::int8: { return sizeof(int8_t); }
        case DataType::int16: { return sizeof(int16_t); }
        case DataType::int32: { return sizeof(int32_t); }
        case DataType::int64: { return sizeof(int64_t); }

        case DataType::float16: { return sizeof(svs::Float16); }
        case DataType::float32: { return sizeof(float); }
        case DataType::float64: { return sizeof(double); }

        case DataType::byte: { return sizeof(std::byte); }
        case DataType::undef: { return 0; }
    }
    // clang-format on
    throw ANNEXCEPTION("Unhandled type!");
}

inline constexpr DataType parse_datatype_floating(std::string_view name) {
    if (name == "float16") {
        return DataType::float16;
    }
    if (name == "float32") {
        return DataType::float32;
    }
    if (name == "float64") {
        return DataType::float64;
    }
    return DataType::undef;
}

inline constexpr DataType parse_datatype_unsigned(std::string_view name) {
    if (name == "uint8") {
        return DataType::uint8;
    }
    if (name == "uint16") {
        return DataType::uint16;
    }
    if (name == "uint32") {
        return DataType::uint32;
    }
    if (name == "uint64") {
        return DataType::uint64;
    }
    return DataType::undef;
}

inline constexpr DataType parse_datatype_signed(std::string_view name) {
    if (name == "int8") {
        return DataType::int8;
    }
    if (name == "int16") {
        return DataType::int16;
    }
    if (name == "int32") {
        return DataType::int32;
    }
    if (name == "int64") {
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
/// @ingroup lib_public_datatype
/// @brief Concept matching a type that has a corresponding data type.
///
template <typename T>
concept HasDataType = has_datatype_v<T>;

///
/// Erased Pointer
///

struct AssertCorrectType {};
inline constexpr AssertCorrectType assert_correct_type{};

class ConstErasedPointer {
  public:
    /// @brief Construct a null pointer with an undefined data type.
    explicit ConstErasedPointer() = default;

    /// @brief Construct a null pointer with an undefined data type.
    ConstErasedPointer(std::nullptr_t) {}

    ///
    /// @brief Construct a type-tagged erased pointer.
    ///
    template <typename T>
    explicit ConstErasedPointer(const T* data)
        : data_{static_cast<const void*>(data)}
        , type_{datatype_v<T>} {}

    ///
    /// @brief Construct a type-tagged erased pointer.
    ///
    /// Providing the incorrect DataType is undefined behavior.
    ///
    ConstErasedPointer(
        AssertCorrectType SVS_UNUSED(assertion), const void* data, DataType type
    )
        : data_{data}
        , type_{type} {}

    ///
    /// @brief Return the underlying type.
    ///
    DataType type() const { return type_; }

    /// @brief Return whether the underlying pointer is null.
    explicit operator bool() const { return data_ != nullptr; }

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

    friend bool operator==(ConstErasedPointer, ConstErasedPointer) = default;

  private:
    const void* data_{nullptr};
    DataType type_{DataType::undef};
};

///
/// Safely get the underlying pointer.
/// Throws `ANNException` if the converted type is incorrect.
///
template <typename T> const T* get(ConstErasedPointer ptr) { return ptr.template get<T>(); }

/// A container for N-dimensional dense arrays with a type-tagged base pointer.
template <size_t N> class AnonymousArray {
  private:
    ConstErasedPointer data_;
    std::array<size_t, N> dims_;

  public:
    explicit AnonymousArray() = default;

    /// @brief Consruct an anonymous array around the pointer.
    template <typename T, typename... Dims>
        requires(sizeof...(Dims) == N)
    explicit AnonymousArray(const T* data, Dims... dims)
        : data_{data}
        , dims_{dims...} {}

    /// @brief Construct an anonymous array around the tagged pointer.
    ///
    /// Passing the incorrect data type is undefined behavior.
    template <typename... Dims>
        requires(sizeof...(Dims) == N)
    AnonymousArray(
        AssertCorrectType assertion, const void* data, DataType type, Dims... dims
    )
        : data_{assertion, data, type}
        , dims_{dims...} {}

    /// @brief Return the logical dimensions of the array.
    std::array<size_t, N> dims() const { return dims_; }

    /// @brief Return the type of each element in the array.
    svs::DataType type() const { return data_.type(); }

    /// @brief Return the base pointer.
    ConstErasedPointer pointer() const { return data_; }

    /// @brief Get the shape of the `i`th dimension.
    size_t size(size_t i) const {
        assert(i < N);
        return dims_[i];
    }

    /// @brief Return the number of elements in a 1-dimensional array.
    size_t size() const
        requires(N == 1)
    {
        return size(0);
    }

    /// @brief Return the based pointer performing a checked cast.
    template <typename T> const T* data() const { return get<T>(data_); }

    /// @brief Return the based pointer performing a un-checked cast.
    template <typename T> const T* data_unchecked() const {
        return data_.template get_unchecked<T>();
    }

    friend bool operator==(AnonymousArray, AnonymousArray) = default;
};

template <typename T, size_t N> const T* get(AnonymousArray<N> array) {
    return array.template data<T>();
}

} // namespace svs

///// Formatting
template <> struct fmt::formatter<svs::DataType> : svs::format_empty {
    auto format(auto x, auto& ctx) const {
        return fmt::format_to(ctx.out(), "{}", svs::name(x));
    }
};
