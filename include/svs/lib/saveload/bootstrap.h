/*
 * Copyright 2024 Intel Corporation
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

// saveload
#include "svs/lib/saveload/core.h"
#include "svs/lib/saveload/load.h"
#include "svs/lib/saveload/save.h"

// svs
#include "svs/lib/datatype.h"
#include "svs/lib/misc.h"
#include "svs/lib/readwrite.h"
#include "svs/lib/uuid.h"

// stl
#include <concepts>
#include <vector>

namespace svs::lib {

/////
///// Built-in Types
/////

// Integers
template <std::integral I> struct Saver<I> {
    static SaveNode save(I x) { return SaveNode(lib::narrow<int64_t>(x)); }
};

template <std::integral I> struct Loader<I> {
    using toml_type = toml::value<int64_t>;
    static I load(const toml::value<int64_t>& v) { return lib::narrow<I>(v.get()); }
};

// Bool
template <> struct Saver<bool> {
    static SaveNode save(bool x) { return SaveNode(x); }
};

template <> struct Loader<bool> {
    using toml_type = toml::value<bool>;
    static bool load(const toml::value<bool> v) { return v.get(); }
};

// Floating Point
template <std::floating_point F> struct Saver<F> {
    static SaveNode save(F x) { return SaveNode(lib::narrow<double>(x)); }
};

template <std::floating_point F> struct Loader<F> {
    using toml_type = toml::value<double>;
    static F load(const toml_type& v) { return lib::narrow_cast<F>(v.get()); }
};

// String-like
template <> struct Saver<std::string> {
    static SaveNode save(const std::string& x) { return SaveNode(x); }
};

template <> struct Saver<std::string_view> {
    static SaveNode save(std::string_view x) { return SaveNode(x); }
};

template <> struct Loader<std::string> {
    using toml_type = toml::value<std::string>;
    static std::string load(const toml_type& value) { return value.get(); }
};

// Filesystem
template <> struct Saver<std::filesystem::path> {
    static SaveNode save(const std::filesystem::path x) {
        return SaveNode(std::string_view(x.native()));
    }
};

template <> struct Loader<std::filesystem::path> {
    using toml_type = toml::value<std::string>;
    static std::filesystem::path load(const toml::value<std::string>& v) {
        return std::filesystem::path(v.get());
    }
};

// Timepoint.
template <> struct Saver<std::chrono::time_point<std::chrono::system_clock>> {
    static SaveNode save(std::chrono::time_point<std::chrono::system_clock> x) {
        auto today = std::chrono::floor<std::chrono::days>(x);
        auto ymd = std::chrono::year_month_day(today);
        auto date = toml::date(
            static_cast<int>(ymd.year()),
            static_cast<unsigned>(ymd.month()),
            static_cast<unsigned>(ymd.day())
        );
        auto hh_mm_ss = std::chrono::hh_mm_ss(x - today);
        auto time = toml::time{
            lib::narrow_cast<uint8_t>(hh_mm_ss.hours().count()),
            lib::narrow_cast<uint8_t>(hh_mm_ss.minutes().count()),
            lib::narrow_cast<uint8_t>(hh_mm_ss.seconds().count()),
        };
        return SaveNode(toml::date_time(date, time));
    }
};

// Vectors
template <typename T, typename Alloc> struct Saver<std::vector<T, Alloc>> {
    static SaveNode save(const std::vector<T, Alloc>& v)
        requires SaveableContextFree<T>
    {
        auto array = toml::array();
        for (const auto& i : v) {
            array.push_back(lib::save(i));
        }
        return SaveNode(std::move(array));
    }

    static SaveNode save(const std::vector<T, Alloc>& v, const SaveContext& ctx) {
        auto array = toml::array();
        for (const auto& i : v) {
            array.push_back(lib::save(i, ctx));
        }
        return SaveNode(std::move(array));
    }
};

template <typename T, typename Alloc> struct Loader<std::vector<T, Alloc>> {
    using toml_type = toml::array;
    template <detail::ArrayLike Array, typename... Args>
    static void do_load(std::vector<T, Alloc>& v, const Array& array, Args&&... args) {
        array.visit([&](auto view) { v.push_back(lib::load<T>(view, args...)); });
    }

    // Without an allocator argument.
    template <detail::ArrayLike Array, typename... Args>
        requires(!lib::first_is<Alloc, Args...>())
    static std::vector<T, Alloc> load(const Array& array, Args&&... args) {
        auto v = std::vector<T, Alloc>();
        do_load(v, array, SVS_FWD(args)...);
        return v;
    }

    // With an allocator argument.
    template <detail::ArrayLike Array, typename... Args>
    static std::vector<T, Alloc>
    load(const Array& array, const Alloc& alloc, Args&&... args) {
        auto v = std::vector<T, Alloc>(alloc);
        do_load(v, array, SVS_FWD(args)...);
        return v;
    }
};

/////
///// DataType
/////

// This needs to go here because the DataType is needed during bootstrapping.
template <> struct Saver<DataType> {
    static SaveNode save(DataType x) { return name(x); }
};

template <> struct Loader<DataType> {
    using toml_type = toml::value<std::string>;
    static DataType load(const toml_type& v) { return parse_datatype(v.get()); }
};

/////
///// UUID
/////

template <> struct Saver<UUID> {
    static SaveNode save(UUID x) { return x.str(); }
};

template <> struct Loader<UUID> {
    using toml_type = toml::value<std::string>;
    static UUID load(const toml_type& val) { return UUID(val.get()); }
};

/////
///// Percent
/////

template <> struct Saver<Percent> {
    static SaveNode save(Percent x) { return SaveNode(x.value()); }
};

template <> struct Loader<Percent> {
    using toml_type = toml::value<double>;
    static Percent load(const toml_type& value) { return Percent(value.get()); }
};

/////
///// Save a full 64-bit unsigned integer
/////

struct FullUnsigned {
  public:
    explicit FullUnsigned(uint64_t value)
        : value_{value} {}
    operator uint64_t() const { return value_; }
    uint64_t value() const { return value_; }

  public:
    uint64_t value_;
};

template <> struct Saver<FullUnsigned> {
    static SaveNode save(FullUnsigned x) {
        return SaveNode(std::bit_cast<int64_t>(x.value()));
    }
};

template <> struct Loader<FullUnsigned> {
    using toml_type = toml::value<int64_t>;
    static FullUnsigned load(const toml_type& v) {
        return FullUnsigned(std::bit_cast<uint64_t>(v.get()));
    }
};

/////
///// BinaryBlob
/////

struct BinaryBlobSerializer {
    static constexpr lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "binary_blob";
};

template <typename T>
    requires(std::is_trivially_copyable_v<T>)
class BinaryBlobSaver : private BinaryBlobSerializer {
  public:
    explicit BinaryBlobSaver(std::span<const T> data)
        : BinaryBlobSerializer()
        , data_{data} {}

    template <typename Alloc>
    explicit BinaryBlobSaver(const std::vector<T, Alloc>& data)
        : BinaryBlobSaver(lib::as_const_span(data)) {}

    SaveTable save(const SaveContext& ctx) const {
        auto path = ctx.generate_name("binary_blob", "bin");
        size_t bytes_written = 0;
        {
            auto ostream = lib::open_write(path);
            bytes_written += lib::write_binary(ostream, data_);
        }

        return SaveTable(
            BinaryBlobSerializer::serialization_schema,
            BinaryBlobSerializer::save_version,
            {{"filename", lib::save(path.filename())},
             {"element_size", lib::save(sizeof(T))},
             {"element_type", lib::save(datatype_v<T>)},
             {"num_elements", lib::save(data_.size())}}
        );
    }

  private:
    std::span<const T> data_;
};

template <typename T, typename Alloc = std::allocator<T>>
    requires(std::is_trivially_copyable_v<T>)
class BinaryBlobLoader : private BinaryBlobSerializer {
  public:
    using BinaryBlobSerializer::save_version;
    using BinaryBlobSerializer::serialization_schema;

    explicit BinaryBlobLoader(size_t num_elements, const Alloc& allocator)
        : BinaryBlobSerializer()
        , data_(num_elements, allocator) {}

    // Implicit conversion to `std::vector`
    operator std::vector<T, Alloc>() && { return std::move(data_); }

    static BinaryBlobLoader load(const LoadTable& table, const Alloc& allocator = {}) {
        auto element_type = load_at<DataType>(table, "element_type");
        constexpr auto expected_element_type = datatype_v<T>;
        if (element_type != expected_element_type) {
            throw ANNEXCEPTION(
                "Element type mismatch! Expected {}, got {}.",
                element_type,
                expected_element_type
            );
        }

        // If this is an unknown data type - the best we can try to do is verify that the
        // element sizes are correct.
        if (element_type == DataType::undef) {
            auto element_size = load_at<size_t>(table, "element_size");
            if (element_size != sizeof(T)) {
                throw ANNEXCEPTION(
                    "Size mismatch for unknown element types. Expected {}, got {}.",
                    element_size,
                    sizeof(T)
                );
            }
        }

        auto num_elements = load_at<size_t>(table, "num_elements");
        auto filename =
            table.context().resolve(load_at<std::filesystem::path>(table, "filename"));

        auto loader = BinaryBlobLoader(num_elements, allocator);
        {
            auto istream = lib::open_read(filename);
            lib::read_binary(istream, loader.data_);
        }
        return loader;
    }

  private:
    std::vector<T, Alloc> data_{};
};

} // namespace svs::lib
