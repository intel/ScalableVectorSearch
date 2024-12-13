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

#include "svs/lib/float16.h"
#include "svs/lib/misc.h"
#include "svs/lib/narrow.h"
#include "svs/lib/readwrite.h"
#include "svs/lib/tuples.h"

#include <functional>
#include <iostream>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>

namespace svs::lib {

/////
///// Optional conversion
/////

namespace detail {
template <typename T, typename U> struct IOConvert {
    using type = T;
};
template <typename U> struct IOConvert<void, U> {
    using type = U;
};
} // namespace detail

template <typename T, typename U>
using io_convert_type_t =
    typename detail::IOConvert<std::remove_const_t<T>, std::remove_const_t<U>>::type;

// Allow the Writers to either write based on the argument type, or try to perform
// conversion to the requested type internally.
template <typename T, typename U> io_convert_type_t<T, U> io_convert(U u) {
    return narrow<io_convert_type_t<T, U>>(u);
}

// We expect the conversion to `Float16` to be lossy.
// Hence, don't require precise narrowing.
template <> inline Float16 io_convert<Float16, float>(float u) { return Float16(u); }

/////
///// Iterator Helpers
/////

template <typename T>
    requires std::is_trivially_copyable_v<T> && std::is_default_constructible_v<T>
class VectorReader {
  public:
    explicit VectorReader(size_t size = 0)
        : buffer_{std::vector<T>(size)} {}
    // Change the number of elements to be read.
    void resize(size_t new_size) { buffer_.resize(new_size); }
    // Return the number of elements to be read.
    [[nodiscard]] size_t size() noexcept { return buffer_.size(); }
    // Read `size()` elements from stream.
    template <typename Stream> void read(Stream& stream) { read_binary(stream, buffer_); }
    // Return a reference to the underlying data.
    std::span<const T, Dynamic> data() noexcept { return {buffer_.data(), buffer_.size()}; }

  private:
    std::vector<T> buffer_;
};

template <typename T>
    requires std::is_trivially_copyable_v<T> && std::is_default_constructible_v<T>
class ValueReader {
  public:
    ValueReader() = default;

    // Read a value of type `T` from the stream.
    template <typename Stream> void read(Stream& stream) { read_binary(stream, value_); }
    const T& data() noexcept { return value_; }

  private:
    T value_;
};

/////
///// Iterator
/////

struct HeterogeneousFileEnd {};

// Call `.data()` on each reader.
template <typename... Readers> constexpr auto getdata(std::tuple<Readers...>& readers) {
    // NOTE: use `decltype(auto) as the trailing return type for the lambda to ensure
    // that the return-by-reference properties of the readers are forwarded properly.
    return map(readers, [](auto& x) -> decltype(auto) { return x.data(); });
}

// Generic class for reading heterogeneously packed binary files.
// The `HeterogeneousFileIterator` accepts an arbitrary number of `Readers`.
// Examples of such reades include the `VectorReader` and `ValueReader` defined above.
template <typename F, typename... Readers> class HeterogeneousFileIterator {
    // List members first to boot-strap type aliases like `value_type`.
  private:
    std::ifstream& stream_;
    std::tuple<Readers...> readers_;
    F postprocess_;
    // For now, require the number of elements to be read to be known ahead of time.
    // We'll likely have to revisit this.
    size_t reads_performed_;
    size_t reads_to_perform_;

  public:
    // Iterator Tags
    using iterator_category = std::input_iterator_tag;
    using value_type = decltype(std::apply(postprocess_, getdata(readers_)));
    using pointer = value_type*;
    using reference = value_type&;
    using const_reference = value_type const&;

    // Constructor
    template <typename Fs, typename... Rs>
    HeterogeneousFileIterator(
        Fs&& postprocess, std::ifstream& stream, size_t reads_to_perform, Rs&&... readers
    )
        : stream_{stream}
        , readers_{std::tuple<Readers...>(std::forward<Rs>(readers)...)}
        , postprocess_{std::forward<Fs>(postprocess)}
        , reads_performed_{0}
        , reads_to_perform_{reads_to_perform} {
        // kickstart reading so first dereference works.
        if (reads_to_perform_ > 0) {
            read();
        }
    }

    void read() {
        foreach (readers_, [&](auto& reader) { reader.read(stream_); })
            ;
    }

    // Iterator Interface
    value_type operator*() { return std::apply(postprocess_, getdata(readers_)); }
    HeterogeneousFileIterator& operator++() {
        ++reads_performed_;
        if (!done()) {
            read();
        }
        return *this;
    }

    [[nodiscard]] bool done() const { return reads_performed_ == reads_to_perform_; }
    [[nodiscard]] bool operator!=(HeterogeneousFileEnd /*unused*/) const {
        return !done();
    };
};

// Helper constructor
template <typename... Readers>
HeterogeneousFileIterator<identity, std::decay_t<Readers>...>
heterogeneous_iterator(std::ifstream& stream, size_t lines_to_read, Readers&&... readers) {
    return {identity(), stream, lines_to_read, std::forward<Readers>(readers)...};
}

template <typename F, typename... Readers>
HeterogeneousFileIterator<std::decay_t<F>, std::decay_t<Readers>...> heterogeneous_iterator(
    F&& f, std::ifstream& stream, size_t lines_to_read, Readers&&... readers
) {
    return {std::forward<F>(f), stream, lines_to_read, std::forward<Readers>(readers)...};
}
} // namespace svs::lib
