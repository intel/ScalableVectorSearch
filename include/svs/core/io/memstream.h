/*
 * Copyright 2026 Intel Corporation
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

#include "svs/core/allocator.h"
#include "svs/lib/array.h" // just for svs::is_view_type_v specialization

#include <cerrno>
#include <cstddef>
#include <filesystem>
#include <istream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <system_error>
#include <type_traits>
#include <version>

#if defined(__cpp_lib_spanstream) && __cpp_lib_spanstream >= 202106L
#include <spanstream>
#define SVS_HAS_STD_SPANSTREAM 1
#else
#define SVS_HAS_STD_SPANSTREAM 0
#endif

namespace svs {
namespace io {

template <typename CharT, typename Traits = std::char_traits<CharT>>
class basic_mmstreambuf : public std::basic_streambuf<CharT, Traits> {
    static_assert(
        sizeof(CharT) == 1, "basic_mmstreambuf requires a 1-byte character type."
    );

  public:
    using char_type = CharT;
    using traits_type = Traits;
    using int_type = typename traits_type::int_type;
    using pos_type = typename traits_type::pos_type;
    using off_type = typename traits_type::off_type;

    explicit basic_mmstreambuf(MMapPtr<CharT> mapping)
        : ptr_{std::move(mapping)} {
        if (ptr_) {
            auto base_ptr = static_cast<char_type*>(ptr_.base());
            this->setg(base_ptr, ptr_.data(), base_ptr + ptr_.size());
            this->setp(&empty_, &empty_); // disallow writing
        } else {
            this->setg(&empty_, &empty_, &empty_); // empty buffer
            this->setp(&empty_, &empty_);          // disallow writing
        }
    }

    basic_mmstreambuf()
        : basic_mmstreambuf(MMapPtr<CharT>{}) {}
    basic_mmstreambuf(const basic_mmstreambuf&) = delete;
    basic_mmstreambuf& operator=(const basic_mmstreambuf&) = delete;
    basic_mmstreambuf(basic_mmstreambuf&&) = default;
    basic_mmstreambuf& operator=(basic_mmstreambuf&&) = default;

    basic_mmstreambuf* open(
        const std::filesystem::path& path,
        std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out
    ) {
        std::error_code ec;
        auto size = std::filesystem::file_size(path, ec);
        if (ec) {
            throw ANNEXCEPTION(
                "Failed to get file size: {} with system error: {}",
                path.string(),
                ec.message()
            );
        }
        if (size == 0) {
            throw ANNEXCEPTION("Cannot memory-map empty file: {}", path.string());
        }
        auto perm =
            (mode & std::ios_base::out) ? MemoryMapper::ReadWrite : MemoryMapper::ReadOnly;
        ptr_ =
            MemoryMapper{perm, MemoryMapper::MustUseExisting}.mmap(path, lib::Bytes{size});
        if (!ptr_) {
            throw ANNEXCEPTION("Failed to memory-map file: {}", path.string());
        }
        auto base_ptr = static_cast<char_type*>(ptr_.base());
        this->setg(base_ptr, ptr_.data(), base_ptr + ptr_.size());
        this->setp(&empty_, &empty_); // disallow writing
        return this;
    }

    basic_mmstreambuf* close() noexcept {
        ptr_.unmap();
        this->setg(&empty_, &empty_, &empty_); // empty buffer
        this->setp(&empty_, &empty_);          // disallow writing
        return this;
    }

    [[nodiscard]] bool is_open() const noexcept { return static_cast<bool>(ptr_); }

    [[nodiscard]] std::size_t size() const noexcept { return ptr_.size(); }

  protected:
    int_type underflow() override {
        if (this->gptr() == this->egptr()) {
            return traits_type::eof();
        }
        return traits_type::to_int_type(*this->gptr());
    }

    pos_type seekoff(
        off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which
    ) override {
        if (!(which & std::ios_base::in)) {
            return pos_type(off_type(-1));
        }

        const off_type current = static_cast<off_type>(this->gptr() - this->eback());
        const off_type end = static_cast<off_type>(this->egptr() - this->eback());

        off_type target = 0;
        switch (dir) {
            case std::ios_base::beg:
                target = off;
                break;
            case std::ios_base::cur:
                target = current + off;
                break;
            case std::ios_base::end:
                target = end + off;
                break;
            default:
                return pos_type(off_type(-1));
        }

        if (target < 0 || target > end) {
            return pos_type(off_type(-1));
        }

        this->setg(this->eback(), this->eback() + target, this->egptr());
        return pos_type(target);
    }

    pos_type seekpos(pos_type sp, std::ios_base::openmode which) override {
        return seekoff(static_cast<off_type>(sp), std::ios_base::beg, which);
    }

    int_type overflow(int_type) override {
        return Traits::eof(); // disallow writing
    }

  private:
    MMapPtr<CharT> ptr_;
    // A dummy character to use as the put area for the streambuf when the mapping is empty
    // or closed. This is necessary to ensure that the put area is always valid, even when
    // the mapping is empty or closed.
    char_type empty_ = char_type{};
};

template <typename CharT, typename Traits = std::char_traits<CharT>>
class basic_mmstream : public std::basic_istream<CharT, Traits> {
  public:
    using streambuf_type = basic_mmstreambuf<CharT, Traits>;

    basic_mmstream()
        : std::basic_istream<CharT, Traits>(nullptr) {
        this->init(&buf_);
    }

    explicit basic_mmstream(MMapPtr<CharT> mapping)
        : std::basic_istream<CharT, Traits>(nullptr)
        , buf_(std::move(mapping)) {
        this->init(&buf_);
    }

    explicit basic_mmstream(
        const std::filesystem::path& path,
        std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out
    )
        : basic_mmstream(MMapPtr<CharT>{}) {
        open(path, mode);
    }

    void open(
        const std::filesystem::path& path,
        std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out
    ) {
        buf_.open(path, mode);
        this->clear();
    }

    void close() noexcept {
        buf_.close();
        this->setstate(std::ios_base::eofbit);
    }

    [[nodiscard]] bool is_open() const noexcept { return buf_.is_open(); }

    [[nodiscard]] std::size_t size() const noexcept { return buf_.size(); }

    [[nodiscard]] streambuf_type* rdbuf() noexcept { return &buf_; }

  private:
    streambuf_type buf_;
};

using mmstreambuf = basic_mmstreambuf<char>;
using mmstream = basic_mmstream<char>;

#if SVS_HAS_STD_SPANSTREAM

template <typename CharT, typename Traits = std::char_traits<CharT>>
using basic_spanbuf = std::basic_spanbuf<CharT, Traits>;

template <typename CharT, typename Traits = std::char_traits<CharT>>
using basic_ispanstream = std::basic_ispanstream<CharT, Traits>;

#else

template <typename CharT, typename Traits = std::char_traits<CharT>>
class basic_spanbuf : public std::basic_streambuf<CharT, Traits> {
    static_assert(sizeof(CharT) == 1, "basic_spanbuf requires a 1-byte character type.");

  public:
    using char_type = CharT;
    using traits_type = Traits;
    using int_type = typename traits_type::int_type;
    using pos_type = typename traits_type::pos_type;
    using off_type = typename traits_type::off_type;
    using span_type = std::span<CharT>;

    basic_spanbuf()
        : basic_spanbuf(span_type{}) {}

    explicit basic_spanbuf(span_type s) { span(s); }

    /// Returns the underlying span.
    [[nodiscard]] span_type span() const noexcept { return data_; }

    /// Updates the underlying span and resets the read position to the beginning.
    void span(span_type s) noexcept {
        data_ = s;
        if (data_.empty()) {
            this->setg(&empty_, &empty_, &empty_);
        } else {
            auto* begin = data_.data();
            this->setg(begin, begin, begin + data_.size());
        }
        this->setp(&empty_, &empty_); // disallow writing
    }

  protected:
    int_type overflow(int_type) override {
        return traits_type::eof(); // disallow writing
    }

    std::basic_streambuf<CharT, Traits>* setbuf(char_type* s, std::streamsize n) override {
        span(span_type{s, static_cast<std::size_t>(n)});
        return this;
    }

    pos_type seekoff(
        off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which
    ) override {
        if (!(which & std::ios_base::in)) {
            return pos_type(off_type(-1));
        }

        const off_type current = static_cast<off_type>(this->gptr() - this->eback());
        const off_type end = static_cast<off_type>(this->egptr() - this->eback());

        off_type target = 0;
        switch (dir) {
            case std::ios_base::beg:
                target = off;
                break;
            case std::ios_base::cur:
                target = current + off;
                break;
            case std::ios_base::end:
                target = end + off;
                break;
            default:
                return pos_type(off_type(-1));
        }

        if (target < 0 || target > end) {
            return pos_type(off_type(-1));
        }

        this->setg(this->eback(), this->eback() + target, this->egptr());
        return pos_type(target);
    }

    pos_type seekpos(pos_type sp, std::ios_base::openmode which) override {
        return seekoff(static_cast<off_type>(sp), std::ios_base::beg, which);
    }

  private:
    span_type data_;
    char_type empty_ = char_type{};
};

template <typename CharT, typename Traits = std::char_traits<CharT>>
class basic_ispanstream : public std::basic_istream<CharT, Traits> {
  public:
    using char_type = CharT;
    using traits_type = Traits;
    using int_type = typename traits_type::int_type;
    using pos_type = typename traits_type::pos_type;
    using off_type = typename traits_type::off_type;
    using streambuf_type = basic_spanbuf<CharT, Traits>;
    using span_type = typename streambuf_type::span_type;

    basic_ispanstream()
        : std::basic_istream<CharT, Traits>(nullptr) {
        this->init(&buf_);
    }

    explicit basic_ispanstream(span_type span)
        : std::basic_istream<CharT, Traits>(nullptr)
        , buf_(span) {
        this->init(&buf_);
    }

    span_type span() const noexcept { return buf_.span(); }
    void span(span_type s) noexcept {
        buf_.span(s);
        this->clear();
    }

    [[nodiscard]] streambuf_type* rdbuf() noexcept { return &buf_; }

  private:
    streambuf_type buf_;
};

#endif

using spanbuf = basic_spanbuf<char>;
using ispanstream = basic_ispanstream<char>;

/// Returns true if @p stream is backed entirely by an in-memory buffer.
///
/// Specifically, returns true when the stream's streambuf is either:
///   - a @c basic_mmstreambuf (memory-mapped file), or
///   - a @c basic_spanbuf (non-owning in-memory span), or
///   - a @c std::basic_stringbuf (std::istringstream / std::stringstream).
template <typename CharT, typename Traits = std::char_traits<CharT>>
[[nodiscard]] bool is_memory_stream(std::basic_istream<CharT, Traits>& stream) noexcept {
    auto* buf = stream.rdbuf();
    if (buf == nullptr) {
        return false;
    }
    if (dynamic_cast<basic_mmstreambuf<CharT, Traits>*>(buf) != nullptr) {
        return true;
    }
    if (dynamic_cast<basic_spanbuf<CharT, Traits>*>(buf) != nullptr) {
        return true;
    }
    if (dynamic_cast<std::basic_stringbuf<CharT, Traits>*>(buf) != nullptr) {
        return true;
    }
    return false;
}

namespace detail {

// A minimal accessor that promotes the protected gptr() method of
// std::basic_streambuf to public visibility.  It adds no data members and no
// virtual functions, so the static_cast below is layout-safe (gptr() reads only
// base-class internal pointers).
template <typename CharT, typename Traits>
struct StreambufAccessor : std::basic_streambuf<CharT, Traits> {
    static CharT* get(std::basic_streambuf<CharT, Traits>* buf) noexcept {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        return static_cast<StreambufAccessor*>(buf)->gptr();
    }

    static CharT* begin(std::basic_streambuf<CharT, Traits>* buf) noexcept {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        return static_cast<StreambufAccessor*>(buf)->eback();
    }

    static CharT* end(std::basic_streambuf<CharT, Traits>* buf) noexcept {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        return static_cast<StreambufAccessor*>(buf)->egptr();
    }
};

} // namespace detail

/// Returns a typed pointer to the current read position of an in-memory stream.
///
/// Works for:
///   - @c basic_mmstreambuf-backed streams (memory-mapped files via @c basic_mmstream)
///   - @c std::basic_stringbuf-backed streams (@c std::istringstream / @c
///   std::stringstream)
///
/// @tparam T        Element type to interpret the raw bytes at the current position as.
/// @tparam CharT    Character type of the stream (must be 1-byte wide).
/// @tparam Traits   Character traits of the stream.
/// @param  stream   The input stream to query.
/// @returns A pointer of type @c T* to the current read position,
///          or @c nullptr if the stream is not in-memory or has no streambuf.
template <typename T, typename CharT, typename Traits = std::char_traits<CharT>>
[[nodiscard]] T* current_ptr(std::basic_istream<CharT, Traits>& stream) noexcept {
    static_assert(sizeof(CharT) == 1, "current_ptr requires a 1-byte character type.");
    if (!is_memory_stream(stream)) {
        return nullptr;
    }

    auto* buf = stream.rdbuf();
    auto begin = detail::StreambufAccessor<CharT, Traits>::begin(buf);
    auto end = detail::StreambufAccessor<CharT, Traits>::end(buf);
    if (begin == end) {
        return nullptr;
    }
    auto raw = detail::StreambufAccessor<CharT, Traits>::get(buf);

    // Return nullptr if the current position is misaligned for the requested type T, to
    // avoid undefined behavior on dereference.
    if (reinterpret_cast<std::uintptr_t>(raw) % alignof(T) != 0) {
        assert(
            false && "current_ptr: current position is misaligned for the requested type T"
        );
        return nullptr;
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<T*>(raw);
}

/// Returns a typed pointer to the beginning of the read position of an in-memory stream.
///
/// Works for:
///   - @c basic_mmstreambuf-backed streams (memory-mapped files via @c basic_mmstream)
///   - @c std::basic_stringbuf-backed streams (@c std::istringstream / @c
///   std::stringstream)
///
/// @tparam T        Element type to interpret the raw bytes at the current position as.
/// @tparam CharT    Character type of the stream (must be 1-byte wide).
/// @tparam Traits   Character traits of the stream.
/// @param  stream   The input stream to query.
/// @returns A pointer of type @c T* to the beginning of the read position,
///          or @c nullptr if the stream is not in-memory or has no streambuf.
template <typename T, typename CharT, typename Traits = std::char_traits<CharT>>
[[nodiscard]] T* begin_ptr(std::basic_istream<CharT, Traits>& stream) noexcept {
    static_assert(sizeof(CharT) == 1, "begin_ptr requires a 1-byte character type.");
    if (!is_memory_stream(stream)) {
        return nullptr;
    }

    auto* buf = stream.rdbuf();
    auto begin = detail::StreambufAccessor<CharT, Traits>::begin(buf);
    auto end = detail::StreambufAccessor<CharT, Traits>::end(buf);
    if (begin == end) {
        return nullptr;
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<T*>(begin);
}

/// Returns a typed pointer to the end of the read position of an in-memory stream.
///
/// Works for:
///   - @c basic_mmstreambuf-backed streams (memory-mapped files via @c basic_mmstream)
///   - @c std::basic_stringbuf-backed streams (@c std::istringstream / @c
///   std::stringstream)
///
/// @tparam T        Element type to interpret the raw bytes at the current position as.
/// @tparam CharT    Character type of the stream (must be 1-byte wide).
/// @tparam Traits   Character traits of the stream.
/// @param  stream   The input stream to query.
/// @returns A pointer of type @c T* to the end of the read position,
///          or @c nullptr if the stream is not in-memory or has no streambuf.
template <typename T, typename CharT, typename Traits = std::char_traits<CharT>>
[[nodiscard]] T* end_ptr(std::basic_istream<CharT, Traits>& stream) noexcept {
    static_assert(sizeof(CharT) == 1, "end_ptr requires a 1-byte character type.");
    if (!is_memory_stream(stream)) {
        return nullptr;
    }

    auto* buf = stream.rdbuf();
    auto begin = detail::StreambufAccessor<CharT, Traits>::begin(buf);
    auto end = detail::StreambufAccessor<CharT, Traits>::end(buf);
    if (begin == end) {
        return nullptr;
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<T*>(end);
}

/// @brief Memory-stream allocator that allocates memory from in-memory streams.
///
/// This is used to construct SVS data structures directly on memory-mapped files or
/// in-memory buffers, without needing to copy data out of the stream into separately
/// allocated memory.
///
/// The allocator does not take ownership of the memory; the caller is responsible for
/// ensuring the memory remains valid for the lifetime of any pointers returned by this
/// allocator.
template <typename T, typename CharT = char, typename Traits = std::char_traits<CharT>>
struct MemoryStreamAllocator {
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    using stream_type = std::basic_istream<CharT, Traits>;

    MemoryStreamAllocator() = default;

    MemoryStreamAllocator(stream_type& stream)
        : stream_(&stream) {
        if (!is_memory_stream(*stream_)) {
            throw std::invalid_argument(
                "MemoryStreamAllocator requires a memory-backed stream."
            );
        }
    }

    template <typename U>
    MemoryStreamAllocator(const MemoryStreamAllocator<U, CharT, Traits>& other)
        : stream_(&other.stream()) {}

    [[nodiscard]] pointer allocate(size_type n) {
        if (stream_ == nullptr) {
            throw std::runtime_error("MemoryStreamAllocator is not properly initialized.");
        }
        T* current = current_ptr<T>(*stream_);
        if (current == nullptr) {
            throw std::runtime_error("Failed to obtain current pointer from memory stream."
            );
        }
        pointer result = current;

        // check for overflow:
        auto off = lib::narrow<typename stream_type::off_type>(n * sizeof(T));

        stream_->seekg(off, std::ios_base::cur);
        if (!*stream_) {
            throw std::runtime_error("Failed to advance memory stream after allocation.");
        }
        return result;
    }

    void deallocate(pointer, size_type) noexcept {
        // No-op since we don't own the memory.
    }

    stream_type& stream() const noexcept { return *stream_; }

  private:
    stream_type* stream_ = nullptr;
};

} // namespace io

template <typename T>
inline constexpr bool is_view_type_v<io::MemoryStreamAllocator<T>> = true;

} // namespace svs
