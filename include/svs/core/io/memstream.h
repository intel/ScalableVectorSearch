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

#include <cerrno>
#include <cstddef>
#include <filesystem>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <system_error>
#include <type_traits>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace svs::io {

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

    basic_mmstreambuf() = default;

    explicit basic_mmstreambuf(
        const std::filesystem::path& path,
        std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out
    ) {
        open(path, mode);
    }

    ~basic_mmstreambuf() override { close(); }

    basic_mmstreambuf(const basic_mmstreambuf&) = delete;
    basic_mmstreambuf& operator=(const basic_mmstreambuf&) = delete;

    basic_mmstreambuf(basic_mmstreambuf&& other) noexcept { move_from(std::move(other)); }

    basic_mmstreambuf& operator=(basic_mmstreambuf&& other) noexcept {
        if (this != &other) {
            close();
            move_from(std::move(other));
        }
        return *this;
    }

    void open(
        const std::filesystem::path& path,
        std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out
    ) {
        close();

        fd_ = ::open(path.c_str(), to_file_mode(mode));
        if (fd_ < 0) {
            throw std::system_error(errno, std::generic_category(), "open() failed");
        }

        struct stat st {};
        if (::fstat(fd_, &st) != 0) {
            const int ec = errno;
            close_fd_only();
            throw std::system_error(ec, std::generic_category(), "fstat() failed");
        }

        if (st.st_size < 0) {
            close_fd_only();
            throw std::runtime_error("Invalid file size");
        }

        mapped_size_ = static_cast<std::size_t>(st.st_size);

        if (mapped_size_ == 0) {
            this->setg(&empty_, &empty_, &empty_);
            return;
        }

        void* mapped =
            ::mmap(nullptr, mapped_size_, to_mmap_prot(mode), MAP_SHARED, fd_, 0);
        if (mapped == MAP_FAILED) {
            const int ec = errno;
            close_fd_only();
            throw std::system_error(ec, std::generic_category(), "mmap() failed");
        }

        data_ = static_cast<char_type*>(mapped);
        this->setg(data_, data_, data_ + mapped_size_);
    }

    void close() noexcept {
        this->setg(nullptr, nullptr, nullptr);

        if (data_ != nullptr) {
            ::munmap(static_cast<void*>(data_), mapped_size_);
            data_ = nullptr;
        }

        mapped_size_ = 0;
        close_fd_only();
    }

    [[nodiscard]] bool is_open() const noexcept { return fd_ >= 0; }

    [[nodiscard]] std::size_t size() const noexcept { return mapped_size_; }

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

  private:
    void close_fd_only() noexcept {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    void move_from(basic_mmstreambuf&& other) noexcept {
        fd_ = other.fd_;
        other.fd_ = -1;

        data_ = other.data_;
        other.data_ = nullptr;

        mapped_size_ = other.mapped_size_;
        other.mapped_size_ = 0;

        empty_ = other.empty_;

        if (data_ != nullptr) {
            this->setg(
                data_,
                data_ + (other.gptr() - other.eback()),
                data_ + (other.egptr() - other.eback())
            );
        } else {
            this->setg(&empty_, &empty_, &empty_);
        }

        other.setg(nullptr, nullptr, nullptr);
    }

    static int to_file_mode(std::ios_base::openmode mode) {
        constexpr auto in_out = std::ios_base::in | std::ios_base::out;
        int flags = 0;
        if ((mode & in_out) == in_out) {
            flags |= O_RDWR;
        } else if (mode & std::ios_base::in) {
            flags |= O_RDONLY;
        } else if (mode & std::ios_base::out) {
            flags |= O_WRONLY;
        }
        return flags;
    }

    static int to_mmap_prot(std::ios_base::openmode mode) {
        int prot = 0;
        if (mode & std::ios_base::in) {
            prot |= PROT_READ;
        }
        if (mode & std::ios_base::out) {
            prot |= PROT_WRITE;
        }
        return prot;
    }

  private:
    int fd_ = -1;
    char_type* data_ = nullptr;
    std::size_t mapped_size_ = 0;
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

    explicit basic_mmstream(const std::filesystem::path& path)
        : std::basic_istream<CharT, Traits>(nullptr)
        , buf_(path) {
        this->init(&buf_);
    }

    void open(const std::filesystem::path& path) {
        buf_.open(path);
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

/// Returns true if @p stream is backed entirely by an in-memory buffer.
///
/// Specifically, returns true when the stream's streambuf is either:
///   - a @c basic_mmstreambuf (memory-mapped file), or
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

} // namespace svs::io
