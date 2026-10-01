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

#include "svs/c/svs_c.h"

#include "error.hpp"

#include <ios>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <vector>

namespace svs::c_runtime {

// Bridges svs_stream_i's read/write callbacks to std::streambuf. Buffered at 64 KiB so the
// callback amortizes across many bytes instead of firing once per byte.
class StreamBuf : public std::streambuf {
  public:
    static constexpr size_t buffer_size = 64 * 1024;
    enum class Direction { read, write };

    static void validate(svs_stream_i stream, bool need_write) {
        if (stream == nullptr) {
            throw std::invalid_argument("Stream pointer cannot be null.");
        }
        if (stream->ops == nullptr) {
            throw std::invalid_argument("Stream interface is not initialized.");
        }
        if (stream->ops->version > svs_get_version()) {
            throw std::invalid_argument("Stream interface version is not supported.");
        }
        if (stream->ops->struct_size < sizeof(svs_stream_ops_t)) {
            throw std::invalid_argument("Incompatible stream interface struct size.");
        }
        if (need_write) {
            if (stream->ops->write == nullptr) {
                throw std::invalid_argument("Stream interface has no write callback.");
            }
        } else {
            if (stream->ops->read == nullptr) {
                throw std::invalid_argument("Stream interface has no read callback.");
            }
        }
    }

    // Holds a value copy of the user's ops table; `self` is referenced and only needs to
    // remain valid until the streaming save/load call returns.
    StreamBuf(const svs_stream_ops_t& ops, void* self, Direction direction)
        : ops_(ops)
        , self_(self)
        , direction_(direction)
        , read_buf_(direction == Direction::read ? buffer_size : 0)
        , write_buf_(direction == Direction::write ? buffer_size : 0) {
        if (direction_ == Direction::write) {
            setp(write_buf_.data(), write_buf_.data() + write_buf_.size());
        }
    }

    StreamBuf(const StreamBuf&) = delete;
    StreamBuf& operator=(const StreamBuf&) = delete;
    StreamBuf(StreamBuf&&) = delete;
    StreamBuf& operator=(StreamBuf&&) = delete;

    // A destructor must never throw; callers that need to observe a final write failure
    // should call pubsync() themselves before the stream goes out of scope.
    ~StreamBuf() override {
        if (direction_ == Direction::write) {
            try {
                flush_write_buffer();
            } catch (...) {}
        }
    }

  protected:
    int_type overflow(int_type ch) override {
        flush_write_buffer();
        if (!traits_type::eq_int_type(ch, traits_type::eof())) {
            *pptr() = traits_type::to_char_type(ch);
            pbump(1);
        }
        return traits_type::not_eof(ch);
    }

    int sync() override {
        flush_write_buffer();
        return 0;
    }

    int_type underflow() override {
        if (gptr() < egptr()) {
            return traits_type::to_int_type(*gptr());
        }
        // Default-initialized to SVS_OK: a 0-byte read is legitimate EOF unless the
        // callback explicitly reported an error, which read()'s return value cannot encode.
        svs_error_desc impl_error{};
        size_t n = ops_.read(self_, read_buf_.data(), read_buf_.size(), &impl_error);
        if (n == 0) {
            if (impl_error.code != SVS_OK) {
                throw coded_error(
                    impl_error.code,
                    "Stream read callback failed: (" + std::to_string(impl_error.code) +
                        ") " + impl_error.message
                );
            }
            return traits_type::eof();
        }
        setg(read_buf_.data(), read_buf_.data(), read_buf_.data() + n);
        return traits_type::to_int_type(*gptr());
    }

    pos_type seekoff(
        off_type off, std::ios_base::seekdir way, std::ios_base::openmode which
    ) override {
        if (off == 0 && way == std::ios_base::cur && which == std::ios_base::out) {
            // tellp() must count bytes handed to the streambuf, not bytes flushed to the
            // callback, or the format's cache-line padding misaligns silently.
            return pos_type(static_cast<off_type>(written_ + (pptr() - pbase())));
        }
        return pos_type(off_type(-1));
    }

  private:
    void flush_write_buffer() {
        auto n = static_cast<size_t>(pptr() - pbase());
        // Reset the put area before the callback: a throw then leaves it empty, so the
        // destructor's flush is a no-op instead of redelivering the same bytes twice.
        setp(write_buf_.data(), write_buf_.data() + write_buf_.size());
        if (n > 0) {
            svs_error_desc impl_error{
                SVS_ERROR_UNKNOWN, "Unknown error in stream write callback"};
            if (!ops_.write(self_, write_buf_.data(), n, &impl_error)) {
                throw coded_error(
                    impl_error.code,
                    "Stream write callback failed: (" + std::to_string(impl_error.code) +
                        ") " + impl_error.message
                );
            }
            written_ += n;
        }
    }

    svs_stream_ops_t ops_;
    void* self_;
    Direction direction_;
    std::vector<char> read_buf_;
    std::vector<char> write_buf_;
    size_t written_ = 0;
};

namespace detail {
// Base ordering trick: a base class initializes before other bases declared after it, so
// this guarantees `buf` exists before std::istream/std::ostream stores its address.
struct StreamBufHolder {
    StreamBuf buf;
    StreamBufHolder(const svs_stream_ops_t& ops, void* self, StreamBuf::Direction direction)
        : buf(ops, self, direction) {}
};
} // namespace detail

class InputStream : private detail::StreamBufHolder, public std::istream {
  public:
    InputStream(const svs_stream_ops_t& ops, void* self)
        : detail::StreamBufHolder(ops, self, StreamBuf::Direction::read)
        , std::istream(&buf) {
        // Without this, the sentry swallows a read-callback exception into a silent
        // badbit instead of rethrowing it; EOF alone only sets eofbit/failbit, not badbit.
        exceptions(std::ios_base::badbit);
    }
};

class OutputStream : private detail::StreamBufHolder, public std::ostream {
  public:
    OutputStream(const svs_stream_ops_t& ops, void* self)
        : detail::StreamBufHolder(ops, self, StreamBuf::Direction::write)
        , std::ostream(&buf) {
        // Without this, the sentry swallows a write-callback exception into a silent
        // badbit instead of rethrowing it, so a failed save would report success.
        exceptions(std::ios_base::badbit);
    }
};

} // namespace svs::c_runtime
