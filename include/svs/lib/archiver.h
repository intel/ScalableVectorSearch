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

// svs
#include "svs/lib/exception.h"

// stl
#include <algorithm>
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>

namespace svs::lib {

// CRTP
template <class Derived> struct Archiver {
    using size_type = uint64_t;

    // TODO: Define CACHELINE_BYTES in a common place
    // rather than duplicating it here and in prefetch.h
    static constexpr auto CACHELINE_BYTES = 64;

    static size_type write_size(std::ostream& os, size_type size) {
        os.write(reinterpret_cast<const char*>(&size), sizeof(size));
        if (!os) {
            throw ANNEXCEPTION("Error writing to stream!");
        }
        return sizeof(size);
    }

    static size_type read_size(std::istream& is, size_type& size) {
        is.read(reinterpret_cast<char*>(&size), sizeof(size));
        if (!is) {
            throw ANNEXCEPTION("Error reading from stream!");
        }
        return sizeof(size);
    }

    static size_type write_name(std::ostream& os, const std::string& name) {
        auto bytes = write_size(os, name.size());
        os.write(name.data(), name.size());
        if (!os) {
            throw ANNEXCEPTION("Error writing to stream!");
        }
        return bytes + name.size();
    }

    static size_type read_name(std::istream& is, std::string& name) {
        size_type size = 0;
        auto bytes = read_size(is, size);
        name.resize(size);
        is.read(name.data(), size);
        if (!is) {
            throw ANNEXCEPTION("Error reading from stream!");
        }
        return bytes + size;
    }

    static void read_from_istream(std::istream& in, std::ostream& out, size_t data_size) {
        // Copy the data in chunks.
        constexpr size_t buffer_size = 1 << 13; // 8KB buffer
        alignas(CACHELINE_BYTES) char buffer[buffer_size];

        size_t bytes_remaining = data_size;
        while (bytes_remaining > 0) {
            size_t to_read = std::min(buffer_size, bytes_remaining);
            in.read(buffer, to_read);
            if (!in) {
                throw ANNEXCEPTION("Error reading from stream!");
            }
            out.write(buffer, to_read);
            bytes_remaining -= to_read;
        }
    }
};

} // namespace svs::lib
