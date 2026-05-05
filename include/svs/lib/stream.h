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

// saveload
#include "svs/lib/saveload/core.h"

// svs
#include "svs/lib/archiver.h"
#include "svs/lib/exception.h"

// stl
#include <cstdint>
#include <sstream>

namespace svs::lib::detail {
template <typename T> auto get_buffer_size(T& ss) {
    if constexpr (requires { ss.rdbuf()->view(); }) {
        return ss.rdbuf()->view().size();
    } else {
        return ss.str().size();
    }
}
} // namespace svs::lib::detail

namespace svs::lib {

struct StreamArchiver : Archiver<StreamArchiver> {
    // SVS_STRM
    static constexpr size_type magic_number = 0x5356535f5354524d;

    static auto read_table(std::istream& is) {
        std::uint64_t tablesize = 0;
        read_size(is, tablesize);

        std::stringstream ss;
        read_from_istream(is, ss, tablesize);

        return toml::parse(ss);
    }

    static void write_table(std::ostream& os, const toml::table& table) {
        std::stringstream ss;
        ss << table << "\n";

        // The best way to get the table size is a c++20 feature:
        // ss.rdbuf()->view().size(),
        // but Apple's Clang 15 doesn't support std::stringbuf::view()
        lib::StreamArchiver::size_type tablesize = detail::get_buffer_size(ss);

        // Get the current position in the stream and compute the padding characters needed
        // to align the (table + tablesize values) to a cache line boundary.
        auto pos = os.tellp();
        // check if position is valid before using it. If it's not valid, we can assume that
        // the stream is at the beginning and thus doesn't need padding.
        pos = std::max(pos, decltype(pos)(0));

        auto padding = (lib::StreamArchiver::CACHELINE_BYTES -
                        (static_cast<size_t>(pos) + sizeof(tablesize) + tablesize) %
                            lib::StreamArchiver::CACHELINE_BYTES) %
                       lib::StreamArchiver::CACHELINE_BYTES;

        ss << std::string(padding, ' ');
        // recompute tablesize after adding padding.
        tablesize = detail::get_buffer_size(ss);

        lib::StreamArchiver::write_size(os, tablesize);
        os << ss.rdbuf();
        if (!os) {
            throw ANNEXCEPTION("Error writing to stream!");
        }
    }
};

} // namespace svs::lib
