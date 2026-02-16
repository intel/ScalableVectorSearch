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

        lib::StreamArchiver::size_type tablesize = ss.rdbuf()->view().size();
        lib::StreamArchiver::write_size(os, tablesize);
        os << ss.rdbuf();
    }
};

} // namespace svs::lib
