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

// fmt
#include "fmt/core.h"
#include "fmt/format.h"
#include "fmt/std.h"

namespace svs {

///
/// Simple base class the formatters can derive from if they only wish to implement empty
/// formatting.
///
struct format_empty {
    constexpr auto parse(fmt::format_parse_context& ctx) -> decltype(ctx.begin()) {
        auto it = ctx.begin();
        auto end = ctx.end();
        if (it == end || *it == '}') {
            return it;
        }
        throw fmt::format_error("invalid format - must be empty!");
    }
};
} // namespace svs
