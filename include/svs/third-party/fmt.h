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

// stl
#include <string_view>

namespace svs {

#define SVS_SHOW_IMPL(f, suffix, show_name, var_name) f(#show_name ": {}" #suffix, var_name)

// Expected transformation:
// SVS_SHOW_STRING_(x) -> fmt::format("x: {}", x_);
#define SVS_SHOW_STRING_(name) SVS_SHOW_IMPL(fmt::format, , name, name##_)

// Expected transformation:
// SVS_SHOW_STRING(x) -> fmt::format("x: {}", x);
#define SVS_SHOW_STRING(name) SVS_SHOW_IMPL(fmt::format, , name, name)

// Expected transformation:
// SVS_SHOW(x) -> fmt::print("x: {}\n", x);
#define SVS_SHOW(name) SVS_SHOW_IMPL(fmt::print, \n, name, name)

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

constexpr std::string_view make_string_view(fmt::string_view v) {
    return std::string_view{v.data(), v.size()};
}

} // namespace svs
