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
