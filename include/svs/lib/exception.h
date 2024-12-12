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

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
#pragma once

///
/// @ingroup lib_public
/// @defgroup lib_exception Exception Classes.
///

#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "svs/lib/preprocessor.h"
#include "svs/lib/tuples.h"
#include "svs/third-party/fmt.h"

// Gather information about the current line.
#define SVS_LINEINFO svs::lib::detail::LineInfo(__LINE__, __FILE__)

///
/// @ingroup lib_exception
/// @brief Construct an exception with line information.
///
#define ANNEXCEPTION(format, ...) \
    svs::lib::ANNException(format " {}", __VA_ARGS__ __VA_OPT__(, ) SVS_LINEINFO)

namespace svs::lib::detail {
struct LineInfo {
    inline LineInfo(int line, std::string_view file)
        : line_{line}
        , file_{file} {}
    int line_;
    std::string_view file_;
};
} // namespace svs::lib::detail

// Enable formatting for LineInfo.
template <> struct fmt::formatter<svs::lib::detail::LineInfo> : svs::format_empty {
    auto format(svs::lib::detail::LineInfo linfo, auto& ctx) const {
        return fmt::format_to(ctx.out(), "(line {} in {})", linfo.line_, linfo.file_);
    }
};

namespace svs {
namespace lib {

///
/// @ingroup lib_exception
/// @brief Generic exception thrown by routines within htelibrary.
///
class ANNException : public std::runtime_error {
  public:
    /// @brief Construct a new excpetion with the given error message.
    inline explicit ANNException(const std::string& message)
        : std::runtime_error{message} {}

    ///
    /// @brief Construct a new exception using the given format string and arguments.
    ///
    /// @param format A fmtlib compatible formatting string.
    /// @param args A collection of arguments to place in the formatting string. All
    ///     argument types must implement fmtlib formatting.
    ///
    /// Construct an ``ANNException`` with all arguments formatted and concatenated as a
    /// string message.
    ///
    /// @example
    /// ```c++
    /// int a = 10;
    /// int b = 20;
    /// throw ANNException("Value mismatch. Expected {}, got {}.", a, b);
    /// ```
    ///
    template <typename... Args>
    SVS_NOINLINE explicit ANNException(fmt::format_string<Args...> format, Args&&... args)
        : std::runtime_error(fmt::format(format, SVS_FWD(args)...)) {}
};
} // namespace lib

using ANNException = lib::ANNException;

} // namespace svs
