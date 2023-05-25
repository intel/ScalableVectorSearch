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

// Gather information about the current line.
#define SVS_LINEINFO svs::lib::detail::LineInfo(__LINE__, __FILE__)

///
/// @ingroup lib_exception
/// @brief Construct an exception with line information.
///
#define ANNEXCEPTION(...) svs::lib::ANNException(SVS_LINEINFO, __VA_ARGS__)

namespace svs {
namespace lib {
namespace detail {
struct LineInfo {
    inline LineInfo(int line, std::string&& file)
        : line_{line}
        , file_{std::move(file)} {}
    int line_;
    std::string file_;
};

inline std::ostream& operator<<(std::ostringstream& stream, const LineInfo& lineinfo) {
    stream << "(line " << lineinfo.line_ << " in " << lineinfo.file_ << ")";
    return stream;
}

template <typename... Args> std::string format_args(Args&&... args) {
    std::ostringstream stream{};
    // Fold all arguments into the stream.
    ((stream << args), ...);
    return stream.str();
}
} // namespace detail

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
    /// @brief Construct a new exception by concatenating the arguments.
    ///
    /// @param args A collection of arguments. The only requirements is that each element
    ///     of type ``Arg`` in the pack overloads ``std::operator<<(std::ostream&, Arg)``.
    ///
    /// Construct an ``ANNException`` with all arguments formatted and concatenated as a
    /// string message.
    ///
    template <typename... Args>
    SVS_NOINLINE explicit ANNException(Args&&... args)
        : std::runtime_error{detail::format_args(std::forward<Args>(args)...)} {}

    // Constructor taking a lineinfo as the first argument.
    // Meant to be called from the ANNEXCEPTION macro.
    template <typename... Args>
    SVS_NOINLINE explicit ANNException(detail::LineInfo&& lineinfo, Args&&... args)
        : std::runtime_error{
              detail::format_args(std::forward<Args>(args)..., ' ', lineinfo)} {}
};
} // namespace lib

using ANNException = lib::ANNException;

} // namespace svs
