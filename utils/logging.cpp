/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
 */
// A utility function used in an integration test for logging initialization.

// svs
#include "svs/core/logging.h"

// svsmain
#include "svsmain.h"

// stl
#include <string_view>

namespace {

constexpr std::string_view HELP = R"(
usage: (1) logging level message
       (2) logging --help

1. Emit a single logging of "message" at the requested level.
2. Print this help message.

Recognized values for `level` are:
"trace", "debug", "info", "warn", "error", "critical", "off"

Use this function in coordinate with the SVS logging environment variables to test logging
initialization.
)";

struct IsHelp {
    bool operator()(std::string_view s) const { return s == "--help" || s == "-h"; }
};

inline constexpr IsHelp is_help{};

} // namespace

int svs_main(const std::vector<std::string>& args) {
    const size_t nargs = args.size();
    if (nargs <= 1) {
        fmt::print("logging: invalid number of arguments\n");
        fmt::print("{}\n", HELP);
        return 0;
    }

    const auto& first = args.at(1);
    if (is_help(first)) {
        fmt::print("{}\n", HELP);
        return 0;
    }

    if (nargs != 3) {
        fmt::print("logging: invalid number of arguments\n");
        fmt::print("{}\n", HELP);
        return 0;
    }

    auto level = svs::logging::detail::level_from_string(first);
    const auto& message = args.at(2);

    svs::logging::log(level, "{}", message);
    return 0;
}

SVS_DEFINE_MAIN();
