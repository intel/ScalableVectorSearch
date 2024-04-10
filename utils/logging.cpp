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
