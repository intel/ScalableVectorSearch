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

// svs
#include "svs/index/vamana/index.h"

// svsmain
#include "svsmain.h"

// format
#include "fmt/core.h"

// stl
#include <string>
#include <string_view>

constexpr std::string_view HELP = R"(
)";

void show_help() { fmt::print("{}\n", HELP); }

///// svsmain
int svs_main(std::vector<std::string> args) {
    const auto& path = args.at(1);

    auto graph = svs::GraphLoader(path).load();
    double s = 0;
    size_t max = 0;
    size_t min = std::numeric_limits<size_t>::max();

    for (uint32_t i = 0, imax = graph.n_nodes(); i < imax; ++i) {
        auto n = graph.get_node_degree(i);
        s += n;
        min = std::min(min, n);
        max = std::max(max, n);
    }

    fmt::print("Max degree: {}\n", max);
    fmt::print("Min degree: {}\n", min);
    fmt::print("Mean degree: {}\n", s / static_cast<double>(graph.n_nodes()));

    return 0;
}

SVS_DEFINE_MAIN();
