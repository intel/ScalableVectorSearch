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
