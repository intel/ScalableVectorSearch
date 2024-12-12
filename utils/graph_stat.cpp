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
