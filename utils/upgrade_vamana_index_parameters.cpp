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
Usage:convert_legacy_vamana_index_parameters generate_vamana_config src dst

Generate a config TOML file for the new Vamana index format.
Upgrades from version v0.0.1 or v0.0.2 to v0.0.3.

Arguments:
    src - The path to the previous config file.
    dst - The destionation directory for the upgraded file.
)";

void show_help() { fmt::print("{}\n", HELP); }

///// svsmain
int svs_main(std::vector<std::string> args) {
    size_t expected = 3;
    size_t nargs = args.size();
    if (nargs != expected) {
        fmt::print("Expected {} args, instead got {}\n", expected, nargs);
        show_help();
        return 1;
    }

    const auto& src = args.at(1);
    const auto& dst = args.at(2);

    svs::lib::save_to_disk(
        svs::lib::load_from_disk<svs::index::vamana::VamanaIndexParameters>(src), dst
    );
    return 0;
}

SVS_DEFINE_MAIN();
