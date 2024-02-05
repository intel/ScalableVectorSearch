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
