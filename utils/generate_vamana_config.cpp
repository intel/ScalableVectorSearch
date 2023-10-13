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
Usage: generate_vamana_config path args...

Generate a config TOML file for the new Vamana index format.
Arguments:
    path - The destination directory for the config file.
    old_metadata_path - The full path to the original metadata.
    graph_max_degree - The maximum degree of the generated graph.
    alpha - The alpha value used for construction.
    max_candidates - The maximum number of candidates for construction.
    construction_window_size - The search window size to use for construction.
    prune_to - The number of candidates to prune to.
    search_window_size - The search window size to use for querying.
    visited_set - Whether the visited set is enabled or now.
)";

void show_help() { fmt::print("{}\n", HELP); }

///// svsmain
int svs_main(std::vector<std::string> args) {
    size_t expected = 10;
    size_t nargs = args.size();
    if (nargs != expected) {
        fmt::print("Expected {} args, instead got {}\n", expected, nargs);
        show_help();
        return 1;
    }

    const auto& path = args.at(1);
    const auto& old_metadata = args.at(2);

    // Read the original entry point.
    auto stream = svs::lib::open_read(old_metadata);
    auto entry_point = svs::lib::read_binary<uint32_t>(stream);

    fmt::print("Using {} as the entry point index.\n", entry_point);

    auto i = 3;
    auto graph_max_degree = std::stoull(args.at(i++));
    auto alpha = std::stof(args.at(i++));
    auto max_candidates = std::stoull(args.at(i++));
    auto construction_window_size = std::stoull(args.at(i++));
    auto prune_to = std::stoull(args.at(i++));
    auto search_window_size = std::stoull(args.at(i++));
    bool visited_set = std::stoull(args.at(i++));

    auto parameters = svs::index::vamana::VamanaConfigParameters{
        graph_max_degree,
        entry_point,
        alpha,
        max_candidates,
        construction_window_size,
        prune_to,
        true,
        search_window_size,
        visited_set};

    svs::lib::save_to_disk(parameters, path);
    return 0;
}

SVS_DEFINE_MAIN();
