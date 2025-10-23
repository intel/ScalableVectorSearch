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
#include "svs/core/io.h"
#include "svs/index/vamana/index.h"
#include "svs/lib/file.h"
#include "svs/lib/file_iterator.h"
#include "svs/lib/timing.h"

// svsmain
#include "svsmain.h"

// format
#include "fmt/core.h"
#include "fmt/ranges.h"

// stl
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

/////
///// Conversion
/////

template <typename T, typename Distance>
void convert(
    svs::lib::Type<T> SVS_UNUSED(dispatch),
    Distance distance,
    const std::filesystem::path& dst,
    const std::filesystem::path& data_path,
    const std::filesystem::path& graph_path,
    float alpha,
    size_t construction_window_size,
    size_t max_candidates,
    size_t num_threads
) {
    auto tic = svs::lib::now();

    auto graph = svs::GraphLoader(graph_path).load();
    auto data = svs::VectorDataLoader<T>(data_path).load();
    auto entry_point = svs::utils::find_medioid(data, num_threads);
    auto index = svs::index::vamana::VamanaIndex{
        std::move(graph),
        std::move(data),
        svs::lib::narrow<uint32_t>(entry_point),
        distance,
        1};

    index.set_alpha(alpha);
    index.set_construction_window_size(construction_window_size);
    index.set_max_candidates(max_candidates);
    fmt::print("Loaded index in {} seconds\n", svs::lib::time_difference(tic));
    tic = svs::lib::now();
    index.save(dst / "config", dst / "graph", dst / "data");
    fmt::print("Saved index in {} seconds\n", svs::lib::time_difference(tic));
}

using Dispatcher = svs::lib::Dispatcher<
    void,
    svs::DataType,
    svs::DistanceType,
    const std::filesystem::path&,
    const std::filesystem::path&,
    const std::filesystem::path&,
    float,
    size_t,
    size_t,
    size_t>;

Dispatcher get_dispatch() {
    auto dispatch = Dispatcher();
    const auto types = svs::lib::Types<float, svs::Float16, uint8_t, int8_t>();
    svs::lib::for_each_type(types, [&]<typename T>(svs::lib::Type<T> SVS_UNUSED(type)) {
        dispatch.register_target(&convert<T, svs::DistanceL2>);
        dispatch.register_target(&convert<T, svs::DistanceIP>);
    });
    return dispatch;
}

/////
///// Main
/////

const size_t expected_nargs = 10;

constexpr std::string_view help = R"(
Usage: assemble_vamana dst data graph eltype distance alpha construction_window_size max_candidates num_threads

Parameters:
dst - The directory where the resulting index will be created.
data - The filepath to the vector dataset.
graph - The filepath to the graph.
eltype - The element type of the dataset. Possible values: {}.
distance - The distance type to use. Possible values: {}.
alpha - Alpha to use for potential construction passes.
construction_window_size - The window size to use for construction passes.
max_candidates - The maximum number of candidates to consider for construction passes.
num_threads - Number of threads to use during the conversion process.

Note that while some of the parameters make references to construction operations, no graph
construction will actually take place.

These parameters exist to bootstrap the conversion process from older indices.
)";

void print_help() { fmt::print(help, "temp", "temp"); }

int svs_main(std::vector<std::string> args) {
    auto nargs = args.size();
    if (nargs != expected_nargs) {
        print_help();
        return 1;
    }

    const auto& dst = args.at(1);
    const auto& data = args.at(2);
    const auto& graph = args.at(3);
    const auto& eltype = svs::parse_datatype(args.at(4));
    const auto& distance = svs::parse_distance_type(args.at(5));

    auto alpha = std::stof(args.at(6));
    auto construction_window_size = std::stoull(args.at(7));
    auto max_candidates = std::stoull(args.at(8));

    auto num_threads = std::stoull(args.at(9));
    get_dispatch().invoke(
        eltype,
        distance,
        dst,
        data,
        graph,
        alpha,
        construction_window_size,
        max_candidates,
        num_threads
    );
    return 0;
}

SVS_DEFINE_MAIN();
