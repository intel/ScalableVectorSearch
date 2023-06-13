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

template <typename Distance, typename T>
void convert(
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
        Distance(),
        1};

    index.set_alpha(alpha);
    index.set_construction_window_size(construction_window_size);
    index.set_max_candidates(max_candidates);
    fmt::print("Loaded index in {} seconds\n", svs::lib::time_difference(tic));
    tic = svs::lib::now();
    index.save(dst / "config", dst / "graph", dst / "data");
    fmt::print("Saved index in {} seconds\n", svs::lib::time_difference(tic));
}

using KeyType = std::tuple<std::string, std::string>;
using ValueType = std::function<void(
    const std::filesystem::path&,
    const std::filesystem::path&,
    const std::filesystem::path&,
    float,
    size_t,
    size_t,
    size_t
)>;

using Hash = svs::lib::TupleHash;

const auto distances =
    svs::meta::Types<svs::distance::DistanceL2, svs::distance::DistanceIP>();
const auto types = svs::meta::Types<float, svs::Float16, uint8_t, int8_t>();

std::string distance_names() {
    auto names = svs::meta::make_vec<std::string>(
        distances,
        []<typename T>(svs::meta::Type<T> /*type*/) { return std::string{T::name}; }
    );
    return fmt::format("{}", fmt::join(names, ", "));
}

std::string eltype_names() {
    auto datatypes = svs::meta::make_vec<svs::DataType>(types, [](auto type) {
        return svs::meta::unwrap(type);
    });
    return svs::lib::format(datatypes);
}

std::unordered_map<KeyType, ValueType, Hash> get_dispatch() {
    auto dispatch = std::unordered_map<KeyType, ValueType, Hash>{};
    svs::meta::for_each_type(
        distances,
        [&]<typename Dist>(svs::meta::Type<Dist> /*unused*/) {
            svs::meta::for_each_type(types, [&]<typename T>(svs::meta::Type<T> type) {
                auto key = KeyType(Dist::name, svs::name<svs::meta::unwrap(type)>());
                dispatch.emplace(key, convert<Dist, T>);
            });
        }
    );
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

void print_help() { fmt::print(help, eltype_names(), distance_names()); }

int svs_main(std::vector<std::string> args) {
    auto nargs = args.size();
    if (nargs != expected_nargs) {
        print_help();
        return 1;
    }

    const auto& dst = args.at(1);
    const auto& data = args.at(2);
    const auto& graph = args.at(3);
    const auto& eltype = args.at(4);
    const auto& distance = args.at(5);

    auto alpha = std::stof(args.at(6));
    auto construction_window_size = std::stoull(args.at(7));
    auto max_candidates = std::stoull(args.at(8));

    auto num_threads = std::stoull(args.at(9));

    auto dispatcher = get_dispatch();
    auto key = KeyType(distance, eltype);
    auto itr = dispatcher.find(key);
    if (itr == dispatcher.end()) {
        fmt::print("Could not find combination ({}, {})", distance, eltype);
        return 1;
    }

    const auto& f = itr->second;
    f(dst, data, graph, alpha, construction_window_size, max_candidates, num_threads);
    return 0;
}

SVS_DEFINE_MAIN();
