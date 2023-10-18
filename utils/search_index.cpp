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

#include "svs/lib/float16.h"
#include "svs/lib/narrow.h"
#include "svs/orchestrators/vamana.h"

#include "svsmain.h"

// format
#include "fmt/core.h"

// stl
#include <chrono>
#include <functional>
#include <map>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

// Hook to provide static dimension
const size_t VectorDimension = svs::Dynamic;

using SearchIndexFunction = std::function<void(
    const std::string&,
    const size_t,
    const size_t,
    const size_t,
    const std::filesystem::path&,
    const std::filesystem::path&,
    const std::filesystem::path&,
    const std::string&,
    const svs::DistanceType
)>;

template <typename E_query, typename E_db>
void search_index(
    const std::string& query_filename,
    const size_t search_window_size,
    const size_t n_neighbors,
    const size_t n_threads,
    const std::filesystem::path& config_path,
    const std::filesystem::path& graph_path,
    const std::filesystem::path& data_path,
    const std::string& result_output_prefix,
    const svs::DistanceType dist_type
) {
    auto index = svs::Vamana::assemble<E_query>(
        config_path,
        svs::GraphLoader(graph_path),
        svs::VectorDataLoader<E_db, VectorDimension>(data_path),
        dist_type,
        n_threads
    );

    index.set_search_window_size(search_window_size);
    const auto query_data = svs::load_data<E_query>(query_filename);
    std::vector<float> latencies(query_data.size());

    fmt::print("Running Search.\n");
    auto tic = svs::lib::now();
    auto query_result = index.search(query_data, n_neighbors);
    auto search_time = svs::lib::time_difference(tic);

    fmt::print("Global search time: {} seconds\n", search_time);
    query_result.save_vecs(fmt::format("{}_idx.ivecs", result_output_prefix));
}

constexpr std::string_view HELP =
    R"(
The required arguments are as follows:
(1) Query Element Type (string). Options: (int8, uint8, float)
(2) Data Element Type (string). Options: (int8, uint8, float, float16)
(3) Query File Path (string). Supported extentions: (.vecs, .bin)
(4) Search Window Size (integer)
(5) Number of neighbors to recall (integer)
(6) Number of threads (integer)
(7) Config directory (string)
(8) Graph directory (string)
(9) Data directory (string)
(10) Result directory (string)
    - Nearest neighbors and performance stats will be created here.
(11) Distance type (string - distance type)

Valid Distance Types: (L2, MIP, Cosine)
)";

int svs_main(std::vector<std::string>&& args) {
    if (args.size() != 12) {
        std::cout << "Expected 11 arguments. Instead, got " << args.size() - 1 << ". "
                  << "The required positional arguments are given below." << std::endl
                  << std::endl
                  << HELP << std::endl;
        return 1;
    }

    size_t i = 1;
    const auto& query_data_type = args[i++];
    const auto& db_data_type = args[i++];
    const auto& query_filename = args[i++];
    const size_t search_window_size = std::stoull(args[i++]);
    const size_t n_neighbors = std::stoull(args[i++]);
    const size_t n_threads = std::stoull(args[i++]);
    const auto& config_path = args[i++];
    const auto& graph_path = args[i++];
    const auto& data_path = args[i++];
    const auto& result_output_prefix = args[i++];
    const auto& distance_type = args[i++];

    // Select distance type.
    svs::DistanceType dist_type{};
    if (distance_type == std::string("L2"))
        dist_type = svs::DistanceType::L2;
    else if (distance_type == std::string("MIP"))
        dist_type = svs::DistanceType::MIP;
    else if (distance_type == std::string("Cosine"))
        dist_type = svs::DistanceType::Cosine;
    else {
        throw ANNEXCEPTION(
            "Unsupported distance type. Valid values: L2/MIP/Cosine. Recieved: ",
            distance_type,
            '!'
        );
    }

    using KeyType = std::pair<std::string, std::string>;
    using ValueType = SearchIndexFunction;
    const auto dispatcher = std::map<KeyType, ValueType>{
        // int8_t queries
        {{"int8", "int8"}, search_index<int8_t, int8_t>},
        {{"int8", "uint8"}, search_index<int8_t, uint8_t>},
        {{"int8", "float"}, search_index<int8_t, float>},
        {{"int8", "float16"}, search_index<int8_t, svs::Float16>},
        // uint8_t queries
        {{"uint8", "int8"}, search_index<uint8_t, int8_t>},
        {{"uint8", "uint8"}, search_index<uint8_t, uint8_t>},
        {{"uint8", "float"}, search_index<uint8_t, float>},
        {{"uint8", "float16"}, search_index<uint8_t, svs::Float16>},
        // float queries
        {{"float", "int8"}, search_index<float, int8_t>},
        {{"float", "uint8"}, search_index<float, uint8_t>},
        {{"float", "float"}, search_index<float, float>},
        {{"float", "float16"}, search_index<float, svs::Float16>}};

    auto it = dispatcher.find({query_data_type, db_data_type});
    if (it == dispatcher.end()) {
        throw ANNEXCEPTION(
            "Unsupported Query and Data type pair: (",
            query_data_type,
            ", ",
            db_data_type,
            ")!"
        );
    }

    // Unpack and call
    const auto& f = it->second;
    f(query_filename,
      search_window_size,
      n_neighbors,
      n_threads,
      config_path,
      graph_path,
      data_path,
      result_output_prefix,
      dist_type);

    return 0;
}

// Include the helper main function.
SVS_DEFINE_MAIN();
