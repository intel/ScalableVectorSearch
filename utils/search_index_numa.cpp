/*
 * Copyright 2024 Intel Corporation
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

#include <chrono>
#include <functional>
#include <map>
#include <numeric>
#include <vector>

#include "svs/core/io.h"
#include "svs/lib/narrow.h"

#include "svs/orchestrators/vamana.h"
#include "svsmain.h"

using SearchIndexFunction = std::function<void(
    const std::string&,
    const size_t,
    const size_t,
    const bool,
    const size_t,
    const std::vector<std::string>&,
    const std::vector<svs::StandardAllocators>&,
    const std::vector<std::string>&,
    const std::vector<svs::StandardAllocators>&,
    const std::vector<std::string>&,
    const std::string&,
    const svs::DistanceType
)>;

template <typename E_query, typename E_db>
    requires svs::is_arithmetic_v<E_query> && svs::is_arithmetic_v<E_db>
void search_index_numa(
    const std::string& query_filename,
    const size_t search_window_size,
    const size_t n_neighbors,
    const bool track_search_stats,
    const size_t n_threads,
    const std::vector<std::string>& index_filenames,
    const std::vector<svs::StandardAllocators>& graph_memory_styles,
    const std::vector<std::string>& graph_filenames,
    const std::vector<svs::StandardAllocators>& data_memory_styles,
    const std::vector<std::string>& data_filenames,
    const std::string& result_output_prefix,
    const svs::DistanceType dist_type
) {
    using Idx = uint32_t;
    const size_t D = svs::Dynamic; // can be set to a specific number to achieve
                                   // better performance

    std::vector<svs::Vamana::Files> files{};
    for (size_t i = 0; i < 2; ++i) {
        files.emplace_back(
            graph_filenames.at(i),
            data_filenames.at(i),
            index_filenames.at(i),
            graph_memory_styles.at(i),
            data_memory_styles.at(i)
        );
    }
    auto index = svs::VamanaNuma::load<E_query, E_db, D>(files, dist_type, n_threads);
    index.set_search_window_size(search_window_size);
    auto query_data = svs::io::auto_load<E_query>(query_filename);
    std::vector<float> latencies(query_data.size());

    std::cout << "Running search" << std::endl;
    auto total_search_time_start = std::chrono::steady_clock::now();
    auto query_result = index.search(query_data, n_neighbors);

    // auto total_search_time_end = std::chrono::steady_clock::now();
    // float total_search_time =
    //     std::chrono::duration<float>(total_search_time_end - total_search_time_start)
    //         .count();

    // auto mean_latency =
    //     std::accumulate(latencies.begin(), latencies.end(), 0.0) / query_data.size();

    // std::cout << "Global search time: " << total_search_time << " seconds" << std::endl;
    // std::cout << "Mean latency: " << mean_latency << " microseconds" << std::endl;

    query_result.save_vecs(result_output_prefix + "_idx.ivecs");
}

/////
///// Main
/////

const std::string HELP =
    R"(
The required arguments are as follows:
(1) Query Element Type (string). Options: (int8, uint8, float)
(2) Data Element Type (string). Options: (int8, uint8, float, float16)
(3) Query File Path (string). Supported extensions: (.vecs, .bin)
(4) Search Window Size (integer)
(5) Number of neighbors to recall (integer)
(6) Unused
(7) Number of threads (integer)
(8) SVS metadata file path (string)
(9) Graph memory style for NUMA node 0 (string - memory style)
(10) Graph file path for NUMA node 0 (string)
(11) Graph memory style for NUMA node 1 (string - memory style)
(12) Graph file path for NUMA node 1 (string)
(13) Data memory style (both nodes) (string - memory style)
(14) Data file path (string)
(15) Result directory (string)
    - Nearest neighbors and performance stats will be created here.
(16) Distance type (string - distance type)

Valid Memory Styles: (dram, memmap)
Valid Distance Types: (L2, MIP Cosine)
)";

int svs_main(std::vector<std::string> args) {
    if (args.size() != 17) {
        std::cout << "Expected 16 arguments. Instead, got " << args.size() - 1 << ". "
                  << "The required positional arguments are given below." << std::endl
                  << std::endl
                  << HELP << std::endl;
        return 1;
    }

    const auto& query_data_type(args[1]);
    const auto& db_data_type(args[2]);
    const auto& query_filename(args[3]);
    const size_t search_window_size = std::stoull(args[4]);
    const size_t n_neighbors = std::stoull(args[5]);
    const bool track_search_stats = svs::lib::narrow<bool>(std::stoul(args[6]));
    const size_t n_threads = std::stoull(args[7]);
    const auto& index_filename(args[8]);
    const auto& graph_memory_style_str_0(args[9]);
    const auto& graph_filename_0(args[10]);
    const auto& graph_memory_style_str_1(args[11]);
    const auto& graph_filename_1(args[12]);
    const auto& data_memory_style_str(args[13]);
    const auto& data_filename(args[14]);
    const auto& result_output_prefix(args[15]);
    const auto& distance_type(args[16]);

    svs::StandardAllocators graph_memory_style_0 =
        svs::select_memory_style(graph_memory_style_str_0);
    svs::StandardAllocators graph_memory_style_1 =
        svs::select_memory_style(graph_memory_style_str_1);
    svs::StandardAllocators data_memory_style =
        svs::select_memory_style(data_memory_style_str);

    svs::DistanceType dist_type{};
    if (distance_type == std::string("L2"))
        dist_type = svs::DistanceType::L2;
    else if (distance_type == std::string("MIP"))
        dist_type = svs::DistanceType::MIP;
    else if (distance_type == std::string("Cosine"))
        dist_type = svs::DistanceType::Cosine;
    else {
        throw ANNEXCEPTION(
            "Unsupported distance type. Valid values: L2/MIP/Cosine. Received: ",
            distance_type,
            '!'
        );
    }

    using KeyType = std::pair<std::string, std::string>;
    using ValueType = SearchIndexFunction;
    const auto dispatcher = std::map<KeyType, ValueType>{
        // int8_t queries
        {{"int8", "int8"}, search_index_numa<int8_t, int8_t>},
        {{"int8", "uint8"}, search_index_numa<int8_t, uint8_t>},
        {{"int8", "float"}, search_index_numa<int8_t, float>},
        {{"int8", "float16"}, search_index_numa<int8_t, svs::Float16>},
        // uint8_t queries
        {{"uint8", "int8"}, search_index_numa<uint8_t, int8_t>},
        {{"uint8", "uint8"}, search_index_numa<uint8_t, uint8_t>},
        {{"uint8", "float"}, search_index_numa<uint8_t, float>},
        {{"uint8", "float16"}, search_index_numa<uint8_t, svs::Float16>},
        // float queries
        {{"float", "int8"}, search_index_numa<float, int8_t>},
        {{"float", "uint8"}, search_index_numa<float, uint8_t>},
        {{"float", "float"}, search_index_numa<float, float>},
        {{"float", "float16"}, search_index_numa<float, svs::Float16>}};

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

    const auto& f = it->second;

    std::vector<std::string> index_filenames{index_filename, index_filename};
    std::vector<svs::StandardAllocators> graph_memory_styles{
        graph_memory_style_0, graph_memory_style_1};
    std::vector<std::string> graph_filenames{graph_filename_0, graph_filename_1};
    std::vector<svs::StandardAllocators> data_memory_styles{
        data_memory_style, data_memory_style};
    std::vector<std::string> data_filenames{data_filename, data_filename};

    f(query_filename,
      search_window_size,
      n_neighbors,
      track_search_stats,
      n_threads,
      index_filenames,
      graph_memory_styles,
      graph_filenames,
      data_memory_styles,
      data_filenames,
      result_output_prefix,
      dist_type);

    return 0;
}

SVS_DEFINE_MAIN();
