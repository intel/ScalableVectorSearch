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

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "svs/orchestrators/vamana.h"

#include "svsmain.h"

using BuildIndexFunction = std::function<void(
    const std::string&,
    const size_t,
    const size_t,
    const size_t,
    float alpha,
    const std::filesystem::path&,
    const std::filesystem::path&,
    const std::filesystem::path&,
    const size_t n_threads,
    const svs::DistanceType
)>;

template <typename E>
    requires svs::is_arithmetic_v<E>
void build_index(
    const std::string& vecs_filename,
    const size_t build_search_window_size,
    const size_t max_degree,
    const size_t max_candidate_pool_size,
    float alpha,
    const std::filesystem::path& config_directory,
    const std::filesystem::path& graph_directory,
    const std::filesystem::path& data_directory,
    const size_t n_threads,
    const svs::DistanceType dist_type
) {
    const size_t D = svs::Dynamic;

    auto tic = std::chrono::steady_clock::now();
    svs::index::vamana::VamanaBuildParameters parameters{
        alpha,
        max_degree,
        build_search_window_size,
        max_candidate_pool_size,
        max_degree,
        true};

    auto index = svs::Vamana::build<E>(
        parameters, svs::VectorDataLoader<E, D>(vecs_filename), dist_type, n_threads
    );
    index.save(config_directory, graph_directory, data_directory);
    auto toc = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = toc - tic;
    std::cout << "Indexing time: " << diff.count() << "s\n";
}

const std::string HELP =
    R"(
The required arguments are as follows:

(1) Data Element Type (string). Options: (int8, uint8, float, float16)
(2) Path to vector dataset (.vecs format) (string).
(3) Search window size to use for graph construction (integer). A larger value will
    yield a higher quality graph at the cost of more compute time.
(4) Maximum degree of the generated graph (integer).
(5) Max candidate pool size (integer). Auxiliary parameter which, if set higher than
    the search window size, may yield a slightly better graph.
(6) Prune threshold parameter (alpha) used for index construction (float).
    If using the L2 distance, a value greater than 1 (e.g. 1.2) should be used.
    If using MIP, use a value less than 1 (such as 0.8).
    If using Cosine, use a value less than 1 (such as 0.8).
(7) Number of threads to use for index construction (integer).
(8) Config directory for saving.
(9) Graph directory for saving.
(10) Data directory for saving.
(11) Distance type (string - distance type).

Valid Distance Types: (L2, MIP, Cosine)
)";

int svs_main(std::vector<std::string> args) {
    if (args.size() != 12) {
        std::cout << "Expected 11 arguments. Instead, got " << args.size() - 1 << ". "
                  << "The required positional arguments are given below." << std::endl
                  << std::endl
                  << HELP << std::endl;
        return 1;
    }

    size_t i = 1;
    const auto& data_type(args[i++]);
    const auto& vecs_filename(args[i++]);
    const size_t build_search_window_size = std::stoull(args[i++]);
    const size_t max_degree = std::stoull(args[i++]);
    const size_t max_candidate_pool_size = std::stoull(args[i++]);
    const float alpha = std::stof(args[i++]);
    const size_t n_threads = std::stoull(args[i++]);
    const std::string& config_directory(args[i++]);
    const std::string& graph_directory(args[i++]);
    const std::string& data_directory(args[i++]);
    const std::string& distance_type(args[i++]);

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

    /////
    ///// Dispatch
    /////

    // Select the entry point based on the passed element type.
    using KeyType = std::string;
    using ValueType = BuildIndexFunction;
    const auto dispatcher = std::unordered_map<KeyType, ValueType>{
        {"int8", build_index<int8_t>},
        {"uint8", build_index<uint8_t>},
        {"float", build_index<float>},
        {"float16", build_index<svs::Float16>}};

    auto it = dispatcher.find(data_type);
    if (it == dispatcher.end()) {
        throw ANNEXCEPTION("Unsupported data type: ", data_type, '.');
    }
    const auto& f = it->second;
    f(vecs_filename,
      build_search_window_size,
      max_degree,
      max_candidate_pool_size,
      alpha,
      config_directory,
      graph_directory,
      data_directory,
      n_threads,
      dist_type);

    return 0;
}

SVS_DEFINE_MAIN();
