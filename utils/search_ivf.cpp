/*
 * Copyright 2025 Intel Corporation
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

#include "svs/core/recall.h"
#include "svs/orchestrators/ivf.h"
#include "svsmain.h"

// stl
#include <functional>
#include <map>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace svs::index::ivf;

template <typename Distance>
using SearchIndexFunction = std::function<void(
    const std::string&,
    const std::string&,
    const size_t,
    const size_t,
    const size_t,
    const size_t,
    const size_t,
    const std::filesystem::path&,
    const std::filesystem::path&,
    const Distance,
    const int
)>;

auto std_dev(std::vector<double> v) {
    double sum = std::accumulate(std::begin(v), std::end(v), 0.0);
    double m = sum / v.size();

    double accum = 0.0;
    std::for_each(std::begin(v), std::end(v), [&](const double d) {
        accum += (d - m) * (d - m);
    });

    double stdev = sqrt(accum / (v.size() - 1));
    return stdev;
}

template <typename T>
auto batch_queries(
    svs::data::SimpleData<T> query_data, size_t num_batches, size_t batchsize
) {
    std::vector<svs::data::SimpleData<T>> query_batch;
    for (size_t batch = 0; batch < num_batches; ++batch) {
        auto this_batch = svs::threads::UnitRange{
            batch * batchsize, std::min((batch + 1) * batchsize, query_data.size())};
        query_batch.push_back(
            svs::data::SimpleData<T>(this_batch.size(), query_data.dimensions())
        );
        for (size_t i = 0; i < this_batch.size(); i++) {
            query_batch[batch].set_datum(i, query_data.get_datum(this_batch[i]));
        }
    }
    return query_batch;
}

template <typename E_query, typename E_db, typename Distance>
void search_index(
    const std::string& query_filename,
    const std::string& gt_filename,
    const size_t n_probes,
    const size_t n_neighbors,
    const size_t batch_size,
    const size_t n_threads,
    const size_t n_inner_threads,
    const std::filesystem::path& clustering_path,
    const std::filesystem::path& data_path,
    const Distance dist_type,
    const int n_reps
) {
    const size_t dim = svs::Dynamic;

    auto data = svs::VectorDataLoader<E_db, dim, svs::lib::Allocator<E_db>>(data_path);

    auto ivf_index = svs::IVF::assemble_from_file<E_query, svs::BFloat16>(
        clustering_path, std::move(data), dist_type, n_threads, n_inner_threads
    );

    const auto query_data = svs::load_data<E_query>(query_filename);
    const auto groundtruth = svs::load_data<uint32_t>(gt_filename);

    ivf_index.set_search_parameters(IVFSearchParameters(n_probes, 1.0));

    size_t batchsize = query_data.size();
    if (batch_size != 0) {
        batchsize = batch_size;
    }

    auto query_results =
        svs::Matrix<uint32_t>{svs::make_dims(query_data.size(), n_neighbors)};
    auto num_batches = svs::lib::div_round_up(query_data.size(), batchsize);
    auto query_batch = batch_queries(query_data, num_batches, batchsize);

    auto tic = svs::lib::now();
    for (size_t batch = 0; batch < num_batches; ++batch) {
        auto query_result = ivf_index.search(query_batch[batch], n_neighbors);
        for (size_t i = 0; i < query_result.n_queries(); i++) {
            for (size_t j = 0; j < n_neighbors; j++) {
                query_results.at(batch * batchsize + i, j) = query_result.index(i, j);
            }
        }
    }
    auto search_time = svs::lib::time_difference(tic);

    std::vector<double> qps;
    for (int i = 0; i < n_reps; i++) {
        tic = svs::lib::now();
        for (size_t batch = 0; batch < num_batches; ++batch) {
            ivf_index.search(query_batch[batch], n_neighbors);
        }
        search_time = svs::lib::time_difference(tic);
        qps.push_back(query_data.size() / search_time);
    }

    fmt::print("Raw QPS: {:7.3f} \n", fmt::join(qps, ", "));
    fmt::print(
        "Batch Size: {}, Recall: {:.4f}, QPS (Avg: {:7.3f}, Max: {:7.3f}, StdDev: {:7.3f} "
        ") "
        "\n",
        batchsize,
        svs::k_recall_at_n(groundtruth, query_results, 10, 10),
        std::reduce(qps.begin(), qps.end()) / qps.size(),
        *std::max_element(qps.begin(), qps.end()),
        std_dev(qps)
    );
}

constexpr std::string_view HELP =
    R"(
The required arguments are as follows:
(1) Query Element Type (string). Options: (int8, uint8, float)
(2) Data Element Type (string). Options: (int8, uint8, float, float16, bfloat16)
(3) Query File Path (string). Supported extentions: (.vecs, .bin)
(4) Groundtruth File Path (string). Supported extentions: (.vecs, .bin)
(5) n_probes (number of clusters to search) (integer)
(6) Number of neighbors to recall (integer)
(7) Batch size (integer)
(8) Number of threads (integer)
(9) Number of intra-query threads (integer)
(10) Clustering directory (string)
(11) Data directory (string)
(12) Number of repetitions to be run for benchmarking purposes (integer)
(13) Distance type (string - distance type)
)";

int svs_main(std::vector<std::string>&& args) {
    if (args.size() != 14) {
        std::cout << "Expected 13 arguments. Instead, got " << args.size() - 1 << ". "
                  << "The required positional arguments are given below." << std::endl
                  << std::endl
                  << HELP << std::endl;
        return 1;
    }

    size_t i = 1;
    const auto& query_data_type = args[i++];
    const auto& db_data_type = args[i++];
    const auto& query_filename = args[i++];
    const auto& gt_filename = args[i++];
    const size_t n_probes = std::stoull(args[i++]);
    const size_t n_neighbors = std::stoull(args[i++]);
    const size_t batch_size = std::stoull(args[i++]);
    const size_t n_threads = std::stoull(args[i++]);
    const size_t n_inner_threads = std::stoull(args[i++]);
    const auto& clustering_path = args[i++];
    const auto& data_path = args[i++];
    const int nreps = std::stoull(args[i++]);
    const auto& distance_type = args[i++];

    auto dist_disp = [&]<typename dist_type>(dist_type dist) {
        using KeyType = std::pair<std::string, std::string>;
        using ValueType = SearchIndexFunction<dist_type>;
        const auto dispatcher = std::map<KeyType, ValueType>{
            {{"float", "float16"}, search_index<float, svs::Float16, dist_type>},
            {{"float", "bfloat16"}, search_index<float, svs::BFloat16, dist_type>},
            {{"float", "float"}, search_index<float, float, dist_type>}};

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
          gt_filename,
          n_probes,
          n_neighbors,
          batch_size,
          n_threads,
          n_inner_threads,
          clustering_path,
          data_path,
          dist,
          nreps);
    };

    if (distance_type == std::string("L2")) {
        dist_disp(svs::distance::DistanceL2{});
    } else if (distance_type == std::string("MIP")) {
        dist_disp(svs::distance::DistanceIP{});
    } else {
        throw ANNEXCEPTION(
            "Unsupported distance type. Valid values: L2/MIP. Received: ",
            distance_type,
            '!'
        );
    }

    return 0;
}

// Include the helper main function.
SVS_DEFINE_MAIN();
