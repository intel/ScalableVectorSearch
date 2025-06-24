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
#include "svs/orchestrators/dynamic_vamana.h"
#include "svs/extensions/vamana/scalar.h"
#include "svs/quantization/scalar/scalar.h"

int main() {
    size_t num_threads = 64;
    size_t search_window_size = 20;
    size_t n_neighbors = 10;
    std::string dfname = "/export/data/mcapot/laion-img-emb-512-1M-cosine.hdf5_train.fvecs";
    std::string qfname = "/export/data/mcapot/laion-img-emb-512-1M-cosine.hdf5_test.fvecs";
    std::string gtfname = "/export/data/mcapot/laion-img-emb-512-1M-cosine.hdf5_neighbors.ivecs";
    using Distance = svs::distance::DistanceIP;
    Distance distance{};

    auto compressor = svs::lib::Lazy([=](svs::threads::ThreadPool auto& threadpool) {
        auto data = svs::VectorDataLoader<float, 512>(dfname).load();
        return svs::quantization::scalar::SQDataset<std::int8_t, 512>::compress(data, threadpool);
    });

    // build ==========================================================
    auto parameters = svs::index::vamana::VamanaBuildParameters{
        0.95,  // alpha
        64,   // graph max degree
        128,  // search window size
        750,  // max candidate pool size
        60,   // prune to degree
        true, // full search history
    };

    auto tic = svs::lib::now();
    svs::Vamana index = svs::Vamana::build<float>(parameters, compressor, distance, num_threads);
    auto build_time = svs::lib::time_difference(tic);
    fmt::print(
        "Vamana index build time: {} Distance: {}\n",
        build_time,
        svs::name(svs::distance_type_v<Distance>)
    );

    // search =========================================================
    const auto query_data = svs::load_data<float>(qfname);
    const auto groundtruth = svs::load_data<int>(gtfname);

    // set search window size
    index.set_search_window_size(search_window_size);
    tic = svs::lib::now();
    auto query_result = index.search(query_data, n_neighbors);
    auto search_time = svs::lib::time_difference(tic);

    std::vector<double> qps;
    for (int i = 0; i < 5; i++) {
        tic = svs::lib::now();
        query_result = index.search(query_data, n_neighbors);
        search_time = svs::lib::time_difference(tic);
        qps.push_back(query_data.size() / search_time);
    }

    // recall =========================================================
    auto recall = svs::k_recall_at_n(groundtruth, query_result, n_neighbors, n_neighbors);
    fmt::print(
        "Vamana Distance: {}, sws: {}, Recall: {}, Max QPS: {:7.3f} \n",
        svs::name(svs::distance_type_v<Distance>),
        search_window_size,
        recall,
        *std::max_element(qps.begin(), qps.end())
    );
}
