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

// SVS
#include "svs/core/recall.h"
#include "svs/extensions/flat/leanvec.h"
#include "svs/extensions/flat/lvq.h"
#include "svs/extensions/vamana/leanvec.h"
#include "svs/extensions/vamana/lvq.h"
#include "svs/orchestrators/dynamic_vamana.h"
#include "svs/orchestrators/exhaustive.h"
#include "svs/orchestrators/vamana.h"

#include "utils.h"

// Alternative main definition
#include "svsmain.h"

// SVS setup and parameters
size_t num_threads = 4;
size_t search_window_size = 20;
size_t n_neighbors = 1;
std::string dfname = "data.vecs";
std::string dfname_f16 = "data_f16.vecs";
std::string qfname = "query.vecs";
std::string gtfname = "gt.vecs";

const std::filesystem::path& config_path = "./config";
const std::filesystem::path& graph_path = "./graph";
const std::filesystem::path& config_path_dynamic = "./config_dynamic";
const std::filesystem::path& graph_path_dynamic = "./graph_dynamic";

void svs_setup() {
    // convert to fp16
    auto reader = svs::io::vecs::VecsReader<float>{dfname};
    auto writer = svs::io::vecs::VecsWriter<svs::Float16>{dfname_f16, reader.ndims()};
    {
        for (auto i : reader) {
            writer << i;
        }
    }
    writer.flush();
}

auto create_lvq_data() {
    auto compressor = svs::lib::Lazy([=](svs::threads::ThreadPool auto& threadpool) {
        auto data = svs::VectorDataLoader<svs::Float16>(dfname_f16).load();
        return svs::quantization::lvq::LVQDataset<4, 4>::compress(data, threadpool, 32);
    });

    auto threadpool = svs::threads::as_threadpool(num_threads);
    auto data = svs::detail::dispatch_load(compressor, threadpool);
    return data;
}

template <typename Data, typename Distance>
void vamana_build(Data& data, Distance distance) {
    auto parameters = svs::index::vamana::VamanaBuildParameters{
        1.2,  // alpha
        64,   // graph max degree
        128,  // search window size
        750,  // max candidate pool size
        60,   // prune to degree
        true, // full search history
    };

    auto tic = svs::lib::now();
    svs::Vamana index = svs::Vamana::build<float>(parameters, data, distance, num_threads);
    auto build_time = svs::lib::time_difference(tic);
    fmt::print(
        "Vamana index build time: {} Distance: {}\n",
        build_time,
        svs::name(svs::distance_type_v<Distance>)
    );
    index.save("config", "graph", "data");
}

template <typename Data, typename Distance>
void vamana_search(Data& data, Distance distance) {
    auto index = svs::Vamana::assemble<float>(
        config_path, svs::GraphLoader(graph_path), data, distance, num_threads
    );

    index.set_search_window_size(search_window_size);
    const auto query_data = svs::load_data<float>(qfname);
    const auto groundtruth = svs::load_data<int>(gtfname);

    auto tic = svs::lib::now();
    auto query_result = index.search(query_data, n_neighbors);
    auto search_time = svs::lib::time_difference(tic);

    std::vector<double> qps;
    for (int i = 0; i < 5; i++) {
        tic = svs::lib::now();
        query_result = index.search(query_data, n_neighbors);
        search_time = svs::lib::time_difference(tic);
        qps.push_back(query_data.size() / search_time);
    }

    auto recall = svs::k_recall_at_n(groundtruth, query_result, n_neighbors, n_neighbors);
    fmt::print(
        "Vamana Distance: {}, sws: {}, Recall: {}, Max QPS: {:7.3f} \n",
        svs::name(svs::distance_type_v<Distance>),
        search_window_size,
        recall,
        *std::max_element(qps.begin(), qps.end())
    );
}

// Alternative main definition
int svs_main(std::vector<std::string> SVS_UNUSED(args)) {
    const size_t dim = 512;
    size_t dataset_size = 100;
    size_t query_size = 10;

    generate_random_data(dim, dataset_size, query_size);
    svs_setup();

    auto data = create_lvq_data();
    vamana_build(data, svs::distance::DistanceL2());
    vamana_search(data, svs::distance::DistanceL2());

    return 0;
}

// Special main providing some helpful utilities.
SVS_DEFINE_MAIN();
