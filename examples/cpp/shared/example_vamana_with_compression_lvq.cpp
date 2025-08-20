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


int main() {
    // STEP 1: Compress Data with LVQ
    //! [Compress data]
    size_t padding = 32;
    const size_t num_threads = 4;
    auto threadpool = svs::threads::as_threadpool(num_threads);
    auto loaded = svs::VectorDataLoader<float>(std::filesystem::path(SVS_DATA_DIR) / "data_f32.svs").load();
    auto data = svs::quantization::lvq::LVQDataset<4, 8>::compress(loaded, threadpool, padding);
    //! [Compress data]

    // STEP 2: Build Vamana Index    
    //! [Index Build]
    auto parameters = svs::index::vamana::VamanaBuildParameters{};
    svs::Vamana index = svs::Vamana::build<float>(parameters, data, svs::distance::DistanceL2(), num_threads);
    //! [Index Build]

    // STEP 3: Search the Index
    //! [Perform Queries]
    const size_t search_window_size = 40;
    const size_t n_neighbors = 10;
    index.set_search_window_size(search_window_size);

    auto queries = svs::load_data<float>(std::filesystem::path(SVS_DATA_DIR) / "queries_f32.fvecs");
    auto results = index.search(queries, n_neighbors);
    //! [Perform Queries]

    //! [Recall]
    auto groundtruth = svs::load_data<int>(std::filesystem::path(SVS_DATA_DIR) / "groundtruth_euclidean.ivecs");
    double recall = svs::k_recall_at_n(groundtruth, results, n_neighbors, n_neighbors);

    fmt::print("Recall@{} = {:.4f}\n", n_neighbors, recall);
    //! [Recall]

    // STEP 4: Saving and reloading the index
    //! [Saving Loading]
    index.save("config", "graph", "data");
    index = svs::Vamana::assemble<float>(
        "config", svs::GraphLoader("graph"), svs::lib::load_from_disk<svs::quantization::lvq::LVQDataset<4, 8>>("data", padding), svs::distance::DistanceL2(), num_threads        
    );
    //! [Saving Loading]
    index.set_search_window_size(search_window_size);
    recall = svs::k_recall_at_n(groundtruth, results, n_neighbors, n_neighbors);

    fmt::print("Recall@{} after saving and reloading = {:.4f}\n", n_neighbors, recall);


    return 0;
}
