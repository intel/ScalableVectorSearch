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
#include "svs/extensions/ivf/lvq.h"
#include "svs/orchestrators/ivf.h"
#include "svs/quantization/lvq/lvq.h"

int main() {
    // STEP 1: Compress Data with LVQ (4-bit primary, 8-bit residual)
    //! [Compress data]
    size_t padding = 32;
    const size_t num_threads = 4;
    auto threadpool = svs::threads::as_threadpool(num_threads);
    auto loaded =
        svs::VectorDataLoader<float>(std::filesystem::path(SVS_DATA_DIR) / "data_f32.svs")
            .load();
    auto data =
        svs::quantization::lvq::LVQDataset<4, 8>::compress(loaded, threadpool, padding);
    //! [Compress data]

    // STEP 2: Build IVF Index
    //! [Index Build]
    const size_t num_clusters = 10;
    auto build_params = svs::index::ivf::IVFBuildParameters(num_clusters, 10, false);

    // Build clustering on uncompressed data
    auto clustering = svs::IVF::build_clustering<float>(
        build_params, loaded, svs::distance::DistanceL2(), num_threads
    );

    // Assemble IVF index with LVQ compressed data
    auto index = svs::IVF::assemble_from_clustering<float>(
        std::move(clustering),
        data,
        svs::distance::DistanceL2(),
        num_threads
    );
    //! [Index Build]

    // STEP 3: Search the Index
    //! [Perform Queries]
    const size_t n_neighbors = 10;
    auto search_params = svs::index::ivf::IVFSearchParameters(
        num_clusters, // n_probes
        n_neighbors   // k_reorder
    );
    index.set_search_parameters(search_params);

    auto queries =
        svs::load_data<float>(std::filesystem::path(SVS_DATA_DIR) / "queries_f32.fvecs");
    auto results = index.search(queries, n_neighbors);
    //! [Perform Queries]

    //! [Recall]
    auto groundtruth = svs::load_data<int>(
        std::filesystem::path(SVS_DATA_DIR) / "groundtruth_euclidean.ivecs"
    );
    double recall = svs::k_recall_at_n(groundtruth, results, n_neighbors, n_neighbors);

    fmt::print("IVF with LVQ<4,8> Recall@{} = {:.4f}\n", n_neighbors, recall);
    //! [Recall]

    // STEP 4: Saving and reloading the index
    //! [Saving Loading]
    index.save("ivf_config", "ivf_data");

    // Reload the index - specify centroid type (float) and data type (LVQDataset)
    using LVQData = svs::quantization::lvq::LVQDataset<4, 8>;
    index = svs::IVF::assemble<float, float, LVQData>(
        "ivf_config",
        "ivf_data",
        svs::distance::DistanceL2(),
        num_threads
    );
    //! [Saving Loading]
    index.set_search_parameters(search_params);
    results = index.search(queries, n_neighbors);
    recall = svs::k_recall_at_n(groundtruth, results, n_neighbors, n_neighbors);

    fmt::print(
        "IVF with LVQ<4,8> Recall@{} after saving and reloading = {:.4f}\n",
        n_neighbors,
        recall
    );

    return 0;
}
