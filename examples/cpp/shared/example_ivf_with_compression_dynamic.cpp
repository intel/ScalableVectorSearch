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
#include "svs/extensions/ivf/leanvec.h"
#include "svs/extensions/ivf/lvq.h"
#include "svs/index/ivf/clustering.h"
#include "svs/leanvec/leanvec.h"
#include "svs/orchestrators/dynamic_ivf.h"
#include "svs/quantization/lvq/lvq.h"

// Alias for blocked LeanVec dataset that supports resize/compact for dynamic IVF
using BlockedLean = svs::leanvec::LeanDataset<
    svs::leanvec::UsingLVQ<4>,
    svs::leanvec::UsingLVQ<8>,
    svs::Dynamic,
    svs::Dynamic,
    svs::data::Blocked<svs::lib::Allocator<std::byte>>>;

int main() {
    // STEP 1: Compress Data with LeanVec, reducing dimensionality to leanvec_dim dimensions
    // and using 4 and 8 bits for primary and secondary levels respectively.
    //! [Compress data]
    const size_t num_threads = 4;
    size_t padding = 32;
    size_t leanvec_dim = 64;
    size_t intra_query_threads = 2;
    auto threadpool = svs::threads::as_threadpool(num_threads);
    auto loaded =
        svs::VectorDataLoader<float>(std::filesystem::path(SVS_DATA_DIR) / "data_f32.svs")
            .load();
    auto data = BlockedLean::reduce(
        loaded,
        std::nullopt,
        threadpool,
        padding,
        svs::lib::MaybeStatic<svs::Dynamic>(leanvec_dim)
    );
    //! [Compress data]

    // STEP 2: Build Dynamic IVF Index
    //! [Index Build]
    const size_t num_clusters = 10;
    auto build_params = svs::index::ivf::IVFBuildParameters(num_clusters, 10, false);

    // Build clustering on uncompressed data
    auto clustering = svs::index::ivf::build_clustering<float>(
        build_params, loaded, svs::distance::DistanceL2(), num_threads, false
    );

    // Generate external IDs for the data
    std::vector<size_t> ids(loaded.size());
    std::iota(ids.begin(), ids.end(), 0);

    // Assemble Dynamic IVF index with LeanVec compressed data
    auto index = svs::DynamicIVF::assemble_from_clustering<float>(
        std::move(clustering),
        data,
        ids,
        svs::distance::DistanceL2(),
        num_threads,
        intra_query_threads
    );
    //! [Index Build]

    // STEP 3: Add and delete vectors as needed.
    //! [Delete vectors]
    size_t num_to_delete = 100;
    std::vector<size_t> ids_delete(num_to_delete);
    for (size_t i = 0; i < ids_delete.size(); ++i) {
        ids_delete[i] = i;
    }

    fmt::print("Deleting {} vectors.\n", ids_delete.size());

    index.delete_points(ids_delete);
    //! [Delete vectors]

    //! [Add vectors]
    // Add the deleted vectors back in.
    auto points =
        svs::data::SimpleData<float, svs::Dynamic>(ids_delete.size(), loaded.dimensions());

    size_t i = 0;
    for (const auto& j : ids_delete) {
        points.set_datum(i, loaded.get_datum(j));
        ++i;
    }
    auto points_const_view = points.cview();

    fmt::print("Adding {} vectors.\n", ids_delete.size());

    index.add_points(points_const_view, ids_delete, false);
    //! [Add vectors]

    //! [Compact]
    // Compact the index to reclaim space from deleted entries
    fmt::print("Compacting index.\n");
    index.compact();
    //! [Compact]

    // STEP 4: Search the Index
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

    fmt::print("Dynamic IVF with LeanVec Recall@{} = {:.4f}\n", n_neighbors, recall);
    fmt::print("Note that recall may be low because this example is using a dummy random "
               "dataset.\n");
    //! [Recall]

    // STEP 5: Saving and reloading the index
    //! [Saving Loading]
    index.save("ivf_dynamic_config", "ivf_dynamic_data");

    // Reload the index - specify centroid type (float) and data type (BlockedLean)
    index = svs::DynamicIVF::assemble<float, float, BlockedLean>(
        "ivf_dynamic_config",
        "ivf_dynamic_data",
        svs::distance::DistanceL2(),
        num_threads,
        intra_query_threads
    );
    //! [Saving Loading]
    index.set_search_parameters(search_params);
    results = index.search(queries, n_neighbors);
    recall = svs::k_recall_at_n(groundtruth, results, n_neighbors, n_neighbors);

    fmt::print(
        "Dynamic IVF with LeanVec Recall@{} after saving and reloading = {:.4f}\n",
        n_neighbors,
        recall
    );

    return 0;
}
