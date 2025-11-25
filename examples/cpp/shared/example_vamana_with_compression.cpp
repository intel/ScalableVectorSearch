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
    // STEP 1: Compress Data with LeanVec, reducing dimensionality to leanvec_dim dimensions
    // and using 4 and 8 bits for primary and secondary levels respectively.
    //! [Compress data]
    const size_t num_threads = 64;
    size_t padding = 32;
    size_t leanvec_dim = 160;
    auto threadpool = svs::threads::as_threadpool(num_threads);
    auto loaded =
        //svs::VectorDataLoader<float, svs::Dynamic, svs::lib::Allocator<float>>("/export/data/ishwarsi/laion/laion_base_1M.fvecs")
        //svs::VectorDataLoader<float, svs::Dynamic, svs::lib::Allocator<float>>("/raid0/ishwarsi/datasets/open-images/oi_base_1M.fvecs")
        svs::VectorDataLoader<float, svs::Dynamic, svs::lib::Allocator<float>>("/raid0/ishwarsi/datasets/rqa/rqa_base_1M.fvecs")
            .load();
    auto learn_queries =
        svs::load_data<float, svs::Dynamic>("/raid0/ishwarsi/datasets/rqa/rqa_learn_query_10k_ood.fvecs");
    auto leanvec_matrix = svs::leanvec::compute_leanvec_matrices_ood<svs::Dynamic, svs::Dynamic>(loaded, learn_queries, svs::lib::MaybeStatic<svs::Dynamic>(leanvec_dim));
    //std::cerr << "data matrix: \n";
    //for(size_t i = 0; i < 100; ++i) {
        //for(size_t j = 0; j < leanvec_dim; ++j) {
            //std::cerr << *(leanvec_matrix.data_matrix_.data() + i * leanvec_dim + j) << "\n";
        //}
    //}
    //std::cerr << "query matrix: \n";
    //for(size_t i = 0; i < 100; ++i) {
        //for(size_t j = 0; j < leanvec_dim; ++j) {
            //std::cerr << *(leanvec_matrix.query_matrix_.data() + i * leanvec_dim + j) << "\n";
        //}
    //}
    //std::cerr << "data ood transform:\n";
    //for(size_t i = 0; i < 100; ++i) {
        //for(size_t k = 0; k < leanvec_dim; ++k) {
            //double tmp = 0.0;
            //for(size_t j = 0; j < loaded.dimensions(); ++j) {
                //tmp += (*(loaded.data() + i * loaded.dimensions() + j)) * (*(leanvec_matrix.data_matrix_.data() + j * leanvec_dim + k ));
            //}
            //std::cerr << tmp << "\n";
        //}
    //}
    auto data = svs::leanvec::LeanDataset<
        svs::leanvec::UsingLVQ<4>,
        svs::leanvec::UsingLVQ<8>,
        svs::Dynamic,
        svs::Dynamic>::
        reduce(
            loaded,
            leanvec_matrix,
            threadpool,
            padding,
            svs::lib::MaybeStatic<svs::Dynamic>(leanvec_dim)
        );
    //! [Compress data]

    // STEP 2: Build Vamana Index
    //! [Index Build]
    auto parameters = svs::index::vamana::VamanaBuildParameters{};
    svs::Vamana index = svs::Vamana::build<float>(
        parameters, data, svs::distance::DistanceIP(), num_threads
    );
    index.save("config", "graph", "data");
    //! [Index Build]

    // STEP 3: Search the Index
    //! [Perform Queries]
    const size_t search_window_size = 450;
    const size_t n_neighbors = 10;
    index.set_search_window_size(search_window_size);

    auto queries =
        svs::load_data<float>("/raid0/ishwarsi/datasets/rqa/rqa_query_10k_ood.fvecs");
        //svs::load_data<float>("/raid0/ishwarsi/datasets/open-images/oi_queries_10k.fvecs");
    auto results = index.search(queries, n_neighbors);
    //! [Perform Queries]

    //! [Recall]
    auto groundtruth = svs::load_data<int>(
        //"/raid0/ishwarsi/datasets/open-images/oi_gtruth_1M.ivecs"
        "/raid0/ishwarsi/datasets/rqa/rqa_1M_gtruth_ood.ivecs"
    );
    double recall = svs::k_recall_at_n(groundtruth, results, n_neighbors, n_neighbors);

    fmt::print("Recall@{} = {:.4f}\n", n_neighbors, recall);
    //! [Recall]

    // STEP 4: Saving and reloading the index
    //! [Saving Loading]
    //index.save("config", "graph", "data");
    //index = svs::Vamana::assemble<float>(
        //"config",
        //svs::GraphLoader("graph"),
        //svs::lib::load_from_disk<svs::leanvec::LeanDataset<
            //svs::leanvec::UsingLVQ<4>,
            //svs::leanvec::UsingLVQ<8>,
            //svs::Dynamic,
            //svs::Dynamic>>("data", padding),
        //svs::distance::DistanceL2(),
        //num_threads
    //);
    ////! [Saving Loading]
    //index.set_search_window_size(search_window_size);
    //recall = svs::k_recall_at_n(groundtruth, results, n_neighbors, n_neighbors);

    //fmt::print("Recall@{} after saving and reloading = {:.4f}\n", n_neighbors, recall);

    return 0;
}
