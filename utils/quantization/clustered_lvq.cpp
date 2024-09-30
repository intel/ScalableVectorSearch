/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
 */
#include "svs/core/data/io.h"
#include "svs/core/kmeans.h"
#include "svs/core/medioid.h"
#include "svs/core/recall.h"
#include "svs/index/vamana/index.h"
#include "svs/quantization/lvq/lvq.h"

#include "svsmain.h"

// Number of bits to use for the primary compressed dataset.
const size_t Primary = 8;
const size_t Residual = 0;

// Dimensionality of the dataset
const size_t Dims = 96; // svs::Dynamic

// Element type of the uncompressed dataset.
using Eltype = svs::Float16;

// Distance function to use.
using Distance = svs::distance::DistanceL2;
using DistanceBiased = svs::quantization::lvq::EuclideanBiased;

// Main function.
int svs_main(std::vector<std::string> args) {
    size_t i = 1;
    const auto& data_path = args.at(i++);
    const auto& graph_path = args.at(i++);
    const auto& query_path = args.at(i++);
    const auto& groundtruth_path = args.at(i++);
    auto num_means = std::stoull(args.at(i++));
    auto num_threads = std::stoull(args.at(i++));

    auto timer = svs::lib::Timer();
    auto data = svs::VectorDataLoader<Eltype, Dims>(data_path).load();
    auto dims = svs::lib::MaybeStatic<Dims>(data.dimensions());

    auto medoid = svs::utils::find_medioid(data, num_threads);
    fmt::print("Medoid: {}\n", medoid);

    auto params = svs::KMeansParameters{num_means, 100'000, 10};
    auto centroids = svs::train(params, data, num_threads);

    auto compressed =
        svs::quantization::lvq::ScaledBiasedDataset<Primary, Dims>(data.size(), dims);
    compressed.set_centroids(centroids.cview());

    auto compress_handle = timer.push_back("compress");
    auto threadpool = svs::threads::NativeThreadPool{num_threads};
    svs::threads::run(
        threadpool,
        svs::threads::DynamicPartition{data.eachindex(), 512},
        [&](auto indices, auto /*tid*/) {
            auto buffer = std::vector<float>(data.dimensions());
            auto codec = svs::quantization::lvq::MinRange<Primary, Dims>(dims);
            for (auto i : indices) {
                const auto& datum = data.get_datum(i);
                auto neighbor = svs::find_nearest(datum, centroids);
                const auto& centroid = centroids.get_datum(neighbor.id());

                // Remove the centroid component from the data.
                for (size_t j = 0, jmax = datum.size(); j < jmax; ++j) {
                    buffer.at(j) = datum[j] - centroid[j];
                }

                uint8_t selector = svs::lib::narrow<uint8_t>(neighbor.id());
                compressed.set_datum(i, codec(svs::lib::as_const_span(buffer), selector));
            }
        }
    );
    compress_handle.finish();
    fmt::print("Done compressing!\n");

    // Construct the index.
    auto index_compressed = svs::index::vamana::VamanaIndex{
        svs::GraphLoader(graph_path).load(),
        svs::quantization::lvq::LVQDataset<Primary, 0, Dims>(std::move(compressed)),
        svs::lib::narrow<uint32_t>(medoid),
        Distance(),
        std::move(threadpool)};

    auto index_native = svs::index::vamana::VamanaIndex{
        svs::GraphLoader(graph_path).load(),
        std::move(data),
        svs::lib::narrow<uint32_t>(medoid),
        Distance{},
        svs::threads::NativeThreadPool{num_threads}};

    auto queries = svs::io::auto_load<float>(query_path);
    auto gt = svs::io::auto_load<uint32_t>(groundtruth_path);
    auto search_window_sizes = std::vector<size_t>{10, 20, 30, 40, 50};
    size_t nloops = 10;

    for (auto sws : search_window_sizes) {
        auto search_handle = timer.push_back("search compressed");
        index_compressed.set_search_window_size(sws);
        auto results = index_compressed.search(queries, 10);
        fmt::print("Compressed Recall = {}\n", svs::k_recall_at_n(gt, results, 10, 10));
        auto key = fmt::format("search {}", sws);
        for (size_t i = 0; i < nloops; ++i) {
            auto handle = timer.push_back(key);
            index_compressed.search(queries, 10);
        }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    for (auto sws : search_window_sizes) {
        auto search_handle = timer.push_back("search native");
        index_native.set_search_window_size(sws);
        auto results = index_native.search(queries, 10);
        fmt::print("Native Recall = {}\n", svs::k_recall_at_n(gt, results, 10, 10));
        auto key = fmt::format("search {}", sws);
        for (size_t i = 0; i < nloops; ++i) {
            auto handle = timer.push_back(key);
            index_native.search(queries, 10);
        }
    }
    timer.print();
    index_compressed.save("config", "graph", "data");
    return 0;
}

SVS_DEFINE_MAIN();
