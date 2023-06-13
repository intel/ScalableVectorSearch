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

// using Distance = svs::distance::DistanceIP;
// using DistanceBiased = svs::quantization::lvq::InnerProductBiased;

// Meta-programming dance
template <typename Primary, svs::data::ImmutableMemoryDataset Data>
auto compress_residual(
    const Primary& primary,
    const svs::data::SimpleData<float>& centroids,
    const Data& data,
    svs::threads::NativeThreadPool& threadpool,
    svs::lib::Timer& timer
) {
    if (primary.size() != data.size()) {
        throw ANNEXCEPTION("Something went wrong!");
    }
    if constexpr (Residual == 0) {
        return svs::index::vamana::NoPostOp();
    } else {
        auto handle = timer.push_back("compressing residual");
        // Thread local decompression buffers.
        auto compressed = svs::quantization::lvq::
            CompressedDataset<svs::quantization::lvq::Signed, Residual, Dims>(
                data.size(), svs::lib::MaybeStatic<Dims>(data.dimensions())
            );

        auto f = [&](auto indices, auto /*tid*/) {
            // Local buffer to hold the full-precision residual.
            auto buffer = std::vector<float>(data.dimensions());
            auto residual_encoder = svs::quantization::lvq::ResidualEncoder<Residual>{};
            for (auto i : indices) {
                auto p = primary.get_datum(i);
                auto d = data.get_datum(i);
                auto centroid = centroids.get_datum(p.get_selector());

                // Get the residual between the original data point and the centroid.
                for (size_t j = 0, jmax = d.size(); j < jmax; ++j) {
                    buffer[j] = d[j] - centroid[j];
                }

                // Compress the residual between the full-precision residual and the
                // primary-compressed residual.
                compressed.set_datum(
                    i,
                    residual_encoder(primary.get_datum(i), svs::lib::as_const_span(buffer))
                );
            }
        };

        svs::threads::run(
            threadpool, svs::threads::DynamicPartition{data.eachindex(), 512}, f
        );

        return svs::index::vamana::ResidualReranker{std::move(compressed)};
    }
}

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

    auto postop = compress_residual(compressed, centroids, data, threadpool, timer);

    // Construct the index.
    auto index_compressed = svs::index::vamana::VamanaIndex{
        svs::GraphLoader(graph_path).load(),
        std::move(compressed),
        svs::lib::narrow<uint32_t>(medoid),
        DistanceBiased{centroids.get_array()},
        std::move(threadpool),
        std::move(postop)};

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
    return 0;
}

SVS_DEFINE_MAIN();
