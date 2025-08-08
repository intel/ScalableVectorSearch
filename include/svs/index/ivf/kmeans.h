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

#pragma once

#include "svs/index/ivf/common.h"

namespace svs::index::ivf {

template <
    typename BuildType,
    data::ImmutableMemoryDataset Data,
    typename Distance,
    threads::ThreadPool Pool,
    std::integral I = uint32_t>
auto kmeans_clustering_impl(
    const IVFBuildParameters& parameters,
    Data& data,
    Distance& distance,
    Pool& threadpool,
    lib::Type<I> SVS_UNUSED(integer_type) = {}
) {
    auto timer = lib::Timer();
    auto kmeans_timer = timer.push_back("Non-hierarchical kmeans clustering");
    auto init_timer = timer.push_back("init");

    constexpr size_t Dims = Data::extent;
    using Alloc = svs::HugepageAllocator<BuildType>;
    size_t ndims = data.dimensions();
    auto num_centroids = parameters.num_centroids_;
    size_t num_training_data =
        lib::narrow<size_t>(std::ceil(data.size() * parameters.training_fraction_));
    if (num_training_data < num_centroids || num_training_data > data.size()) {
        throw ANNEXCEPTION(
            "Invalid number of training data: {}, num_centroids: {}, total data size: "
            "{}\n",
            num_training_data,
            num_centroids,
            data.size()
        );
    }

    // The cluster centroids
    auto centroids = data::SimpleData<BuildType>{num_centroids, ndims};
    auto data_train = data::SimpleData<BuildType, Dims, Alloc>{num_training_data, ndims};
    auto matmul_results =
        data::SimpleData<float>{parameters.minibatch_size_, num_centroids};
    auto rng = std::mt19937(parameters.seed_);

    std::vector<size_t> v(num_training_data);
    generate_unique_ids(v, data.size(), rng);
    threads::parallel_for(
        threadpool,
        threads::StaticPartition{num_training_data},
        [&](auto indices, auto /*tid*/) {
            for (auto i : indices) {
                data_train.set_datum(i, data.get_datum(v[i]));
            }
        }
    );

    v.resize(num_centroids);
    generate_unique_ids(v, data_train.size(), rng);
    threads::parallel_for(
        threadpool,
        threads::StaticPartition{num_centroids},
        [&](auto indices, auto /*tid*/) {
            for (auto i : indices) {
                centroids.set_datum(i, data_train.get_datum(v[i]));
            }
        }
    );
    init_timer.finish();

    auto centroids_fp32 = kmeans_training(
        parameters, data_train, distance, centroids, matmul_results, rng, threadpool, timer
    );

    auto final_assignments_time = timer.push_back("final assignments");
    auto assignments = std::vector<size_t>(data.size());
    auto batchsize = parameters.minibatch_size_;
    auto num_batches = lib::div_round_up(data.size(), batchsize);

    std::vector<float> data_norm;
    if constexpr (std::is_same_v<Distance, distance::DistanceL2>) {
        generate_norms(data, data_norm, threadpool);
    }
    std::vector<float> centroids_norm;
    if constexpr (std::is_same_v<Distance, distance::DistanceL2>) {
        generate_norms(centroids_fp32, centroids_norm, threadpool);
    }

    auto data_batch = data::SimpleData<BuildType, Dims, Alloc>{batchsize, ndims};
    for (size_t batch = 0; batch < num_batches; ++batch) {
        auto this_batch = threads::UnitRange{
            batch * batchsize, std::min((batch + 1) * batchsize, data.size())};
        auto data_batch_view = data::make_view(data, this_batch);
        convert_data(data_batch_view, data_batch, threadpool);
        centroid_assignment(
            data_batch,
            data_norm,
            this_batch,
            distance,
            centroids,
            centroids_norm,
            assignments,
            matmul_results,
            threadpool,
            timer
        );
    }

    auto clusters = std::vector<std::vector<I>>(num_centroids);
    for (auto i : data.eachindex()) {
        clusters[assignments[i]].push_back(i);
    }
    final_assignments_time.finish();
    kmeans_timer.finish();
    svs::logging::debug("{}", timer);
    fmt::print("kmeans clustering time: {}\n", lib::as_seconds(timer.elapsed()));
    return std::make_tuple(centroids, std::move(clusters));
}

template <
    typename BuildType,
    data::ImmutableMemoryDataset Data,
    typename Distance,
    threads::ThreadPool Pool,
    std::integral I = uint32_t>
auto kmeans_clustering(
    const IVFBuildParameters& parameters,
    Data& data,
    Distance& distance,
    Pool& threadpool,
    lib::Type<I> integer_type = {}
) {
    return kmeans_clustering_impl<BuildType>(
        parameters, data, distance, threadpool, integer_type
    );
}
} // namespace svs::index::ivf
