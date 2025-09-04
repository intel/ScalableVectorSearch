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
    lib::Type<I> SVS_UNUSED(integer_type) = {},
    svs::logging::logger_ptr logger = svs::logging::get()
) {
    auto timer = lib::Timer();
    auto kmeans_timer = timer.push_back("Non-hierarchical kmeans clustering");
    auto init_timer = timer.push_back("init");

    constexpr size_t Dims = Data::extent;
    using Alloc = svs::HugepageAllocator<BuildType>;
    size_t ndims = data.dimensions();
    auto num_centroids = parameters.num_centroids_;

    // Step 1: Create training set
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
    auto rng = std::mt19937(parameters.seed_);
    std::vector<size_t> v(num_training_data);
    auto data_train =
        make_training_set<BuildType, Data, Alloc>(data, v, num_training_data, rng, threadpool);

    // Step 2: Init centroids by randomly selecting from training set
    v.resize(num_centroids);
    auto centroids =
        init_centroids<BuildType>(data_train, v, num_centroids, rng, threadpool);
    auto matmul_results =
        data::SimpleData<float>{parameters.minibatch_size_, num_centroids};
    init_timer.finish();

    // Step 3: K-means training
    auto centroids_fp32 = kmeans_training(
        parameters, data_train, distance, centroids, matmul_results, rng, threadpool, timer
    );

    auto final_assignments_time = timer.push_back("final assignments");
    auto assignments = std::vector<size_t>(data.size());
    auto batchsize = parameters.minibatch_size_;
    auto num_batches = lib::div_round_up(data.size(), batchsize);

    auto data_norm = maybe_compute_norms<Distance>(data, threadpool);
    auto centroids_norm = maybe_compute_norms<Distance>(centroids_fp32, threadpool);

    // Step 4: Assign training data to clusters
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

    // Step 5: Assign all data to clusters
    auto clusters = group_assignments(assignments, num_centroids, data);
    final_assignments_time.finish();
    kmeans_timer.finish();
    svs::logging::debug(logger, "{}", timer);
    svs::logging::debug(
        logger, "kmeans clustering time: {}\n", lib::as_seconds(timer.elapsed())
    );
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
    lib::Type<I> integer_type = {},
    svs::logging::logger_ptr logger = svs::logging::get()
) {
    return kmeans_clustering_impl<BuildType>(
        parameters, data, distance, threadpool, integer_type, std::move(logger)
    );
}
} // namespace svs::index::ivf
