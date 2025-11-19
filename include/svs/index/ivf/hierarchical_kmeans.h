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

// stdlib
#include <cmath>

namespace svs::index::ivf {

/// @brief Calculate the number of level 2 clusters for each level 1 cluster
///        Distributes num_clusters among num_level1_clusters proportional to cluster sizes,
///        then randomly assigns any remaining centroids to non-empty clusters.
template <std::integral I = uint32_t>
auto calc_level2_clusters(
    size_t num_clusters,
    size_t num_level1_clusters,
    std::vector<std::vector<I>>& clusters_level1,
    size_t num_training_data,
    std::mt19937& rng
) {
    // Proportional allocation
    auto num_level2_clusters = std::vector<size_t>(num_level1_clusters);
    size_t total_centroids_l2 = 0;
    if (num_training_data == 0) {
        throw ANNEXCEPTION("No training data for hierarchical kmeans\n");
        return num_level2_clusters;
    }

    for (size_t cluster = 0; cluster < num_level1_clusters; cluster++) {
        num_level2_clusters[cluster] =
            (1.0 * clusters_level1[cluster].size()) / num_training_data * num_clusters;
        total_centroids_l2 += num_level2_clusters[cluster];
    }

    while (total_centroids_l2 < num_clusters) {
        size_t j = rng() % num_level1_clusters;
        if (!clusters_level1[j].empty()) {
            num_level2_clusters[j]++;
            total_centroids_l2++;
        }
    }
    return num_level2_clusters;
}

/// @brief Main hierarchical k-means clustering logic
template <
    typename BuildType,
    data::ImmutableMemoryDataset Data,
    typename Distance,
    threads::ThreadPool Pool,
    std::integral I = uint32_t>
auto hierarchical_kmeans_clustering_impl(
    const IVFBuildParameters& parameters,
    Data& data,
    Distance& distance,
    Pool& threadpool,
    lib::Type<I> SVS_UNUSED(integer_type) = {},
    svs::logging::logger_ptr logger = svs::logging::get(),
    bool train_only = false
) {
    auto timer = lib::Timer();
    auto kmeans_timer = timer.push_back("Hierarchical kmeans clustering");
    auto init_timer = timer.push_back("init");

    constexpr size_t Dims = Data::extent;
    using Alloc = svs::HugepageAllocator<BuildType>;
    size_t ndims = data.dimensions();
    auto num_clusters = parameters.num_centroids_;

    size_t num_level1_clusters = parameters.hierarchical_level1_clusters_;
    if (num_level1_clusters == 0) {
        num_level1_clusters = std::sqrt(num_clusters);
    }

    svs::logging::debug(logger, "Level1 clusters: {}\n", num_level1_clusters);

    // Step 1: Create training set
    size_t num_training_data =
        lib::narrow<size_t>(std::ceil(data.size() * parameters.training_fraction_));
    if (num_training_data < num_clusters || num_training_data > data.size()) {
        throw ANNEXCEPTION(
            "Invalid number of training data: {}, num_clusters: {}, total data size: "
            "{}\n",
            num_training_data,
            num_clusters,
            data.size()
        );
    }
    auto rng = std::mt19937(parameters.seed_);
    std::vector<size_t> v(num_training_data);
    auto data_train = make_training_set<BuildType, Data, Alloc>(
        data, v, num_training_data, rng, threadpool
    );

    // Step 2: Init centroids for level 1
    v.resize(num_level1_clusters);
    auto centroids_level1 =
        init_centroids<BuildType>(data_train, v, num_level1_clusters, rng, threadpool);
    auto matmul_results_level1 =
        data::SimpleData<float>{parameters.minibatch_size_, num_level1_clusters};

    init_timer.finish();

    // Step 3: Train level 1 k-means
    auto level1_training_time = timer.push_back("Level1 training");
    auto centroids_level1_fp32 = kmeans_training(
        parameters,
        data_train,
        distance,
        centroids_level1,
        matmul_results_level1,
        rng,
        threadpool,
        timer
    );
    auto assignments_level1 = std::vector<size_t>(data_train.size());
    auto batchsize = parameters.minibatch_size_;
    auto num_batches = lib::div_round_up(data_train.size(), batchsize);

    // Step 4: Assign training data to clusters
    auto data_norm = maybe_compute_norms<Distance>(data_train, threadpool);
    auto centroids_level1_norm =
        maybe_compute_norms<Distance>(centroids_level1_fp32, threadpool);

    for (size_t batch = 0; batch < num_batches; ++batch) {
        auto this_batch = threads::UnitRange{
            batch * batchsize, std::min((batch + 1) * batchsize, data_train.size())};
        auto data_batch = data::make_view(data_train, this_batch);
        centroid_assignment(
            data_batch,
            data_norm,
            this_batch,
            distance,
            centroids_level1,
            centroids_level1_norm,
            assignments_level1,
            matmul_results_level1,
            threadpool,
            timer
        );
    }
    auto clusters_level1 =
        group_assignments(assignments_level1, num_level1_clusters, data_train);

    std::vector<std::vector<I>> clusters_level1_all;

    // Declare timer outside of block to avoid scope issues
    auto all_assignments_time = timer.push_back("level1 all assignments");

    if (!train_only) {
        // Step 5: Assign all data to clusters
        auto assignments_level1_all = std::vector<size_t>(data.size());

        batchsize = parameters.minibatch_size_;
        num_batches = lib::div_round_up(data.size(), batchsize);

        data_norm = maybe_compute_norms<Distance>(data, threadpool);
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
                centroids_level1,
                centroids_level1_norm,
                assignments_level1_all,
                matmul_results_level1,
                threadpool,
                timer
            );
        }
        auto all_assignments_cluster = timer.push_back("level1 all assignments clusters");
        clusters_level1_all =
            group_assignments(assignments_level1_all, num_level1_clusters, data);
        all_assignments_cluster.finish();
    } else {
        // For train_only, create empty clusters
        clusters_level1_all.resize(num_level1_clusters);
    }

    all_assignments_time.finish();
    level1_training_time.finish();

    // Step 6: Train level 2 clusters for each level 1 cluster
    auto level2_training_time = timer.push_back("Level2 training");
    auto num_level2_clusters = calc_level2_clusters(
        num_clusters, num_level1_clusters, clusters_level1, num_training_data, rng
    );

    auto centroids_final = data::SimpleData<BuildType>{num_clusters, ndims};
    auto clusters_final = std::vector<std::vector<I>>(num_clusters);

    size_t max_data_per_cluster = 0;
    if (!train_only) {
        for (size_t cluster = 0; cluster < num_level1_clusters; cluster++) {
            max_data_per_cluster =
                clusters_level1_all[cluster].size() > max_data_per_cluster
                    ? clusters_level1_all[cluster].size()
                    : max_data_per_cluster;
        }
    } else {
        // In train_only mode, use training clusters for Level 2 training
        for (size_t cluster = 0; cluster < num_level1_clusters; cluster++) {
            max_data_per_cluster = clusters_level1[cluster].size() > max_data_per_cluster
                                       ? clusters_level1[cluster].size()
                                       : max_data_per_cluster;
        }
    }
    auto data_level2 =
        data::SimpleData<BuildType, Dims, Alloc>{max_data_per_cluster, ndims};
    auto assignments_level2_all = std::vector<size_t>(max_data_per_cluster);

    size_t cluster_start = 0;
    for (size_t cluster = 0; cluster < num_level1_clusters; cluster++) {
        size_t num_clusters_l2 = num_level2_clusters[cluster];
        size_t num_assignments_l2 = clusters_level1[cluster].size();
        size_t num_assignments_l2_all =
            train_only ? 0 : clusters_level1_all[cluster].size();

        auto matmul_results_level2 =
            data::SimpleData<float>{parameters.minibatch_size_, num_clusters_l2};
        auto data_train_level2 = data::SimpleData<BuildType>{num_assignments_l2, ndims};

        // Prepare level 2 training data
        threads::parallel_for(
            threadpool,
            threads::StaticPartition{num_assignments_l2},
            [&](auto indices, auto /*tid*/) {
                for (auto i : indices) {
                    data_train_level2.set_datum(
                        i, data_train.get_datum(clusters_level1[cluster][i])
                    );
                }
            }
        );

        v.resize(num_clusters_l2);
        auto centroids_level2 = init_centroids<BuildType>(
            data_train_level2, v, num_clusters_l2, rng, threadpool
        );

        auto centroids_level2_fp32 = kmeans_training(
            parameters,
            data_train_level2,
            distance,
            centroids_level2,
            matmul_results_level2,
            rng,
            threadpool,
            timer
        );

        auto all_assignments_level2 = timer.push_back("level2 all assignments");

        if (!train_only) {
            // Only do Level 2 assignments if not in train_only mode
            threads::parallel_for(
                threadpool,
                threads::StaticPartition{num_assignments_l2_all},
                [&](auto indices, auto /*tid*/) {
                    for (auto i : indices) {
                        data_level2.set_datum(
                            i, data.get_datum(clusters_level1_all[cluster][i])
                        );
                    }
                }
            );

            batchsize = parameters.minibatch_size_;
            num_batches = lib::div_round_up(num_assignments_l2_all, batchsize);

            data_norm = maybe_compute_norms<Distance>(data_level2, threadpool);
            auto centroids_level2_norm =
                maybe_compute_norms<Distance>(centroids_level2_fp32, threadpool);
            for (size_t batch = 0; batch < num_batches; ++batch) {
                auto this_batch = threads::UnitRange{
                    batch * batchsize,
                    std::min((batch + 1) * batchsize, num_assignments_l2_all)};
                auto data_batch = data::make_view(data_level2, this_batch);
                centroid_assignment(
                    data_batch,
                    data_norm,
                    this_batch,
                    distance,
                    centroids_level2,
                    centroids_level2_norm,
                    assignments_level2_all,
                    matmul_results_level2,
                    threadpool,
                    timer
                );
            }

            for (size_t i = 0; i < num_assignments_l2_all; i++) {
                clusters_final[cluster_start + assignments_level2_all[i]].push_back(
                    clusters_level1_all[cluster][i]
                );
            }
        }

        threads::parallel_for(
            threadpool,
            threads::StaticPartition{num_clusters_l2},
            [&](auto indices, auto /*tid*/) {
                for (auto i : indices) {
                    centroids_final.set_datum(
                        cluster_start + i, centroids_level2_fp32.get_datum(i)
                    );
                }
            }
        );

        cluster_start += num_clusters_l2;
        all_assignments_level2.finish();
    }

    level2_training_time.finish();

    kmeans_timer.finish();
    svs::logging::debug(logger, "{}", timer);
    svs::logging::debug(
        logger,
        "Hierarchical kmeans clustering time: {}\n",
        lib::as_seconds(timer.elapsed())
    );

    return std::make_tuple(std::move(centroids_final), std::move(clusters_final));
}

template <
    typename BuildType,
    data::ImmutableMemoryDataset Data,
    typename Distance,
    threads::ThreadPool Pool,
    std::integral I = uint32_t>
auto hierarchical_kmeans_clustering(
    const IVFBuildParameters& parameters,
    Data& data,
    Distance& distance,
    Pool& threadpool,
    lib::Type<I> integer_type = {},
    svs::logging::logger_ptr logger = svs::logging::get(),
    bool train_only = false
) {
    return hierarchical_kmeans_clustering_impl<BuildType>(
        parameters, data, distance, threadpool, integer_type, std::move(logger), train_only
    );
}

} // namespace svs::index::ivf
