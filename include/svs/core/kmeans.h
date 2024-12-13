/*
 * Copyright 2023 Intel Corporation
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

#include "svs/concepts/data.h"
#include "svs/concepts/distance.h"
#include "svs/core/data/simple.h"
#include "svs/core/data/view.h"
#include "svs/core/distance/euclidean.h"
#include "svs/core/logging.h"
#include "svs/lib/exception.h"
#include "svs/lib/neighbor.h"
#include "svs/lib/threads/threadpool.h"
#include "svs/lib/timing.h"
#include "svs/lib/type_traits.h"

#include <random>

namespace svs {
///
/// @brief Return the nearest neighbor using the L2 distance.
///
/// @param query The data vector to find nearest neighbor from.
/// @param data The pool of candidates from which to find the nearest neighbors.
///
template <typename Query, data::ImmutableMemoryDataset Data>
Neighbor<size_t> find_nearest(const Query& query, const Data& data) {
    auto f = distance::DistanceL2{};
    auto nearest = type_traits::sentinel_v<Neighbor<size_t>, std::less<>>;
    for (auto i : data.eachindex()) {
        auto d = distance::compute(f, query, data.get_datum(i));
        nearest = std::min(nearest, Neighbor<size_t>(i, d));
    }
    return nearest;
}

template <data::ImmutableMemoryDataset Data, data::ImmutableMemoryDataset Centroids>
double mean_squared_error(
    const Data& data, const Centroids& centroids, threads::NativeThreadPool& threadpool
) {
    threads::SequentialTLS<double> sums(0, threadpool.size());
    threads::run(
        threadpool,
        threads::DynamicPartition{data.size(), 256},
        [&](auto indices, auto tid) {
            auto& this_sum = sums.at(tid);
            for (auto i : indices) {
                auto nearest_centroid = find_nearest(data.get_datum(i), centroids);
                this_sum += nearest_centroid.distance();
            }
        }
    );
    double final_sum = 0;
    sums.visit([&final_sum](double s) { final_sum += s; });
    return final_sum / data.size();
}

struct MeanSquaredErrorCallback {
    template <typename... Args> void operator()(Args&&... args) const {
        return mean_squared_error(std::forward<Args>(args)...);
    }
};

const size_t KMEANS_DEFAULT_SEED = 0xc0ffee;

/// @brief Parameters controlling the k-means algortihm.
struct KMeansParameters {
    KMeansParameters(
        size_t clusters_,
        size_t minibatch_size_,
        size_t epochs_,
        size_t seed_ = KMEANS_DEFAULT_SEED
    )
        : clusters{clusters_}
        , minibatch_size{minibatch_size_}
        , epochs{epochs_}
        , seed{seed_} {}
    /// The target number of clusters in the final result.
    size_t clusters;
    /// The size of each minibatch.
    size_t minibatch_size;
    /// The number of epochs of training.
    size_t epochs;
    /// The initial seed for the random number generator.
    size_t seed;
};

template <data::ImmutableMemoryDataset Data>
void process_batch(
    const Data& data,
    data::SimpleData<float>& centroids,
    std::vector<int64_t>& counts,
    std::vector<int64_t>& old_counts,
    std::vector<size_t>& assignments,
    threads::NativeThreadPool& threadpool,
    lib::Timer& timer
) {
    assignments.resize(data.size());

    // Find the nearest centroid to each element in the sampled dataset.
    // Store the results in `assignments`.
    auto generate_assignments = timer.push_back("generate assignments");
    threads::run(
        threadpool,
        threads::DynamicPartition{data.size(), 128},
        [&](auto indices, auto /*tid*/) {
            for (auto i : indices) {
                assignments[i] = find_nearest(data.get_datum(i), centroids).id();
            }
        }
    );
    generate_assignments.finish();

    auto adjust_centroids = timer.push_back("adjust centroids");
    for (auto i : data.eachindex()) {
        auto datum = data.get_datum(i);
        auto assignment = assignments[i];

        old_counts.at(assignment)++;
        auto& count = counts.at(assignment);
        count++;

        float lr = 1.0F / count;
        auto this_centroid = centroids.get_datum(assignment);
        for (size_t p = 0, pmax = this_centroid.size(); p < pmax; ++p) {
            this_centroid[p] = (1 - lr) * this_centroid[p] + lr * datum[p];
        }
    }
    adjust_centroids.finish();
}

template <data::ImmutableMemoryDataset Data, typename Callback = lib::donothing>
data::SimpleData<float> train_impl(
    const KMeansParameters& parameters,
    const Data& data,
    threads::NativeThreadPool& threadpool,
    Callback&& post_epoch_callback = lib::donothing()
) {
    size_t ndims = data.dimensions();
    auto num_clusters = parameters.clusters;

    // The cluster centroids
    auto centroids = data::SimpleData<float>{num_clusters, ndims};
    auto rng = std::mt19937_64(parameters.seed);
    auto distribution = std::uniform_int_distribution<size_t>(0, data.size());
    std::unordered_set<size_t> seen{};
    for (size_t i = 0; i < num_clusters; ++i) {
        // Pick a vector a random.
        auto j = distribution(rng);
        while (seen.contains(j)) {
            j = distribution(rng);
        }
        centroids.set_datum(i, data.get_datum(j));
        seen.insert(j);
    }

    // Book-keeping
    auto counts = std::vector<int64_t>(num_clusters);
    auto old_counts = std::vector<int64_t>(num_clusters);
    auto assignments = std::vector<size_t>();
    auto timer = lib::Timer();
    for (size_t epoch = 0; epoch < parameters.epochs; ++epoch) {
        auto epoch_timer = timer.push_back("epoch");
        auto batchsize = parameters.minibatch_size;
        auto num_batches = lib::div_round_up(data.size(), batchsize);
        for (size_t batch = 0; batch < num_batches; ++batch) {
            auto batch_timer = timer.push_back("mini batch");
            auto this_batch = threads::UnitRange{
                batch * batchsize, std::min((batch + 1) * batchsize, data.size())};
            process_batch(
                data::make_const_view(data, this_batch),
                centroids,
                counts,
                old_counts,
                assignments,
                threadpool,
                timer
            );
        }

        // Run the requested post-epoch callback.
        post_epoch_callback(data, centroids, threadpool, timer);

        auto cleanup_handle = timer.push_back("clean up");
        for (size_t i = 0, imax = counts.size(); i < imax; ++i) {
            counts[i] = old_counts[i];
            old_counts[i] = 0;
        }
    }
    svs::logging::debug("{}", timer);
    return centroids;
}

template <
    data::ImmutableMemoryDataset Data,
    typename ThreadpoolProto,
    typename Callback = lib::donothing>
data::SimpleData<float> train(
    const KMeansParameters& parameters,
    const Data& data,
    ThreadpoolProto&& threadpool_proto,
    Callback&& post_epoch_callback = lib::donothing()
) {
    auto threadpool =
        threads::as_threadpool(std::forward<ThreadpoolProto>(threadpool_proto));
    return train_impl(
        parameters, data, threadpool, std::forward<Callback>(post_epoch_callback)
    );
}
} // namespace svs
