/*
 * Copyright 2026 Intel Corporation
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

#if SVS_ENABLE_MULTI_VECTOR

// svs
#include "svs/lib/threads.h"
#include "svs/lib/timing.h"
#include "svs/quantization/gem/codebook.h"

// stl
#include <chrono>
#include <random>
#include <utility>
#include <vector>

// Opt-in wall-clock timing for svs::quantization::gem::quantize/train_codebook. Nothing
// in the library includes this header; a caller that wants phase timings does.
namespace svs::quantization::gem {

/// Wall-clock seconds for each phase, measured with std::chrono::steady_clock.
struct QuantizeTimings {
    double sample = 0.0;
    double train = 0.0;
    double normalize = 0.0;
    double assign = 0.0;
    double ground_table = 0.0;
    double coarse = 0.0;
    double total = 0.0;
    std::vector<double> train_iterations;
};

namespace detail {

// Times the interval between before_/after_iteration and appends it to `seconds`, which
// outlives the observer; satisfies KMeansObserver.
struct IterationTimer {
    std::vector<double>& seconds;
    std::chrono::steady_clock::time_point start;

    void before_iteration(size_t /*iter*/) { start = std::chrono::steady_clock::now(); }
    void after_iteration(size_t /*iter*/) {
        seconds.push_back(
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
        );
    }
};

} // namespace detail

/// train_codebook, plus a QuantizeTimings for sample/train/normalize/train_iterations.
template <TokenVectorStore Data, threads::ThreadPool Pool>
std::pair<TrainedCodebook, QuantizeTimings> train_codebook_timed(
    const Data& tokens, const CodebookParameters& parameters, Pool& threadpool
) {
    auto total_start = std::chrono::steady_clock::now();
    QuantizeTimings timings;

    std::mt19937 rng(parameters.seed);

    auto sample_start = std::chrono::steady_clock::now();
    auto sampled = sample_training_tokens(tokens, parameters, rng, threadpool);
    timings.sample =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - sample_start)
            .count();

    auto train_start = std::chrono::steady_clock::now();
    auto kmeans_result = train_centroids(
        sampled.trainset,
        sampled.K,
        parameters.kmeans_iterations,
        parameters.minibatch_size,
        rng,
        threadpool,
        detail::IterationTimer{timings.train_iterations, {}},
        parameters.kmeans_algorithm,
        parameters.emeans_epochs
    );
    timings.train =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - train_start)
            .count();

    auto normalize_start = std::chrono::steady_clock::now();
    svs::lib::Timer scratch_timer;
    svs::kmeans::normalize_centroids(kmeans_result.centroids, threadpool, scratch_timer);
    timings.normalize =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - normalize_start)
            .count();

    timings.total =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start)
            .count();

    auto trained = TrainedCodebook{
        std::move(kmeans_result.centroids),
        kmeans_result.objective,
        sampled.num_sampled_passages,
        sampled.num_training_tokens};
    return {std::move(trained), std::move(timings)};
}

/// quantize, plus the full QuantizeTimings (train_codebook_timed's phases, then
/// assign/ground_table/coarse/total). Identical result to quantize on the same input.
template <TokenVectorStore Data, threads::ThreadPool Pool>
std::pair<Quantized, QuantizeTimings>
quantize_timed(const Data& tokens, const CodebookParameters& parameters, Pool& threadpool) {
    auto total_start = std::chrono::steady_clock::now();

    auto [trained, timings] = train_codebook_timed(tokens, parameters, threadpool);

    auto assign_start = std::chrono::steady_clock::now();
    auto ids = assign_tokens(tokens, trained.centroids, parameters, threadpool);
    timings.assign =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - assign_start)
            .count();

    size_t K = trained.centroids.size();

    // Timed with the pairwise table build; neither has its own QuantizeTimings field.
    auto ground_start = std::chrono::steady_clock::now();
    auto ground = build_ground_structures(trained.centroids, parameters, threadpool);
    timings.ground_table =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - ground_start)
            .count();

    auto coarse_start = std::chrono::steady_clock::now();
    auto coarse = coarse_clusters(trained.centroids, parameters, threadpool);
    timings.coarse =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - coarse_start)
            .count();

    timings.total =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start)
            .count();

    auto quantized = Quantized{
        std::move(trained.centroids),
        std::move(ground.fine_centroids),
        std::move(ground.centroid_table),
        std::move(ids),
        std::move(coarse.coarse_of_fine),
        coarse.num_coarse,
        K,
        trained.objective,
        trained.num_sampled_passages,
        trained.num_training_tokens};
    return {std::move(quantized), std::move(timings)};
}

} // namespace svs::quantization::gem

#else

#error \
    "svs/quantization/gem/codebook_timed.h needs SVS_ENABLE_MULTI_VECTOR=1 (cmake: SVS_EXPERIMENTAL_ENABLE_MULTI_VECTOR=YES); it pulls in Intel MKL through codebook.h."

#endif // SVS_ENABLE_MULTI_VECTOR
