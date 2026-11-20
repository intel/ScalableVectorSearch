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

// svs
#include "svs/concepts/data.h"
#include "svs/concepts/distance.h"
#include "svs/core/data/simple.h"
#include "svs/core/data/view.h"
#include "svs/core/distance.h"
#include "svs/core/kmeans_training.h"
#include "svs/core/logging.h"
#include "svs/index/ivf/sorted_buffer.h"
#include "svs/lib/exception.h"
#include "svs/lib/neighbor.h"
#include "svs/lib/threads/threadpool.h"
#include "svs/lib/timing.h"
#include "svs/lib/type_traits.h"

// external
#include "tsl/robin_set.h"

// Intel(R) MKL
#include <mkl.h>

// stl
#include <random>

// Common definitions.
namespace svs::index::ivf {

// Re-exports the generic k-means names so existing svs::index::ivf::* call sites compile.
using svs::kmeans::centroid_adjustment;
using svs::kmeans::centroid_assignment;
using svs::kmeans::centroid_split;
using svs::kmeans::compute_matmul;
using svs::kmeans::convert_data;
using svs::kmeans::EPSILON;
using svs::kmeans::generate_norms;
using svs::kmeans::generate_unique_ids;
using svs::kmeans::init_centroids;
using svs::kmeans::is_ip_like_v;
using svs::kmeans::is_l2_v;
using svs::kmeans::make_training_set;
using svs::kmeans::maybe_compute_norms;
using svs::kmeans::normalize_centroids;

/// Minimum training sample multiplier for clustering algorithms.
/// When training data size is small relative to the number of clusters, we ensure
/// at least (num_clusters * MIN_TRAINING_SAMPLE_MULTIPLIER) samples are used for
/// training to maintain clustering quality. This prevents degenerate cases where
/// training_fraction would produce insufficient samples.
constexpr size_t MIN_TRAINING_SAMPLE_MULTIPLIER = 2;

/// @brief Parameters controlling the IVF build/k-means algortihm.
struct IVFBuildParameters {
  public:
    /// The target number of clusters in the final result.
    size_t num_centroids_ = 1000;
    /// The size of each minibatch.
    size_t minibatch_size_ = 10'000;
    /// The number of iterations used in kmeans training.
    size_t num_iterations_ = 10;
    /// Use hierarchical Kmeans
    bool is_hierarchical_ = true;
    /// Fraction of dataset used for training
    float training_fraction_ = 0.1;
    // Level1 clusters for hierarchical kmeans (use heuristic when 0)
    size_t hierarchical_level1_clusters_ = 0;
    /// The initial seed for the random number generator.
    size_t seed_ = 0xc0ffee;

  public:
    IVFBuildParameters() = default;
    IVFBuildParameters(
        size_t num_centroids,
        size_t minibatch_size = 10'000,
        size_t num_iterations = 10,
        bool is_hierarchical = true,
        float training_fraction = 0.1,
        size_t hierarchical_level1_clusters = 0,
        size_t seed = 0xc0ffee
    )
        : num_centroids_{num_centroids}
        , minibatch_size_{minibatch_size}
        , num_iterations_{num_iterations}
        , is_hierarchical_{is_hierarchical}
        , training_fraction_{training_fraction}
        , hierarchical_level1_clusters_{hierarchical_level1_clusters}
        , seed_{seed} {}

    // Chain setters to help with construction.
    SVS_CHAIN_SETTER_(IVFBuildParameters, num_centroids);
    SVS_CHAIN_SETTER_(IVFBuildParameters, minibatch_size);
    SVS_CHAIN_SETTER_(IVFBuildParameters, num_iterations);
    SVS_CHAIN_SETTER_(IVFBuildParameters, is_hierarchical);
    SVS_CHAIN_SETTER_(IVFBuildParameters, training_fraction);
    SVS_CHAIN_SETTER_(IVFBuildParameters, hierarchical_level1_clusters);
    SVS_CHAIN_SETTER_(IVFBuildParameters, seed);

    // Comparison
    friend constexpr bool
    operator==(const IVFBuildParameters&, const IVFBuildParameters&) = default;

    // Saving and Loading.
    static constexpr svs::lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "ivf_build_parameters";
    lib::SaveTable save() const {
        return lib::SaveTable(
            serialization_schema,
            save_version,
            {
                SVS_LIST_SAVE_(num_centroids),
                SVS_LIST_SAVE_(minibatch_size),
                SVS_LIST_SAVE_(num_iterations),
                SVS_LIST_SAVE_(is_hierarchical),
                SVS_LIST_SAVE_(training_fraction),
                SVS_LIST_SAVE_(hierarchical_level1_clusters),
                {"seed", lib::save(lib::FullUnsigned(seed_))},
            }
        );
    }

    static IVFBuildParameters load(const lib::ContextFreeLoadTable& table) {
        return IVFBuildParameters(
            SVS_LOAD_MEMBER_AT_(table, num_centroids),
            SVS_LOAD_MEMBER_AT_(table, minibatch_size),
            SVS_LOAD_MEMBER_AT_(table, num_iterations),
            SVS_LOAD_MEMBER_AT_(table, is_hierarchical),
            SVS_LOAD_MEMBER_AT_(table, training_fraction),
            SVS_LOAD_MEMBER_AT_(table, hierarchical_level1_clusters),
            lib::load_at<lib::FullUnsigned>(table, "seed")
        );
    }
};

/// @brief Parameters controlling the IVF search algorithm.
struct IVFSearchParameters {
  public:
    /// The number of nearest clusters to be explored
    size_t n_probes_ = 1;
    /// Level of reordering or reranking done when using compressed datasets
    float k_reorder_ = 1.0;

  public:
    IVFSearchParameters() = default;

    IVFSearchParameters(size_t n_probes, float k_reorder)
        : n_probes_{n_probes}
        , k_reorder_{k_reorder} {}

    SVS_CHAIN_SETTER_(IVFSearchParameters, n_probes);
    SVS_CHAIN_SETTER_(IVFSearchParameters, k_reorder);

    // Saving and Loading.
    static constexpr lib::Version save_version{0, 0, 0};
    static constexpr std::string_view serialization_schema = "ivf_search_parameters";
    lib::SaveTable save() const {
        return lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(n_probes), SVS_LIST_SAVE_(k_reorder)}
        );
    }

    static IVFSearchParameters load(const lib::ContextFreeLoadTable& table) {
        return IVFSearchParameters{
            SVS_LOAD_MEMBER_AT_(table, n_probes), SVS_LOAD_MEMBER_AT_(table, k_reorder)};
    }

    constexpr friend bool
    operator==(const IVFSearchParameters&, const IVFSearchParameters&) = default;
};

/// @brief Forward to svs::kmeans::kmeans_training using the IVF build parameters this
/// index has always taken; kept so existing IVF call sites need no change.
template <
    data::ImmutableMemoryDataset Data,
    typename T,
    typename Distance,
    threads::ThreadPool Pool>
auto kmeans_training(
    const IVFBuildParameters& parameters,
    Data& data,
    Distance& distance,
    data::SimpleData<T>& centroids,
    data::SimpleData<float>& matmul_results,
    std::mt19937& rng,
    Pool& threadpool,
    lib::Timer& timer
) {
    auto kmeans_parameters = svs::kmeans::KMeansTrainingParameters{
        parameters.num_iterations_, parameters.minibatch_size_};
    return svs::kmeans::kmeans_training(
        kmeans_parameters, data, distance, centroids, matmul_results, rng, threadpool, timer
    );
}

template <
    data::ImmutableMemoryDataset Queries,
    typename Centroids,
    typename MatMulResults,
    threads::ThreadPool Pool>
void compute_centroid_distances(
    const Queries& queries,
    const Centroids& centroids,
    MatMulResults& matmul_results,
    Pool& threadpool
) {
    using TQ = typename Queries::element_type;
    using TC = typename Centroids::element_type;
    size_t num_centroids = centroids.size();
    size_t num_queries = queries.size();
    data::SimpleData<TC> queries_conv;

    // Convert if Queries and Centroids datatypes are not same
    if constexpr (!std::is_same_v<TC, TQ>) {
        queries_conv = convert_data<TC>(queries, threadpool);
    }

    threads::parallel_for(
        threadpool,
        threads::StaticPartition(num_centroids),
        [&](auto is, auto tid) {
            auto batch = threads::UnitRange{is};
            if constexpr (!std::is_same_v<TC, TQ>) {
                compute_matmul(
                    queries_conv.data(),
                    centroids.get_datum(batch.start()).data(),
                    matmul_results[tid].data(),
                    num_queries,
                    batch.size(),
                    queries.dimensions()
                );
            } else {
                compute_matmul(
                    queries.get_datum(0).data(),
                    centroids.get_datum(batch.start()).data(),
                    matmul_results[tid].data(),
                    num_queries,
                    batch.size(),
                    queries.dimensions()
                );
            }
        }
    );
}

/// @brief Assign all points to clusters according to assignments
template <std::integral I = uint32_t, typename Data>
std::vector<std::vector<I>> group_assignments(
    const std::vector<size_t>& assignments, size_t num_clusters, const Data& data
) {
    std::vector<std::vector<I>> clusters(num_clusters);
    for (auto i : data.eachindex()) {
        clusters[assignments[i]].push_back(i);
    }
    return clusters;
}

/// @brief Perform cluster assignment for data given pre-trained centroids
///
/// @tparam BuildType The numeric type used for matrix operations (float, Float16, BFloat16)
/// @tparam Data The dataset type
/// @tparam Centroids The centroids dataset type
/// @tparam Distance The distance metric type
/// @tparam Pool The thread pool type
/// @tparam I The integer type for cluster indices
///
/// @param data The dataset to assign to clusters
/// @param centroids The pre-trained centroids
/// @param distance The distance metric
/// @param threadpool The thread pool for parallel execution
/// @param minibatch_size Size of each processing batch (default: 10000)
/// @param integer_type Type tag for cluster indices (default: uint32_t)
///
/// @return A vector of vectors where each inner vector contains the indices of data
///         points assigned to that cluster
template <
    typename BuildType,
    data::ImmutableMemoryDataset Data,
    data::ImmutableMemoryDataset Centroids,
    typename Distance,
    threads::ThreadPool Pool,
    std::integral I = uint32_t>
auto cluster_assignment(
    Data& data,
    Centroids& centroids,
    Distance& distance,
    Pool& threadpool,
    size_t minibatch_size = 10'000,
    lib::Type<I> SVS_UNUSED(integer_type) = {}
) {
    size_t ndims = data.dimensions();
    size_t num_centroids = centroids.size();

    if (data.dimensions() != centroids.dimensions()) {
        throw ANNEXCEPTION(
            "Data and centroids must have the same dimensions! Data dims: {}, Centroids "
            "dims: {}",
            data.dimensions(),
            centroids.dimensions()
        );
    }

    // Allocate memory for assignments and matmul results
    auto assignments = std::vector<size_t>(data.size());
    auto matmul_results = data::SimpleData<float>{minibatch_size, num_centroids};

    // Convert centroids to BuildType if necessary
    using CentroidType = typename Centroids::element_type;
    data::SimpleData<BuildType> centroids_build;
    if constexpr (!std::is_same_v<BuildType, CentroidType>) {
        centroids_build = convert_data<BuildType>(centroids, threadpool);
    } else {
        centroids_build =
            data::SimpleData<BuildType>{centroids.size(), centroids.dimensions()};
        convert_data(centroids, centroids_build, threadpool);
    }

    // Compute norms if using L2 distance
    auto data_norm = maybe_compute_norms<Distance>(data, threadpool);
    auto centroids_norm = maybe_compute_norms<Distance>(centroids_build, threadpool);

    // Process data in batches
    size_t batchsize = minibatch_size;
    size_t num_batches = lib::div_round_up(data.size(), batchsize);

    using Alloc = svs::HugepageAllocator<BuildType>;
    auto data_batch = data::SimpleData<BuildType, Data::extent, Alloc>{batchsize, ndims};

    for (size_t batch = 0; batch < num_batches; ++batch) {
        auto this_batch = threads::UnitRange{
            batch * batchsize, std::min((batch + 1) * batchsize, data.size())};
        auto data_batch_view = data::make_view(data, this_batch);
        convert_data(data_batch_view, data_batch, threadpool);

        // Use the existing centroid_assignment function to compute assignments
        auto timer = lib::Timer();
        centroid_assignment(
            data_batch,
            data_norm,
            this_batch,
            distance,
            centroids_build,
            centroids_norm,
            assignments,
            matmul_results,
            threadpool,
            timer
        );
    }

    // Group assignments into clusters
    return group_assignments<I>(assignments, num_centroids, data);
}

template <typename Query, typename Dist, typename MatMulResults, typename Buffer>
void search_centroids(
    const Query& query,
    Dist& SVS_UNUSED(dist),
    const MatMulResults& matmul_results,
    Buffer& buffer,
    size_t query_id,
    const std::vector<float>& centroids_norm,
    size_t num_threads
) {
    unsigned int count = 0;
    buffer.clear();
    if constexpr (is_ip_like_v<Dist>) {
        for (size_t j = 0; j < num_threads; j++) {
            auto distance = matmul_results[j].get_datum(query_id);
            for (size_t k = 0; k < distance.size(); k++) {
                buffer.insert({count, distance[k]});
                count++;
            }
        }
    } else if constexpr (is_l2_v<Dist>) {
        float query_norm = distance::norm_square(query);
        for (size_t j = 0; j < num_threads; j++) {
            auto distance = matmul_results[j].get_datum(query_id);
            for (size_t k = 0; k < distance.size(); k++) {
                float dist = query_norm + centroids_norm[count] - (2 * distance[k]);
                buffer.insert({count, dist});
                count++;
            }
        }
    } else {
        static_assert(
            sizeof(Dist) == 0, "Only L2, MIP, and Cosine distances supported in IVF search!"
        );
    }
}

template <
    typename Query,
    typename Dist,
    typename Cluster,
    typename BufferCentroids,
    typename BufferLeaves,
    threads::ThreadPool Pool>
void search_leaves(
    const Query& query,
    Dist& dist,
    const Cluster& cluster,
    const BufferCentroids& buffer_centroids,
    BufferLeaves& buffer_leaves,
    Pool& threadpool_inner
) {
    for (size_t j = 0; j < buffer_leaves.size(); j++) {
        buffer_leaves[j].clear();
    }
    distance::maybe_fix_argument(dist, query);
    threads::parallel_for(
        threadpool_inner,
        threads::DynamicPartition(buffer_centroids.size(), 1),
        [&](auto js, auto tid_inner) {
            for (auto j : js) {
                auto candidate = buffer_centroids[j];
                auto cluster_id = candidate.id();

                // Compute the distance between the query and each leaf element.
                cluster.on_leaves(
                    [&](const auto& datum, unsigned int /*gid*/, unsigned int lid) {
                        auto distance = distance::compute(dist, query, datum);
                        buffer_leaves[tid_inner].insert({cluster_id, distance, lid});
                    },
                    cluster_id
                );
            }
        }
    );
}

} // namespace svs::index::ivf
