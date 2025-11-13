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

// Small epsilon value used for floating-point comparisons to avoid precision
// issues.  The value 1/1024 (approximately 0.0009765625) is chosen as a reasonable
// threshold for numerical stability in algorithms such as k-means clustering, where exact
constexpr double EPSILON = 1.0 / 1024.0;

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

// Helper functions to convert data from one type to another using threadpool
template <typename D1, typename D2, threads::ThreadPool Pool>
void convert_data(D1& src, D2& dst, Pool& threadpool) {
    // Note: Destination size can be bigger than the source as we preallocate a bigger
    // buffer and reuse it to reduce the cost of frequent allocations
    if (src.size() > dst.size() || src.dimensions() != dst.dimensions()) {
        throw ANNEXCEPTION(
            "Unexpected data shapes sizes: {}, {}; dims: {}, {}!",
            src.size(),
            dst.size(),
            src.dimensions(),
            dst.dimensions()
        );
    }

    threads::parallel_for(
        threadpool,
        threads::StaticPartition{src.size()},
        [&](auto indices, auto /*tid*/) {
            for (auto i : indices) {
                dst.set_datum(i, src.get_datum(i));
            }
        }
    );
}

template <typename T, data::ImmutableMemoryDataset Data, threads::ThreadPool Pool>
auto convert_data(Data& src, Pool& threadpool) {
    auto dst = svs::data::SimpleData<T>(src.size(), src.dimensions());
    convert_data(src, dst, threadpool);
    return dst;
}

// Partial specialization to preserve the dimensionality and Allocation type
template <typename T, size_t Extent, typename Alloc, threads::ThreadPool Pool>
auto convert_data(svs::data::SimpleData<float, Extent, Alloc>& src, Pool& threadpool) {
    using allocator_type = svs::lib::rebind_allocator_t<T, Alloc>;
    allocator_type rebound_alloctor = {};

    auto dst = svs::data::SimpleData<T, Extent, allocator_type>(
        src.size(), src.dimensions(), rebound_alloctor
    );
    convert_data(src, dst, threadpool);
    return dst;
}

template <typename T, data::ImmutableMemoryDataset Data> auto convert_data(Data& src) {
    auto dst = svs::data::SimpleData<T>(src.size(), src.dimensions());
    auto threadpool = threads::as_threadpool(1);
    convert_data(src, dst, threadpool);
    return dst;
}

template <typename T>
void compute_matmul(
    const T* data, const T* centroids, float* results, size_t m, size_t n, size_t k
) {
    // Validate parameters to avoid Intel MKL errors
    if (m == 0 || n == 0 || k == 0) {
        return; // Nothing to compute
    }

    // Check for integer overflow when casting to int (MKL requirement)
    constexpr size_t max_int = static_cast<size_t>(std::numeric_limits<int>::max());
    if (m > max_int || n > max_int || k > max_int) {
        throw ANNEXCEPTION(
            "Matrix dimensions too large for Intel MKL GEMM: m={}, n={}, k={}", m, n, k
        );
    }

    if constexpr (std::is_same_v<T, float>) {
        // Cast size_t parameters to int for MKL GEMM functions
        int m_int = static_cast<int>(m);
        int n_int = static_cast<int>(n);
        int k_int = static_cast<int>(k);

        cblas_sgemm(
            CblasRowMajor, // CBLAS_LAYOUT layout
            CblasNoTrans,  // CBLAS_TRANSPOSE TransA
            CblasTrans,    // CBLAS_TRANSPOSE TransB
            m_int,         // const int M
            n_int,         // const int N
            k_int,         // const int K
            1.0f,          // float alpha (explicitly float)
            data,          // const float* A
            k_int,         // const int lda
            centroids,     // const float* B
            k_int,         // const int ldb
            0.0f,          // const float beta (explicitly float)
            results,       // float* c
            n_int          // const int ldc
        );
    } else if constexpr (std::is_same_v<T, BFloat16>) {
        // Intel MKL BFloat16 GEMM requires careful parameter casting to avoid parameter
        // errors Ensure all integer parameters are properly cast to int (MKL expects int,
        // not size_t)
        int m_int = static_cast<int>(m);
        int n_int = static_cast<int>(n);
        int k_int = static_cast<int>(k);

        cblas_gemm_bf16bf16f32(
            CblasRowMajor,              // CBLAS_LAYOUT layout
            CblasNoTrans,               // CBLAS_TRANSPOSE TransA
            CblasTrans,                 // CBLAS_TRANSPOSE TransB
            m_int,                      // const int M
            n_int,                      // const int N
            k_int,                      // const int K
            1.0f,                       // float alpha (explicitly float)
            (const uint16_t*)data,      // const *uint16_t A
            k_int,                      // const int lda
            (const uint16_t*)centroids, // const uint16_t* B
            k_int,                      // const int ldb
            0.0f,                       // const float beta (explicitly float)
            results,                    // float* c
            n_int                       // const int ldc
        );
    } else if constexpr (std::is_same_v<T, Float16>) {
        // Intel MKL Float16 GEMM requires careful parameter casting to avoid parameter
        // errors Ensure all integer parameters are properly cast to int (MKL expects int,
        // not size_t)
        int m_int = static_cast<int>(m);
        int n_int = static_cast<int>(n);
        int k_int = static_cast<int>(k);

        cblas_gemm_f16f16f32(
            CblasRowMajor,              // CBLAS_LAYOUT layout
            CblasNoTrans,               // CBLAS_TRANSPOSE TransA
            CblasTrans,                 // CBLAS_TRANSPOSE TransB
            m_int,                      // const int M
            n_int,                      // const int N
            k_int,                      // const int K
            1.0f,                       // float alpha (explicitly float)
            (const uint16_t*)data,      // const *uint16_t A
            k_int,                      // const int lda
            (const uint16_t*)centroids, // const uint16_t* B
            k_int,                      // const int ldb
            0.0f,                       // const float beta (explicitly float)
            results,                    // float* c
            n_int                       // const int ldc
        );
    } else {
        throw ANNEXCEPTION("GEMM type not supported!");
    }
}

inline static void
generate_unique_ids(std::vector<size_t>& ids, size_t id_range, std::mt19937& rng) {
    size_t n = ids.size();
    tsl::robin_set<size_t> seen;
    seen.reserve(n);

    while (seen.size() < n) {
        auto j = rng() % id_range;
        seen.insert(j);
    }

    size_t i = 0;
    for (auto it = seen.begin(); it != seen.end(); ++it, ++i) {
        ids[i] = *it;
    }
}

template <typename T, threads::ThreadPool Pool>
void normalize_centroids(
    data::SimpleData<T>& centroids, Pool& threadpool, lib::Timer& timer
) {
    auto normalize_centroids_t = timer.push_back("normalize centroids");
    threads::parallel_for(
        threadpool,
        threads::StaticPartition{centroids.size()},
        [&](auto indices, auto /*tid*/) {
            for (auto i : indices) {
                auto datum = centroids.get_datum(i);
                float norm = distance::norm(datum);
                if (norm != 0.0) {
                    float norm_inv = 1.0 / norm;
                    for (size_t j = 0; j < datum.size(); j++) {
                        datum[j] = datum[j] * norm_inv;
                    }
                }
            }
        }
    );
    normalize_centroids_t.finish();
}

template <
    data::ImmutableMemoryDataset Data,
    typename T,
    typename Distance,
    threads::ThreadPool Pool>
void centroid_assignment(
    Data& data,
    std::vector<float>& data_norm,
    threads::UnitRange<uint64_t> batch_range,
    Distance& SVS_UNUSED(distance),
    data::SimpleData<T>& centroids,
    std::vector<float>& centroids_norm,
    std::vector<size_t>& assignments,
    data::SimpleData<float>& matmul_results,
    Pool& threadpool,
    lib::Timer& timer
) {
    using DataType = typename Data::element_type;
    using CentroidType = T;

    // Convert data to match centroid type if necessary
    data::SimpleData<CentroidType> data_conv;
    if constexpr (!std::is_same_v<CentroidType, DataType>) {
        data_conv = convert_data<CentroidType>(data, threadpool);
    }

    auto generate_assignments = timer.push_back("generate assignments");
    threads::parallel_for(
        threadpool,
        threads::StaticPartition{batch_range.size()},
        [&](auto indices, auto /*tid*/) {
            auto range = threads::UnitRange(indices);
            if constexpr (!std::is_same_v<CentroidType, DataType>) {
                compute_matmul(
                    data_conv.get_datum(range.start()).data(),
                    centroids.data(),
                    matmul_results.get_datum(range.start()).data(),
                    range.size(),
                    centroids.size(),
                    data.dimensions()
                );
            } else {
                compute_matmul(
                    data.get_datum(range.start()).data(),
                    centroids.data(),
                    matmul_results.get_datum(range.start()).data(),
                    range.size(),
                    centroids.size(),
                    data.dimensions()
                );
            }
            if constexpr (std::is_same_v<Distance, distance::DistanceIP>) {
                for (auto i : indices) {
                    auto nearest =
                        type_traits::sentinel_v<Neighbor<size_t>, std::greater<>>;
                    auto dists = matmul_results.get_datum(i);
                    for (size_t j = 0; j < centroids.size(); j++) {
                        nearest = std::max(nearest, Neighbor<size_t>(j, dists[j]));
                    }
                    assignments[batch_range.start() + i] = nearest.id();
                }
            } else if constexpr (std::is_same_v<Distance, distance::DistanceL2>) {
                for (auto i : indices) {
                    auto nearest = type_traits::sentinel_v<Neighbor<size_t>, std::less<>>;
                    auto dists = matmul_results.get_datum(i);
                    for (size_t j = 0; j < centroids.size(); j++) {
                        auto dist = data_norm[batch_range.start() + i] + centroids_norm[j] -
                                    2 * dists[j];
                        nearest = std::min(nearest, Neighbor<size_t>(j, dist));
                    }
                    assignments[batch_range.start() + i] = nearest.id();
                }
            } else {
                throw ANNEXCEPTION("Only L2 and MIP distances supported in IVF build!");
            }
        }
    );
    generate_assignments.finish();
}

template <data::ImmutableMemoryDataset Data, typename T, threads::ThreadPool Pool>
void centroid_adjustment(
    Data& data,
    data::SimpleData<T>& centroids,
    std::vector<size_t>& assignments,
    std::vector<size_t>& counts,
    Pool& threadpool,
    lib::Timer& timer
) {
    auto adjust_centroids = timer.push_back("adjust centroids");
    size_t n_threads = threadpool.size();
    size_t n_centroids = centroids.size();

    threads::parallel_for(
        threadpool,
        threads::StaticPartition{n_threads},
        [&](auto /*indices*/, auto tid) {
            size_t centroid_start = (n_centroids * tid) / n_threads;
            size_t centroid_end = (n_centroids * (tid + 1)) / n_threads;
            for (auto i : data.eachindex()) {
                auto assignment = assignments[i];
                if (assignment >= centroid_start && assignment < centroid_end) {
                    counts.at(assignment)++;
                    auto datum = data.get_datum(i);
                    auto this_centroid = centroids.get_datum(assignment);
                    for (size_t p = 0, pmax = this_centroid.size(); p < pmax; ++p) {
                        this_centroid[p] += lib::narrow_cast<float>(datum[p]);
                    }
                }
            }
        }
    );

    threads::parallel_for(
        threadpool,
        threads::StaticPartition{n_centroids},
        [&](auto indices, auto /*tid*/) {
            for (auto i : indices) {
                if (counts.at(i) != 0) {
                    auto this_centroid = centroids.get_datum(i);
                    float norm = 1.0 / (counts.at(i) + 1);
                    // float norm = 1.0 / counts.at(i);
                    for (size_t p = 0, pmax = this_centroid.size(); p < pmax; ++p) {
                        this_centroid[p] *= norm;
                    }
                }
            }
        }
    );
    adjust_centroids.finish();
}

template <data::ImmutableMemoryDataset Data, typename T, threads::ThreadPool Pool>
void centroid_split(
    Data& data,
    data::SimpleData<T>& centroids,
    std::vector<size_t>& counts,
    std::mt19937& rng,
    Pool& SVS_UNUSED(threadpool),
    lib::Timer& timer
) {
    auto split_centroids = timer.push_back("split centroids");

    auto num_centroids = centroids.size();
    auto dims = centroids.dimensions();
    auto num_data = data.size();

    auto distribution = std::uniform_real_distribution<float>(0.0, 1.0);
    for (size_t i = 0; i < num_centroids; i++) {
        if (counts.at(i) == 0) {
            size_t j;
            for (j = 0; true; j = (j + 1) % num_centroids) {
                if (counts.at(j) == 0) {
                    continue;
                }
                float p = counts.at(j) / float(num_data);
                float r = distribution(rng);
                if (r < p) {
                    break;
                }
            }
            centroids.set_datum(i, centroids.get_datum(j));
            for (size_t k = 0; k < dims; k++) {
                if (k % 2 == 0) {
                    centroids.get_datum(i)[k] *= 1 + EPSILON;
                    centroids.get_datum(j)[k] *= 1 - EPSILON;
                } else {
                    centroids.get_datum(i)[k] *= 1 - EPSILON;
                    centroids.get_datum(j)[k] *= 1 + EPSILON;
                }
            }
            counts.at(i) = counts.at(j) / 2;
            counts.at(j) -= counts.at(i);
        }
    }
    split_centroids.finish();
}

template <typename Data, threads::ThreadPool Pool>
void generate_norms(Data& data, std::vector<float>& norms, Pool& threadpool) {
    norms.resize(data.size());
    threads::parallel_for(
        threadpool,
        threads::StaticPartition{data.size()},
        [&](auto indices, auto /*tid*/) {
            for (auto i : indices) {
                norms[i] = distance::norm_square(data.get_datum(i));
            }
        }
    );
}

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
    auto training_timer = timer.push_back("Kmeans training");
    data::SimpleData<float> centroids_fp32 = convert_data<float>(centroids, threadpool);

    if constexpr (std::is_same_v<Distance, distance::DistanceIP>) {
        normalize_centroids(centroids_fp32, threadpool, timer);
    }

    auto assignments = std::vector<size_t>(data.size());
    std::vector<float> data_norm;
    if constexpr (std::is_same_v<Distance, distance::DistanceL2>) {
        generate_norms(data, data_norm, threadpool);
    }
    std::vector<float> centroids_norm;

    for (size_t iter = 0; iter < parameters.num_iterations_; ++iter) {
        auto iter_timer = timer.push_back("iteration");
        auto batchsize = parameters.minibatch_size_;
        auto num_batches = lib::div_round_up(data.size(), batchsize);
        if constexpr (std::is_same_v<Distance, distance::DistanceL2>) {
            generate_norms(centroids_fp32, centroids_norm, threadpool);
        }

        // Convert from fp32 to fp16/bf16
        convert_data(centroids_fp32, centroids, threadpool);

        for (size_t batch = 0; batch < num_batches; ++batch) {
            auto this_batch = threads::UnitRange{
                batch * batchsize, std::min((batch + 1) * batchsize, data.size())};
            auto data_batch = data::make_view(data, this_batch);
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

        // Convert back to fp32
        convert_data(centroids, centroids_fp32, threadpool);

        auto counts = std::vector<size_t>(centroids.size());
        centroid_adjustment(data, centroids_fp32, assignments, counts, threadpool, timer);

        centroid_split(data, centroids_fp32, counts, rng, threadpool, timer);

        if constexpr (std::is_same_v<Distance, distance::DistanceIP>) {
            normalize_centroids(centroids_fp32, threadpool, timer);
        }
    }

    // Finally call the conversion to get the updated centroids
    // after adjustment and split
    convert_data(centroids_fp32, centroids, threadpool);
    training_timer.finish();
    return centroids_fp32;
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

/// @brief Generate a random subset of data for training
template <typename BuildType, typename Data, typename Alloc, threads::ThreadPool Pool>
data::SimpleData<BuildType, Data::extent, Alloc> make_training_set(
    const Data& data,
    std::vector<size_t>& ids,
    size_t num_training,
    std::mt19937& rng,
    Pool& threadpool
) {
    data::SimpleData<BuildType, Data::extent, Alloc> trainset(
        num_training, data.dimensions()
    );
    generate_unique_ids(ids, data.size(), rng);
    threads::parallel_for(
        threadpool,
        threads::StaticPartition{num_training},
        [&](auto indices, auto /*tid*/) {
            for (auto i : indices)
                trainset.set_datum(i, data.get_datum(ids[i]));
        }
    );
    return trainset;
}

/// @brief Initialize centroids randomly from the training set
template <typename BuildType, typename Data, threads::ThreadPool Pool>
data::SimpleData<BuildType> init_centroids(
    const Data& trainset,
    std::vector<size_t>& ids,
    size_t num_centroids,
    std::mt19937& rng,
    Pool& threadpool
) {
    data::SimpleData<BuildType> centroids(num_centroids, trainset.dimensions());
    generate_unique_ids(ids, trainset.size(), rng);
    threads::parallel_for(
        threadpool,
        threads::StaticPartition{num_centroids},
        [&](auto indices, auto) {
            for (auto i : indices)
                centroids.set_datum(i, trainset.get_datum(ids[i]));
        }
    );
    return centroids;
}

/// @brief Compute norms for L2 distance if needed
template <typename Distance, typename Data, threads::ThreadPool Pool>
std::vector<float> maybe_compute_norms(const Data& data, Pool& threadpool) {
    std::vector<float> norms;
    if constexpr (std::is_same_v<Distance, distance::DistanceL2>) {
        generate_norms(data, norms, threadpool);
    }
    return norms;
}

/// @brief Assign all points to clusters according to assignments
template <std::integral I = uint32_t, typename Data>
std::vector<std::vector<I>> group_assignments(
    const std::vector<size_t>& assignments, size_t num_clusters, const Data& data
) {
    std::vector<std::vector<I>> clusters(num_clusters);
    for (auto i : data.eachindex())
        clusters[assignments[i]].push_back(i);
    return clusters;
}

/// @brief Perform cluster assignment for data given pre-trained centroids
///
/// @tparam BuildType The numeric type used for matrix operations (float, Float16, BFloat16)
/// @tparam Data The dataset type
/// @tparam Centroids The centroids dataset type
/// @tparam Distance The distance metric type (DistanceIP or DistanceL2)
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
    if constexpr (std::is_same_v<Dist, distance::DistanceIP>) {
        for (size_t j = 0; j < num_threads; j++) {
            auto distance = matmul_results[j].get_datum(query_id);
            for (size_t k = 0; k < distance.size(); k++) {
                buffer.insert({count, distance[k]});
                count++;
            }
        }
    } else if constexpr (std::is_same_v<Dist, distance::DistanceL2>) {
        float query_norm = distance::norm_square(query);
        for (size_t j = 0; j < num_threads; j++) {
            auto distance = matmul_results[j].get_datum(query_id);
            for (size_t k = 0; k < distance.size(); k++) {
                float dist = query_norm + centroids_norm[count] - 2 * distance[k];
                buffer.insert({count, dist});
                count++;
            }
        }
    } else {
        throw ANNEXCEPTION("Only L2 and MIP distances supported in IVF search!");
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
