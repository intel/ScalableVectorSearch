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
#include <svs/runtime/api_defs.h>
#include <svs/runtime/training.h>

#include <cstddef>
#include <istream>
#include <ostream>

namespace svs {
namespace runtime {
namespace v0 {

// Abstract interface for IVF (Inverted File) indices.
struct SVS_RUNTIME_API IVFIndex {
    virtual ~IVFIndex();

    /// @brief Parameters for building an IVF index.
    struct BuildParams {
        /// The number of centroids/clusters to create
        size_t num_centroids = Unspecify<size_t>();
        /// Minibatch size for k-means clustering
        size_t minibatch_size = Unspecify<size_t>();
        /// Number of iterations for k-means clustering
        size_t num_iterations = Unspecify<size_t>();
        /// Whether to use hierarchical clustering
        OptionalBool is_hierarchical = Unspecify<bool>();
        /// Fraction of data to use for training (0.0 to 1.0)
        float training_fraction = Unspecify<float>();
        /// Number of level-1 clusters for hierarchical clustering
        size_t hierarchical_level1_clusters = Unspecify<size_t>();
        /// Random seed for clustering
        size_t seed = Unspecify<size_t>();
    };

    /// @brief Parameters for IVF search operations.
    struct SearchParams {
        /// The number of nearest clusters to be explored during search
        size_t n_probes = Unspecify<size_t>();
        /// Level of reordering/reranking done when using compressed datasets (multiplier)
        float k_reorder = Unspecify<float>();
    };

    /// @brief Perform k-NN search on the index.
    virtual Status search(
        size_t n,
        const float* x,
        size_t k,
        float* distances,
        size_t* labels,
        const SearchParams* params = nullptr
    ) const noexcept = 0;

    /// @brief Utility function to check storage kind support.
    static Status check_storage_kind(StorageKind storage_kind) noexcept;

    /// @brief Build an IVF index from data.
    static Status build(
        IVFIndex** index,
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        size_t n,
        const float* data,
        const BuildParams& params,
        const SearchParams& default_search_params,
        size_t num_threads = 0,
        size_t intra_query_threads = 1
    ) noexcept;

    /// @brief Build an IVF index from data (uses default search parameters).
    static Status build(
        IVFIndex** index,
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        size_t n,
        const float* data,
        const BuildParams& params,
        size_t num_threads = 0,
        size_t intra_query_threads = 1
    ) noexcept;

    /// @brief Destroy an IVF index.
    static Status destroy(IVFIndex* index) noexcept;

    /// @brief Save the index to a stream.
    virtual Status save(std::ostream& out) const noexcept = 0;

    /// @brief Set the number of threads used for index operations.
    virtual Status set_num_threads(size_t num_threads) noexcept = 0;

    /// @brief Get the number of threads used for index operations.
    virtual Status get_num_threads(size_t* num_threads) const noexcept = 0;

    /// @brief Load an IVF index from a stream.
    static Status load(
        IVFIndex** index,
        std::istream& in,
        MetricType metric,
        StorageKind storage_kind,
        size_t num_threads = 0,
        size_t intra_query_threads = 1
    ) noexcept;
};

/// @brief Specialization for building LeanVec-based IVF indices.
struct SVS_RUNTIME_API IVFIndexLeanVec : public IVFIndex {
    /// @brief Build a LeanVec IVF index with specified leanvec dimensions.
    static Status build(
        IVFIndex** index,
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        size_t n,
        const float* data,
        size_t leanvec_dims,
        const IVFIndex::BuildParams& params = {},
        const IVFIndex::SearchParams& default_search_params = {},
        size_t num_threads = 0,
        size_t intra_query_threads = 1
    ) noexcept;

    /// @brief Build a LeanVec IVF index with provided training data.
    static Status build(
        IVFIndex** index,
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        size_t n,
        const float* data,
        const LeanVecTrainingData* training_data,
        const IVFIndex::BuildParams& params = {},
        const IVFIndex::SearchParams& default_search_params = {},
        size_t num_threads = 0,
        size_t intra_query_threads = 1
    ) noexcept;
};

} // namespace v0
} // namespace runtime
} // namespace svs
