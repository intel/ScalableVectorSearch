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
    ///
    /// @param n Number of query vectors.
    /// @param x Pointer to query vectors (row-major, n x dimensions).
    /// @param k Number of nearest neighbors to find.
    /// @param distances Output array for distances (n x k).
    /// @param labels Output array for neighbor IDs (n x k).
    /// @param params Optional search parameters (uses defaults if nullptr).
    /// @return Status indicating success or error.
    virtual Status search(
        size_t n,
        const float* x,
        size_t k,
        float* distances,
        size_t* labels,
        const SearchParams* params = nullptr
    ) const noexcept = 0;
};

/// @brief Abstract interface for static IVF indices (read-only after construction).
struct SVS_RUNTIME_API StaticIVFIndex : public IVFIndex {
    /// @brief Utility function to check storage kind support.
    static Status check_storage_kind(StorageKind storage_kind) noexcept;

    /// @brief Build a static IVF index from data.
    ///
    /// @param index Output pointer to the created index.
    /// @param dim Dimensionality of vectors.
    /// @param metric Distance metric to use.
    /// @param storage_kind Storage type for the dataset.
    /// @param n Number of vectors in the dataset.
    /// @param data Pointer to vector data (row-major, n x dim).
    /// @param params Build parameters for clustering.
    /// @param default_search_params Default search parameters.
    /// @param num_threads Number of threads for building and searching.
    /// @param intra_query_threads Number of threads for intra-query parallelism.
    /// @return Status indicating success or error.
    static Status build(
        StaticIVFIndex** index,
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        size_t n,
        const float* data,
        const IVFIndex::BuildParams& params = {},
        const IVFIndex::SearchParams& default_search_params = {},
        size_t num_threads = 0,
        size_t intra_query_threads = 1
    ) noexcept;

    /// @brief Destroy a static IVF index.
    static Status destroy(StaticIVFIndex* index) noexcept;

    /// @brief Save the index to a stream.
    virtual Status save(std::ostream& out) const noexcept = 0;

    /// @brief Load a static IVF index from a stream.
    ///
    /// @param index Output pointer to the loaded index.
    /// @param in Input stream containing the serialized index.
    /// @param metric Distance metric to use.
    /// @param storage_kind Storage type for the dataset.
    /// @param num_threads Number of threads for searching.
    /// @param intra_query_threads Number of threads for intra-query parallelism.
    /// @return Status indicating success or error.
    static Status load(
        StaticIVFIndex** index,
        std::istream& in,
        MetricType metric,
        StorageKind storage_kind,
        size_t num_threads = 0,
        size_t intra_query_threads = 1
    ) noexcept;
};

/// @brief Abstract interface for dynamic IVF indices (supports add/delete).
struct SVS_RUNTIME_API DynamicIVFIndex : public IVFIndex {
    /// @brief Utility function to check storage kind support.
    static Status check_storage_kind(StorageKind storage_kind) noexcept;

    /// @brief Build a dynamic IVF index.
    ///
    /// @param index Output pointer to the created index.
    /// @param dim Dimensionality of vectors.
    /// @param metric Distance metric to use.
    /// @param storage_kind Storage type for the dataset.
    /// @param n Number of initial vectors (can be 0 for empty index).
    /// @param data Pointer to initial vector data (can be nullptr if n=0).
    /// @param labels Pointer to labels for initial vectors (can be nullptr if n=0).
    /// @param params Build parameters for clustering.
    /// @param default_search_params Default search parameters.
    /// @param num_threads Number of threads for operations.
    /// @param intra_query_threads Number of threads for intra-query parallelism.
    /// @return Status indicating success or error.
    static Status build(
        DynamicIVFIndex** index,
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        size_t n,
        const float* data,
        const size_t* labels,
        const IVFIndex::BuildParams& params = {},
        const IVFIndex::SearchParams& default_search_params = {},
        size_t num_threads = 0,
        size_t intra_query_threads = 1
    ) noexcept;

    /// @brief Destroy a dynamic IVF index.
    static Status destroy(DynamicIVFIndex* index) noexcept;

    /// @brief Add vectors to the index.
    ///
    /// @param n Number of vectors to add.
    /// @param labels Pointer to labels for the new vectors.
    /// @param x Pointer to vector data (row-major, n x dimensions).
    /// @param reuse_empty Whether to reuse empty slots from deleted vectors.
    /// @return Status indicating success or error.
    virtual Status
    add(size_t n, const size_t* labels, const float* x, bool reuse_empty = false
    ) noexcept = 0;

    /// @brief Remove vectors from the index by ID.
    ///
    /// @param n Number of vectors to remove.
    /// @param labels Pointer to labels of vectors to remove.
    /// @return Status indicating success or error.
    virtual Status remove(size_t n, const size_t* labels) noexcept = 0;

    /// @brief Remove vectors matching a selector.
    ///
    /// @param num_removed Output: number of vectors actually removed.
    /// @param selector Filter to determine which vectors to remove.
    /// @return Status indicating success or error.
    virtual Status
    remove_selected(size_t* num_removed, const IDFilter& selector) noexcept = 0;

    /// @brief Check if an ID exists in the index.
    ///
    /// @param exists Output: true if the ID exists.
    /// @param id The ID to check.
    /// @return Status indicating success or error.
    virtual Status has_id(bool* exists, size_t id) const noexcept = 0;

    /// @brief Consolidate the index (clean up deleted entries).
    virtual Status consolidate() noexcept = 0;

    /// @brief Compact the index (reclaim memory from deleted entries).
    ///
    /// @param batchsize Number of entries to process per batch.
    /// @return Status indicating success or error.
    virtual Status compact(size_t batchsize = 1'000'000) noexcept = 0;

    /// @brief Save the index to a stream.
    virtual Status save(std::ostream& out) const noexcept = 0;

    /// @brief Load a dynamic IVF index from a stream.
    ///
    /// @param index Output pointer to the loaded index.
    /// @param in Input stream containing the serialized index.
    /// @param metric Distance metric to use.
    /// @param storage_kind Storage type for the dataset.
    /// @param num_threads Number of threads for operations.
    /// @param intra_query_threads Number of threads for intra-query parallelism.
    /// @return Status indicating success or error.
    static Status load(
        DynamicIVFIndex** index,
        std::istream& in,
        MetricType metric,
        StorageKind storage_kind,
        size_t num_threads = 0,
        size_t intra_query_threads = 1
    ) noexcept;
};

} // namespace v0
} // namespace runtime
} // namespace svs
