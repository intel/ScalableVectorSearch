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

// Include the IVF index
#include "svs/index/ivf/clustering.h"
#include "svs/index/ivf/index.h"

// svs
#include "svs/concepts/distance.h"
#include "svs/core/data.h"
#include "svs/core/loading.h"
#include "svs/core/logging.h"
#include "svs/core/query_result.h"
#include "svs/core/translation.h"
#include "svs/lib/boundscheck.h"
#include "svs/lib/invoke.h"
#include "svs/lib/misc.h"
#include "svs/lib/threads.h"

// stdlib
#include <filesystem>
#include <memory>
#include <vector>

namespace svs::index::ivf {

///
/// Metadata tracking the state of a particular data index for DynamicIVFIndex.
/// The following states have the given meaning for their corresponding slot:
///
/// * Valid: Valid and present in the associated dataset.
/// * Empty: Available slot that can be used for new data or reclaimed after deletion.
///
enum class IVFSlotMetadata : uint8_t { Empty = 0x00, Valid = 0x01 };

///
/// @brief Dynamic cluster implementation using blocked data for resizeability
///
/// Similar to DenseCluster but uses BlockedData to support dynamic operations
///
template <typename Data, std::integral I> struct DynamicDenseCluster {
    using data_type = Data;
    using index_type = I;

    template <typename Callback>
    void on_leaves(Callback&& f, size_t prefetch_offset) const {
        size_t p = prefetch_offset;
        for (size_t i = 0; i < data_.size(); ++i) {
            if (p < data_.size()) {
                data_.prefetch(p);
                ++p;
            }
            f(accessor(data_, i), ids_[i], i);
        }
    }

    auto get_datum(size_t id) const { return data_.get_datum(id); }
    auto get_secondary(size_t id) const { return data_.get_secondary(id); }
    auto get_global_id(size_t local_id) const { return ids_[local_id]; }
    const Data& view_cluster() const { return data_; }
    Data& view_cluster() { return data_; }

    // Allow resizing for dynamic operations
    void resize(size_t new_size) {
        data_.resize(new_size);
        ids_.resize(new_size);
    }

    size_t size() const { return data_.size(); }
    size_t capacity() const { return data_.capacity(); }

  public:
    Data data_;
    std::vector<I> ids_;
};

///
/// @brief Dynamic IVF Index with insertion and deletion support
///
/// Uses the same cluster framework as static IVF (DenseClusteredDataset pattern)
/// but with BlockedData allocators for resizeability.
///
/// @tparam Centroids The type of centroid storage
/// @tparam Cluster Type representing cluster storage (DynamicDenseCluster with BlockedData)
/// @tparam Dist The distance functor used to compare queries with the elements
/// @tparam ThreadPoolProto Thread pool prototype type
///
/// An IVF index implementation that supports dynamic insertion and deletion of vectors
/// while maintaining the inverted file structure for efficient similarity search.
///
template <typename Centroids, typename Cluster, typename Dist, typename ThreadPoolProto>
class DynamicIVFIndex {
  public:
    // Traits
    static constexpr bool supports_insertions = true;
    static constexpr bool supports_deletions = true;
    static constexpr bool supports_saving = true;
    static constexpr bool needs_id_translation = true;

    // Type Aliases
    using Idx = typename Cluster::index_type;
    using Data = typename Cluster::data_type;
    using internal_id_type = size_t;
    using external_id_type = size_t;
    using distance_type = Dist;
    using centroids_type = Centroids;
    using cluster_type = Cluster;
    using search_parameters_type = IVFSearchParameters;
    using compare = distance::compare_t<Dist>;

    // Thread-related type aliases
    using InterQueryThreadPool = threads::ThreadPoolHandle;
    using IntraQueryThreadPool = threads::DefaultThreadPool;

  private:
    // Core IVF components (same structure as static IVF)
    centroids_type centroids_;
    std::vector<cluster_type> clusters_; // Each cluster contains data_ and ids_

    // Metadata tracking for dynamic operations
    std::vector<IVFSlotMetadata> status_; // Status of each global slot
    std::vector<size_t> id_to_cluster_;   // Maps global ID to cluster index
    std::vector<size_t> id_in_cluster_;   // Maps global ID to position in cluster
    size_t first_empty_ = 0;
    size_t prefetch_offset_ = 8;

    // Translation and distance
    IDTranslator translator_;
    distance_type distance_;

    // Threading infrastructure (same as static IVF)
    InterQueryThreadPool inter_query_threadpool_;
    const size_t intra_query_thread_count_;
    std::vector<IntraQueryThreadPool> intra_query_threadpools_;

    // Search infrastructure (same as static IVF)
    std::vector<data::SimpleData<float>> matmul_results_;
    std::vector<float> centroids_norm_;
    search_parameters_type search_parameters_{};

    // Logger
    svs::logging::logger_ptr logger_;

  public:
    /// @brief Construct a Dynamic IVF Index from clusters
    ///
    /// @param centroids Centroid collection for space partitioning
    /// @param clusters Vector of cluster data structures (each with data_ and ids_)
    /// @param external_ids External IDs for all vectors
    /// @param distance_function Distance metric for similarity computation
    /// @param threadpool_proto Primary thread pool prototype
    /// @param intra_query_thread_count Number of threads for intra-query parallelism
    /// @param logger Logger for per-index logging customization
    template <typename ExternalIds, typename TP>
    DynamicIVFIndex(
        centroids_type centroids,
        std::vector<cluster_type> clusters,
        const ExternalIds& external_ids,
        Dist distance_function,
        TP threadpool_proto,
        const size_t intra_query_thread_count = 1,
        svs::logging::logger_ptr logger = svs::logging::get()
    )
        : centroids_{std::move(centroids)}
        , clusters_{std::move(clusters)}
        , status_()
        , id_to_cluster_()
        , id_in_cluster_()
        , first_empty_{0}
        , prefetch_offset_{8}
        , translator_()
        , distance_{std::move(distance_function)}
        , inter_query_threadpool_{threads::as_threadpool(std::move(threadpool_proto))}
        , intra_query_thread_count_{intra_query_thread_count}
        , logger_{std::move(logger)} {
        // Initialize metadata structures
        size_t total_size = 0;
        for (const auto& cluster : clusters_) {
            for (size_t pos = 0; pos < cluster.ids_.size(); ++pos) {
                total_size =
                    std::max(total_size, static_cast<size_t>(cluster.ids_[pos]) + 1);
            }
        }

        status_.resize(total_size, IVFSlotMetadata::Valid);
        id_to_cluster_.resize(total_size);
        id_in_cluster_.resize(total_size);
        first_empty_ = total_size;

        // Build reverse mapping from global ID to cluster location
        for (size_t cluster_idx = 0; cluster_idx < clusters_.size(); ++cluster_idx) {
            const auto& cluster = clusters_[cluster_idx];
            for (size_t pos = 0; pos < cluster.ids_.size(); ++pos) {
                Idx global_id = cluster.ids_[pos];
                id_to_cluster_[global_id] = cluster_idx;
                id_in_cluster_[global_id] = pos;
            }
        }

        // Initialize ID translation
        translator_.insert(
            external_ids, threads::UnitRange<size_t>(0, external_ids.size())
        );

        // Initialize thread pools and search infrastructure
        validate_thread_configuration();
        initialize_thread_pools();
        initialize_search_buffers();
        initialize_distance_metadata();
    }

    /// @brief Constructor with pre-existing translator (for loading from saved state)
    template <typename TP>
    DynamicIVFIndex(
        centroids_type centroids,
        std::vector<cluster_type> clusters,
        IDTranslator translator,
        Dist distance_function,
        TP threadpool_proto,
        const size_t intra_query_thread_count = 1,
        svs::logging::logger_ptr logger = svs::logging::get()
    )
        : centroids_{std::move(centroids)}
        , clusters_{std::move(clusters)}
        , status_()
        , id_to_cluster_()
        , id_in_cluster_()
        , first_empty_{0}
        , prefetch_offset_{8}
        , translator_{std::move(translator)}
        , distance_{std::move(distance_function)}
        , inter_query_threadpool_{threads::as_threadpool(std::move(threadpool_proto))}
        , intra_query_thread_count_{intra_query_thread_count}
        , logger_{std::move(logger)} {
        // Initialize metadata structures based on cluster contents
        size_t total_size = 0;
        for (const auto& cluster : clusters_) {
            for (size_t pos = 0; pos < cluster.ids_.size(); ++pos) {
                total_size =
                    std::max(total_size, static_cast<size_t>(cluster.ids_[pos]) + 1);
            }
        }

        status_.resize(total_size, IVFSlotMetadata::Valid);
        id_to_cluster_.resize(total_size);
        id_in_cluster_.resize(total_size);
        first_empty_ = total_size;

        // Build reverse mapping from global ID to cluster location
        for (size_t cluster_idx = 0; cluster_idx < clusters_.size(); ++cluster_idx) {
            const auto& cluster = clusters_[cluster_idx];
            for (size_t pos = 0; pos < cluster.ids_.size(); ++pos) {
                Idx global_id = cluster.ids_[pos];
                id_to_cluster_[global_id] = cluster_idx;
                id_in_cluster_[global_id] = pos;
            }
        }

        // Initialize thread pools and search infrastructure
        validate_thread_configuration();
        initialize_thread_pools();
        initialize_search_buffers();
        initialize_distance_metadata();
    }

    ///// Basic Properties /////

    /// @brief Get logger
    svs::logging::logger_ptr get_logger() const { return logger_; }

    /// @brief Return the number of valid entries in the index
    size_t size() const { return translator_.size(); }

    /// @brief Return the number of centroids/clusters
    size_t num_clusters() const { return centroids_.size(); }

    /// @brief Return the logical number of dimensions
    size_t dimensions() const { return centroids_.dimensions(); }

    /// @brief Get index name
    std::string name() const { return "Dynamic IVF Index"; }

    ///// Search Parameters /////

    /// @brief Get current search parameters
    search_parameters_type get_search_parameters() const { return search_parameters_; }

    /// @brief Set search parameters
    void set_search_parameters(const search_parameters_type& params) {
        search_parameters_ = params;
    }

    ///// Threading Configuration /////

    /// @brief Get number of threads for inter-query parallelism
    size_t get_num_threads() const { return inter_query_threadpool_.size(); }

    /// @brief Get number of threads for intra-query parallelism
    size_t get_num_intra_query_threads() const { return intra_query_thread_count_; }

    /// @brief Set threadpool for inter-query parallelism
    void set_threadpool(InterQueryThreadPool threadpool) {
        if (threadpool.size() != inter_query_threadpool_.size()) {
            throw std::runtime_error(
                "Threadpool change not supported - thread count must remain constant"
            );
        }
        inter_query_threadpool_ = std::move(threadpool);
    }

    /// @brief Get threadpool handle
    InterQueryThreadPool& get_threadpool_handle() { return inter_query_threadpool_; }

    /// @brief Get const threadpool handle
    const InterQueryThreadPool& get_threadpool_handle() const {
        return inter_query_threadpool_;
    }

    ///// Index Translation /////

    /// @brief Translate external ID to internal ID
    size_t translate_external_id(size_t e) const { return translator_.get_internal(e); }

    /// @brief Translate internal ID to external ID
    size_t translate_internal_id(size_t i) const { return translator_.get_external(i); }

    /// @brief Check whether external ID exists
    bool has_id(size_t e) const { return translator_.has_external(e); }

    /// @brief Get the raw data for external id
    auto get_datum(size_t e) const {
        size_t internal_id = translate_external_id(e);
        size_t cluster_idx = id_to_cluster_[internal_id];
        size_t pos = id_in_cluster_[internal_id];
        return clusters_[cluster_idx].get_datum(pos);
    }

    /// @brief Iterate over all external IDs
    template <typename F> void on_ids(F&& f) const {
        for (size_t i = 0; i < status_.size(); ++i) {
            if (is_valid(i)) {
                f(translator_.get_external(i));
            }
        }
    }

    /// @brief Get external IDs (compatibility method)
    auto external_ids() const {
        std::vector<size_t> ids;
        ids.reserve(size());
        on_ids([&ids](size_t id) { ids.push_back(id); });
        return ids;
    }

    ///// Insertion /////

    /// @brief Add points to the index
    ///
    /// New points are assigned to clusters based on nearest centroid.
    /// Empty slots from previous deletions can be reused if reuse_empty is enabled.
    ///
    /// @param points Dataset of points to add
    /// @param external_ids External IDs for the points
    /// @param reuse_empty Whether to reuse empty slots from deletions
    /// @return Vector of internal IDs where points were inserted
    template <typename Points, class ExternalIds>
    std::vector<size_t> add_points(
        const Points& points, const ExternalIds& external_ids, bool reuse_empty = false
    ) {
        const size_t num_points = points.size();
        const size_t num_ids = external_ids.size();

        if (num_points != num_ids) {
            throw ANNEXCEPTION(
                "Number of points ({}) not equal to number of external ids ({})!",
                num_points,
                num_ids
            );
        }

        // Assign each point to its nearest centroid
        std::vector<size_t> assigned_clusters(num_points);
        assign_to_clusters(points, assigned_clusters);

        // Allocate global IDs
        std::vector<size_t> global_ids = allocate_ids(num_points, reuse_empty);

        // Try to update ID translation
        translator_.insert(external_ids, global_ids);

        // Insert points into their assigned clusters
        insert_into_clusters(points, global_ids, assigned_clusters);

        return global_ids;
    }

    ///// Deletion /////

    /// @brief Delete entries by external ID
    ///
    /// Entries are marked as Empty and can be reused immediately.
    /// Call compact() periodically to reclaim memory and reorganize clusters.
    ///
    /// @param ids Container of external IDs to delete
    /// @return Number of entries deleted
    template <typename T> size_t delete_entries(const T& ids) {
        translator_.check_external_exist(ids.begin(), ids.end());

        for (auto external_id : ids) {
            size_t internal_id = translator_.get_internal(external_id);
            assert(internal_id < status_.size());
            assert(status_[internal_id] == IVFSlotMetadata::Valid);
            status_[internal_id] = IVFSlotMetadata::Empty;
            first_empty_ = std::min(first_empty_, internal_id);
        }

        translator_.delete_external(ids);
        return ids.size();
    }

    ///// Compaction /////

    /// @brief Compact the data structure
    ///
    /// Compact removes all empty slots, rebuilding the index structure
    /// for optimal memory usage and search performance.
    ///
    /// @param batch_size Granularity at which points are shuffled (unused for IVF)
    void compact(size_t SVS_UNUSED(batch_size) = 1'000) {
        // Collect all valid indices
        auto valid_indices = nonmissing_indices();

        // Rebuild clusters compactly, removing empty slots
        rebuild_clusters_compact(valid_indices);

        // Update metadata
        size_t new_size = valid_indices.size();
        status_.resize(new_size);
        // After compaction, all retained entries are valid
        std::fill(status_.begin(), status_.end(), IVFSlotMetadata::Valid);
        id_to_cluster_.resize(new_size);
        id_in_cluster_.resize(new_size);
        first_empty_ = new_size;

        svs::logging::info(logger_, "Compaction complete: {} valid entries", new_size);
    }

    ///// Search /////

    /// @brief Perform similarity search
    ///
    /// Search process:
    /// 1. Find n_probe nearest centroids for each query
    /// 2. Search within those clusters, skipping empty entries
    /// 3. Return top-k neighbors
    ///
    /// @param results View for storing search results
    /// @param queries Query vectors
    /// @param search_parameters Search configuration
    /// @param cancel Optional cancellation predicate
    template <data::ImmutableMemoryDataset Queries>
    void search(
        QueryResultView<size_t> results,
        const Queries& queries,
        const search_parameters_type& search_parameters,
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
    ) {
        validate_query_batch_size(queries.size());

        size_t num_neighbors = results.n_neighbors();
        size_t buffer_leaves_size = search_parameters.k_reorder_ * num_neighbors;

        // Phase 1: Compute distances to centroids
        compute_centroid_distances(
            queries, centroids_, matmul_results_, inter_query_threadpool_
        );

        // Phase 2: Process queries in parallel
        threads::parallel_for(
            inter_query_threadpool_,
            threads::StaticPartition(queries.size()),
            [&](auto is, auto tid) {
                // Check for cancellation
                if (cancel()) {
                    return;
                }

                // Create buffers for this thread
                auto buffer_centroids = create_centroid_buffer(search_parameters.n_probes_);
                auto buffer_leaves = create_leaf_buffers(buffer_leaves_size);

                // Search for each query
                for (auto query_idx : is) {
                    search_single_query(
                        queries,
                        query_idx,
                        results,
                        buffer_centroids,
                        buffer_leaves,
                        search_parameters,
                        tid
                    );
                }
            }
        );
    }

    ///// Saving /////

    static constexpr lib::Version save_version = lib::Version(0, 0, 0);

    void save(
        const std::filesystem::path& config_directory,
        const std::filesystem::path& data_directory
    ) {
        // Compact before saving to remove empty slots
        compact();

        // Save configuration
        lib::save_to_disk(
            lib::SaveOverride([&](const lib::SaveContext& ctx) {
                return lib::SaveTable(
                    "dynamic_ivf_config",
                    save_version,
                    {
                        {"name", lib::save(name())},
                        {"translation", lib::save(translator_, ctx)},
                        {"num_clusters", lib::save(clusters_.size())},
                    }
                );
            }),
            config_directory
        );

        // Save centroids and cluster data
        lib::save_to_disk(centroids_, data_directory / "centroids");

        for (size_t i = 0; i < clusters_.size(); ++i) {
            auto cluster_path = data_directory / fmt::format("cluster_{}", i);
            lib::save_to_disk(clusters_[i].data_, cluster_path);

            auto ids_path = data_directory / fmt::format("cluster_ids_{}", i);
            lib::save_to_disk(clusters_[i].ids_, ids_path);
        }
    }

  private:
    ///// Helper Methods /////

    void validate_thread_configuration() {
        if (intra_query_thread_count_ < 1) {
            throw std::invalid_argument("Intra-query thread count must be at least 1");
        }
    }

    void initialize_thread_pools() {
        for (size_t i = 0; i < inter_query_threadpool_.size(); i++) {
            intra_query_threadpools_.push_back(
                threads::as_threadpool(intra_query_thread_count_)
            );
        }
    }

    void initialize_search_buffers() {
        auto batches =
            std::vector<threads::UnitRange<uint64_t>>(inter_query_threadpool_.size());

        threads::parallel_for(
            inter_query_threadpool_,
            threads::StaticPartition(centroids_.size()),
            [&](auto is, auto tid) { batches[tid] = threads::UnitRange{is}; }
        );

        for (size_t i = 0; i < inter_query_threadpool_.size(); i++) {
            matmul_results_.emplace_back(MAX_QUERY_BATCH_SIZE, batches[i].size());
        }
    }

    void initialize_distance_metadata() {
        if constexpr (std::is_same_v<std::remove_cvref_t<Dist>, distance::DistanceL2>) {
            centroids_norm_.reserve(centroids_.size());
            for (size_t i = 0; i < centroids_.size(); ++i) {
                centroids_norm_.push_back(distance::norm_square(centroids_.get_datum(i)));
            }
        }
    }

    void validate_query_batch_size(size_t query_size) const {
        if (query_size > MAX_QUERY_BATCH_SIZE) {
            throw std::runtime_error(fmt::format(
                "Query batch size {} exceeds maximum allowed {}",
                query_size,
                MAX_QUERY_BATCH_SIZE
            ));
        }
    }

    auto create_centroid_buffer(size_t n_probes) const {
        return SortedBuffer<Idx, compare>(n_probes, distance::comparator(distance_));
    }

    auto create_leaf_buffers(size_t buffer_size) const {
        std::vector<SortedBuffer<Idx, compare>> buffers;
        buffers.reserve(intra_query_thread_count_);
        for (size_t j = 0; j < intra_query_thread_count_; j++) {
            buffers.push_back(
                SortedBuffer<Idx, compare>(buffer_size, distance::comparator(distance_))
            );
        }
        return buffers;
    }

    bool is_empty(size_t i) const { return status_[i] == IVFSlotMetadata::Empty; }

    bool is_valid(size_t i) const { return status_[i] == IVFSlotMetadata::Valid; }

    std::vector<size_t> nonmissing_indices() const {
        std::vector<size_t> indices;
        indices.reserve(size());
        for (size_t i = 0; i < status_.size(); ++i) {
            if (is_valid(i)) {
                indices.push_back(i);
            }
        }
        return indices;
    }

    template <typename Points>
    void assign_to_clusters(const Points& points, std::vector<size_t>& assignments) {
        // For each point, find nearest centroid
        for (size_t i = 0; i < points.size(); ++i) {
            auto point = points.get_datum(i);
            float min_dist = std::numeric_limits<float>::max();
            size_t best_cluster = 0;

            for (size_t c = 0; c < centroids_.size(); ++c) {
                auto centroid = centroids_.get_datum(c);
                float dist = distance::compute(distance_, point, centroid);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }

            assignments[i] = best_cluster;
        }
    }

    std::vector<size_t> allocate_ids(size_t count, bool reuse_empty) {
        std::vector<size_t> ids;
        ids.reserve(count);

        // Try to find empty slots if reuse is enabled
        if (reuse_empty) {
            for (size_t i = 0; i < status_.size() && ids.size() < count; ++i) {
                if (is_empty(i)) {
                    ids.push_back(i);
                    status_[i] = IVFSlotMetadata::Valid; // Mark as valid when reusing
                }
            }
        }

        // Allocate new slots as needed
        size_t current_size = status_.size();
        while (ids.size() < count) {
            ids.push_back(current_size++);
        }

        // Resize metadata if we added new slots
        if (current_size > status_.size()) {
            status_.resize(current_size, IVFSlotMetadata::Valid);
            id_to_cluster_.resize(current_size);
            id_in_cluster_.resize(current_size);
            first_empty_ = current_size;
        }

        return ids;
    }

    template <typename Points>
    void insert_into_clusters(
        const Points& points,
        const std::vector<size_t>& global_ids,
        const std::vector<size_t>& assigned_clusters
    ) {
        for (size_t i = 0; i < points.size(); ++i) {
            size_t global_id = global_ids[i];
            size_t cluster_idx = assigned_clusters[i];

            // Add to cluster
            auto& cluster = clusters_[cluster_idx];

            size_t pos = cluster.size();
            cluster.resize(cluster.size() + 1);
            cluster.data_.set_datum(pos, points.get_datum(i));
            cluster.ids_.push_back(static_cast<Idx>(global_id));

            // Update metadata
            status_[global_id] = IVFSlotMetadata::Valid;
            id_to_cluster_[global_id] = cluster_idx;
            id_in_cluster_[global_id] = pos;
        }
    }

    void rebuild_clusters_compact(const std::vector<size_t>& valid_indices) {
        // Group valid indices by cluster
        // cluster_valid_indices[cluster_idx] contains pairs of (new_id, old_id)
        std::vector<std::vector<std::pair<size_t, size_t>>> cluster_valid_indices(
            clusters_.size()
        );

        // Collect all mappings: (external_id, new_internal_id)
        // NOTE: This must be done BEFORE we modify the translator
        std::vector<Idx> external_ids;
        std::vector<size_t> new_internal_ids;
        external_ids.reserve(valid_indices.size());
        new_internal_ids.reserve(valid_indices.size());

        for (size_t new_id = 0; new_id < valid_indices.size(); ++new_id) {
            size_t old_id = valid_indices[new_id];
            size_t cluster_idx = id_to_cluster_[old_id];
            cluster_valid_indices[cluster_idx].push_back({new_id, old_id});

            // Save the external ID mapping for later
            auto external_id = translator_.get_external(old_id);
            external_ids.push_back(external_id);
            new_internal_ids.push_back(new_id);
        }

        // Phase 1: Clear the translator completely
        // This is simpler and safer than trying to selectively delete entries
        translator_ = IDTranslator();

        // Phase 2: Rebuild clusters and update metadata
        for (size_t cluster_idx = 0; cluster_idx < clusters_.size(); ++cluster_idx) {
            const auto& indices = cluster_valid_indices[cluster_idx];
            if (indices.empty()) {
                clusters_[cluster_idx].data_ =
                    Data(0, clusters_[cluster_idx].data_.dimensions());
                clusters_[cluster_idx].ids_.clear();
                continue;
            }

            Data new_data(indices.size(), clusters_[cluster_idx].data_.dimensions());
            std::vector<Idx> new_ids;
            new_ids.reserve(indices.size());

            for (size_t pos = 0; pos < indices.size(); ++pos) {
                auto [new_global_id, old_global_id] = indices[pos];
                size_t old_cluster = id_to_cluster_[old_global_id];
                size_t old_pos = id_in_cluster_[old_global_id];

                new_data.set_datum(pos, clusters_[old_cluster].data_.get_datum(old_pos));
                new_ids.push_back(static_cast<Idx>(new_global_id));

                // Update metadata
                id_to_cluster_[new_global_id] = cluster_idx;
                id_in_cluster_[new_global_id] = pos;
            }

            clusters_[cluster_idx].data_ = std::move(new_data);
            clusters_[cluster_idx].ids_ = std::move(new_ids);
        }

        // Phase 3: Re-add all IDs to the translator with their new internal IDs
        translator_.insert(external_ids, new_internal_ids, false);
    }

    template <typename Queries>
    void search_single_query(
        const Queries& queries,
        size_t query_idx,
        QueryResultView<size_t>& results,
        auto& buffer_centroids,
        auto& buffer_leaves,
        const search_parameters_type& search_parameters,
        size_t tid
    ) {
        // Find nearest centroids
        auto query = queries.get_datum(query_idx);
        search_centroids(
            query,
            distance_,
            matmul_results_,
            buffer_centroids,
            tid,
            centroids_norm_,
            get_num_threads()
        );

        // Search within selected clusters
        size_t n_probes = std::min(search_parameters.n_probes_, buffer_centroids.size());

        for (size_t probe_idx = 0; probe_idx < n_probes; ++probe_idx) {
            size_t cluster_idx = buffer_centroids[probe_idx].id();
            search_cluster(query, cluster_idx, buffer_leaves[0]);
        }

        // Write results (translating to external IDs)
        size_t num_neighbors = results.n_neighbors();
        for (size_t i = 0; i < std::min(num_neighbors, buffer_leaves[0].size()); ++i) {
            size_t internal_id = buffer_leaves[0][i].id();
            float dist = buffer_leaves[0][i].distance();
            size_t external_id = translate_internal_id(internal_id);

            results.set(Neighbor<size_t>(external_id, dist), query_idx, i);
        }

        // Fill remaining slots with invalid neighbors if needed
        for (size_t i = buffer_leaves[0].size(); i < num_neighbors; ++i) {
            results.set(
                Neighbor<size_t>(
                    std::numeric_limits<size_t>::max(), std::numeric_limits<float>::max()
                ),
                query_idx,
                i
            );
        }
    }

    template <typename Query>
    void search_cluster(const Query& query, size_t cluster_idx, auto& buffer) {
        const auto& cluster = clusters_[cluster_idx];

        for (size_t pos = 0; pos < cluster.size(); ++pos) {
            Idx global_id = cluster.ids_[pos];

            // Skip empty entries
            if (!is_valid(global_id)) {
                continue;
            }

            auto datum = cluster.data_.get_datum(pos);
            float dist = distance::compute(distance_, query, datum);
            buffer.insert({global_id, dist});
        }
    }
};

} // namespace svs::index::ivf
