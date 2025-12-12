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

// Include the IVF index and clustering
#include "svs/index/ivf/clustering.h"
#include "svs/index/ivf/index.h"

// svs
#include "svs/concepts/distance.h"
#include "svs/core/logging.h"
#include "svs/core/query_result.h"
#include "svs/core/translation.h"
#include "svs/lib/misc.h"
#include "svs/lib/threads.h"

// stdlib
#include <filesystem>
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
/// @brief Dynamic IVF Index with insertion and deletion support
///
/// @tparam Centroids The type of centroid storage
/// @tparam Cluster Type representing cluster storage (DenseCluster with BlockedData)
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
    Cluster clusters_; // Cluster container

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
    search_parameters_type search_parameters_;

    // Logger
    svs::logging::logger_ptr logger_;

  public:
    /// @brief Construct a new Dynamic IVF Index
    ///
    /// @param centroids Centroid collection for space partitioning
    /// @param clusters Cluster container
    /// @param external_ids External IDs for all vectors
    /// @param distance_function Distance metric for similarity computation
    /// @param threadpool_proto Primary thread pool prototype
    /// @param intra_query_thread_count Number of threads for intra-query parallelism
    /// @param logger Logger for per-index logging customization
    template <typename ExternalIds, typename TP>
    DynamicIVFIndex(
        centroids_type centroids,
        Cluster clusters,
        const ExternalIds& external_ids,
        Dist distance_function,
        TP threadpool_proto,
        const size_t intra_query_thread_count = 1,
        svs::logging::logger_ptr logger = svs::logging::get()
    )
        : centroids_{std::move(centroids)}
        , clusters_{std::move(clusters)}
        , first_empty_{0}
        , prefetch_offset_{8}
        , distance_{std::move(distance_function)}
        , inter_query_threadpool_{threads::as_threadpool(std::move(threadpool_proto))}
        , intra_query_thread_count_{intra_query_thread_count}
        , logger_{std::move(logger)} {
        // Initialize metadata structures
        size_t total_size = 0;
        for (size_t cluster_idx = 0; cluster_idx < clusters_.size(); ++cluster_idx) {
            const auto& cluster = clusters_[cluster_idx];
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
        Cluster clusters,
        IDTranslator translator,
        Dist distance_function,
        TP threadpool_proto,
        const size_t intra_query_thread_count = 1,
        svs::logging::logger_ptr logger = svs::logging::get()
    )
        : centroids_{std::move(centroids)}
        , clusters_{std::move(clusters)}
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

    /// @brief Get raw data by cluster and local position (for extension compatibility)
    auto get_datum(size_t cluster_idx, size_t local_pos) const {
        return clusters_[cluster_idx].get_datum(local_pos);
    }

    /// @brief Get secondary data by cluster and local position (for LeanVec)
    auto get_secondary(size_t cluster_idx, size_t local_pos) const {
        return clusters_[cluster_idx].data_.get_secondary(local_pos);
    }

    ///// Distance

    /// @brief Compute the distance between an external vector and a vector in the index.
    template <typename Query> double get_distance(size_t id, const Query& query) const {
        // Check if id exists
        if (!has_id(id)) {
            throw ANNEXCEPTION("ID {} does not exist in the index!", id);
        }

        // Verify dimensions match
        const size_t query_size = query.size();
        const size_t index_vector_size = dimensions();
        if (query_size != index_vector_size) {
            throw ANNEXCEPTION(
                "Incompatible dimensions. Query has {} while the index expects {}.",
                query_size,
                index_vector_size
            );
        }

        // Translate external ID to internal ID and get cluster location
        size_t internal_id = translate_external_id(id);
        size_t cluster_idx = id_to_cluster_[internal_id];
        size_t pos = id_in_cluster_[internal_id];

        // Call extension for distance computation
        return svs::index::ivf::extensions::get_distance_ext(
            clusters_, distance_, cluster_idx, pos, query
        );
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

    /// @brief Consolidate the data structure (no-op for IVF).
    ///
    /// In the IVF index implementation, deletion marks entries as Empty in metadata,
    /// making them invalid for searches. These empty slots can be reused by add_points.
    /// This method is a no-op for compatibility with the dynamic index interface.
    ///
    void consolidate() {
        // No-op: Deleted entries are marked Empty and excluded from searches
    }

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

    /// Translate internal IDs to external IDs in search results.
    /// This method converts all IDs in the result view from internal (global) IDs
    /// to external IDs using the ID map.
    ///
    /// @param ids Result indices to translate (2D array)
    template <class Dims, class Base>
        requires(std::tuple_size_v<Dims> == 2)
    void translate_to_external(DenseArray<size_t, Dims, Base>& ids) {
        threads::parallel_for(
            inter_query_threadpool_,
            threads::StaticPartition{getsize<0>(ids)},
            [&](const auto is, auto /*tid*/) {
                for (auto i : is) {
                    for (size_t j = 0, jmax = getsize<1>(ids); j < jmax; ++j) {
                        auto internal = lib::narrow_cast<Idx>(ids.at(i, j));
                        ids.at(i, j) = translate_internal_id(internal);
                    }
                }
            }
        );
    }

    /// @brief Perform similarity search
    ///
    /// Search Process:
    /// 1. Inter-query parallel: Distribute queries across primary threads
    /// 2. For each query: Find n_probe nearest centroids
    /// 3. Intra-query parallel: Explore identified clusters using inner threads
    /// 4. Combine results from all explored clusters (skipping empty entries)
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
        const lib::DefaultPredicate& SVS_UNUSED(cancel) = lib::Returns(lib::Const<false>())
    ) {
        validate_query_batch_size(queries.size());

        size_t num_neighbors = results.n_neighbors();
        size_t buffer_leaves_size = static_cast<size_t>(
            search_parameters.k_reorder_ * static_cast<float>(num_neighbors)
        );

        // Phase 1: Inter-query parallel - Compute distances to centroids
        compute_centroid_distances(
            queries, centroids_, matmul_results_, inter_query_threadpool_
        );

        // Phase 2: Process queries in parallel
        threads::parallel_for(
            inter_query_threadpool_,
            threads::StaticPartition(queries.size()),
            [&](auto is, auto tid) {
                // Initialize search buffers
                auto buffer_centroids = create_centroid_buffer(search_parameters.n_probes_);
                auto buffer_leaves = create_leaf_buffers(buffer_leaves_size);

                // Prepare cluster search scratch space (distance copy)
                // Pass cluster data (not centroids) to support quantized datasets
                auto scratch = extensions::per_thread_batch_search_setup(
                    clusters_[0].data_, distance_
                );

                // Execute search with intra-query parallelism
                // Pass cluster data as first parameter to enable dataset-specific overrides
                extensions::per_thread_batch_search(
                    clusters_[0].data_,
                    *this,
                    buffer_centroids,
                    buffer_leaves,
                    scratch,
                    queries,
                    results,
                    threads::UnitRange{is},
                    tid,
                    search_centroids_closure(),
                    search_leaves_closure()
                );
            }
        );

        // Convert internal IDs to external IDs
        this->translate_to_external(results.indices());
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

    /// @brief Assign points to their nearest centroids using parallel processing
    ///
    /// Uses centroid_assignment with batching to handle matmul_results size constraints.
    /// Processes points in batches for efficient parallel centroid assignment.
    ///
    /// @param points Dataset to assign to clusters
    /// @param assignments Output vector for cluster assignments
    template <typename Points>
    void assign_to_clusters(const Points& points, std::vector<size_t>& assignments) {
        size_t num_points = points.size();
        size_t num_centroids = centroids_.size();

        // Compute norms if using L2 distance
        auto data_norm = maybe_compute_norms<Dist>(points, inter_query_threadpool_);

        // Determine batch size based on matmul_results capacity
        // matmul_results_ is sized for queries, reuse for point assignment
        size_t batch_size = matmul_results_[0].size(); // Number of queries it can hold
        size_t num_batches = lib::div_round_up(num_points, batch_size);

        // Create a local matmul buffer for assignments (batch_size x num_centroids)
        auto matmul_buffer = data::SimpleData<float>{batch_size, num_centroids};
        auto timer = lib::Timer();

        // Process points in batches
        for (size_t batch = 0; batch < num_batches; ++batch) {
            auto batch_range = threads::UnitRange{
                batch * batch_size, std::min((batch + 1) * batch_size, num_points)};

            // Use centroid_assignment to compute assignments for this batch
            centroid_assignment(
                const_cast<Points&>(points), // centroid_assignment expects non-const
                data_norm,
                batch_range,
                distance_,
                centroids_,
                centroids_norm_,
                assignments,
                matmul_buffer,
                inter_query_threadpool_,
                timer
            );
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

    ///// Search Closures /////

    /// @brief Create closure for searching centroids
    auto search_centroids_closure() const {
        return [this](const auto& query, auto& buffer_centroids, size_t query_idx) {
            search_centroids(
                query,
                distance_,
                matmul_results_,
                buffer_centroids,
                query_idx,
                centroids_norm_,
                get_num_threads()
            );
        };
    }

    /// @brief Create closure for searching clusters/leaves
    auto search_leaves_closure() {
        return [this](
                   const auto& query,
                   auto& distance,
                   const auto& buffer_centroids,
                   auto& buffer_leaves,
                   size_t tid
               ) {
            // Use the common search_leaves function
            search_leaves(
                query,
                distance,
                *this,
                buffer_centroids,
                buffer_leaves,
                intra_query_threadpools_[tid]
            );
        };
    }

  public:
    /// @brief Cluster accessor interface for search_leaves
    /// This method provides filtered access to cluster leaves, skipping empty entries
    ///
    /// Note: For DynamicIVFIndex, we pass the global_id as the local_id (3rd parameter)
    /// because the common search_leaves function will combine it with cluster_id to get
    /// the final ID. Our get_global_id() just returns the local_id unchanged.
    template <typename Callback> void on_leaves(Callback&& f, size_t cluster_id) const {
        const auto& cluster = clusters_[cluster_id];
        for (size_t i = 0; i < cluster.size(); ++i) {
            Idx global_id = cluster.ids_[i];

            // Skip empty entries
            if (!is_valid(global_id)) {
                continue;
            }

            auto datum = cluster.data_.get_datum(i);
            // Pass global_id as the local_id (3rd param) since get_global_id returns it
            // unchanged
            f(datum, 0 /* unused gid */, global_id);
        }
    }

    /// @brief Get global ID for a point (identity function for dynamic IVF)
    size_t get_global_id(size_t /*cluster_id*/, size_t local_id) const { return local_id; }
};

///
/// @brief Build a DynamicIVFIndex from clustering and data
///
template <
    typename Centroids,
    data::ImmutableMemoryDataset SourceData,
    typename Distance,
    typename ThreadPoolProto>
auto build_dynamic_ivf(
    Centroids centroids,
    const index::ivf::Clustering<Centroids, uint32_t>& clustering,
    const SourceData& source_data,
    std::span<const size_t> ids,
    Distance distance,
    ThreadPoolProto threadpool_proto,
    const size_t intra_query_thread_count = 1
) {
    using I = uint32_t;
    using ElementType = typename SourceData::element_type;
    // Use BlockedData with default lib::Allocator for dynamic operations
    using BlockedDataType =
        data::SimpleData<ElementType, Dynamic, data::Blocked<lib::Allocator<ElementType>>>;

    auto threadpool = threads::as_threadpool(std::move(threadpool_proto));

    // Use a small block size for IVF clusters (1MB instead of 1GB default)
    // With many clusters, large blocks cause excessive memory usage
    auto blocking_params = data::BlockingParameters{
        .blocksize_bytes = lib::PowerOfTwo(20) // 2^20 = 1MB
    };
    auto blocked_allocator = data::Blocked<lib::Allocator<ElementType>>(
        blocking_params, lib::Allocator<ElementType>()
    );

    // Use DenseClusteredDataset to create clusters, just like static IVF
    auto dense_clusters = DenseClusteredDataset<Centroids, I, BlockedDataType>(
        clustering, source_data, threadpool, blocked_allocator
    );

    // Create the index
    return DynamicIVFIndex<
        Centroids,
        decltype(dense_clusters),
        Distance,
        decltype(threadpool)>(
        std::move(centroids),
        std::move(dense_clusters),
        ids,
        std::move(distance),
        std::move(threadpool),
        intra_query_thread_count
    );
}

} // namespace svs::index::ivf
