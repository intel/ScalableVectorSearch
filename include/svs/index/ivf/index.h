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
#include "svs/core/loading.h"
#include "svs/core/query_result.h"
#include "svs/index/ivf/clustering.h"
#include "svs/index/ivf/extensions.h"
#include "svs/index/ivf/hierarchical_kmeans.h"
#include "svs/index/ivf/kmeans.h"
#include "svs/lib/timing.h"

// format
#include "fmt/core.h"

// stl
#include <memory>
#include <mutex>
#include <random>
#include <vector>

namespace svs::index::ivf {

// Forward declaration of BatchIterator
template <typename Index, typename QueryType> class BatchIterator;

// The maximum batch size for queries is set to 10,000 to balance memory usage and
// performance. This value was chosen based on empirical testing to avoid excessive memory
// allocation while supporting large batch operations typical in high-throughput
// environments.
constexpr size_t MAX_QUERY_BATCH_SIZE = 10000;

///
/// @brief Search scratchspace used by the IVF index.
///
/// These can be pre-allocated and passed to the index when performing externally
/// threaded searches to reduce allocations.
///
/// **NOTE**: The members ``buffer_centroids``, ``buffer_leaves``, and ``scratch`` are part
/// of the public API for this class. Users are free to access and manipulate these objects.
/// However, doing so incorrectly can yield undefined-behavior.
///
template <typename BufferCentroids, typename BufferLeaves, typename Scratch>
struct IVFScratchspace {
  public:
    // Members
    BufferCentroids buffer_centroids;
    BufferLeaves buffer_leaves;
    Scratch scratch;

    // Constructors
    IVFScratchspace(
        BufferCentroids buffer_centroids_, BufferLeaves buffer_leaves_, Scratch scratch_
    )
        : buffer_centroids{std::move(buffer_centroids_)}
        , buffer_leaves{std::move(buffer_leaves_)}
        , scratch{std::move(scratch_)} {}
};

/// @brief IVF (Inverted File) Index implementation for efficient similarity search
///
/// This class implements an IVF index structure that partitions the search space using
/// centroids and supports a two-level hierarchical threading model for parallel search
/// operations:
///
/// Threading Model Details:
/// ----------------------
/// 1. Inter-Query Parallelism (Outer Threading):
///    - Distributes query batches across multiple threads
///    - Each thread processes its assigned queries independently
///    - Used for initial centroid search to find n_probe nearest centroids
///    - Managed by inter_query_threadpool_
///
/// 2. Intra-Query Parallelism (Inner Threading):
///    - Activated after finding n_probe nearest centroids for a query
///    - Parallelizes cluster exploration for each query
///    - Multiple threads explore different clusters of the same query concurrently
///    - Each outer thread has its own pool of inner threads (intra_query_threadpools_)
///    - Enables parallel processing of leaf nodes within identified clusters
///
/// Search Process Flow:
/// ------------------
/// 1. Queries are distributed across outer threads for initial processing
/// 2. Each outer thread finds n_probe nearest centroids for its assigned queries
/// 3. For each query, its n_probe clusters are distributed across inner threads
/// 4. Inner threads concurrently explore assigned clusters to find nearest neighbors
///
/// @tparam Centroids Type representing centroid storage
/// @tparam Cluster Type representing cluster storage
/// @tparam Dist Distance metric type
/// @tparam ThreadPoolProto Thread pool prototype type
///
template <typename Centroids, typename Cluster, typename Dist, typename ThreadPoolProto>
class IVFIndex {
  public:
    using Idx = typename Cluster::index_type;
    using Data = typename Cluster::data_type;
    using search_parameters_type = IVFSearchParameters;

    // Thread-related type aliases for clarity
    using InterQueryThreadPool = threads::ThreadPoolHandle;  // For inter-query parallelism
    using IntraQueryThreadPool = threads::DefaultThreadPool; // For intra-query parallelism

    // Scratchspace type for external threading
    using buffer_centroids_type = SortedBuffer<Idx, distance::compare_t<Dist>>;
    using buffer_leaves_type = std::vector<SortedBuffer<Idx, distance::compare_t<Dist>>>;
    using inner_scratch_type =
        svs::tag_t<extensions::per_thread_batch_search_setup>::result_t<Data, Dist>;
    using scratchspace_type =
        IVFScratchspace<buffer_centroids_type, buffer_leaves_type, inner_scratch_type>;

    /// @brief Construct a new IVF Index
    ///
    /// @param centroids Collection of centroids for space partitioning
    /// @param cluster Cluster storage implementation
    /// @param distance_function Distance metric for similarity computation
    /// @param threadpool_proto Primary thread pool prototype for inter-query parallelism
    /// @param intra_query_thread_count Number of threads for intra-query parallelism
    ///     (default: 1)
    /// @param logger logger for per-index logging customization.
    /// @throws std::invalid_argument if thread configuration is invalid
    IVFIndex(
        Centroids centroids,
        Cluster cluster,
        Dist distance_function,
        ThreadPoolProto threadpool_proto,
        const size_t intra_query_thread_count = 1,
        svs::logging::logger_ptr logger = svs::logging::get()
    )
        : centroids_{std::move(centroids)}
        , cluster_{std::move(cluster)}
        , cluster0_{cluster_.view_cluster(0)}
        , distance_{std::move(distance_function)}
        , inter_query_threadpool_{threads::as_threadpool(std::move(threadpool_proto))}
        , intra_query_thread_count_{intra_query_thread_count}
        , logger_{std::move(logger)} {
        validate_thread_configuration();
        initialize_thread_pools();
        initialize_search_buffers();
        initialize_distance_metadata();
    }

    ///// Index Information /////

    /// @brief Indicates whether internal IDs need translation to external IDs
    static constexpr bool needs_id_translation = false;

    /// @brief Get the total number of vectors in the index
    size_t size() const {
        size_t total = 0;
        for (size_t i = 0; i < centroids_.size(); ++i) {
            total += cluster_.view_cluster(i).size();
        }
        return total;
    }

    /// @brief Get the number of clusters/centroids in the index
    size_t num_clusters() const { return centroids_.size(); }

    /// @brief Get the dimensionality of the indexed vectors
    size_t dimensions() const { return centroids_.dimensions(); }

    /// @brief Get the index type name
    std::string name() const { return "IVFIndex"; }

    /// @brief Getter method for logger
    svs::logging::logger_ptr get_logger() const { return logger_; }

    ///// Threading Configuration /////

    /// @brief Indicates if the number of threads can be changed at runtime
    static constexpr bool can_change_threads() { return false; }

    /// @brief Get the number of threads used for inter-query parallelism
    size_t get_num_threads() const { return inter_query_threadpool_.size(); }

    /// @brief Get the number of threads used for intra-query cluster exploration
    size_t get_num_intra_query_threads() const { return intra_query_thread_count_; }

    /// @brief Set a new thread pool for inter-query parallelism
    /// @throws std::runtime_error if thread count differs from original
    void set_threadpool(InterQueryThreadPool threadpool) {
        if (threadpool.size() != inter_query_threadpool_.size()) {
            throw std::runtime_error("Threadpool change not supported for IVFIndex - "
                                     "thread count must remain constant");
        }
        inter_query_threadpool_ = std::move(threadpool);
    }

    /// @brief Get the thread pool handle for inter-query parallelism
    InterQueryThreadPool& get_threadpool_handle() { return inter_query_threadpool_; }

    ///// Search Parameters /////

    /// @brief Get current search parameters
    search_parameters_type get_search_parameters() const { return search_parameters_; }

    /// @brief Update search parameters
    void set_search_parameters(const search_parameters_type& search_parameters) {
        search_parameters_ = search_parameters;
    }

    ///// ID Mapping /////

    /// @brief Check if an ID exists in the index
    bool has_id(size_t id) const {
        return id < id_to_cluster_.size() && id_to_cluster_[id] != SIZE_MAX;
    }

    ///// Distance Computation /////

    /// @brief Compute the distance between a query vector and a vector in the index
    template <typename Query> double get_distance(size_t id, const Query& query) const {
        // Thread-safe lazy initialization of ID mapping
        std::call_once(*id_mapping_init_flag_, [this]() { initialize_id_mapping(); });

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

        // Get cluster and position
        size_t cluster_id = id_to_cluster_[id];
        size_t pos = id_in_cluster_[id];

        // Fix distance argument if needed
        auto distance_copy = distance_;
        svs::distance::maybe_fix_argument(distance_copy, query);

        // Call extension for distance computation
        return svs::index::ivf::extensions::get_distance_ext(
            cluster_, distance_copy, cluster_id, pos, query
        );
    }

    /// @brief Return scratch space resources for external threading
    /// @param sp Search parameters to configure the scratchspace
    /// @param num_neighbors Number of neighbors to return (default: 10)
    scratchspace_type
    scratchspace(const search_parameters_type& sp, size_t num_neighbors = 10) const {
        size_t buffer_leaves_size =
            static_cast<size_t>(sp.k_reorder_ * static_cast<float>(num_neighbors));
        return scratchspace_type{
            create_centroid_buffer(sp.n_probes_),
            create_leaf_buffers(buffer_leaves_size),
            extensions::per_thread_batch_search_setup(cluster0_, distance_)};
    }

    /// @brief Return scratch space resources for external threading with default parameters
    scratchspace_type scratchspace() const { return scratchspace(get_search_parameters()); }

    /// @brief Perform a nearest neighbor search for a single query using provided scratch
    /// space
    ///
    /// Operations performed:
    /// * Compute centroid distances for the single query
    /// * Search centroids to find n_probes nearest clusters
    /// * Search within selected clusters to find k nearest neighbors
    ///
    /// Results will be present in the scratch.buffer_leaves[0] data structure.
    /// The caller is responsible for extracting and processing results.
    ///
    /// **Note**: It is the caller's responsibility to ensure that the scratch space has
    /// been initialized properly to return the requested number of neighbors.
    ///
    template <typename Query> void search(const Query& query, scratchspace_type& scratch) {
        // Compute centroid distances for the single query
        // Create a 1-query view and compute matmul_results
        auto query_view = data::ConstSimpleDataView<float>(query.data(), 1, query.size());
        compute_centroid_distances(
            query_view, centroids_, matmul_results_, inter_query_threadpool_
        );

        // Wrapper lambdas that drop query_idx and tid parameters
        auto search_centroids_fn = [&](const auto& q, auto& buf) {
            search_centroids_closure()(q, buf, 0);
        };
        auto search_leaves_fn =
            [&](const auto& q, auto& dist, const auto& buf_cent, auto& buf_leaves) {
                search_leaves_closure()(q, dist, buf_cent, buf_leaves, 0);
            };

        extensions::single_search(
            cluster0_,
            cluster_,
            scratch.buffer_centroids,
            scratch.buffer_leaves,
            scratch.scratch,
            query,
            search_centroids_fn,
            search_leaves_fn
        );
    }

    ///// Batch Iterator /////

    /// @brief Create a batch iterator for retrieving neighbors in batches.
    ///
    /// The iterator allows incremental retrieval of neighbors, expanding the search
    /// space on each call to `next()`. This is useful for applications that need
    /// to process neighbors in batches or implement early termination.
    ///
    /// @tparam QueryType The element type of the query vector.
    /// @param query The query vector as a span.
    /// @param extra_search_buffer_capacity Additional buffer capacity for the search.
    /// @return A BatchIterator for the given query.
    ///
    template <typename QueryType>
    auto make_batch_iterator(
        std::span<const QueryType> query, size_t extra_search_buffer_capacity = 0
    ) {
        return BatchIterator(*this, query, extra_search_buffer_capacity);
    }

    ///// Search Implementation /////

    /// @brief Search closure for centroid distance computation
    /// @return Function object handling initial centroid search phase (inter-query
    /// parallel)
    auto search_centroids_closure() const {
        return [&](const auto& query, auto& buffer, size_t id) {
            search_centroids(
                query,
                distance_,
                matmul_results_,
                buffer,
                id,
                centroids_norm_,
                get_num_threads()
            );
        };
    }

    /// @brief Search closure for cluster traversal
    /// @return Function object handling cluster exploration (intra-query parallel)
    auto search_leaves_closure() const {
        return [&](const auto& query,
                   auto& distance,
                   const auto& buffer_centroids,
                   auto& buffer_leaves,
                   size_t tid) {
            search_leaves(
                query,
                distance,
                cluster_,
                buffer_centroids,
                buffer_leaves,
                intra_query_threadpools_[tid]
            );
        };
    }

    /// @brief Perform similarity search for given queries
    ///
    /// Search Process:
    /// 1. Inter-query parallel: Distribute queries across primary threads
    /// 2. For each query: Find n_probe nearest centroids
    /// 3. Intra-query parallel: Explore identified clusters using inner threads
    /// 4. Combine results from all explored clusters
    ///
    /// @tparam Idx Index type for results
    /// @tparam Queries Query dataset type
    /// @param results View for storing search results
    /// @param queries Query vectors to search for
    /// @param search_parameters Search configuration parameters
    /// @param cancel Optional cancellation predicate
    /// @throws std::runtime_error if query batch size exceeds limits
    template <typename Idx, data::ImmutableMemoryDataset Queries>
    void search(
        QueryResultView<Idx> results,
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

                // Prepare cluster search scratch space
                auto scratch =
                    extensions::per_thread_batch_search_setup(cluster0_, distance_);

                // Execute search with intra-query parallelism
                extensions::per_thread_batch_search(
                    cluster0_,
                    cluster_,
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
    }

  private:
    ///// Core Components /////
    Centroids centroids_;
    Cluster cluster_;
    Data cluster0_;
    Dist distance_;

    ///// ID Mapping for get_distance /////
    // Maps ID -> cluster_id
    mutable std::vector<size_t> id_to_cluster_{};
    // Maps ID -> position within cluster
    mutable std::vector<size_t> id_in_cluster_{};
    // Thread-safe initialization flag for ID mapping (wrapped in unique_ptr for movability)
    mutable std::unique_ptr<std::once_flag> id_mapping_init_flag_{
        std::make_unique<std::once_flag>()};

    ///// Threading Infrastructure /////
    InterQueryThreadPool inter_query_threadpool_; // Handles parallelism across queries
    const size_t intra_query_thread_count_;       // Number of threads per query processing
    mutable std::vector<IntraQueryThreadPool>
        intra_query_threadpools_; // Per-query parallel cluster exploration

    ///// Search Data /////
    mutable std::vector<data::SimpleData<float>> matmul_results_;
    std::vector<float> centroids_norm_;
    search_parameters_type search_parameters_;

    // SVS logger for per index logging
    svs::logging::logger_ptr logger_;

    ///// Initialization Methods /////

    void validate_thread_configuration() {
        if (intra_query_thread_count_ < 1) {
            throw std::invalid_argument("Intra-query thread count must be at least 1");
        }
    }

    void initialize_thread_pools() {
        // Create thread pools for intra-query (cluster-level) parallelism
        for (size_t i = 0; i < inter_query_threadpool_.size(); i++) {
            intra_query_threadpools_.push_back(
                threads::as_threadpool(intra_query_thread_count_)
            );
        }
    }

    void initialize_search_buffers() {
        // Initialize matmul result buffers for centroid distance computation
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
        // Precalculate centroid norms for L2 distance
        if constexpr (std::is_same_v<std::remove_cvref_t<Dist>, distance::DistanceL2>) {
            centroids_norm_.reserve(centroids_.size());
            for (size_t i = 0; i < centroids_.size(); i++) {
                centroids_norm_.push_back(distance::norm_square(centroids_.get_datum(i)));
            }
        }
    }

    void initialize_id_mapping() const {
        // Build ID-to-location mapping from cluster data
        // Compute total size by summing all cluster sizes
        size_t total_size = 0;
        size_t num_clusters = centroids_.size();
        for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id) {
            total_size += cluster_.view_cluster(cluster_id).size();
        }

        // Initialize mapping vectors with sentinel value
        id_to_cluster_.resize(total_size, SIZE_MAX);
        id_in_cluster_.resize(total_size, SIZE_MAX);

        // Populate mappings
        for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id) {
            auto cluster_view = cluster_.view_cluster(cluster_id);
            size_t cluster_size = cluster_view.size();
            for (size_t pos = 0; pos < cluster_size; ++pos) {
                size_t id = cluster_.get_global_id(cluster_id, pos);
                id_to_cluster_[id] = cluster_id;
                id_in_cluster_[id] = pos;
            }
        }
    }

    ///// Helper Methods /////

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
        return SortedBuffer<Idx, distance::compare_t<Dist>>(
            n_probes, distance::comparator(distance_)
        );
    }

    auto create_leaf_buffers(size_t buffer_size) const {
        std::vector<SortedBuffer<Idx, distance::compare_t<Dist>>> buffers;
        buffers.reserve(intra_query_thread_count_);
        for (size_t j = 0; j < intra_query_thread_count_; j++) {
            buffers.push_back(SortedBuffer<Idx, distance::compare_t<Dist>>(
                buffer_size, distance::comparator(distance_)
            ));
        }
        return buffers;
    }
};

/// @brief Build an IVF clustering using either hierarchical or flat k-means
///
/// This function builds an IVF (Inverted File) clustering structure by:
/// 1. Loading the input data using the provided data prototype
/// 2. Performing either hierarchical or flat k-means clustering based on parameters
/// 3. Measuring and logging the build performance
///
/// @tparam BuildType The data type used for building (e.g., float, float16, bfloat16)
/// @tparam DataProto Type of the data prototype for loading
/// @tparam Distance Distance metric type
/// @tparam ThreadpoolProto Thread pool prototype type
///
/// @param parameters IVF build configuration parameters
/// @param data_proto Data prototype for loading the dataset
/// @param distance Distance metric for clustering
/// @param threadpool_proto Thread pool for parallel processing
/// @param logger logger for logging customization.
///
/// @return Clustering object containing centroids and cluster assignments
///
template <
    typename BuildType,
    typename DataProto,
    typename Distance,
    typename ThreadpoolProto>
auto build_clustering(
    const IVFBuildParameters& parameters,
    const DataProto& data_proto,
    Distance distance,
    ThreadpoolProto threadpool_proto,
    bool train_only = false,
    svs::logging::logger_ptr logger = svs::logging::get()
) {
    auto threadpool = threads::as_threadpool(std::move(threadpool_proto));
    auto data = svs::detail::dispatch_load(data_proto, threadpool);

    // Start timing the clustering process
    auto tic = svs::lib::now();
    data::SimpleData<BuildType> centroids;
    std::vector<std::vector<uint32_t>> clusters;

    using Idx = lib::Type<uint32_t>;
    // Choose clustering method based on parameters
    if (parameters.is_hierarchical_) {
        std::tie(centroids, clusters) = hierarchical_kmeans_clustering<BuildType>(
            parameters, data, distance, threadpool, Idx{}, logger, train_only
        );
    } else {
        std::tie(centroids, clusters) = kmeans_clustering<BuildType>(
            parameters, data, distance, threadpool, Idx{}, logger, train_only
        );
    }

    // Create and validate clustering
    Clustering clustering(std::move(centroids), std::move(clusters));

    // Log performance metrics
    auto build_time = svs::lib::time_difference(tic);
    svs::logging::debug(logger, "IVF build time: {} seconds\n", build_time);

    svs::logging::debug(
        logger, "IVF Clustering Stats: {}", clustering.statistics().report("\n")
    );

    return clustering;
}

/// @brief Assemble an IVF index from an existing clustering
///
/// This function creates a complete IVF index by:
/// 1. Loading the dataset
/// 2. Creating dense cluster representations
/// 3. Constructing the final IVF index with parallel search support
///
/// @tparam Clustering Type of the clustering structure
/// @tparam DataProto Type of the data prototype for loading
/// @tparam Distance Distance metric type
/// @tparam ThreadpoolProto Thread pool prototype type
///
/// @param clustering Existing clustering structure
/// @param data_proto Data prototype for loading the dataset
/// @param distance Distance metric for searching
/// @param threadpool_proto Thread pool for parallel processing
/// @param intra_query_thread_count Number of threads for intra-query parallelism (default:
///     1)
/// @param logger logger for logging customization.
///
/// @return Fully constructed IVF index ready for searching
///
template <
    typename Clustering,
    typename DataProto,
    typename Distance,
    typename ThreadpoolProto>
auto assemble_from_clustering(
    Clustering clustering,
    const DataProto& data_proto,
    Distance distance,
    ThreadpoolProto threadpool_proto,
    const size_t intra_query_thread_count = 1,
    svs::logging::logger_ptr logger = svs::logging::get()
) {
    // Initialize timing infrastructure
    auto timer = lib::Timer();
    auto assemble_timer = timer.push_back("Total Assembling time");

    // Phase 1: Load dataset
    auto data_load_timer = timer.push_back("Data loading");
    auto threadpool = threads::as_threadpool(std::move(threadpool_proto));
    auto data = svs::detail::dispatch_load(data_proto, threadpool);
    data_load_timer.finish();

    // Phase 2: Create dense cluster representation
    auto dense_cluster_timer = timer.push_back("Dense clustering");
    using centroids_type = data::SimpleData<typename Clustering::T>;
    using data_type = typename decltype(data)::lib_alloc_data_type;

    auto dense_clusters = DenseClusteredDataset<centroids_type, uint32_t, data_type>(
        clustering, data, threadpool, lib::Allocator<std::byte>()
    );
    dense_cluster_timer.finish();

    // Phase 3: Construct IVF index
    auto index_build_timer = timer.push_back("IVF index construction");
    auto ivf_index = IVFIndex(
        std::move(clustering.centroids()),
        std::move(dense_clusters),
        std::move(distance),
        std::move(threadpool),
        intra_query_thread_count,
        logger
    );
    index_build_timer.finish();

    // Log timing results
    assemble_timer.finish();
    svs::logging::debug(logger, "{}", timer);
    return ivf_index;
}

/// @brief Assemble an IVF index from a saved clustering file
///
/// This function loads a previously saved clustering from disk and creates
/// a complete IVF index. It's a convenience wrapper around assemble_from_clustering
/// that handles the clustering loading step.
///
/// @tparam Centroids Type of the centroid data
/// @tparam DataProto Type of the data prototype for loading
/// @tparam Distance Distance metric type
/// @tparam ThreadpoolProto Thread pool prototype type
///
/// @param clustering_path Path to the saved clustering file
/// @param data_proto Data prototype for loading the dataset
/// @param distance Distance metric for searching
/// @param threadpool_proto Thread pool for parallel processing
/// @param n_inner_threads Number of threads for intra-query parallelism (default: 1)
/// @param intra_query_thread_count Number of threads for intra-query parallelism (default:
///     1)
/// @param logger logger for logging customization.
///
/// @return Fully constructed IVF index ready for searching
///
template <
    typename Centroids,
    typename DataProto,
    typename Distance,
    typename ThreadpoolProto>
auto assemble_from_file(
    const std::filesystem::path& clustering_path,
    const DataProto& data_proto,
    Distance distance,
    ThreadpoolProto threadpool_proto,
    const size_t intra_query_thread_count = 1,
    svs::logging::logger_ptr logger = svs::logging::get()
) {
    // Define the clustering type
    using centroids_type = data::SimpleData<Centroids>;

    // Initialize thread pool and load clustering from disk
    auto threadpool = threads::as_threadpool(std::move(threadpool_proto));
    auto clustering = svs::lib::load_from_disk<Clustering<centroids_type, uint32_t>>(
        clustering_path, threadpool
    );

    // Delegate to the main assembly function
    return assemble_from_clustering(
        std::move(clustering),
        data_proto,
        std::move(distance),
        std::move(threadpool),
        intra_query_thread_count,
        std::move(logger)
    );
}

} // namespace svs::index::ivf
