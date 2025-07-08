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
#include <random>
#include <vector>

namespace svs::index::ivf {

const size_t MAX_QUERY_BATCH_SIZE = 10000;

template <typename Centroids, typename Cluster, typename Dist, typename ThreadPoolProto>
class IVFIndex {
  public:
    using Idx = typename Cluster::index_type;
    using Data = typename Cluster::data_type;
    using search_parameters_type = IVFSearchParameters;

    IVFIndex(
        Centroids centroids,
        Cluster cluster,
        Dist distance_function,
        ThreadPoolProto threadpool_proto,
        size_t n_inner_threads = 1
    )
        : centroids_{std::move(centroids)}
        , cluster_{std::move(cluster)}
        , cluster0_{cluster_.view_cluster(0)}
        , distance_{std::move(distance_function)}
        , threadpool_{threads::as_threadpool(std::move(threadpool_proto))}
        , n_inner_threads_{n_inner_threads} {
        // Initialize threadpools for intra-query parallelism
        for (size_t i = 0; i < threadpool_.size(); i++) {
            threadpool_inner_.push_back(threads::as_threadpool(n_inner_threads_));
        }

        // The first level of search to find the n_probes nearest centroids is done
        // using matmul (single thread) and thread batching on the number of centroids.
        // Matmul results of each batch with appropriate sizes are initialized here,
        // assuming that the static partition will keep the batching same.
        auto batches = std::vector<threads::UnitRange<uint64_t>>(threadpool_.size());
        threads::parallel_for(
            threadpool_,
            threads::StaticPartition(centroids_.size()),
            [&](auto is, auto tid) { batches[tid] = threads::UnitRange{is}; }
        );
        for (size_t i = 0; i < threadpool_.size(); i++) {
            matmul_results_.emplace_back(MAX_QUERY_BATCH_SIZE, batches[i].size());
        }

        // Precalculate centroid norms (required in L2 distances)
        if constexpr (std::is_same_v<Dist, distance::DistanceL2>) {
            for (size_t i = 0; i < centroids_.size(); i++) {
                centroids_norm_.push_back(distance::norm_square(centroids_.get_datum(i)));
            }
        }
    }

    ///// Threading Interface

    static constexpr bool can_change_threads() { return false; }
    ///
    /// @brief Return the current number of threads used for search.
    ///
    size_t get_num_threads() const { return threadpool_.size(); }

    void set_threadpool(threads::ThreadPoolHandle threadpool) {
        if (threadpool.size() != threadpool_.size()) {
            throw ANNEXCEPTION("Threadpool change not supported for IVFIndex!");
        }

        threadpool_ = std::move(threadpool);
    }

    ///
    /// @brief Destroy the original thread pool and set to the provided one.
    ///
    /// @param threadpool An acceptable thread pool.
    ///
    /// @copydoc threadpool_requirements
    ///
    template <threads::ThreadPool Pool>
    void set_threadpool(Pool threadpool)
        requires(!std::is_same_v<Pool, threads::ThreadPoolHandle>)
    {
        set_threadpool(threads::ThreadPoolHandle(std::move(threadpool)));
    }

    ///
    /// @brief Return the current thread pool handle.
    ///
    threads::ThreadPoolHandle& get_threadpool_handle() { return threadpool_; }

    size_t size() const { return centroids_.size(); }
    size_t dimensions() const { return centroids_.dimensions(); }

    ///// Search Parameter Setting
    search_parameters_type get_search_parameters() const { return search_parameters_; }

    void set_search_parameters(const search_parameters_type& search_parameters) {
        search_parameters_ = search_parameters;
    }

    auto search_centroids_closure() {
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

    auto search_leaves_closure() {
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
                threadpool_inner_[tid]
            );
        };
    }

    // Search
    template <typename Idx, data::ImmutableMemoryDataset Queries>
    void search(
        QueryResultView<Idx> results,
        const Queries& queries,
        const search_parameters_type& search_parameters,
        const lib::DefaultPredicate& SVS_UNUSED(cancel) = lib::Returns(lib::Const<false>())
    ) {
        if (queries.size() > MAX_QUERY_BATCH_SIZE) {
            throw ANNEXCEPTION(
                "Number of queries {} higher than expected {}, increase value of "
                "MAX_QUERY_BATCH_SIZE",
                queries.size(),
                MAX_QUERY_BATCH_SIZE
            );
        }

        size_t num_neighbors = results.n_neighbors();
        size_t buffer_leaves_size = search_parameters.k_reorder_ * num_neighbors;
        compute_centroid_distances(queries, centroids_, matmul_results_, threadpool_);

        threads::parallel_for(
            threadpool_,
            threads::StaticPartition(queries.size()),
            [&](auto is, auto tid) {
                auto buffer_centroids = SortedBuffer<Idx, distance::compare_t<Dist>>(
                    search_parameters.n_probes_, distance::comparator(distance_)
                );
                std::vector<SortedBuffer<Idx, distance::compare_t<Dist>>> buffer_leaves;
                for (size_t j = 0; j < n_inner_threads_; j++) {
                    buffer_leaves.push_back(SortedBuffer<Idx, distance::compare_t<Dist>>(
                        buffer_leaves_size, distance::comparator(distance_)
                    ));
                }

                // Pre-allocate scratch space needed by the dataset implementation.
                auto scratch =
                    extensions::per_thread_batch_search_setup(cluster0_, distance_);

                // Perform a search over the batch of queries.
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

    std::string name() const { return "IVFIndex"; }

  private:
    Centroids centroids_;
    Cluster cluster_;
    Data cluster0_;
    Dist distance_;
    threads::ThreadPoolHandle threadpool_;
    const size_t n_inner_threads_ = 1;
    std::vector<threads::DefaultThreadPool> threadpool_inner_;
    std::vector<data::SimpleData<float>> matmul_results_;
    std::vector<float> centroids_norm_;

    // Tunable Parameters
    search_parameters_type search_parameters_{};
};

template <
    typename BuildType,
    typename DataProto,
    typename Distance,
    typename ThreadpoolProto>
auto build_clustering(
    const IVFBuildParameters& parameters,
    const DataProto& data_proto,
    Distance distance,
    ThreadpoolProto threadpool_proto
) {
    auto threadpool = threads::as_threadpool(std::move(threadpool_proto));
    auto data = svs::detail::dispatch_load(data_proto, threadpool);

    auto tic = svs::lib::now();
    data::SimpleData<BuildType> centroids;
    std::vector<std::vector<uint32_t>> clusters;
    if (parameters.is_hierarchical_) {
        std::tie(centroids, clusters) = hierarchical_kmeans_clustering<BuildType>(
            parameters, data, distance, threadpool
        );
    } else {
        std::tie(centroids, clusters) =
            kmeans_clustering<BuildType>(parameters, data, distance, threadpool);
    }

    Clustering clustering(std::move(centroids), std::move(clusters));
    auto build_time = svs::lib::time_difference(tic);
    fmt::print("IVF build time: {} seconds\n", build_time);

    auto logger = svs::logging::get();
    svs::logging::debug(
        logger, "IVF Clustering Stats: {}", clustering.statistics().report("\n")
    );

    return clustering;
}

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
    const size_t n_inner_threads = 1
) {
    auto timer = lib::Timer();
    auto assemble_timer = timer.push_back("Total Assembling time");
    auto data_load_timer = timer.push_back("Data loading");
    auto threadpool = threads::as_threadpool(std::move(threadpool_proto));
    auto data = svs::detail::dispatch_load(data_proto, threadpool);
    data_load_timer.finish();

    auto dense_cluster_timer = timer.push_back("Dense clustering");
    using centroids_type = data::SimpleData<typename Clustering::T>;
    using data_type = typename decltype(data)::lib_alloc_data_type;

    auto dense_clusters = DenseClusteredDataset<centroids_type, uint32_t, data_type>(
        clustering, data, threadpool, lib::Allocator<std::byte>()
    );
    dense_cluster_timer.finish();

    auto index_build_timer = timer.push_back("IVF index construction");
    auto ivf_index = IVFIndex(
        std::move(clustering.centroids()),
        std::move(dense_clusters),
        std::move(distance),
        std::move(threadpool),
        n_inner_threads
    );
    index_build_timer.finish();
    assemble_timer.finish();
    svs::logging::debug("{}", timer);
    return ivf_index;
}

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
    const size_t n_inner_threads = 1
) {
    using centroids_type = data::SimpleData<Centroids>;
    auto threadpool = threads::as_threadpool(std::move(threadpool_proto));
    auto clustering = svs::lib::load_from_disk<Clustering<centroids_type, uint32_t>>(
        clustering_path, threadpool
    );

    return assemble_from_clustering(
        std::move(clustering),
        data_proto,
        std::move(distance),
        std::move(threadpool),
        n_inner_threads
    );
}

} // namespace svs::index::ivf
