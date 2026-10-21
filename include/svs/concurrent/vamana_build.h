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

// local
#include "svs/concepts/data.h"
#include "svs/concepts/distance.h"
#include "svs/concurrent/greedy_search.h"
#include "svs/concurrent/prune.h"
#include "svs/concurrent/spinlock.h"
#include "svs/core/logging.h"
#include "svs/index/vamana/build_params.h"
#include "svs/index/vamana/extensions.h"
#include "svs/index/vamana/search_buffer.h"
#include "svs/index/vamana/search_tracker.h"
#include "svs/lib/boundscheck.h"
#include "svs/lib/exception.h"
#include "svs/lib/misc.h"
#include "svs/lib/narrow.h"
#include "svs/lib/neighbor.h"
#include "svs/lib/threads/threadlocal.h"
#include "svs/lib/threads/threadpool.h"
#include "svs/lib/timing.h"
#include "svs/third-party/fmt.h"

// external
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

// stdlib
#include <algorithm>
#include <concepts>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>

namespace svs::index::vamana::concurrent {

// Optional search tracker to get full history of graph search.
template <typename Idx> class OptionalTracker {
  public:
    using set_type = tsl::robin_set<Neighbor<Idx>, IDHash, IDEqual>;
    using const_iterator = typename set_type::const_iterator;

  private:
    std::optional<set_type> neighbors_;

  public:
    ///// Constructors
    OptionalTracker(bool enable)
        : neighbors_{std::nullopt} {
        if (enable) {
            neighbors_.emplace();
        }
    }

    ///// Methods
    bool enabled() const { return neighbors_.has_value(); }
    size_t size() const { return enabled() ? (*neighbors_).size() : 0; }

    const_iterator begin() const { return neighbors_.value().begin(); }
    const_iterator end() const { return neighbors_.value().end(); }

    void clear() {
        // This method is safe to call even if the tracker isn't being used.
        if (enabled()) {
            (*neighbors_).clear();
        }
    }

    ///// Search Tracker API
    void visited(const Neighbor<Idx>& neighbor, size_t SVS_UNUSED(distance_computations)) {
        if (enabled()) {
            (*neighbors_).insert(neighbor);
        }
    }
};

// Define an auxiliary struct to disambiguate constructor calls.
struct BackedgeBufferParameters {
    size_t bucket_size_;
    size_t num_buckets_;
};

///
/// @brief A helper type for managing synchronization and parallelism of backedges
///
/// The big idea is to use locking over coarse regions of indices.
/// This still provides synchronized access to individual entries, but allows parallelized
/// access to multiple buckets.
///
template <typename Idx> class BackedgeBuffer {
  public:
    // Map an vertex to it's expanded adjacency list.
    using set_type = tsl::robin_set<Idx>;
    using map_type = tsl::robin_map<Idx, set_type>;

  private:
    // The number of elements assigned to each bucket - starting sequentially from zero.
    // Used to determine which bucket an index belongs to.
    size_t bucket_size_;
    std::vector<map_type> buckets_;
    std::vector<std::mutex> bucket_locks_;

  public:
    ///// Constructors
    BackedgeBuffer(BackedgeBufferParameters parameters)
        : bucket_size_{parameters.bucket_size_}
        , buckets_(parameters.num_buckets_)
        , bucket_locks_{parameters.num_buckets_} {}

    BackedgeBuffer(size_t num_elements, size_t bucket_size)
        : BackedgeBuffer(BackedgeBufferParameters{
              bucket_size, lib::div_round_up(num_elements, bucket_size)}) {}

    // Add a point.
    void add_edge(Idx src, Idx dst) {
        // Get the bucket that the source vertex belongs to.
        size_t bucket = src / bucket_size_;
        // The bucket array is sized once from the graph size snapshotted at
        // VamanaBuilder construction. A concurrent add_points may have grown the
        // graph past that snapshot, so a greedy-search neighbor `src` can fall
        // beyond our bucket range. That node belongs to another in-flight add and
        // gets its own back-edges from that add's builder; drop the overflow edge
        // here rather than indexing out of bounds — consistent with the
        // best-effort dropping already done on AddEdgeResult::Full.
        if (bucket >= buckets_.size()) {
            return;
        }
        // Lock the bucket and update the adjacency list.
        std::lock_guard lock(bucket_locks_.at(bucket));

        // The "try_emplace" method will default construct the set if it doesn't exist.
        // Whether or not the set existed to begin with, we get an iterator to the set
        // which we can then add the destination to.
        auto& map = buckets_.at(bucket);
        auto [iterator, _] = map.try_emplace(src);
        iterator.value().insert(dst);
    }

    // Return the underlying buckets directly.
    // Buckets can be iterated over to add back edges.
    std::vector<map_type>& buckets() { return buckets_; }

    // Return the number of buckets in the buffer
    size_t num_buckets() const {
        assert(buckets_.size() == bucket_locks_.size());
        return buckets_.size();
    }

    // Reset the container for another iteration.
    void reset() {
        for (size_t i = 0, imax = num_buckets(); i < imax; ++i) {
            std::lock_guard lock(bucket_locks_.at(i));
            buckets_.at(i).clear();
        }
    }
};

template <
    graphs::MemoryGraph Graph,
    data::ImmutableMemoryDataset Data,
    typename Dist,
    threads::ThreadPool Pool,
    typename Eligible = lib::ReturnsTrueType>
class VamanaBuilder {
  public:
    // Type Aliases
    using Idx = typename Graph::index_type;
    using search_buffer_type = SearchBuffer<Idx, distance::compare_t<Dist>>;

    template <typename T> using set_type = tsl::robin_set<T>;

    using update_type =
        threads::SequentialTLS<std::vector<std::pair<Idx, std::vector<Idx>>>>;

    /// Constructor
    VamanaBuilder(
        Graph& graph,
        const Data& data,
        Dist distance_function,
        const VamanaBuildParameters& params,
        Pool& threadpool,
        GreedySearchPrefetchParameters prefetch_hint = {},
        svs::logging::logger_ptr logger = svs::logging::get(),
        logging::Level level = logging::Level::Debug,
        Eligible eligible = {},
        std::span<const Idx> additional_candidates = {}
    )
        : graph_{graph}
        , data_{data}
        , distance_function_{std::move(distance_function)}
        , params_{params}
        , prefetch_hint_{prefetch_hint}
        , threadpool_{threadpool}
        , backedge_buffer_{data.size(), 1000}
        , eligible_{std::move(eligible)}
        , additional_candidates_{additional_candidates} {
        // Print all parameters
        svs::logging::log(
            logger,
            level,
            "Vamana Build Parameters: alpha={}, graph_max_degree={}, "
            "max_candidate_pool_size={}, prune_to={}, window_size={}, "
            "use_full_search_history={}",
            params.alpha,
            params.graph_max_degree,
            params.max_candidate_pool_size,
            params.prune_to,
            params.window_size,
            params.use_full_search_history
        );
        // Note: graph/data size invariant (graph_.n_nodes() == data_.size()) is
        // maintained under mutation_mutex_ in add_points(). During concurrent
        // add_points() calls, sizes may temporarily differ between lock release
        // and this point, but both are always >= the slots we will operate on.
    }

    void construct(
        float alpha,
        Idx entry_point,
        logging::Level level = logging::Level::Trace,
        logging::logger_ptr logger = svs::logging::get()
    ) {
        construct(
            alpha, entry_point, threads::UnitRange<size_t>{0, data_.size()}, level, logger
        );
    }

    template <typename R>
    void construct(
        float alpha,
        Idx entry_point,
        const R& range,
        logging::Level level = logging::Level::Trace,
        logging::logger_ptr logger = svs::logging::get()
    ) {
        size_t num_nodes = range.size();
        if (num_nodes == 0) {
            return;
        }

        // One node per worker
        const size_t batchsize = std::min<size_t>(threadpool_.size(), num_nodes);
        const size_t num_batches = lib::div_round_up(num_nodes, batchsize);

        std::vector entry_points{entry_point};

        // Runtime variables
        double search_time = 0;
        double reverse_time = 0;
        unsigned progress_counter = 0;

        svs::logging::log(logger, level, "Number of syncs: {}", num_batches);
        svs::logging::log(logger, level, "Batch Size: {}", batchsize);

        // The base point for iteration.
        auto&& base = range.begin();
        auto timer = lib::Timer();
        for (size_t batch_id = 0; batch_id < num_batches; ++batch_id) {
            // Set up batch parameters
            auto start = std::min(num_nodes, batchsize * batch_id) + base;
            auto stop = std::min(num_nodes, batchsize * (batch_id + 1)) + base;
            auto batch = threads::IteratorPair{start, stop};

            // Perform search.
            // N.B. - We purposely pass "params_.alpha" instead of the external "alpha"
            // because it seems to generally yield better results.
            auto x = timer.push_back("generate neighbors");
            generate_neighbors(batch, params_.alpha, entry_points, timer);
            search_time += lib::as_seconds(x.finish());

            auto y = timer.push_back("reverse edges");
            std::vector<Idx> retry;
            add_reverse_edges(batch, alpha, timer, &retry);
            while (!retry.empty()) {
                std::sort(retry.begin(), retry.end());
                retry.erase(std::unique(retry.begin(), retry.end()), retry.end());
                auto nodes = std::move(retry);
                retry.clear();
                // These insertions are still Pending. Preserve published adjacency;
                // reselect parents if deletion won the backlink admission race.
                generate_neighbors(nodes, params_.alpha, entry_points, timer);
                add_reverse_edges(nodes, alpha, timer, &retry);
            }
            reverse_time += lib::as_seconds(y.finish());

            auto this_progress = lib::narrow_cast<double>(batch_id) * 1e2 /
                                 lib::narrow_cast<double>(num_batches);
            if (this_progress > progress_counter && batch_id > 0) {
                auto total_elapsed_time = lib::as_seconds(timer.elapsed());
                auto num_batches_f = lib::narrow_cast<double>(num_batches);
                auto batch_id_f = lib::narrow_cast<double>(batch_id);

                double estimated_remaining_time =
                    total_elapsed_time * (num_batches_f / batch_id_f - 1);
                constexpr std::string_view message = "Completed round {} of {}. "
                                                     "Search Time: {:.4}s, "
                                                     "Reverse Time: {:.4}s, "
                                                     "Total Time: {:.4}s, "
                                                     "Estimated Remaining Time: {:.4}s";

                svs::logging::log(
                    logger,
                    level,
                    message,
                    batch_id + 1,
                    num_batches,
                    search_time,
                    reverse_time,
                    total_elapsed_time,
                    estimated_remaining_time
                );
                search_time = 0;
                reverse_time = 0;
                progress_counter += 1;
            }
        }
        svs::logging::log(
            logger, level, "Completed pass using window size {}.", params_.window_size
        );
        svs::logging::log(logger, level, "{}", timer);
    }

    ///
    /// Generate Adjacency lists for new collection of nodes.
    /// As far as the algorithm is concerned, this implements the search and heuristic
    /// pruning for the vertices.
    ///
    /// Addition of back edges is saved for another step.
    ///
    template <typename /*std::ranges::random_access_range*/ R>
    void generate_neighbors(
        const R& indices,
        float alpha,
        const std::vector<Idx>& entry_points,
        lib::Timer& timer
    ) {
        auto range = threads::StaticPartition{indices};

        auto main = timer.push_back("main");
        threads::parallel_for(
            threadpool_,
            range,
            [&](const auto& local_indices, uint64_t SVS_UNUSED(tid)) {
                // Scratch space.
                std::vector<Neighbor<Idx>> pool{};
                std::vector<Idx> pruned_results{};
                auto search_buffer = search_buffer_type{params_.window_size};

                // Enable use of the visited filter of the search buffer.
                // It seems to help in high-window-size scenarios.
                search_buffer.enable_visited_set();
                set_type<Idx> visited{};
                auto tracker = OptionalTracker<Idx>(params_.use_full_search_history);

                // Unpack adaptor.
                auto build_adaptor = extensions::build_adaptor(data_, distance_function_);
                auto&& graph_search_distance = build_adaptor.graph_search_distance();
                auto&& general_distance = build_adaptor.general_distance();
                auto general_accessor = build_adaptor.general_accessor();

                for (auto node_id : local_indices) {
                    pool.clear();
                    search_buffer.clear();
                    visited.clear();
                    tracker.clear();

                    const auto& graph_search_query =
                        build_adaptor.access_query_for_graph_search(data_, node_id);

                    // Perform the greedy search.
                    // The search tracker will be used if it is enabled.
                    {
                        auto accessor = build_adaptor.graph_search_accessor();
                        concurrent::greedy_search(
                            graph_,
                            data_,
                            accessor,
                            graph_search_query,
                            graph_search_distance,
                            search_buffer,
                            vamana::EntryPointInitializer{lib::as_const_span(entry_points)},
                            NeighborBuilder(),
                            tracker,
                            prefetch_hint_
                        );
                    }

                    const auto& post_search_query = build_adaptor.modify_post_search_query(
                        data_, node_id, graph_search_query
                    );

                    // If the query and distance functors are sufficiently different for the
                    // graph search and the general case, then we *may* need to reapply fix
                    // argument before we can do any further distance computations.
                    //
                    // Decide whether we need to make this call.
                    if constexpr (decltype(build_adaptor)::refix_argument_after_search) {
                        distance::maybe_fix_argument(general_distance, post_search_query);
                    }

                    auto modify_distance = [&](NeighborLike auto const& n) {
                        return build_adaptor.post_search_modify(
                            data_, general_distance, post_search_query, n
                        );
                    };

                    // If the full search history is to be used, then use the tracker to
                    // populate the candidate pool.
                    //
                    // Otherwise, pull results directly out of the search buffer.
                    if (tracker.enabled()) {
                        for (const auto& neighbor : tracker) {
                            if (!eligible_(neighbor.id())) {
                                continue;
                            }
                            pool.push_back(modify_distance(neighbor));
                            visited.insert(neighbor.id());
                        }
                    } else {
                        for (size_t i = 0, imax = search_buffer.size(); i < imax; ++i) {
                            const auto& neighbor = search_buffer[i];
                            if (!eligible_(neighbor.id())) {
                                continue;
                            }
                            pool.push_back(modify_distance(neighbor));
                            visited.insert(neighbor.id());
                        }
                    }

                    // Ready in-flight insertions need not yet be reachable from the
                    // entry point. Deduplicate only against candidates found by search.
                    for (auto id : additional_candidates_) {
                        if (id != node_id && eligible_(id) && visited.insert(id).second) {
                            pool.emplace_back(
                                id,
                                distance::compute(
                                    general_distance,
                                    post_search_query,
                                    general_accessor(data_, id)
                                )
                            );
                        }
                    }

                    // Read all peers in the shared batch
                    if (indices.size() > 1) {
                        for (auto raw_id : indices) {
                            const auto id = lib::narrow_cast<Idx>(raw_id);
                            if (id != node_id && eligible_(id) &&
                                visited.insert(id).second) {
                                pool.emplace_back(
                                    id,
                                    distance::compute(
                                        general_distance,
                                        post_search_query,
                                        general_accessor(data_, id)
                                    )
                                );
                            }
                        }
                    }

                    if constexpr (!std::is_same_v<Eligible, lib::ReturnsTrueType>) {
                        // The retained anchor is usable even when all old labels
                        // have been deleted or displaced from the search window.
                        for (auto id : entry_points) {
                            if (id != node_id && eligible_(id) &&
                                visited.insert(id).second) {
                                pool.emplace_back(
                                    id,
                                    distance::compute(
                                        general_distance,
                                        post_search_query,
                                        general_accessor(data_, id)
                                    )
                                );
                            }
                        }
                    }

                    // Greedy search stays outside the lock. Protect the current
                    // adjacency from its first read through the pruned replacement.
                    auto node_lock = graph_.lock_node(node_id);
                    // Add neighbors of the query that are not part of `visited`.
                    for (auto id : graph_.get_node(node_id)) {
                        if (!eligible_(id)) {
                            continue;
                        }
                        assert(id != node_id);
                        // Try to emplace the node id into the visited set.
                        // If the id was inserted, then it didn't already exist in the
                        // visited set and we need to add it to the candidate pool.
                        auto [_, inserted] = visited.emplace(id);
                        if (inserted) {
                            pool.emplace_back(
                                id,
                                distance::compute(
                                    general_distance,
                                    post_search_query,
                                    general_accessor(data_, id)
                                )
                            );
                        }
                    }

                    std::sort(
                        pool.begin(),
                        pool.end(),
                        TotalOrder(distance::comparator(general_distance))
                    );
                    pool.resize(std::min(pool.size(), params_.max_candidate_pool_size));

                    pruned_results.clear();
                    heuristic_prune_neighbors(
                        prune_strategy(distance_function_),
                        params_.graph_max_degree,
                        alpha,
                        data_,
                        general_accessor,
                        general_distance,
                        node_id,
                        lib::as_const_span(pool),
                        pruned_results
                    );
                    graph_.replace_node(
                        node_id, lib::as_const_span(pruned_results), node_lock
                    );
                }
            }
        );

        main.finish();
    }

    ///
    /// Add reverse edges to the graph.
    ///
    template <typename /*std::ranges::random_access_range*/ R>
    void add_reverse_edges(
        const R& indices, float alpha, lib::Timer& timer, std::vector<Idx>* retry = nullptr
    ) {
        std::mutex retry_mutex;
        auto retry_node = [&](Idx id) {
            if (retry) {
                std::lock_guard lock{retry_mutex};
                retry->push_back(id);
            }
        };
        // Apply backedges to all new candidate adjacency lists.
        // If adding an edge to the graph will cause it to violate the maximum degree
        // constraint, save the excess to the backedge buffer.
        auto backedge_timer = timer.push_back("backedge generation");
        auto range = threads::StaticPartition{indices};
        backedge_buffer_.reset();
        threads::parallel_for(
            threadpool_,
            range,
            [&](const auto& is, uint64_t SVS_UNUSED(tid)) {
                for (auto node_id : is) {
                    for (auto other_id : graph_.get_node(node_id)) {
                        // graph_.add_edge is atomic under node_locks_[other_id].
                        // If it reports Full, route to the overflow buffer —
                        // no TOCTOU race between a pre-check and the insert.
                        auto result = graph_.add_edge(other_id, node_id, eligible_);
                        if (result == graphs::AddEdgeResult::Full) {
                            backedge_buffer_.add_edge(other_id, node_id);
                        } else if (result == graphs::AddEdgeResult::Rejected) {
                            retry_node(node_id);
                        }
                    }
                }
            }
        );
        backedge_timer.finish();

        // For all vertices that now exceed the max degree requirement, run the pruning
        // procedure on the union of their current adjacency list as well as any extra edges
        // that were recorded in the previous process.
        //
        // Take care to avoid duplicate entries.
        auto prune_timer = timer.push_back("pruning backedges");
        threads::parallel_for(
            threadpool_,
            threads::DynamicPartition{backedge_buffer_.buckets(), 1},
            [&](auto& buckets, uint64_t SVS_UNUSED(tid)) {
                // Thread local auxiliary data structures.
                std::vector<Neighbor<Idx>> candidates{};
                std::vector<Idx> pruned_results{};
                auto build_adaptor = extensions::build_adaptor(data_, distance_function_);

                auto general_accessor = build_adaptor.general_accessor();
                auto&& general_distance = build_adaptor.general_distance();

                auto cmp = distance::comparator(general_distance);
                for (auto& bucket : buckets) {
                    for (const auto& kv : bucket) {
                        // The ``neighbors`` class is a set.
                        auto src = kv.first;
                        const auto& neighbors = kv.second;
                        auto node_lock = graph_.lock_node(src);
                        if (!eligible_(src)) {
                            // Full was only queued, not yet committed. Deletion
                            // may have won since the first backlink attempt.
                            for (auto n : neighbors) {
                                retry_node(n);
                            }
                            continue;
                        }
                        const auto& src_data = general_accessor(data_, src);
                        distance::maybe_fix_argument(general_distance, src_data);

                        // Helper lambda to make distance computations look a little
                        // cleaner.
                        auto make_neighbor = [&](auto i) {
                            return Neighbor<Idx>{
                                i,
                                distance::compute(
                                    general_distance, src_data, general_accessor(data_, i)
                                )};
                        };

                        candidates.clear();
                        // Add the overflow candidates.
                        for (auto n : neighbors) {
                            if (eligible_(n)) {
                                candidates.push_back(make_neighbor(n));
                            }
                        }

                        // Add the old adjacency list.
                        // Existing tombstones may still route to live nodes; insertion
                        // eligibility must not remove them from the pruning pool.
                        for (auto n : graph_.get_node(src)) {
                            if (!neighbors.contains(n)) {
                                candidates.push_back(make_neighbor(n));
                            }
                        }
                        std::sort(candidates.begin(), candidates.end(), TotalOrder(cmp));
                        candidates.resize(
                            std::min(candidates.size(), params_.max_candidate_pool_size)
                        );

                        pruned_results.clear();
                        heuristic_prune_neighbors(
                            prune_strategy(distance_function_),
                            params_.prune_to,
                            alpha,
                            data_,
                            general_accessor,
                            general_distance,
                            src,
                            lib::as_const_span(candidates),
                            pruned_results
                        );
                        graph_.replace_node(
                            src, lib::as_const_span(pruned_results), node_lock
                        );
                    }
                }
            }
        );
    }

  private:
    /// The graph being constructed.
    Graph& graph_;
    /// The dataset we're building the graph over.
    const Data& data_;
    /// The distance function to use.
    Dist distance_function_;
    /// Parameters regarding index construction.
    VamanaBuildParameters params_;
    /// Prefetch parameters to use during the graph search.
    GreedySearchPrefetchParameters prefetch_hint_;
    /// Worker threadpool.
    Pool& threadpool_;
    /// Overflow backedge buffer.
    BackedgeBuffer<Idx> backedge_buffer_;
    [[no_unique_address]] Eligible eligible_;
    /// Optional initialized candidates; the caller owns the span through construct().
    std::span<const Idx> additional_candidates_;
};
} // namespace svs::index::vamana::concurrent
