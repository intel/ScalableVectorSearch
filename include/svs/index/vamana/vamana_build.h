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
#include "svs/core/logging.h"
#include "svs/index/vamana/build_params.h"
#include "svs/index/vamana/extensions.h"
#include "svs/index/vamana/greedy_search.h"
#include "svs/index/vamana/prune.h"
#include "svs/index/vamana/search_buffer.h"
#include "svs/index/vamana/search_tracker.h"
#include "svs/lib/boundscheck.h"
#include "svs/lib/exception.h"
#include "svs/lib/narrow.h"
#include "svs/lib/neighbor.h"
#include "svs/lib/spinlock.h"
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
#include <optional>
#include <tuple>
#include <vector>

namespace svs::index::vamana {

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
    // The number of elements assigned to each bucket - starting sequentialy from zero.
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
    threads::ThreadPool Pool>
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
        GreedySearchPrefetchParameters prefetch_hint = {}
    )
        : graph_{graph}
        , data_{data}
        , distance_function_{std::move(distance_function)}
        , params_{params}
        , prefetch_hint_{prefetch_hint}
        , threadpool_{threadpool}
        , vertex_locks_(data.size())
        , backedge_buffer_{data.size(), 1000} {
        // Check class invariants.
        if (graph_.n_nodes() != data_.size()) {
            throw ANNEXCEPTION(
                "Expected graph to be pre-allocated with {} vertices!", data_.size()
            );
        }
    }

    void
    construct(float alpha, Idx entry_point, logging::Level level = logging::Level::Info) {
        construct(alpha, entry_point, threads::UnitRange<size_t>{0, data_.size()}, level);
    }

    template <typename R>
    void construct(
        float alpha,
        Idx entry_point,
        const R& range,
        logging::Level level = logging::Level::Info
    ) {
        auto logger = svs::logging::get();
        size_t num_nodes = range.size();
        size_t num_batches = std::max(
            size_t{40}, lib::div_round_up(num_nodes, lib::narrow_cast<size_t>(64 * 64))
        );
        size_t batchsize = lib::div_round_up(num_nodes, num_batches);
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

            // Perform search.
            // N.B. - We purposely pass "params_.alpha" instead of the external "alpha"
            // because it seems to generally yield better results.
            auto x = timer.push_back("generate neighbors");
            generate_neighbors(
                threads::IteratorPair{start, stop}, params_.alpha, entry_points, timer
            );
            search_time += lib::as_seconds(x.finish());

            auto y = timer.push_back("reverse edges");
            add_reverse_edges(threads::IteratorPair{start, stop}, alpha, timer);
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

        update_type updates{threadpool_.size()};
        auto main = timer.push_back("main");
        threads::run(threadpool_, range, [&](const auto& local_indices, uint64_t tid) {
            // Thread local variables
            auto& thread_local_updates = updates.at(tid);

            // Scratch space.
            std::vector<Neighbor<Idx>> pool{};
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
                    greedy_search(
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
                        pool.push_back(modify_distance(neighbor));
                        visited.insert(neighbor.id());
                    }
                } else {
                    for (size_t i = 0, imax = search_buffer.size(); i < imax; ++i) {
                        const auto& neighbor = search_buffer[i];
                        pool.push_back(modify_distance(neighbor));
                        visited.insert(neighbor.id());
                    }
                }

                // Add neighbors of the query that are not part of `visited`.
                for (auto id : graph_.get_node(node_id)) {
                    assert(id != node_id);
                    // Try to emplace the node id into the visited set.
                    // If the id was inserted, then it didn't already exist in the visited
                    // set and we need to add it to the candidate pool.
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

                // Prune and wait for an update.
                thread_local_updates.emplace_back(node_id, std::vector<Idx>{});
                auto& pruned_results = thread_local_updates.back().second;
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
            }
        });

        main.finish();

        // Apply updates.
        auto update = timer.push_back("updates");
        threads::run(threadpool_, [&](uint64_t tid) {
            const auto& thread_local_updates = updates.at(tid);
            for (auto [node_id, update] : thread_local_updates) {
                graph_.replace_node(node_id, update);
            }
        });
    }

    ///
    /// Add reverse edges to the graph.
    ///
    template <typename /*std::ranges::random_access_range*/ R>
    void add_reverse_edges(const R& indices, float alpha, lib::Timer& timer) {
        // Apply backedges to all new candidate adjacency lists.
        // If adding an edge to the graph will cause it to violate the maximum degree
        // constraint, save the excess to the backedge buffer.
        auto backedge_timer = timer.push_back("backedge generation");
        auto range = threads::StaticPartition{indices};
        backedge_buffer_.reset();
        threads::run(threadpool_, range, [&](const auto& is, uint64_t SVS_UNUSED(tid)) {
            for (auto node_id : is) {
                for (auto other_id : graph_.get_node(node_id)) {
                    std::lock_guard lock{vertex_locks_[other_id]};
                    if (graph_.get_node_degree(other_id) < params_.graph_max_degree) {
                        graph_.add_edge(other_id, node_id);
                    } else {
                        backedge_buffer_.add_edge(other_id, node_id);
                    }
                }
            }
        });
        backedge_timer.finish();

        // For all vertices that now exceed the max degree requirement, run the pruning
        // procedure on the union of their current adjacency list as well as any extra edges
        // that were recorded in the previous process.
        //
        // Take care to avoid duplicate entries.
        auto prune_timer = timer.push_back("pruning backedges");
        threads::run(
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
                            candidates.push_back(make_neighbor(n));
                        }

                        // Add the old adjacency list.
                        for (auto n : graph_.get_node(src)) {
                            if (!neighbors.contains(n)) {
                                candidates.push_back(make_neighbor(n));
                            }
                        }
                        std::sort(candidates.begin(), candidates.end(), TotalOrder(cmp));
                        candidates.resize(
                            std::min(candidates.size(), params_.max_candidate_pool_size)
                        );

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
                        graph_.replace_node(src, pruned_results);
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
    /// Per-vertex locks.
    std::vector<SpinLock> vertex_locks_;
    /// Overflow backedge buffer.
    BackedgeBuffer<Idx> backedge_buffer_;
};
} // namespace svs::index::vamana
