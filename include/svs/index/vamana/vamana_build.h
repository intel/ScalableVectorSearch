/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

#pragma once

// local
#include "svs/concepts/data.h"
#include "svs/concepts/distance.h"
#include "svs/index/vamana/build_params.h"
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
#include <tuple>
#include <vector>

namespace svs::index::vamana {

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

    using self_distance = SelfDistance<Dist, data::const_value_type_t<Data>>;
    using build_distance_type = typename self_distance::type;

    using update_type =
        threads::SequentialTLS<std::vector<std::pair<Idx, std::vector<Idx>>>>;

    /// Constructor
    VamanaBuilder(
        Graph& graph,
        const Data& data,
        Dist distance_function,
        const VamanaBuildParameters& params,
        Pool& threadpool
    )
        : graph_{graph}
        , data_{data}
        , distance_function_{self_distance::modify(distance_function)}
        , params_{params}
        , threadpool_{threadpool}
        , vertex_locks_(data.size())
        , overflow_flags_(data.size())
        , overflow_maps_{threadpool.size()} {
        // Check class invariants.
        if (graph_.n_nodes() != data_.size()) {
            throw ANNEXCEPTION(
                "Expected graph to be pre-allocated with ", data_.size(), " vertices!"
            );
        }
    }

    void construct(float alpha, Idx entry_point, bool verbose = true) {
        construct(alpha, entry_point, threads::UnitRange<size_t>{0, data_.size()}, verbose);
    }

    template <typename R>
    void construct(float alpha, Idx entry_point, const R& range, bool verbose = true) {
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

        if (verbose) {
            fmt::print("Number of syncs: {}\n", num_batches);
            fmt::print("Batch Size: {}\n", batchsize);
        }

        // The base point for iteration.
        auto&& base = range.begin();
        auto timer = lib::Timer();
        for (size_t batch_id = 0; batch_id < num_batches; ++batch_id) {
            // Set up batch parameters
            auto start = std::min(num_nodes, batchsize * batch_id) + base;
            auto stop = std::min(num_nodes, batchsize * (batch_id + 1)) + base;

            // Perform search.
            auto x = timer.push_back("generate neighbors");
            generate_neighbors(threads::IteratorPair{start, stop}, entry_points, timer);
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
                if (verbose) {
                    constexpr std::string_view message =
                        "Completed round {} of {}. "
                        "Search Time: {:.4}s, "
                        "Reverse Time: {:.4}s, "
                        "Total Time: {:.4}s, "
                        "Estimated Remaining Time: {:.4}s\n";

                    fmt::print(
                        message,
                        batch_id + 1,
                        num_batches,
                        search_time,
                        reverse_time,
                        total_elapsed_time,
                        estimated_remaining_time
                    );
                }
                search_time = 0;
                reverse_time = 0;
                progress_counter += 1;
            }
        }
        if (verbose) {
            fmt::print("Completed pass using window size {}.\n", params_.window_size);
            timer.print();
        }
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
        const R& indices, const std::vector<Idx>& entry_points, lib::Timer& timer
    ) {
        auto range = threads::StaticPartition{indices};

        update_type updates{threadpool_.size()};
        auto main = timer.push_back("main");
        threads::run(threadpool_, range, [&](const auto& local_indices, uint64_t tid) {
            // Thread local variables
            auto& thread_local_updates = updates.at(tid);
            auto distance_function = threads::shallow_copy(distance_function_);
            std::vector<SearchNeighbor<Idx>> pool{};
            search_buffer_type search_buffer{params_.window_size};
            set_type<Idx> visited{};

            for (auto node_id : local_indices) {
                pool.clear();
                search_buffer.clear();
                visited.clear();
                const auto query = data_.get_datum(node_id);

                // Perform the greedy search.
                greedy_search(
                    graph_, data_, query, distance_function, search_buffer, entry_points
                );

                // Pull out the visited candidates from the search buffer.
                auto upper =
                    std::min(search_buffer.size(), params_.max_candidate_pool_size);
                for (size_t i = 0; i < upper; ++i) {
                    auto neighbor = search_buffer[i];
                    pool.push_back(neighbor);
                    visited.insert(neighbor.id_);
                }

                // Add neighbors of the query that are not part of `visisted`.
                distance::maybe_fix_argument(distance_function, query);
                for (auto id : graph_.get_node(node_id)) {
                    if (id != node_id && !visited.contains(id)) {
                        auto dist = distance::compute(
                            distance_function, query, data_.get_datum(id)
                        );
                        visited.insert(id);
                        pool.push_back(SearchNeighbor<Idx>(id, dist));
                    }
                }

                std::sort(
                    pool.begin(), pool.end(), distance::comparator(distance_function)
                );
                pool.resize(std::min(pool.size(), params_.max_candidate_pool_size));

                // Prune and wait for an update.
                thread_local_updates.emplace_back(node_id, std::vector<Idx>{});
                auto& pruned_results = thread_local_updates.back().second;
                heuristic_prune_neighbors(
                    params_.graph_max_degree,
                    //(95 * params_.graph_max_degree) / 100,
                    params_.alpha,
                    data_,
                    distance_function,
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
        // constraint, save the excess to the `overflow` associative data structure
        // in the case that the underlying graph can't support out-degrees higher than
        // the specified max degree.
        auto range = threads::StaticPartition{indices};

        // Clear all overflow maps.
        clear_threadlocal_overflow();
        clear_overflow_flags();

        auto backedges = timer.push_back("backedge generation");
        threads::run(threadpool_, range, [&](const auto& is, uint64_t tid) {
            auto& overflow_map = overflow_maps_.at(tid);
            for (auto node_id : is) {
                for (auto other_id : graph_.get_node(node_id)) {
                    std::lock_guard lock{vertex_locks_[other_id]};
                    if (graph_.get_node_degree(other_id) < params_.graph_max_degree) {
                        graph_.add_edge(other_id, node_id);
                    } else {
                        // Mark this node as being full.
                        // We can use relaxed semantics because all threads are only ever
                        // setting the atomic variables in this case.
                        //
                        // It doesn't matter if both try to write to the same variable
                        // because they are writing the same value.
                        set_overflow_flag(other_id, true);
                        auto [position, _] = overflow_map.try_emplace(other_id);
                        position.value().insert(node_id);
                    }
                }
            }
        });
        backedges.finish();

        // For all vertices that now exceed the max degree requirement, run the pruning
        // procedure on the union of their current adjacency list as well as any extra edges
        // that were recorded in the previous process.
        //
        // Take care to avoid duplicate entries.
        //
        // TODO: There's an interesting balance to this batch size.
        // If it's too large, than load balancing is really bad.
        // If it's too small, than we fight over the atomic variable running our load
        // balancing algorithms in `threads::run`.
        const size_t prune_batchsize = 1000;
        auto prune_handle = timer.push_back("pruning backedges");
        threads::run(
            threadpool_,
            threads::DynamicPartition{overflow_flags_.size(), prune_batchsize},
            [&](const auto node_indices, uint64_t SVS_UNUSED(tid)) {
                // Thread local auxiliary data structures.
                search_buffer_type buffer{params_.max_candidate_pool_size};
                std::vector<Idx> pruned_results{};
                auto distance_function = threads::shallow_copy(distance_function_);
                for (const auto node_id : node_indices) {
                    // If this entry is not marked as having overflowed, then move on to
                    // the next one.
                    if (!get_overflow_flag(node_id)) {
                        continue;
                    }

                    buffer.clear();
                    const auto node_data = data_.get_datum(node_id);
                    distance::maybe_fix_argument(distance_function, node_data);

                    // Helper lambda to make distance computations look a little cleaner.
                    auto make_neighbor = [&](auto i) {
                        return SearchNeighbor<Idx>{
                            i,
                            distance::compute(
                                distance_function, node_data, data_.get_datum(i)
                            )};
                    };

                    // Add the current neighbors to the list of potential neighbors.
                    for (auto n : graph_.get_node(node_id)) {
                        buffer.insert(make_neighbor(n));
                    }

                    // Snoop through all thread local overflows and add any additional
                    // neighobrs.
                    overflow_maps_.visit([&](const auto& map) {
                        if (auto itr = map.find(node_id); itr != map.end()) {
                            for (const auto& other_id : itr->second) {
                                buffer.insert(make_neighbor(other_id));
                            }
                        }
                    });

                    heuristic_prune_neighbors(
                        params_.graph_max_degree,
                        //(95 * params_.graph_max_degree) / 100,
                        alpha,
                        data_,
                        distance_function,
                        node_id,
                        buffer.view(),
                        pruned_results
                    );
                    graph_.replace_node(node_id, pruned_results);
                }
            }
        );
    }

    ///
    /// Clear all thread-local overflow maps.
    ///
    void clear_threadlocal_overflow() {
        threads::run(threadpool_, [&](uint64_t tid) { overflow_maps_.at(tid).clear(); });
    }

    ///
    /// Set overflow flag for node `i` to `value` using the provided memory ordering.
    ///
    /// By default, `std::memory_order_relaxed` is used because all threads are all trying
    /// to set flags, or purely reading flags.
    ///
    /// In the former case, there is no race if two threads set a flag because the end
    /// result of a flag being set is the same.
    ///
    /// In the latter, there is no race condition for read-only data.
    ///
    void set_overflow_flag(
        Idx i, bool value, std::memory_order ordering = std::memory_order_relaxed
    ) {
        getindex(overflow_flags_, i).store(value, ordering);
    }

    bool
    get_overflow_flag(Idx i, std::memory_order ordering = std::memory_order_relaxed) const {
        return getindex(overflow_flags_, i).load(ordering);
    }

    ///
    /// Clear all overflow flags.
    ///
    void clear_overflow_flags() {
        threads::run(
            threadpool_,
            threads::StaticPartition(overflow_flags_.size()),
            [&](const auto& is, uint64_t /*unused*/) {
                for (const auto& i : is) {
                    set_overflow_flag(i, false);
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
    build_distance_type distance_function_;
    /// Parameters regarding index construction.
    VamanaBuildParameters params_;
    /// Worker threadpool.
    Pool& threadpool_;
    /// Per-vertex locks.
    std::vector<SpinLock> vertex_locks_;
    /// Per-vertex overflow.
    std::vector<std::atomic<bool>> overflow_flags_;
    threads::SequentialTLS<tsl::robin_map<Idx, tsl::robin_set<Idx>>> overflow_maps_;
};
} // namespace svs::index::vamana
