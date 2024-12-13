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
#include "svs/core/distance.h"
#include "svs/core/graph.h"
#include "svs/index/vamana/extensions.h"
#include "svs/index/vamana/prune.h"
#include "svs/lib/array.h"
#include "svs/lib/threads.h"
#include "svs/lib/timing.h"

// external
#include "tsl/robin_set.h"

// stdlib
#include <span>
#include <unordered_set>

namespace svs::index::vamana {

///
/// Parameters controlling aspects of the graph consolidation process.
///
/// * `update_batch_size`: The algorithm for graph consolidation is a two-phase algorithm
///   over batches of the dataset to facalitate parallelism.
///
///   The first phase is a read-only phase where updates for the graph are prepared in an
///   auxiliary data structure. The second phase commits these updates to the graph.
///
///   This multi-phase approach allows for parallelism with both phases without worrying
///   about mutating the graph while reading from it.
///
///   This parameter controlls how large of a batch is processed during each phase.
///
/// * `prune_to`: The number of candidates to prune to.
///
/// * `alpha`: The pruning parameter to use when constructing a new set of neighbors for
///   vertices with at least one deleted neighbor.
///
struct ConsolidationParameters {
    size_t update_batch_size;
    size_t prune_to;
    size_t max_candidate_pool_size;
    float alpha;
};

///
/// A helper struct to store pending updates for the consolidated graph.
/// Represents pending updates using several large allocations rather than many small ones.
///
template <std::integral I> class BulkUpdate {
  public:
    // Constructor
    BulkUpdate(size_t max_batch_size, size_t prune_to)
        : neighbors_{make_dims(max_batch_size, prune_to)}
        , lengths_{max_batch_size}
        , needs_update_{max_batch_size} {}

    ///
    /// Pre-conditions:
    /// * 0 <= src < max_batch_size
    /// * neighbors.size() <= prune_to
    /// * May be called concurrently from multiple threads as long as `src` is
    ///   unique for each thread.
    ///
    template <typename Allocator>
    void insert(size_t src, const std::vector<I, Allocator>& neighbors) {
        assert(neighbors.size() <= getsize<1>(neighbors_));
        assert(src < getsize<0>(neighbors_));

        needs_update_.at(src) = true;
        // N.B.: There is no way we get even close to crossing what can be
        // expressed as a 32-bit unsigned number.
        lengths_.at(src) = lib::narrow_cast<I>(neighbors.size());
        std::copy(neighbors.begin(), neighbors.end(), neighbors_.slice(src).begin());
    }

    ///
    /// Return `true` if index `src` has a pending update.
    ///
    bool needs_update(size_t src) const { return needs_update_.at(src); }

    ///
    /// Return the pending update for `src`.
    /// Pre-conditions:
    /// * `needs_update(src) == true`
    ///
    std::span<const I> get_update(size_t src) const {
        assert(needs_update(src));
        return neighbors_.slice(src).first(lengths_.at(src));
    }

    ///
    /// Prepare the data structure for another batch of processing.
    /// **Preconditions:**
    ///
    /// * Must be called from a single thread.
    ///
    void prepare() { std::fill(needs_update_.begin(), needs_update_.end(), false); }

  private:
    Matrix<I> neighbors_;
    Vector<I> lengths_;
    // N.B.: Use a `Vector` instead of a `std::vector` because `std::vector`
    // specialized on `bool` and will thus cannot be guarentee coherent updates from
    // multiple threads.
    Vector<bool> needs_update_;
};

template <std::integral I> struct ConsolidateThreadLocal {
    template <typename T> using allocator_type = threads::CacheAlignedAllocator<T>;

    // Type Aliases
    using set_type = tsl::robin_set<I, std::hash<I>, std::equal_to<I>, allocator_type<I>>;
    using neighbor_vector_type = std::vector<Neighbor<I>, allocator_type<Neighbor<I>>>;

    // Members
    set_type all_candidates{};
    neighbor_vector_type valid_candidates{};
    std::vector<I, allocator_type<I>> final_candidates{};
};

template <
    graphs::MemoryGraph Graph,
    data::ImmutableMemoryDataset Data,
    threads::ThreadPool Pool,
    typename Distance>
class GraphConsolidator {
  public:
    // Type Aliases
    using I = typename Graph::index_type;
    using graph_neighbor_container = typename Graph::const_value_type;
    using datum_type = typename Data::const_value_type;

    using Compare = typename Distance::compare;
    using scratch_type = ConsolidateThreadLocal<I>;

    using set_type = typename scratch_type::set_type;
    using neighbor_vector_type = typename scratch_type::neighbor_vector_type;

    // Members
  private:
    Graph& graph_;
    const Data& data_;
    Pool& threadpool_;
    const Distance& distance_;
    ConsolidationParameters params_;

  public:
    // Constructor
    GraphConsolidator(
        Graph& graph,
        const Data& data,
        Pool& threadpool,
        const Distance& distance,
        const ConsolidationParameters& params
    )
        : graph_{graph}
        , data_{data}
        , threadpool_{threadpool}
        , distance_{distance}
        , params_{params} {
        assert(graph.n_nodes() == data.size());
    }

    ///
    /// Add all neighbors and neighbor-of-deleted-neighbors to the set `all_candidates`.
    ///
    /// @param all_candidates In-out parameter. After this function call, `all_candidates`
    ///        will contain the full set of neighbor candidate indices.
    /// @param neighbors The current neighbors of the vertex being processed.
    /// @param is_deleted Callable functor returning `true` of a vertex is deleted.
    ///
    template <typename Deleted>
    void populate_candidates(
        set_type& all_candidates,
        const graph_neighbor_container& neighbors,
        const Deleted& is_deleted
    ) const {
        all_candidates.clear();
        for (auto dst : neighbors) {
            if (is_deleted(dst)) {
                const auto& others = graph_.get_node(dst);
                all_candidates.insert(others.begin(), others.end());
            } else {
                all_candidates.insert(dst);
            }
        }
    }

    template <data::AccessorFor<Data> Accessor, typename SelfDistance, typename Deleted>
    void filter_candidates(
        neighbor_vector_type& valid_candidates,
        const set_type& all_candidates,
        const datum_type& src_data,
        const Accessor& accessor,
        SelfDistance& distance,
        const Deleted& is_deleted
    ) const {
        distance::maybe_fix_argument(distance, src_data);
        valid_candidates.clear();
        for (auto dst : all_candidates) {
            if (is_deleted(dst)) {
                continue;
            }

            valid_candidates.push_back(
                {dst, distance::compute(distance, src_data, accessor(data_, dst))}
            );
        }

        std::sort(valid_candidates.begin(), valid_candidates.end(), Compare{});
    }

    template <typename Deleted>
    void generate_updates(
        const threads::UnitRange<size_t>& global_ids,
        const threads::UnitRange<size_t>& local_ids,
        BulkUpdate<I>& update_buffer,
        ConsolidateThreadLocal<I>& tls,
        const Deleted& is_deleted
    ) const {
        auto& [all_candidates, valid_candidates, final_candidates] = tls;

        auto build_adaptor = extensions::build_adaptor(data_, distance_);

        auto accessor = build_adaptor.general_accessor();
        auto&& general_distance = build_adaptor.general_distance();

        for (auto i : local_ids) {
            size_t src = global_ids[i];

            if (is_deleted(src)) {
                continue;
            }

            // Determine if any of the neighbors of this node are deleted.
            const auto& neighbors = graph_.get_node(src);
            if (std::none_of(neighbors.begin(), neighbors.end(), is_deleted)) {
                continue;
            }

            // Add all neighbors and neighbors-of-deleted-neighbors.
            populate_candidates(all_candidates, neighbors, is_deleted);

            // Insert non-deleted candidates into the vector to prepare for pruning.
            filter_candidates(
                valid_candidates,
                all_candidates,
                accessor(data_, src),
                accessor,
                general_distance,
                is_deleted
            );

            size_t new_candidate_size =
                std::min(valid_candidates.size(), params_.max_candidate_pool_size);
            valid_candidates.resize(new_candidate_size);
            heuristic_prune_neighbors(
                prune_strategy(distance_),
                params_.prune_to,
                params_.alpha,
                data_,
                accessor,
                general_distance,
                src,
                lib::as_const_span(valid_candidates),
                final_candidates
            );
            update_buffer.insert(i, final_candidates);
        }
    }

    ///
    /// Write pending updates to the graph.
    ///
    void apply_updates(
        BulkUpdate<I>& update_buffer,
        const threads::UnitRange<size_t>& global_ids,
        const threads::UnitRange<size_t>& local_ids
    ) {
        for (auto i : local_ids) {
            if (update_buffer.needs_update(i)) {
                graph_.replace_node(global_ids[i], update_buffer.get_update(i));
            }
        }
    }

    template <typename Delete> void operator()(const Delete& is_deleted) {
        // Allocate necessary scratch space.
        BulkUpdate<I> update_buffer{params_.update_batch_size, params_.prune_to};
        threads::SequentialTLS<ConsolidateThreadLocal<I>> tls{threadpool_.size()};

        const size_t num_nodes = graph_.n_nodes();
        const size_t update_batch_size = std::min(params_.update_batch_size, num_nodes);
        const size_t thread_batch_size = 500;

        size_t start = 0;
        while (start < num_nodes) {
            size_t stop = std::min(num_nodes, start + update_batch_size);

            // Generate updates.
            update_buffer.prepare();
            threads::UnitRange global_ids{start, stop};
            threads::run(
                threadpool_,
                threads::DynamicPartition{global_ids.eachindex(), thread_batch_size},
                [&](const auto& local_ids, uint64_t tid) {
                    auto& thread_local_scratch = tls.at(tid);
                    generate_updates(
                        global_ids,
                        threads::UnitRange(local_ids),
                        update_buffer,
                        thread_local_scratch,
                        is_deleted
                    );
                }
            );

            // Write back results.
            threads::run(
                threadpool_,
                threads::DynamicPartition{global_ids.eachindex(), thread_batch_size},
                [&](const auto& local_ids, uint64_t /*tid*/) {
                    apply_updates(update_buffer, global_ids, threads::UnitRange(local_ids));
                }
            );

            // Prepare for the next iteration.
            start = stop;
        }
    }
};

template <
    graphs::MemoryGraph Graph,
    data::ImmutableMemoryDataset Data,
    threads::ThreadPool Pool,
    typename Distance,
    typename Deleted>
void consolidate(
    Graph& graph,
    const Data& data,
    Pool& threadpool,
    size_t prune_to,
    size_t max_candidate_pool_size,
    float alpha,
    const Distance& distance,
    Deleted&& is_deleted
) {
    ConsolidationParameters params{200'000, prune_to, max_candidate_pool_size, alpha};
    auto consolidator = GraphConsolidator{graph, data, threadpool, distance, params};
    consolidator(is_deleted);
}

} // namespace svs::index::vamana
