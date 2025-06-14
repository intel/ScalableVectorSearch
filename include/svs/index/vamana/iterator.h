/*
 * Copyright 2024 Intel Corporation
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
#include "svs/index/index.h"
#include "svs/index/vamana/dynamic_index.h"
#include "svs/index/vamana/index.h"
#include "svs/lib/scopeguard.h"

// stl
#include <unordered_set>
#include <vector>

namespace svs::index::vamana {

/// @brief A graph search initializer that uses the existing contents of the search buffer
/// to initialize the next round of graph search.
///
/// If a previous search exited with an exception, this initializer allows restarting
/// the search from scratch using the traditional method.
template <std::integral I> struct RestartInitializer {
    template <
        typename Buffer,
        typename Computer,
        graphs::ImmutableMemoryGraph Graph,
        typename Builder>
    void operator()(
        Buffer& buffer,
        const Computer& computer,
        const Graph& graph,
        const Builder& builder,
        vamana::NullTracker tracker // Compile error for non-NullTracker.
    ) const {
        // Restart the search from scratch if requested.
        if (hard_restart_) {
            vamana::EntryPointInitializer<I>{
                entry_points_}(buffer, computer, graph, builder, tracker);
            return;
        }

        // Otherwise, sort the buffer to prepare for the next search.
        buffer.sort();
    }

    // Entry points for restarting the search from scratch.
    std::span<const I> entry_points_;
    bool hard_restart_;
};

namespace detail {
constexpr void checkdims(size_t query_size, size_t index_dims) {
    if (query_size != index_dims) {
        throw ANNEXCEPTION(
            "Incompatible dimensions. Query has {} while the index expects {}.",
            query_size,
            index_dims
        );
    }
}
} // namespace detail

/// @brief A batch iterator for retrieving neighbors from the index in batches.
///
/// This iterator abstracts the process of retrieving neighbors in fixed-size batches
/// while maintaining internal state for efficient graph traversal.
template <typename Index, typename QueryType> class BatchIterator {
  public:
    static_assert(
        std::is_trivially_copyable_v<QueryType>,
        "The batch iterator requires a trivial (no-throw copy constructible) type to "
        "provide its exception guarantees"
    );

    // Public type aliases
    using scratchspace_type = index::scratchspace_t<Index>;
    using internal_id_type = typename Index::internal_id_type;
    using external_id_type = size_t;

    // The value type yielded by the iterator.
    using value_type = Neighbor<external_id_type>;

  private:
    // Private type aliases
    using result_buffer_type = std::vector<value_type>;

    /// @brief Copies results from the scratch buffer to the results buffer.
    /// Ensures that only unique neighbors are added to the results buffer.
    void copy_from_scratch(size_t batch_size) {
        results_.clear();
        const auto& buffer = scratchspace_.buffer;
        for (size_t i = 0, imax = buffer.size(); i < imax; ++i) {
            auto neighbor = buffer[i];
            auto result = yielded_.insert(neighbor.id());
            if (result.second /* inserted */) {
                // Rollback insertion into the yielded set if push_back throws.
                auto guard = lib::make_dismissable_scope_guard([&]() noexcept {
                    yielded_.erase(result.first);
                });
                results_.push_back(adapt(neighbor));
                guard.dismiss();
            }

            // Stop if the requested batch size is reached.
            if (results_.size() == batch_size) {
                break;
            }
        }
    }

    /// @brief Initializes the search buffer with the configured capacity.
    void initialize_buffer() {
        auto config = SearchBufferConfig{0, extra_search_buffer_capacity_};
        scratchspace_.buffer.change_maxsize(config);
        scratchspace_.buffer.clear();
    }

    /// @brief Increments the search window and capacity by `batch_size` for the next
    /// iteration.
    void increment_buffer(size_t batch_size) {
        auto config = scratchspace_.buffer.config();
        config.increment(batch_size);
        scratchspace_.buffer.change_maxsize(config);
    }

  public:
    using size_type = typename result_buffer_type::size_type;
    using reference = value_type&;
    using const_reference = const value_type&;

    /// Random-access iterator to `value_type` over the current batch of results.
    using iterator = typename result_buffer_type::iterator;
    /// Random-access iterator to `const value_type` over the current batch of results.
    using const_iterator = typename result_buffer_type::const_iterator;

    /// @brief Constructs a batch iterator for the given query over the index.
    /// @param parent The index to search.
    /// @param query The query data.
    /// @param extra_search_buffer_capacity Additional buffer capacity for the search.
    ///     When not provided, ``svs::ITERATOR_EXTRA_BUFFER_CAPACITY_DEFAULT = 100`` is
    ///     used.
    BatchIterator(
        const Index& parent,
        std::span<const QueryType> query,
        size_t extra_search_buffer_capacity = svs::UNSIGNED_INTEGER_PLACEHOLDER
    )
        : parent_{&parent}
        , query_{query.begin(), query.end()}
        , scratchspace_{parent_->scratchspace()} {
        detail::checkdims(query.size(), parent.dimensions());

        extra_search_buffer_capacity_ =
            extra_search_buffer_capacity == svs::UNSIGNED_INTEGER_PLACEHOLDER
                ? svs::ITERATOR_EXTRA_BUFFER_CAPACITY_DEFAULT
                : extra_search_buffer_capacity;
        initialize_buffer();
    }

    /// @brief Updates the iterator with a new query.
    /// Resets the internal state and restarts the search when `next(...)` is called.
    void update(std::span<const QueryType> newquery) {
        detail::checkdims(newquery.size(), parent_->dimensions());
        assert(newquery.size() == query_.size());

        std::copy(newquery.begin(), newquery.end(), query_.begin());

        initialize_buffer();
        restart_search_ = true;
        iteration_ = 0;
        yielded_.clear();
        results_.clear();
        is_exhausted_ = false;
    }

    /// @brief Adapts an internal neighbor to an external neighbor.
    template <NeighborLike N> svs::Neighbor<external_id_type> adapt(N internal) const {
        if constexpr (Index::needs_id_translation) {
            return Neighbor<external_id_type>{
                parent_->translate_internal_id(internal.id()), internal.distance()};
        } else {
            return internal;
        }
    }

    /// @brief Returns an iterator to the beginning of the results.
    iterator begin() { return results_.begin(); }
    /// @brief Returns an iterator to the end of the results.
    iterator end() { return results_.end(); }
    /// @copydoc begin()
    const_iterator begin() const { return results_.begin(); }
    /// @copydoc end()
    const_iterator end() const { return results_.end(); }
    /// @copydoc begin()
    const_iterator cbegin() const { return results_.cbegin(); }
    /// @copydoc begin()
    const_iterator cend() const { return results_.cend(); }

    /// @brief Returns a span over the current batch of neighbors.
    /// The span is invalidated by calls to `next(...)`.
    std::span<const value_type> contents() const { return lib::as_const_span(results_); }

    /// @brief Returns the number of buffered results.
    size_t size() const { return results_.size(); }

    /// @brief Return the batch number corresponding to the current buffer.
    size_t batch_number() const { return iteration_; }

    /// @brief Returns whether iterator can find more neighbors or not for the given query.
    ///
    /// The iterator is considered done when all the available nodes have been yielded or
    /// when the search can not find any more neighbors. The transition from not done to
    /// done will be triggered by a call to ``next()``. The contents of ``batch_number()``
    /// and ``parameters_for_current_iteration()`` will then remain unchanged by subsequent
    /// invocations of ``next()``.
    bool done() const { return (yielded_.size() == parent_->size() || is_exhausted_); }

    /// @brief Forces the next iteration to restart the search from scratch.
    void restart_next_search() { restart_search_ = true; }

    /// @brief Returns the search parameters used for the current batch.
    vamana::VamanaSearchParameters parameters_for_current_iteration() const {
        auto& buffer = scratchspace_.buffer;
        auto& prefetch = scratchspace_.prefetch_parameters;
        return VamanaSearchParameters{
            buffer.config(),
            buffer.visited_set_enabled(),
            prefetch.lookahead,
            prefetch.step};
    }

    /// @brief Prepares the next batch of neighbors (up to ``batch_size``) from the index.
    /// Handles exceptions gracefully and ensures iterator state consistency.
    void next(
        size_t batch_size,
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
    ) {
        if (done()) {
            results_.clear();
            return;
        }

        increment_buffer(batch_size);

        bool restart_search_copy = std::exchange(restart_search_, true);

        parent_->experimental_escape_hatch([&]<std::integral I>(
                                               const auto& graph,
                                               const auto& data,
                                               const auto& SVS_UNUSED(distance),
                                               std::span<const I> entry_points
                                           ) {
            auto search_closure =
                [&](const auto& query, const auto& accessor, auto& d, auto& buffer) {
                    constexpr vamana::extensions::UsesReranking<
                        std::remove_const_t<std::remove_reference_t<decltype(data)>>>
                        uses_reranking{};
                    if constexpr (uses_reranking()) {
                        distance::maybe_fix_argument(d, query);
                        for (size_t j = 0, jmax = buffer.size(); j < jmax; ++j) {
                            auto& neighbor = buffer[j];
                            auto id = neighbor.id();
                            auto new_distance =
                                distance::compute(d, query, data.get_primary(id));
                            neighbor.set_distance(new_distance);
                        }
                        buffer.sort();
                    }

                    vamana::greedy_search(
                        graph,
                        data,
                        accessor,
                        query,
                        d,
                        buffer,
                        RestartInitializer<I>{entry_points, restart_search_copy},
                        parent_->internal_search_builder(),
                        scratchspace_.prefetch_parameters,
                        cancel
                    );

                    if constexpr (Index::needs_id_translation) {
                        buffer.cleanup();
                    }
                };

            extensions::single_search(
                data,
                scratchspace_.buffer,
                scratchspace_.scratch,
                lib::as_const_span(query_),
                search_closure
            );
        });

        ++iteration_;
        restart_search_ = false;
        copy_from_scratch(batch_size);
        // If result is empty after calling next(), mark the iterator as exhausted.
        // The iterator will not be able to find any more neighbors.
        if (results_.size() == 0 && batch_size > 0) {
            is_exhausted_ = true;
        }
    }

  private:
    const Index* parent_;                               // The index being accessed.
    std::vector<QueryType> query_;                      // Local buffer for the query.
    scratchspace_type scratchspace_;                    // Scratch space for search.
    std::vector<Neighbor<external_id_type>> results_{}; // Filtered results from search.
    std::unordered_set<internal_id_type> yielded_{};    // Set of yielded neighbors.
    size_t iteration_ = 0;                              // Current iteration number.
    bool restart_search_ = true; // Whether the next search should restart from scratch.
    size_t extra_search_buffer_capacity_ =
        svs::UNSIGNED_INTEGER_PLACEHOLDER; // Extra buffer capacity for the next search.
    bool is_exhausted_ = false;            // Whether the iterator is exhausted.
};

// Deduction Guides
template <typename Index, typename QueryType>
BatchIterator(const Index*, std::span<const QueryType>) -> BatchIterator<Index, QueryType>;

} // namespace svs::index::vamana
