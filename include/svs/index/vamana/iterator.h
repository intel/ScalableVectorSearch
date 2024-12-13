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
#include "svs/index/vamana/iterator_schedule.h"
#include "svs/lib/scopeguard.h"

// stl
#include <unordered_set>
#include <vector>

namespace svs::index::vamana {

/// A graph search initializer that uses the original contents of the search buffer
/// to kick-start the next round of graph search.
///
/// If a previous search exited with an exception, then we need to be able to restart search
/// from scratch using the traditional method.
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
        // Start again from scratch if requested.
        if (hard_restart_) {
            vamana::EntryPointInitializer<I>{
                entry_points_}(buffer, computer, graph, builder, tracker);
            return;
        }

        // Happy path - we can reuse the contents of the search buffer.
        buffer.soft_clear();
    }

    // Entry Points in case we need to restart from scratch.
    std::span<const I> entry_points_;
    bool hard_restart_;
};

/////
///// Concrete Schedules
/////

namespace detail {
constexpr void checkdims(size_t query_size, size_t index_dims) {
    if (query_size != index_dims) {
        throw ANNEXCEPTION(
            "Incompatible dimensions. Query has {} while the index expectes {}.",
            query_size,
            index_dims
        );
    }
}
} // namespace detail

template <typename Index, typename QueryType, IteratorSchedule Schedule = DefaultSchedule>
class BatchIterator {
  public:
    static_assert(
        std::is_trivially_copyable_v<QueryType>,
        "The batch iterator requires a trivial (no-throw copy constructible) type to "
        "provide its exception guarentees"
    );

    // Public type-aliases
    using scratchspace_type = index::scratchspace_t<Index>;
    using internal_id_type = typename Index::internal_id_type;
    using external_id_type = size_t;

    /// @brief The value yielded by this container's iterator interface.
    using value_type = Neighbor<external_id_type>;

  private:
    // Private type-aliases
    using result_buffer_type = std::vector<value_type>;

    // Clear the results buffer and copy into it the contents of the scratch buffer.
    void copy_from_scratch(size_t max_candidates) {
        results_.clear();
        const auto& buffer = scratchspace_.buffer;
        for (size_t i = 0, imax = buffer.size(); i < imax; ++i) {
            auto neighbor = buffer[i];
            // Unfortunately, lambdas cannot capture structured bindings.
            // So we need to manually unpack the result from `insert`.
            auto result = yielded_.insert(neighbor.id());
            if (result.second /*inserted*/) {
                // If push_back throws - rollback the insertion into the `yielded` set.
                auto guard = lib::make_dismissable_scope_guard([&]() noexcept {
                    // `std::unordered_set` documented to not throw exceptions on `erase`.
                    yielded_.erase(result.first);
                });
                results_.push_back(adapt(neighbor));
                guard.dismiss();
            }

            // Break if we've yielded the requested number of candidates.
            if (results_.size() == max_candidates) {
                break;
            }
        }
    }

  public:
    using size_type = typename result_buffer_type::size_type;
    using reference = value_type&;
    using const_reference = const value_type&;

    /// A random-access, contiguous iterator to ``value_type`` over the current batch of
    /// results.
    using iterator = typename result_buffer_type::iterator;
    /// A random-access, contiguous iterator to ``cosnt value_type`` over the current batch
    /// of results.
    using const_iterator = typename result_buffer_type::const_iterator;

    BatchIterator(const Index& parent, std::span<const QueryType> query, Schedule schedule)
        : parent_{&parent}
        , query_{query.begin(), query.end()}
        , scratchspace_{parent.scratchspace()}
        , schedule_{std::move(schedule)} {
        // TODO: Can we delegate the dimensionality check to the index?
        detail::checkdims(query.size(), parent.dimensions());

        // Update the newly allocated scratchspace with the search parameters for the first
        // iteration.
        size_t max_candidates = schedule_.max_candidates(0);
        scratchspace_.apply(schedule_.for_iteration(0));

        // Perform a traditional graph search to initialize the internal scratchspace.
        // The internal state of the scratch space is preserved for reuse on future runs.
        parent.search(lib::as_const_span(query_), scratchspace_);
        copy_from_scratch(max_candidates);
    }

    void update(std::span<const QueryType> newquery) { update(newquery, schedule_); }

    void update(std::span<const QueryType> newquery, const Schedule& new_schedule) {
        detail::checkdims(newquery.size(), parent_->dimensions());
        assert(newquery.size() == query_.size());

        // Perform the initial search.
        // Make sure it completes successfully before updating local state to provide
        // a basic exception guarenteee.

        // We're reusing the scratchspace to conduct the new search.
        // If search fails and the caller retries `next` with the previous query, we will
        // need to completely restart search.
        restart_search_ = true;
        size_t max_candidates = new_schedule.max_candidates(0);
        scratchspace_.apply(schedule_.for_iteration(0));
        parent_->search(newquery, scratchspace_);

        // Use the copy-and-swap idiom for the new schedule to support copy-constructors
        // that can throw.
        //
        // If ``update`` is called with the original schedule, then we don't need to update.
        bool schedule_needs_update = &new_schedule != &schedule_;
        if (schedule_needs_update) {
            if constexpr (std::is_nothrow_copy_assignable_v<Schedule>) {
                schedule_ = new_schedule;
            } else {
                // Schedules are required to be nothrow swappable.
                // Copy-construction can throw, but throwing here is still okay.
                auto tmp = new_schedule;
                using std::swap;
                swap(schedule_, tmp);
            }
        }

        // At this point - search is successful. We're committed to updating the query.
        // Copy should not throw as long as the QueryType is trivially copyable.
        std::copy(newquery.begin(), newquery.end(), query_.begin());

        // Clearing data structures should not throw.
        iteration_ = 0;
        yielded_.clear();
        copy_from_scratch(max_candidates);
        restart_search_ = false;

        // Schedules must be no-throw copy assignable.
        schedule_ = new_schedule;
    }

    // Hook to perform late index translation.
    // TODO: The dynamic index *really* needs a better interface for this.
    template <NeighborLike N> svs::Neighbor<external_id_type> adapt(N internal) const {
        if constexpr (Index::needs_id_translation) {
            return Neighbor<external_id_type>{
                parent_->translate_internal_id(internal.id()), internal.distance()};
        } else {
            return internal;
        }
    }

    /// Return an iterator to the beginning.
    iterator begin() { return results_.begin(); }
    /// Return an iterato to the end.
    iterator end() { return results_.end(); }
    /// @copydoc begin()
    const_iterator begin() const { return results_.begin(); }
    /// @copydoc end()
    const_iterator end() const { return results_.end(); }
    /// @copydoc begin()
    const_iterator cbegin() const { return results_.cbegin(); }
    /// @copydoc begin()
    const_iterator cend() const { return results_.cend(); }

    /// @brief Return a span over the current batch of neighbors.
    ///
    /// The return span will be invalidated by calls to ``next()``.
    std::span<const value_type> contents() const { return lib::as_const_span(results_); }

    /// @brief Return the number of buffered results.
    size_t size() const { return results_.size(); }

    /// @brief Return the current batch number contained in the buffer.
    size_t batch() const { return iteration_; }

    /// @brief Return whether the entire entries in the index have been yielded.
    ///
    /// The transition from not done to done will be triggered by a call to ``next()``.
    /// The contents of ``batch()`` and ``parameters_for_current_iteration()`` will then
    /// remain unchanged by subsequent invocations of ``next()``.
    bool done() const { return yielded_.size() == parent_->size(); }

    /// @brief Require that the next iteration start search from scratch.
    ///
    /// This method is mainly used for testing purposes and to identify possible issues
    /// related to maintaining state in the internal scratchspace.
    ///
    /// The assignment is only valid for the next invocation of `next`, after which the
    /// default behavior of the `BatchIterator` is restored.
    void restart_next_search() { restart_search_ = true; }

    /// @brief Return a const-ref to the underlying schedule.
    const Schedule& schedule() const { return schedule_; }

    /// @brief Return a mutable reference to the underlying schedule
    Schedule& schedule() { return schedule_; }

    /// @brief Return the parameters used for the most recent iteration.
    vamana::VamanaSearchParameters parameters_for_current_iteration() const {
        return schedule_.for_iteration(iteration_);
    }

    /// @brief Retrieve the next batch of neighbors from the index.
    ///
    /// This function provides the basic exception guarantee with the following semantics:
    ///
    /// * If an exception is thrown during search, the contents and batch number of the
    ///   ``BatchIterator`` remain unchanged. This can be detected by using ``batch()``
    ///   to determine the current batch buffered in this container.
    ///
    /// * If an exception is thrown after search (this should be extremely rare), then the
    ///   contents and batch number will be incremented but the contents of container may
    ///   be less than the schedule's requested batch size.
    ///
    ///   This may cause some de-sync between the yielded elements and the batch size
    ///   requested by the schedule, but the iterator should continue to work.
    ///
    /// After this function returns, a new batch of neighbors can be retrieved using this
    /// container's iterator interface.
    ///
    /// If ``done()`` returns ``true`` prior to calling this function, then the internal
    /// result buffer will be emptied and no new neighbors will be returned.
    ///
    /// @sa ``done()``
    void next() {
        if (done()) {
            results_.clear();
            return;
        }

        // Increment the number of neighbors to return.
        // Defer actually incrementing of the member variable until after search is
        // completed to maintain class invariants in the presence of exceptions.
        size_t this_iteration = iteration_ + 1;
        size_t max_candidates = schedule_.max_candidates(this_iteration);
        const auto& p = schedule_.for_iteration(this_iteration);
        scratchspace_.apply(p);

        // Conservatively set the `restart_search_` so that if an exception is thrown during
        // search, we begin from scratch on the next round.
        //
        // Make a copy of the old value for capturing in the lambda below.
        bool restart_search_copy = std::exchange(restart_search_, true);

        // Rerun search using the stashed contents of the search buffer.
        parent_->experimental_escape_hatch([&]<std::integral I>(
                                               const auto& graph,
                                               const auto& data,
                                               const auto& SVS_UNUSED(distance),
                                               std::span<const I> entry_points
                                           ) {
            auto search_closure =
                [&](const auto& query, const auto& accessor, auto& d, auto& buffer) {
                    vamana::greedy_search(
                        graph,
                        data,
                        accessor,
                        query,
                        d,
                        buffer,
                        RestartInitializer<I>{entry_points, restart_search_copy},
                        parent_->internal_search_builder(),
                        vamana::GreedySearchPrefetchParameters{
                            p.prefetch_lookahead_,
                            p.prefetch_step_,
                        }
                    );

                    // TODO: Need a better way of identifying if we're working with
                    // the dynamic index or not.
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

        // Search was successful.
        // We do not need to start the next search from scratch.
        ++iteration_;
        restart_search_ = false;
        copy_from_scratch(max_candidates);
    }

  private:
    // The index we are accessing.
    const Index* parent_;
    // Local buffer for the query.
    std::vector<QueryType> query_;
    // Scratchspace for search.
    // TODO: Provide a better API for scratch spaces.
    scratchspace_type scratchspace_;
    // Filtered results from search.
    std::vector<Neighbor<external_id_type>> results_{};
    // Yielded neighbors.
    // Keep track of internal IDs since (generally speaking), internal IDs should have
    // a smaller width than external IDs.
    std::unordered_set<internal_id_type> yielded_{};
    // Which iteration is currently being processed.
    size_t iteration_ = 0;
    // If an exception is thrown during search,
    bool restart_search_ = false;
    // The search buffer schedule.
    Schedule schedule_;
};

// Deduction Guides
template <typename Index, typename QueryType, IteratorSchedule Schedule>
BatchIterator(const Index*, std::span<const QueryType>, Schedule)
    -> BatchIterator<Index, QueryType, Schedule>;

} // namespace svs::index::vamana
