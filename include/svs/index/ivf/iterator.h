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
#include "svs/index/ivf/extensions.h"
#include "svs/lib/misc.h"
#include "svs/lib/neighbor.h"
#include "svs/lib/scopeguard.h"

// stl
#include <optional>
#include <unordered_set>
#include <vector>

namespace svs::index::ivf {

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

/// @brief A batch iterator for retrieving neighbors from the IVF index in batches.
///
/// This iterator abstracts the process of retrieving neighbors in fixed-size batches
/// while maintaining internal state for efficient IVF traversal. Unlike Vamana's graph
/// traversal, IVF maintains state through centroid buffer capacity to expand the search
/// space in subsequent iterations.
///
/// @tparam Index The IVF index type (IVFIndex or DynamicIVFIndex)
/// @tparam QueryType The element type of the query vector
template <typename Index, typename QueryType> class BatchIterator {
  public:
    static_assert(
        std::is_trivially_copyable_v<QueryType>,
        "The batch iterator requires a trivial (no-throw copy constructible) type to "
        "provide its exception guarantees"
    );

    // Public type aliases
    using scratchspace_type = typename Index::scratchspace_type;
    using internal_id_type = size_t;
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
        const auto& buffer = scratchspace_->buffer_leaves[0];
        for (size_t i = 0, imax = buffer.size(); i < imax; ++i) {
            auto neighbor = buffer[i];
            auto internal_id = neighbor.id();
            auto result = yielded_.insert(internal_id);
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

    /// @brief Initializes the scratchspace with the configured capacity.
    void initialize_scratchspace() {
        // Create scratchspace with current n_probes and buffer capacity
        scratchspace_ =
            parent_->scratchspace(search_params_, extra_search_buffer_capacity_);
    }

    /// @brief Increments the search window by `batch_size` for the next iteration.
    /// This expands n_probes to search more centroids and increases buffer capacity.
    void increment_search_params(size_t batch_size) {
        // Increase n_probes to explore more clusters in subsequent iterations
        // This is similar to how Vamana increases the search window
        search_params_.n_probes_ =
            std::min(search_params_.n_probes_ + batch_size, parent_->num_clusters());

        // Increase buffer capacity to hold more results
        extra_search_buffer_capacity_ += batch_size;

        // Reinitialize scratchspace with new parameters
        initialize_scratchspace();
    }

  public:
    using size_type = typename result_buffer_type::size_type;
    using reference = value_type&;
    using const_reference = const value_type&;

    /// Random-access iterator to `value_type` over the current batch of results.
    using iterator = typename result_buffer_type::iterator;
    /// Random-access iterator to `const value_type` over the current batch of results.
    using const_iterator = typename result_buffer_type::const_iterator;

    /// @brief Constructs a batch iterator for the given query over the IVF index.
    /// @param parent The IVF index to search.
    /// @param query The query data.
    /// @param search_params Initial search parameters.
    /// @param extra_search_buffer_capacity Additional buffer capacity for the search.
    BatchIterator(
        Index& parent,
        std::span<const QueryType> query,
        const typename Index::search_parameters_type& search_params,
        size_t extra_search_buffer_capacity = 0
    )
        : parent_{&parent}
        , query_{query.begin(), query.end()}
        , search_params_{search_params}
        , extra_search_buffer_capacity_{extra_search_buffer_capacity} {
        detail::checkdims(query.size(), parent.dimensions());
        initialize_scratchspace();
    }

    /// @brief Constructs a batch iterator with default search parameters.
    /// @param parent The IVF index to search.
    /// @param query The query data.
    /// @param extra_search_buffer_capacity Additional buffer capacity for the search.
    BatchIterator(
        Index& parent,
        std::span<const QueryType> query,
        size_t extra_search_buffer_capacity = 0
    )
        : BatchIterator(
              parent,
              query,
              [&parent]() {
                  // Start with a reasonable initial n_probes for iteration
                  // Use 10% of clusters or at least 5, whichever gives more coverage
                  auto params = parent.get_search_parameters();
                  params.n_probes_ = std::max<size_t>(
                      params.n_probes_,
                      std::min(
                          parent.num_clusters(),
                          std::max<size_t>(5, parent.num_clusters() / 10)
                      )
                  );
                  return params;
              }(),
              extra_search_buffer_capacity
          ) {}

    /// @brief Updates the iterator with a new query.
    /// Resets the internal state and restarts the search when `next(...)` is called.
    void update(std::span<const QueryType> newquery) {
        detail::checkdims(newquery.size(), parent_->dimensions());
        assert(newquery.size() == query_.size());

        std::copy(newquery.begin(), newquery.end(), query_.begin());

        // Reset search parameters to initial values with reasonable n_probes
        search_params_ = parent_->get_search_parameters();
        search_params_.n_probes_ = std::max<size_t>(
            search_params_.n_probes_,
            std::min(
                parent_->num_clusters(), std::max<size_t>(5, parent_->num_clusters() / 10)
            )
        );
        extra_search_buffer_capacity_ = 0;
        initialize_scratchspace();
        restart_search_ = true;
        iteration_ = 0;
        yielded_.clear();
        results_.clear();
        is_exhausted_ = false;
    }

    /// @brief Adapts an internal neighbor to an external neighbor.
    /// For dynamic IVF, translates internal IDs to external IDs.
    template <NeighborLike N> svs::Neighbor<external_id_type> adapt(N internal) const {
        if constexpr (Index::needs_id_translation) {
            return Neighbor<external_id_type>{
                parent_->translate_internal_id(internal.id()), internal.distance()};
        } else {
            return Neighbor<external_id_type>{internal.id(), internal.distance()};
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
    /// The iterator is considered done when all the available nodes have been yielded,
    /// when all centroids have been searched, or when the search can not find any more
    /// neighbors.
    bool done() const {
        // The iterator is done when:
        // 1. All vectors in the index have been yielded, or
        // 2. The search has been exhausted (no new results after expanding search)
        // Note: We don't consider n_probes >= num_clusters as done, because we can still
        // yield results from previously searched clusters that weren't returned yet.
        return (yielded_.size() == parent_->size() || is_exhausted_);
    }

    /// @brief Forces the next iteration to restart the search from scratch.
    void restart_next_search() { restart_search_ = true; }

    /// @brief Returns the search parameters used for the current batch.
    typename Index::search_parameters_type parameters_for_current_iteration() const {
        return search_params_;
    }

    /// @brief Prepares the next batch of neighbors (up to ``batch_size``) from the index.
    /// Handles exceptions gracefully and ensures iterator state consistency.
    void next(
        size_t batch_size,
        const lib::DefaultPredicate& SVS_UNUSED(cancel) = lib::Returns(lib::Const<false>())
    ) {
        if (done()) {
            results_.clear();
            return;
        }

        // Always increment search parameters before search to ensure buffer capacity
        // On first call, this sets up the initial buffer; on subsequent calls, it expands
        increment_search_params(batch_size);

        // Perform search using single_search with scratchspace
        parent_->search(lib::as_const_span(query_), *scratchspace_);

        ++iteration_;
        restart_search_ = false;
        copy_from_scratch(batch_size);

        // If result is empty after calling next(), mark the iterator as exhausted.
        if (results_.size() == 0 && batch_size > 0) {
            is_exhausted_ = true;
        }
    }

  private:
    Index* parent_;                                        // The index being accessed.
    std::vector<QueryType> query_;                         // Local buffer for the query.
    std::optional<scratchspace_type> scratchspace_;        // Scratch space for search.
    typename Index::search_parameters_type search_params_; // Current search parameters.
    std::vector<Neighbor<external_id_type>> results_{};    // Filtered results from search.
    std::unordered_set<internal_id_type> yielded_{};       // Set of yielded neighbors.
    size_t iteration_ = 0;                                 // Current iteration number.
    bool restart_search_ = true; // Whether the next search should restart from scratch.
    size_t extra_search_buffer_capacity_ = 0; // Extra buffer capacity for the next search.
    bool is_exhausted_ = false;               // Whether the iterator is exhausted.
};

// Deduction Guides
template <typename Index, typename QueryType>
BatchIterator(const Index*, std::span<const QueryType>) -> BatchIterator<Index, QueryType>;

template <typename Index, typename QueryType>
BatchIterator(
    const Index*,
    std::span<const QueryType>,
    const typename Index::search_parameters_type&,
    size_t
) -> BatchIterator<Index, QueryType>;

} // namespace svs::index::ivf
