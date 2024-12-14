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

#include "svs/index/vamana/filter.h"
#include "svs/lib/datatype.h"
#include "svs/lib/neighbor.h"
#include "svs/lib/prefetch.h"
#include "svs/lib/threads/threadlocal.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace svs::index::vamana {

// Forward declaration
template <typename Idx, typename Cmp> class SearchBuffer;

class SearchBufferConfig {
    // Class Invariants:
    // * search_window_size_ <= total_capacity_;
  private:
    size_t search_window_size_{0};
    size_t total_capacity_{0};

    // Tags to bypass invariant checks.
    struct BypassInvariantCheckTag {};

  public:
    constexpr SearchBufferConfig() = default;

    SearchBufferConfig(
        BypassInvariantCheckTag, size_t search_window_size, size_t total_capacity
    )
        : search_window_size_{search_window_size}
        , total_capacity_{total_capacity} {}

    SearchBufferConfig(size_t search_window_size, size_t total_capacity)
        : SearchBufferConfig(
              BypassInvariantCheckTag(), search_window_size, total_capacity
          ) {
        check_invariants();
    }
    /*not explicit*/ SearchBufferConfig(size_t search_window_size)
        : SearchBufferConfig(
              BypassInvariantCheckTag(), search_window_size, search_window_size
          ) {}

    // Increment both the window size and capacity.
    void increment(size_t by) {
        search_window_size_ += by;
        total_capacity_ += by;
    }

    void increment(SearchBufferConfig by) {
        search_window_size_ += by.search_window_size_;
        total_capacity_ += by.total_capacity_;
    }

    // Read-only accessors.
    size_t get_search_window_size() const { return search_window_size_; }
    size_t get_total_capacity() const { return total_capacity_; }

    void check_invariants() const {
        if (search_window_size_ > total_capacity_) {
            throw ANNEXCEPTION(
                "Improper configuration for search buffer! Effective size ({}) cannot be "
                "less than capacity ({}).",
                search_window_size_,
                total_capacity_
            );
        }
    }

    friend bool operator==(SearchBufferConfig, SearchBufferConfig) = default;
};

///
/// @brief Class used to store search results for static greedy search.
///
/// @tparam Idx Type used to uniquely identify DB vectors
/// @tparam Cmp Type of the comparison function used to sort neighbors by distance.
///
template <typename Idx, typename Cmp = std::less<>> class SearchBuffer {
  public:
    // External type aliases
    using index_type = Idx;
    using value_type = SearchNeighbor<Idx>;
    using reference = value_type&;
    using const_reference = const value_type&;
    using compare_type = Cmp;

    using vector_type = std::vector<value_type, threads::CacheAlignedAllocator<value_type>>;
    using iterator = typename vector_type::iterator;
    using const_iterator = typename vector_type::const_iterator;

    /// A visited filter with 65,535 entries with a memory footpring of 128 kiB.
    using set_type = VisitedFilter<Idx, 16>;

    ///
    /// @brief Initialize a buffer with zero capacity.
    ///
    /// In order to use a buffer that has been default constructed, use the
    /// @ref change_maxsize(size_t) method.
    ///
    SearchBuffer() = default;

    ///
    /// @brief Construct a search buffer with the target capacity and comparison function.
    ///
    /// @param config The configuration for split region of interest (ROI) size and total
    ///     capacity.
    /// @param compare The functor used to compare two ``SearchNeighbor``s together.
    /// @param enable_visited Whether or not the visited set is enabled.
    ///
    explicit SearchBuffer(
        SearchBufferConfig config, Cmp compare = {}, bool enable_visited = false
    )
        : compare_{std::move(compare)}
        , search_window_size_{config.get_search_window_size()}
        , capacity_{config.get_total_capacity()}
        , candidates_{capacity_ + 1}
        , visited_{std::nullopt} {
        if (enable_visited) {
            enable_visited_set();
        }
    }

    ///
    /// @brief Construct a search buffer with the target capacity and comparison function.
    ///
    /// @param size The number of valid elements to return from a search operation.
    /// @param compare The functor used to compare two ``SearchNeighbor``s together.
    /// @param enable_visited Whether or not the visited set is enabled.
    ///
    explicit SearchBuffer(size_t size, Cmp compare = Cmp{}, bool enable_visited = false)
        : SearchBuffer{SearchBufferConfig(size), std::move(compare), enable_visited} {}

    ///
    /// @brief Perform an efficient copy.
    ///
    /// Copy the portions of the SearchBuffer that matter for the purposes of scratch
    /// space.
    ///
    /// Perserves the sizes of various containers but not necessarily the values.
    ///
    SearchBuffer shallow_copy() const {
        // We care about the contents of the buffer - just its size.
        // Therefore, we can construct a new buffer from scratch.
        return SearchBuffer{config(), compare_, visited_set_enabled()};
    }

    // TODO: Allow this construction to be noexcept since the pre-conditions for the
    // search buffer are already satisfied.
    SearchBufferConfig config() const {
        return SearchBufferConfig{search_window_size_, capacity_};
    }

    ///
    /// @brief Change the target number of elements to return after search.
    ///
    /// @param new_size The new number of elements to return.
    ///
    /// Post conditions:
    /// - The capacity of the search buffer will be set to the new size.
    /// - The actual size (number of contained elements) will be the minimum of the current
    ///   size and the new size.
    ///
    void change_maxsize(size_t new_size) {
        search_window_size_ = new_size;
        capacity_ = new_size;
        candidates_.resize(new_size + 1);
        size_ = std::min(size_, new_size);
    }

    ///
    /// @brief Change the target number of elements to return after search.
    ///
    /// @param config The new configuration for the buffer.
    ///
    /// Post conditions:
    /// - The capacity of the search buffer will be set to the new capacity.
    /// - The actual size (number of contained elements) will be the minimum of the current
    ///   size and the new size.
    ///
    void change_maxsize(SearchBufferConfig config) {
        search_window_size_ = config.get_search_window_size();
        capacity_ = config.get_total_capacity();
        candidates_.resize(capacity_ + 1);
        size_ = std::min(size_, capacity_);
    }

    ///
    /// @brief Prepare the buffer for a new search operation.
    ///
    void clear() {
        size_ = 0;
        best_unvisited_ = 0;
        if (visited_set_enabled()) {
            visited_->reset();
        }
    }

    void soft_clear() {
        bool use_visited_set = visited_set_enabled();
        if (use_visited_set) {
            visited_->reset();
        }

        // Mark all neighbors as no longer visited.
        // Use a loop over `[0, size_)` since this is likely called after growing the buffer
        // and the contents of the extended elements are invalid.
        for (size_t i = 0; i < size_; ++i) {
            auto& neighbor = candidates_[i];
            neighbor.clear_visited();
            if (use_visited_set) {
                visited_->emplace(neighbor.id());
            }
        }

        // Reset the best unvisited back to the beginning.
        best_unvisited_ = 0;
    }

    /// @brief Return the current number of valid elements in the buffer.
    size_t size() const { return size_; }

    /// @brief Return the maximum number of neighbors that can be held by the buffer.
    size_t capacity() const { return capacity_; }

    /// @brief Return whether or not the buffer is full of valid elements.
    bool full() const { return size() == capacity(); }

    /// @brief Access the neighbor at position `i`.
    reference operator[](size_t i) { return candidates_[i]; }

    /// @brief Access the neighbor at position `i`.
    const_reference operator[](size_t i) const { return candidates_[i]; }

    /// @brief Return the furtherst valid neighbor.
    reference back() { return candidates_[size_ - 1]; }

    /// @brief Return the furtherst valid neighbor.
    const_reference back() const { return candidates_[size_ - 1]; }

    /// @brief Return the position of the best unvisited neighbor.
    size_t best_unvisited() const { return best_unvisited_; }

    /// @brief Return a view of the backing data for this buffer.
    std::span<const value_type> view() const {
        return std::span<const value_type>(candidates_.data(), size());
    }

    ///
    /// @brief Return `true` if the search buffer has reached its terminating condition.
    ///
    /// Note: If `done()` evaluates to `true`, do not try to extract further candidates
    /// from it using `next()`.
    ///
    bool done() const { return best_unvisited_ == std::min(size_, search_window_size_); }

    ///
    /// @brief Return the best unvisited neighbor in the buffer.
    ///
    /// The returned result will be convertible to `Neighbor<Idx>`.
    ///
    /// Pre-conditions:
    /// * `search_buffer.done()` must evaluate to `false`, otherwise an out of bounds
    ///   access will occur.
    ///
    /// Post-conditions:
    /// * The returned neighbor will be marked as visited.
    ///
    const_reference next() {
        // Get the best unvisited node
        SearchNeighbor<Idx>& node = candidates_[best_unvisited_];
        node.set_visited();

        // Increment `best_unvisited_` until it's equal to the size OR until we encounter
        // an unvisited node.
        size_t upper = std::min(size(), search_window_size_);
        while (++best_unvisited_ != upper && candidates_[best_unvisited_].visited()) {}
        return node;
    }

    ///
    /// @brief Place the neighbor at the end of the search buffer if `full() != true`.
    ///
    /// Otherwise, do nothing.
    ///
    void push_back(value_type neighbor) {
        if (!full()) {
            *end() = neighbor;
            ++size_;
        }
    }

    // Define iterators.
    constexpr const_iterator begin() const noexcept { return candidates_.begin(); }
    constexpr const_iterator end() const noexcept { return begin() + size(); }
    constexpr iterator begin() noexcept { return candidates_.begin(); }
    constexpr iterator end() noexcept { return begin() + size(); }

    // **Preconditions:**
    //
    // (1) underlying vector must be at least `max_n_element_` long.
    void unsafe_insert(value_type neighbor, iterator index) {
        // Copying an element to `*end()` is okay because we maintain the invariant that the
        // underlying vector is always 1 longer than the length of the maximum number of
        // elements.
        std::copy_backward(index, end(), end() + 1);
        (*index) = neighbor;
    }

    ///
    /// @brief Return ``true`` if a neighbor with the given distance can be skipped.
    ///
    /// In other words, if it can be known ahead of time that inserting a neighbor with
    /// the given distance will not change the state of the buffer, than this method
    /// returns ``true``.
    ///
    bool can_skip(float distance) const {
        return compare_(back().distance(), distance) && full();
    }

    ///
    /// @brief Insert the neighbor into the buffer.
    ///
    /// @param neighbor The neighbor to insert.
    ///
    /// @returns The position where the neighbor was inserted.
    ///
    size_t insert(value_type neighbor) {
        if (can_skip(neighbor.distance())) {
            return size();
        }
        return insert_inner(neighbor);
    }

    size_t insert_inner(value_type neighbor) {
        const auto start = begin();
        // Binary search to the first location where `distance` is less than the stored
        // neighbor.
        auto pos = std::lower_bound(
            start,
            end(),
            neighbor.distance(),
            [&](const value_type& other, const float& d) {
                return !compare_(d, other.distance());
            }
        );

        // Because repeat ids can exist, we have to search until we're sure no repeat will
        // be found.
        //
        // To do that, we start one before the insertion position.
        // Because each instance of repeat ids should have the same distance, we only need
        // to look at `ids` until the buffer elements have a distance less than the
        // distance of the current node.
        if (pos != start) {
            auto back = pos;
            do {
                --back;
                const auto& candidate = *back;
                if (compare_(candidate.distance(), neighbor.distance())) {
                    break;
                } else if (candidate.id_ == neighbor.id()) {
                    return size() + 1;
                }
            } while (back != start);
        }

        // We're explicitly avoiding moving the underlying range in this implementaion,
        // so we don't need to worry about iterator invalidation.
        //
        // Nevertheless, probably safe to compute the actual index because messing around
        // with the internal state of the buffer.
        size_t i = pos - start;
        unsafe_insert(neighbor, pos);
        size_ = std::min(size_ + 1, capacity());
        best_unvisited_ = std::min(best_unvisited_, i);
        return i;
    }

    ///
    /// @brief Sort the elements in the buffer according to the internal comparison functor.
    ///
    void sort() { std::sort(begin(), end(), compare_); }

    ///// Visited API

    ///
    /// @brief Return ``true`` if the visited set is enabled. Otherwise, return ``false``.
    ///
    bool visited_set_enabled() const { return visited_.has_value(); }

    ///
    /// @brief Enable use of the visited set when performing greedy searcher.
    ///
    /// Visited set use does not affect accuracy but may affect performance.
    ///
    void enable_visited_set() {
        if (!visited_set_enabled()) {
            visited_.emplace();
        }
    }

    ///
    /// @brief Disable use of the visited set when performing greedy searches.
    /// Visited set use does not affect accuracy but may affect performance.
    ///
    void disable_visited_set() {
        if (visited_set_enabled()) {
            visited_.reset();
        }
    }

    /// @brief Enable or disable the visited set based on the argument.
    void configure_visited_set(bool enable) {
        if (enable) {
            enable_visited_set();
        } else {
            disable_visited_set();
        }
    }

    ///
    /// @brief Return `true` if key `i` has definitely been marked as visited. Otherwise
    /// ``false``.
    ///
    /// This function is allowed to spuriously return ``false``.
    ///
    bool is_visited(Idx i) const { return visited_set_enabled() && unsafe_is_visited(i); }

    void prefetch_visited(Idx i) const {
        if (visited_set_enabled()) {
            unsafe_prefetch_visited(i);
        }
    }

    bool emplace_visited(Idx i) {
        return visited_set_enabled() && unsafe_emplace_visited(i);
    }

    // Unsafe implementations.
    bool unsafe_is_visited(Idx i) const {
        assert(visited_);
        return visited_->contains(i);
    }

    void unsafe_prefetch_visited(Idx i) const {
        assert(visited_);
        visited_->prefetch(i);
    }

    bool unsafe_emplace_visited(Idx i) {
        assert(visited_);
        return visited_->emplace(i);
    }

  private:
    // The comparison functor.
    [[no_unique_address]] Cmp compare_ = Cmp{};
    // The current number of valid neighbors.
    size_t size_ = 0;
    // The index of the lowest (w.r.t ``compare_`) unvisited neighbor.
    size_t best_unvisited_ = 0;
    // The size of region of interest (determines stopping conditions).
    size_t search_window_size_ = 0;
    // The maximum capacity of the buffer.
    size_t capacity_ = 0;
    // Storage for the neighbors.
    vector_type candidates_ = {};
    // The visited set. Implemented as a `std::optional` to allow enablind and disabling
    // without always requiring allocation of the data structure.
    std::optional<set_type> visited_{std::nullopt};
};

} // namespace svs::index::vamana
