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

#include "svs/lib/datatype.h"
#include "svs/lib/neighbor.h"
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

///
/// @brief Class used to store search results for static greedy search.
///
/// @tparam Idx Type used to uniquely identify DB vectors
/// @tparam Cmp Type of the comparison function used to sort neighbors by distance.
///
template <typename Idx, typename Cmp = std::less<>> class SearchBuffer {
  public:
    // External type aliases
    using value_type = SearchNeighbor<Idx>;
    using reference = value_type&;
    using const_reference = const value_type&;

    using vector_type = std::vector<value_type, threads::CacheAlignedAllocator<value_type>>;
    using iterator = typename vector_type::iterator;
    using const_iterator = typename vector_type::const_iterator;

    using set_type = std::unordered_set<Idx>;

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
    /// @param size The number of valid elements to return from a search operation.
    /// @param compare The functor used to compare two ``SearchNeighbor``s together.
    /// @param enable_visited Whether or not the visited set is enabled.
    ///
    explicit SearchBuffer(size_t size, Cmp compare = Cmp{}, bool enable_visited = false)
        : compare_{std::move(compare)}
        , capacity_{size}
        , candidates_{size + 1}
        , visited_{std::nullopt} {
        if (enable_visited) {
            enable_visited_set();
        }
    }

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
        return SearchBuffer{capacity(), compare_, visited_set_enabled()};
    }

    ///
    /// @brief Change the target number of elements to return after search.
    ///
    /// @param new_size The new number of elements to return.
    ///
    void change_maxsize(size_t new_size) {
        capacity_ = new_size;
        candidates_.resize(new_size + 1);
    }

    ///
    /// @brief Prepare the buffer for a new search operation.
    ///
    void clear() {
        size_ = 0;
        best_unvisited_ = 0;
        if (visited_set_enabled()) {
            visited_->clear();
        }
    }

    ///
    /// @brief Return the current number of valid elements in the buffer.
    ///
    size_t size() const { return size_; }

    ///
    /// @brief Return the maximum number of neighbors that can be held by the buffer.
    ///
    size_t capacity() const { return capacity_; }

    bool full() const { return size() == capacity(); }

    ///
    /// @brief Access the neighbor at position `i`.
    ///
    reference operator[](size_t i) { return candidates_[i]; }

    ///
    /// @brief Access the neighbor at position `i`.
    ///
    const_reference operator[](size_t i) const { return candidates_[i]; }

    ///
    /// @brief Return the furtherst valid neighbor.
    ///
    reference back() { return candidates_[size_ - 1]; }

    ///
    /// @brief Return the furtherst valid neighbor.
    ///
    const_reference back() const { return candidates_[size_ - 1]; }

    ///
    /// @brief Return the position of the best unvisited neighbor.
    ///
    size_t best_unvisited() const { return best_unvisited_; }

    ///
    /// @brief Return a view of the backing data for this buffer.
    ///
    std::span<const value_type> view() const {
        return std::span<const value_type>(candidates_.data(), size());
    }

    ///
    /// @brief Return `true` if the search buffer has reached its terminating condition.
    ///
    /// Note: If `done()` evaluates to `true`, do not try to extract further candidates
    /// from it using `next()`.
    ///
    bool done() const { return best_unvisited_ == size_; }

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
        while (++best_unvisited_ != size() && candidates_[best_unvisited_].visited()) {}
        set_visited(node.id());
        return node;
    }

    ///
    /// @brief Place the neighbor at the end of the search buffer if `full() != true`.
    ///
    /// Otherwise, do nothing.
    ///
    void push_back(value_type neighbor) {
        if (size_ < capacity_) {
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
        size_ = std::min(size_ + 1, capacity_);
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

    ///
    /// @brief Return `true` if key `i` has been marked as visited. Otherwise ``false``.
    ///
    /// This function will always return `false` if the visited set is disabled.
    ///
    bool visited(Idx i) const {
        // Short circuiting logic avoids the undefined access if the visited set is not
        // enabled.
        return visited_set_enabled() && visited_->contains(i);
    }

    ///
    /// @brief Mark the key `i` as visited if the visited set is enabled.
    ///
    void set_visited(Idx i) {
        if (visited_set_enabled()) {
            visited_->insert(i);
        }
    }

  private:
    // The comparison functor.
    [[no_unique_address]] Cmp compare_ = Cmp{};
    // The current number of valid neighbors.
    size_t size_ = 0;
    // The index of the lowest (w.r.t ``compare_`) unvisited neighbor.
    size_t best_unvisited_ = 0;
    // The maximum number of neighbors that can be stored.
    size_t capacity_ = 0;
    // Storage for the neighbors.
    vector_type candidates_ = {};
    // The visited set. Implemented as a `std::optional` to allow enablind and disabling
    // without always requiring allocation of the data structure.
    std::optional<set_type> visited_{std::nullopt};
};

} // namespace svs::index::vamana
