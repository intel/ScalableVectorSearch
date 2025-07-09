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

namespace svs::index::ivf {
///
/// @brief Class used to store search results for static greedy search.
///
/// @tparam Idx Type used to uniquely identify DB vectors
/// @tparam Cmp Type of the comparison function used to sort neighbors by distance.
///
template <typename Idx, typename Cmp = std::less<>> class SortedBuffer {
  public:
    // External type aliases
    using value_type = IVFNeighbor<Idx>;
    using reference = value_type&;
    using const_reference = const value_type&;
    using compare_type = Cmp;

    using vector_type = std::vector<value_type, threads::CacheAlignedAllocator<value_type>>;
    using iterator = typename vector_type::iterator;
    using const_iterator = typename vector_type::const_iterator;

    /// A visited filter with 65,535 entries with a memory footpring of 128 kiB.
    using set_type = index::vamana::VisitedFilter<uint32_t, 16>;

    ///
    /// @brief Initialize a buffer with zero capacity.
    ///
    /// In order to use a buffer that has been default constructed, use the
    /// @ref change_maxsize(size_t) method.
    ///
    SortedBuffer() = default;

    ///
    /// @brief Construct a search buffer with the target capacity and comparison function.
    ///
    /// @param size The number of valid elements to return from a search operation.
    /// @param compare The functor used to compare two ``Neighbor``s together.
    ///
    explicit SortedBuffer(size_t size, Cmp compare = {})
        : compare_{std::move(compare)}
        , capacity_{size}
        , candidates_{capacity_ + 1} {}

    ///
    /// @brief Perform an efficient copy.
    ///
    /// Copy the portions of the SortedBuffer that matter for the purposes of scratch
    /// space.
    ///
    /// Perserves the sizes of various containers but not necessarily the values.
    ///
    SortedBuffer shallow_copy() const { return SortedBuffer{capacity_, compare_}; }

    ///
    /// @brief Change the target number of elements to return after search.
    ///
    /// @param new_size The new number of elements to return.
    ///
    /// Post conditions
    /// - The capacity of the search buffer will be set to the new size.
    /// - The actual size (number of valid elements) will be the minimum of the current
    ///   size and the new size.
    ///
    void change_maxsize(size_t new_size) {
        capacity_ = new_size;
        candidates_.resize(new_size + 1);
        size_ = std::min(size_, new_size);
    }

    ///
    /// @brief Prepare the buffer for a new search operation.
    ///
    void clear() { size_ = 0; }

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

    // Define iterators.
    constexpr const_iterator begin() const noexcept { return candidates_.begin(); }
    constexpr const_iterator end() const noexcept { return begin() + size(); }
    constexpr iterator begin() noexcept { return candidates_.begin(); }
    constexpr iterator end() noexcept { return begin() + size(); }

    void unsafe_insert(value_type neighbor, iterator index) {
        std::copy_backward(index, end(), end() + 1);
        (*index) = neighbor;
    }

    ///
    /// @brief Return ``true`` if a neighbor with the given distance can be skipped.
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

        size_t i = pos - start;
        unsafe_insert(neighbor, pos);
        size_ = std::min(size_ + 1, capacity());
        return i;
    }

    ///
    /// @brief Sort the elements in the buffer according to the internal comparison functor.
    ///
    void sort() { std::sort(begin(), end(), compare_); }

    ///
    /// @brief Return ``true`` if the visited set is enabled. Otherwise, return ``false``.
    ///
    bool visited_set_enabled() const { return visited_.has_value(); }

  private:
    // The comparison functor.
    [[no_unique_address]] Cmp compare_ = Cmp{};
    // The current number of valid neighbors.
    size_t size_ = 0;
    // The maximum capacity of the buffer.
    size_t capacity_ = 0;
    // Storage for the neighbors.
    vector_type candidates_ = {};
    // The visited set. Implemented as a `std::optional` to allow enablind and disabling
    // without always requiring allocation of the data structure.
    std::optional<set_type> visited_{std::nullopt};
};

} // namespace svs::index::ivf
