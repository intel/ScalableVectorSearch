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

#include "svs/index/vamana/search_buffer.h"
#include "svs/lib/boundscheck.h"
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
/// A search buffer that allows entries to be predicated out.
/// The search buffer will still nagivate through those entries, but it won't include
/// them in its final result.
///
template <typename Idx, typename Cmp = std::less<>> class MutableBuffer {
  public:
    // Type Aliases
    using index_type = Idx;
    using value_type = PredicatedSearchNeighbor<Idx>;
    using reference = value_type&;
    using const_reference = const value_type&;
    using vector_type = std::vector<value_type, threads::CacheAlignedAllocator<value_type>>;
    using iterator = typename vector_type::iterator;
    using const_iterator = typename vector_type::const_iterator;
    using compare_type = Cmp;
    using filter_type = VisitedFilter<Idx, 16>;

  private:
    ///// Invariants:
    //
    // 1. 1 <= target_valid_ <= valid_capacity_
    //
    //    The second inequality is enforced by the `BufferConfig` class.
    //    TODO: Enforce the first in `BufferConfig` as well.
    //
    // 2. best_unvisited_ <= size().
    // 3. roi_end_ <= size().
    // 4. valid_ <= valid_capacity_;
    //
    // 5. WHEN (5A): `valid_ < target_valid_` (haven't yet found enough valid neighbors)
    //    THEN: `roi_end_ == size()`: Region of interest points to the end of the buffer.
    //    ELSE (5B): `valid_ >= target_valid_`
    //    THEN: The number of valid between the start of the buffer and `roi_end_ - 1` is
    //          exactly `valid_`.
    //
    // 6. WHEN: `valid_ == valid_capacity_`
    //    THEN: `back()` must be valid.
    //
    // 7. The number of valid elements in `[begin(), end())` is equal to `valid_`.

    [[no_unique_address]] Cmp compare_{};
    // Equivalent of the `search_window_size_` in the traditional search buffer.
    uint16_t target_valid_ = 0;
    // Number of valid elements can are configured to contain.
    // Equivalent to the `search_buffer_capacity_` in the traditional search buffer.
    uint16_t valid_capacity_ = 0;
    // Index of the best unvisited candidate.
    uint16_t best_unvisited_ = 0;
    // One past the "target_valid_"th entry.
    uint16_t roi_end_ = 0;
    // number of valid neighbors.
    uint16_t valid_ = 0;
    // A buffer of candidates.
    // Unlike the static buffer, this container *does* dynamically change size and does not
    // reserve one-past-the-end for copying neighbors.
    vector_type candidates_{};
    // An optional visited filter.
    std::optional<filter_type> visited_{std::nullopt};

  public:
    MutableBuffer() = default;

    /// Construct a new buffer with the given buffer configuration.
    explicit MutableBuffer(
        SearchBufferConfig config, Cmp compare = Cmp{}, bool enable_visited = false
    )
        : compare_{std::move(compare)}
        , target_valid_{lib::narrow<uint16_t>(config.get_search_window_size())}
        , valid_capacity_{lib::narrow<uint16_t>(config.get_total_capacity())}
        , candidates_{valid_capacity_} {
        candidates_.clear();
        if (enable_visited) {
            enable_visited_set();
        }
    }

    /// Construct a new buffer with the given size and capacity.
    explicit MutableBuffer(size_t size, Cmp compare = Cmp{}, bool enable_visited = false)
        : MutableBuffer{SearchBufferConfig{size}, std::move(compare), enable_visited} {}

    /// Copy the portions of the MutableBuffer that matter for the purposes of scratch
    /// space.
    ///
    /// Perserves the sizes of various containers but not necessarily the values.
    MutableBuffer shallow_copy() const {
        // We don't care about the contents of the buffer - just its size.
        // Therefore, we can construct a new buffer from scratch.
        return MutableBuffer{config(), compare_, visited_set_enabled()};
    }

    // TODO: Allow this construction to be noexcept.
    SearchBufferConfig config() const {
        return SearchBufferConfig{target_valid_, valid_capacity_};
    }

    ///
    /// @brief Change the target number of elements to return after search.
    ///
    /// @param config The new configuration for the buffer.
    ///
    /// Post conditions:
    /// - The target valid capacity for the search buffer will be set to the new capacity.
    /// - The actual size (number of contained elements both valid and invalid) will be the
    ///   minimum of the current size and the new capacity.
    ///
    void change_maxsize(SearchBufferConfig config) {
        // Use temporary variables to ensure the given configuration is valid before
        // committing.
        //
        // `lib::narrow` may throw.
        uint16_t target_valid_temp = lib::narrow<uint16_t>(config.get_search_window_size());
        uint16_t valid_capacity_temp = lib::narrow<uint16_t>(config.get_total_capacity());

        // If the new capacity is lower then the current size, shrink the buffer to fit.
        if (valid_capacity_temp < candidates_.size()) {
            candidates_.resize(valid_capacity_temp);
        }

        // Commit the new sizes. Integer assignment is `noexcept`, so we don't need to worry
        // about an exception breaking class invariants.
        target_valid_ = target_valid_temp;
        valid_capacity_ = valid_capacity_temp;
    }

    // Change the maximum number of elements that can be in the search buffer.
    void change_maxsize(size_t new_size) { change_maxsize(SearchBufferConfig{new_size}); }

    /// @brief Prepare the buffer for a new search operation.
    void clear() {
        candidates_.clear();
        best_unvisited_ = 0;
        roi_end_ = 0;
        valid_ = 0;
        if (visited_set_enabled()) {
            visited_->reset();
        }
    }

    void soft_clear() {
        bool use_visited_set = visited_set_enabled();
        if (use_visited_set) {
            visited_->reset();
        }

        for (auto& neighbor : candidates_) {
            neighbor.clear_visited();
            if (use_visited_set) {
                visited_->emplace(neighbor.id());
            }
        }

        best_unvisited_ = 0;
    }

    size_t capacity() const { return candidates_.capacity(); }

    /// @brief Return the number of valid elements currently contained in the buffer.
    size_t valid() const { return valid_; }

    /// @brief Return the target number of valid candidates.
    size_t target() const { return valid_capacity_; }

    /// @brief Return whether or not the buffer contains its target number of candidates.
    bool full() const { return valid() == target(); }

    /// @brief Return the candidate at index `i`.
    ///
    /// Element will only be valid if:
    /// (A) cleanup() has been invoked.
    /// (B) 0 <= i < valid();
    reference operator[](size_t i) { return candidates_[i]; }

    /// @brief Return the candidate at index `i`.
    ///
    /// Element will only be valid if:
    /// (A) cleanup() has been invoked.
    /// (B) 0 <= i < valid();
    const_reference operator[](size_t i) const { return candidates_[i]; }

    /// @brief Return the last candidate, whether or not it is valid.
    ///
    /// It is undefined behavior to call this on an empty buffer.
    reference back() { return candidates_.back(); }

    /// @brief Return the last candidate, whether or not it is valid.
    ///
    /// It is undefined behavior to call this on an empty buffer.
    const_reference back() const { return candidates_.back(); }

    /// @brief Return the index of the best unvisited candidate.
    size_t best_unvisited() const { return best_unvisited_; }

    /// @brief Return `true` if the search buffer has reached its terminating condition.
    ///
    /// N.B.: If `done()` evaluates to `true`, do not try to extract further candidates
    /// from it using `next()`.
    bool done() const {
        // Until we've reached the target number of valid elements, we have to ignore
        // the state of `roi_end_`.
        return best_unvisited_ == ((valid() < target_valid_) ? size() : roi_end_);
    }

    /// @brief Return the best unvisited neighbor in the buffer.
    /// The returned result will be convertible to `Neighbor<Idx>`.
    ///
    /// Pre-conditions:
    /// * `search_buffer.done()` must evaluate to `false`, otherwise an out of bounds
    ///   access will occur.
    ///
    /// Post-conditions:
    /// * The returned neighbor will be marked as visited.
    const_reference next() {
        // Get the best unvisited node
        reference node = getindex(candidates_, best_unvisited_);
        node.set_visited();

        // Increment `best_unvisited_` until it's equal to the size OR until we encounter
        // an unvisited node.
        while (++best_unvisited_ != roi_end_ && candidates_[best_unvisited_].visited()) {}
        return node;
    }

    /// Place the neighbor at the end of the search buffer if `full() != true`.
    /// Otherwise, do nothing.
    ///
    /// NOTE: `push_back` does not necessarily maintain the required invariants by this
    /// class.
    ///
    /// These invariants are restored upon calling `sort()`.
    ///
    /// Therefore, sequencies of `push_back` must always be followed by a call to `sort`.
    void push_back(value_type neighbor) {
        // Follow the contract of not appending more valid elements than the capacity
        // allows.
        bool valid = neighbor.valid();
        if (full() && valid) {
            return;
        }
        candidates_.push_back(neighbor);
        if (valid) {
            ++valid_;
        }
        roi_end_ = candidates_.size();
    }

    // Define iterators.
    constexpr const_iterator begin() const noexcept { return candidates_.begin(); }
    constexpr const_iterator end() const noexcept { return begin() + size(); }
    constexpr iterator begin() noexcept { return candidates_.begin(); }
    constexpr iterator end() noexcept { return begin() + size(); }

    ///
    /// Insert `neighbor` into the candidates buffer at position `pos`.
    /// Afterwards, try to shrink the candidates buffer
    ///
    /// May invalidate any iterators for `candidates_`.
    ///
    void unsafe_insert(value_type neighbor, iterator pos, size_t offset) {
        // Insert the neighbor into the buffer.
        // The rest of this function fixes the invariants that got broken as a result
        // of this operation.
        candidates_.insert(pos, neighbor);

        // Enough candidates for ROI to kick in (5B instead of 5A).
        bool at_threshold = valid() == target_valid_;
        bool this_valid = neighbor.valid();
        // Is this candidate begin inserted before the end of the ROI.
        bool inserted_below = offset < roi_end_;
        if (valid() < target_valid_) {
            // Maintain invariant 5A
            roi_end_ = size();
        } else if (at_threshold) {
            // Several options:
            // (1) This is the entry that causes us to cross the valid threshold. If so,
            //     then we need to move `roi_end_` to the last valid entry;
            // (2) We've previsouly crossed the threshold and this entry is skipped.
            //     If this is the case, then we must bump `roi_end_` if this neighbor
            //     is inserted below.
            if (this_valid) {
                assert(roi_end_ == size() - 1);
                // Case 1: This neighbor caused us to cross the threshold.
                //
                // In this case, the `roi_end_` USED to point to one past the end.
                // After the insertion, it points to the end exactly.
                //
                // To maintain 5B, move `roi_end_` backwards until a valid neighbor is
                // reached, then increment by 1 to point to one-past the end.
                while (!svs::getindex(candidates_, roi_end_).valid()) {
                    --roi_end_;
                }
                ++roi_end_;
            } else if (inserted_below) {
                // Case 2: We've previously crossed the threshold and this neighbor is
                // skipped.
                //
                // In this case, `roi_end_` used to point to one-past the last valid element
                // (according to 5B).
                //
                // Since we inserted below (moving everything above by 1), we can restore
                // 5B by incrementing `roi_end_`.
                ++roi_end_;
            }
        } else {
            // At this point - we are in split-buffer territory.
            // We can assume that before the insertion, invariante 5B held.
            //
            // Therefore, we only need to fix 5B if we inserted below the previous
            // `roi_end_`.
            if (inserted_below) {
                // Assert that 5B held prior to this invocation of `unsafe_insert`.
                assert(candidates_.at(roi_end_).valid());

                // If this neighbor is not valid, then the number of valid neighbors in
                // `[0, roi_end_]` (with right inclusion) has not changed.
                // We can restore 5B by incrementing `roi_end_`.
                if (!this_valid) {
                    ++roi_end_;
                } else {
                    // In this case, the number of neighbors in `[0, roi_end_`] (with
                    // right inclusion) has been incremented by 1.
                    //
                    // We restore 5B by walking back until the next previous valid neighbor
                    // is found, then step forward by 1 to point to one-past the end.
                    while (!svs::getindex(candidates_, --roi_end_).valid()) {}
                    ++roi_end_;
                }
            }
        }

        // Restore Invariant 6.
        bool no_split_buffer = target_valid_ == valid_capacity_;
        if (no_split_buffer) {
            // When using 5A: the ROI already points to the end - so resizing does nothing.
            // Furthermore, when using 5A, `valid() > target_valid_` is always false, so
            // the branch is never taken.
            //
            // When using 5B: `roi_end_` points to one-past the `target_valid_`th entry
            // Shrinking `candidates_` to `roi_end_` is valid
            //
            // Prior to this call to, invariants 4 and 7 must have held.
            // Therefore, the side-effect of adding this neighbor increased the number of
            // valid elements by at most 1 (i.e., `slack() <= 1`).
            //
            // From this, we deduce that the extra valid neighbor used to be located at or
            // above `roi_end_` and was therefore implicitly dropped when shrinking
            // `candidates_`.
            candidates_.resize(roi_end_);
            if (slack() > 0) {
                assert(slack() == 1);
                valid_ -= 1;
            }
        } else if (slack() == 0) {
            // From invariant 7, we know `candidates_` contains `valid_` number of elements.
            // We can restore invariant 6 simply by finding the last valid neighbor and
            // resizing appropriately.
            candidates_.resize(walk_back(candidates_.size() - 1) + 1);
        } else if (slack() >= 0) {
            // We know that invariant 6 must have held prior to this invocation of
            // `unsafe_insert`.
            //
            // Therefore the last neighbor in `candidates_` is valid and must be dropped.
            // We can restore invariant 6 by starting at this neighbor and walking back
            // until the previous valid neighbor is found.
            assert(slack() == 1);
            assert(back().valid());
            candidates_.resize(walk_back(candidates_.size() - 2) + 1);
            //                                                ^
            //                                                |
            //                                    Not a typo. It should be "two".

            // We dropped a valid element off the back when shrinking.
            valid_ -= 1;
        }
    }

    bool can_skip(float distance) const {
        // If not full - we cannot skip appending this item.
        // If we are full, then rely on invariant 6 to compare with the last valid element.
        return full() && compare_(back().distance(), distance);
    }

    /// Insert the given neighbor into the search buffer.
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
                }
                if (candidate.id_ == neighbor.id()) {
                    return size() + 1;
                }
            } while (back != start);
        }

        // Note: Iterators *may* be invalidated as a result of `unsafe_insert`.
        // Hoist out the insertion position as an index before inserting.
        size_t i = pos - start;
        valid_ += static_cast<size_t>(neighbor.valid());
        unsafe_insert(neighbor, pos, i);
        best_unvisited_ = std::min(best_unvisited_, lib::narrow_cast<uint16_t>(i));
        return i;
    }

    // Sort all stored elements in the buffer.
    void sort() {
        assert(valid() <= valid_capacity_);

        // Put the elements in ascending order.
        std::sort(begin(), end(), compare_);

        // Now - fix our invariants.
        if (valid_ < target_valid_) {
            // Maintain 5A
            // Invariant 6 has not activated.
            roi_end_ = size();
            return;
        }

        // Maintain 5B
        size_t valid_count = 0;
        for (size_t i = 0; i < size(); ++i) {
            if (candidates_[i].valid()) {
                ++valid_count;
                if (valid_count == target_valid_) {
                    // One past the last valid element.
                    roi_end_ = i + 1;
                    break;
                }
            }
        }

        // Check if invariant 6 is active.
        // If so, drop invalid elements off the end until a valid element is found.
        if (slack() == 0) {
            assert(!candidates_.empty());
            while (!back().valid()) {
                candidates_.pop_back();
                assert(!candidates_.empty());
            }
        }
    }

    // TODO: Switch over to using iterators for the return values to avoid this.
    void cleanup() {
        auto new_end =
            std::remove_if(begin(), end(), [](const auto& x) { return !x.valid(); });
        candidates_.resize(new_end - begin());
    }

    /// @brief Return the size of the underlying vector.
    ///
    /// N.B.: THIS FUNCTION IS EASY TO CALL INCORRECTLY.
    ///
    /// At any given point, the size of the underlying vector and the number of valid
    /// candidates in the buffer can be very different.
    ///
    /// These two are only the same after invoking `clean()`.
    /// That is the only context in which non-internal interaction with the search buffer
    /// should operate.
    size_t size() const { return candidates_.size(); }

    ///// Visited API
    bool visited_set_enabled() const { return visited_.has_value(); }
    void enable_visited_set() {
        if (!visited_set_enabled()) {
            visited_.emplace();
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

    void disable_visited_set() {
        if (visited_set_enabled()) {
            visited_.reset();
        }
    }

    bool is_visited(Idx i) const { return visited_set_enabled() && unsafe_is_visited(i); }
    void prefetch_visited(Idx i) const {
        if (visited_set_enabled()) {
            unsafe_prefetch_visited(i);
        }
    }
    bool emplace_visited(Idx i) {
        return visited_set_enabled() && unsafe_emplace_visited(i);
    }

    // Unsafe API
    bool unsafe_is_visited(Idx i) const {
        assert(visited_);
        return visited_->contains(i);
    }
    void unsafe_prefetch_visited(Idx i) const {
        assert(visited_);
        return visited_->prefetch(i);
    }
    bool unsafe_emplace_visited(Idx i) {
        assert(visited_);
        return visited_->emplace(i);
    }

  private:
    /// Return how many more valid candidates exist than required.
    /// If the number of valid candidates is *less* than the target, a negative number
    /// is returned.
    int64_t slack() const {
        return lib::narrow_cast<int64_t>(valid()) - lib::narrow_cast<int64_t>(target());
    }

    /// Return the index of the first preceding valid candidate beginning at the provided
    /// index.
    ///
    /// Requires:
    /// * `i` is in the range `[0, size())`.
    /// * There exists at least one valid candidate in `[0, i)`.
    size_t walk_back(size_t i) const {
        while (!candidates_[i].valid()) {
            --i;
        }
        return i;
    }
};

template <typename Idx, typename Cmp>
std::ostream& operator<<(std::ostream& io, const MutableBuffer<Idx, Cmp>& buffer) {
    return io << "MutableBuffer<" << datatype_v<Idx> << ">("
              << "target_valid = " << buffer.target()
              << ", best_unvisited = " << buffer.best_unvisited()
              << ", valid = " << buffer.valid() << ", size = " << buffer.size() << ")";
}

} // namespace svs::index::vamana
