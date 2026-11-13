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
#include "svs/lib/misc.h"
#include "svs/lib/neighbor.h"
#include "svs/lib/threads/threadlocal.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace svs::index::vamana {

/// @brief Snapshot of the `MutableBuffer` search state passed to its stop predicate.
///
/// Only constructed when an unvisited candidate exists, so `best_unvisited_distance` is
/// always the distance of a real candidate.
struct MutableBufferState {
    /// Number of valid candidates in the buffer, i.e. the results available to the user.
    size_t valid;
    /// Number of candidates, valid or not, offered to the buffer by `push_back()` or
    /// `insert()` since the last `clear()`, including dropped ones.
    /// Counted only if the stop predicate declares `counts_total_candidates = true`.
    size_t total_candidates;
    /// Number of valid candidates preceding the best unvisited one, all of them visited.
    size_t valid_visited;
    /// Distance of the best unvisited candidate to the query.
    float best_unvisited_distance;
    /// Target number of valid candidates (search window size).
    size_t search_window;
};

/// @brief Whether the buffer must count all candidates for the stop predicate.
template <typename Stop>
inline constexpr bool counts_total_candidates_v =
    requires { requires Stop::counts_total_candidates; };

/// @brief Default stop predicate: done once the best `search_window` valid candidates
/// have all been visited.
struct DefaultStop {
    constexpr bool operator()(const MutableBufferState& state) const {
        return state.valid_visited >= state.search_window;
    }
};

/// @brief Stop predicate for range search.
///
/// Visits at least the search window, then continues while the best unvisited candidate
/// is within `radius`. A distance `d` is within the radius if `compare(d, radius)`.
template <typename Cmp = std::less<>> struct RangeStop {
    float radius;
    [[no_unique_address]] Cmp compare{};

    constexpr bool operator()(const MutableBufferState& state) const {
        return state.valid_visited >= state.search_window &&
               !compare(state.best_unvisited_distance, radius);
    }
};

/// @brief Stop predicate that also stops when the search is cancelled (e.g. on timeout).
///
/// The search then ends gracefully: the buffer keeps the best candidates found so far.
template <typename Stop> struct CancelableStop {
    static constexpr bool counts_total_candidates = counts_total_candidates_v<Stop>;

    [[no_unique_address]] Stop stop;
    const lib::DefaultPredicate* cancel;

    bool operator()(const MutableBufferState& state) const {
        return stop(state) || (*cancel)();
    }
};

///
/// A search buffer that allows entries to be predicated out.
/// The search buffer will still navigate through those entries, but it won't include
/// them in its final result.
///
/// The search terminates when no unvisited candidate is left or when the `Stop` predicate,
/// invoked with a `MutableBufferState`, returns `true`.
///
template <typename Idx, typename Cmp = std::less<>, typename Stop = DefaultStop>
class MutableBuffer {
  public:
    // Type Aliases
    using index_type = Idx;
    using value_type = PredicatedSearchNeighbor<Idx>;
    using reference = value_type&;
    using const_reference = const value_type&;
    using compare_type = Cmp;
    using stop_type = Stop;

    using vector_type = std::vector<value_type, threads::CacheAlignedAllocator<value_type>>;
    using iterator = typename vector_type::iterator;
    using const_iterator = typename vector_type::const_iterator;

    /// A visited filter with 65,536 entries with a memory footprint of 128 kiB.
    using set_type = VisitedFilter<Idx, 16>;

    /// Total capacity value that lets the buffer grow without bound.
    static constexpr size_t unbounded_capacity = std::numeric_limits<size_t>::max();

  protected:
    ///// Invariants:
    //
    // 1. 1 <= target_valid_ <= valid_capacity_
    //
    //    The second inequality is enforced by the `BufferConfig` class.
    //    TODO: Enforce the first in `BufferConfig` as well.
    //
    // 2. best_unvisited_ <= size().
    // 3. `best_unvisited_` is the first unvisited entry (or `size()` if there is none).
    // 4. valid_ <= valid_capacity_;
    //
    // 5. `valid_visited_` is the number of valid elements in `[0, best_unvisited_)`.
    //
    // 6. WHEN: `valid_ == valid_capacity_`
    //    THEN: `back()` must be valid.
    //
    // 7. The number of valid elements in `[begin(), end())` is equal to `valid_`.

    [[no_unique_address]] Cmp compare_{};
    // Equivalent of the `search_window_size_` in the traditional search buffer.
    size_t target_valid_ = 0;
    // Number of valid elements can are configured to contain.
    // Equivalent to the `search_buffer_capacity_` in the traditional search buffer.
    size_t valid_capacity_ = 0;
    // Index of the best unvisited candidate.
    size_t best_unvisited_ = 0;
    // Number of valid neighbors before `best_unvisited_`.
    size_t valid_visited_ = 0;
    // Number of candidates offered to the buffer.
    size_t total_candidates_ = 0;
    // number of valid neighbors.
    size_t valid_ = 0;
    // A buffer of candidates.
    // Unlike the static buffer, this container *does* dynamically change size and does not
    // reserve one-past-the-end for copying neighbors.
    vector_type candidates_{};
    // An optional visited filter.
    std::optional<set_type> visited_{std::nullopt};
    [[no_unique_address]] Stop stop_{};

  public:
    MutableBuffer() = default;

    /// Construct a new buffer with the given buffer configuration.
    explicit MutableBuffer(
        SearchBufferConfig config,
        Cmp compare = Cmp{},
        bool enable_visited = false,
        Stop stop = Stop{}
    )
        : compare_{std::move(compare)}
        , target_valid_{config.get_search_window_size()}
        , valid_capacity_{config.get_total_capacity()}
        , stop_{std::move(stop)} {
        candidates_.reserve(
            valid_capacity_ == unbounded_capacity ? target_valid_ : valid_capacity_
        );
        if (enable_visited) {
            enable_visited_set();
        }
    }

    /// Construct a new buffer with the given size and capacity.
    explicit MutableBuffer(
        size_t size, Cmp compare = Cmp{}, bool enable_visited = false, Stop stop = Stop{}
    )
        : MutableBuffer{
              SearchBufferConfig{size},
              std::move(compare),
              enable_visited,
              std::move(stop)} {}

    /// Estimate heap memory footprint of a search buffer with the given configuration.
    static constexpr size_t estimate_memory_footprint(
        const SearchBufferConfig& config, bool enable_visited = false
    ) {
        // The SearchBuffer contains a vector of candidates and an optional visited set. The
        // size of the vector is determined by the total capacity of the buffer.
        const auto candidates_num = config.get_total_capacity();

        // Calculate the size of the candidates vector, taking into account the alignment of
        // the CacheAlignedAllocator.
        constexpr size_t alignment = threads::CacheAlignedAllocator<value_type>::alignment;
        const auto candidates_size =
            alignment * lib::div_round_up(sizeof(value_type) * candidates_num, alignment);

        auto result = candidates_size;

        // VisitedFilter has a static filter capacity, which is a compile-time constant. The
        // size of the visited set is determined by the filter capacity of the
        // VisitedFilter, which is a compile-time constant.
        if (enable_visited) {
            result += set_type::filter_capacity * sizeof(typename set_type::value_type);
        }

        return result;
    }

    /// Estimate the memory footprint of a search buffer with the given size.
    static constexpr size_t
    estimate_memory_footprint(size_t size, bool enable_visited = false) {
        return estimate_memory_footprint(SearchBufferConfig(size), enable_visited);
    }

    /// Copy the portions of the MutableBuffer that matter for the purposes of scratch
    /// space.
    ///
    /// Preserves the sizes of various containers but not necessarily the values.
    MutableBuffer shallow_copy() const {
        // We don't care about the contents of the buffer - just its size.
        // Therefore, we can construct a new buffer from scratch.
        return MutableBuffer{config(), compare_, visited_set_enabled(), stop_};
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
        size_t target_valid_temp = config.get_search_window_size();
        size_t valid_capacity_temp = config.get_total_capacity();

        // If the new capacity is lower then the current size, shrink the buffer to fit.
        if (valid_capacity_temp < candidates_.size()) {
            candidates_.resize(valid_capacity_temp);
            restore_best_unvisited();
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
        valid_visited_ = 0;
        total_candidates_ = 0;
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
        valid_visited_ = 0;
    }

    size_t capacity() const { return candidates_.capacity(); }

    /// @brief Return the number of valid elements currently contained in the buffer.
    size_t valid() const { return valid_; }

    /// @brief Return the target valid capacity as equivalent to buffer capacity
    size_t target_capacity() const { return valid_capacity_; }

    /// @brief Return the target valid candidates as equivalent to the search window
    size_t target_window() const { return target_valid_; }

    /// @brief Return whether or not the buffer contains its target number of candidates.
    bool full() const { return valid() == target_capacity(); }

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

    /// @brief Return the number of valid candidates preceding the best unvisited one.
    size_t valid_visited() const { return valid_visited_; }

    /// @brief Return the number of candidates offered to the buffer since `clear()`.
    /// Only counted for stop predicates that opt in.
    size_t total_candidates() const { return total_candidates_; }

    /// @brief Return the stop predicate.
    const Stop& stop_predicate() const { return stop_; }

    /// @brief Return `true` if the search buffer has reached its terminating condition.
    ///
    /// N.B.: If `done()` evaluates to `true`, do not try to extract further candidates
    /// from it using `next()`.
    bool done() const {
        return best_unvisited_ == size() ||
               stop_(MutableBufferState{
                   .valid = valid_,
                   .total_candidates = total_candidates_,
                   .valid_visited = valid_visited_,
                   .best_unvisited_distance = candidates_[best_unvisited_].distance(),
                   .search_window = target_valid_});
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
        valid_visited_ += static_cast<size_t>(node.valid());

        // Increment `best_unvisited_` until it's equal to the size OR until we encounter
        // an unvisited node.
        const size_t end = size();
        while (++best_unvisited_ != end) {
            const auto& candidate = candidates_[best_unvisited_];
            if (!candidate.visited()) {
                break;
            }
            valid_visited_ += static_cast<size_t>(candidate.valid());
        }
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
    /// Therefore, sequences of `push_back` must always be followed by a call to `sort`.
    void push_back(value_type neighbor) {
        // Follow the contract of not appending more valid elements than the capacity
        // allows.
        bool valid = neighbor.valid();
        count_candidate();
        if (full() && valid) {
            return;
        }
        candidates_.push_back(neighbor);
        if (valid) {
            ++valid_;
        }
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
    void unsafe_insert(value_type neighbor, iterator pos) {
        // Insert the neighbor into the buffer.
        // The rest of this function fixes the invariants that got broken as a result
        // of this operation.
        candidates_.insert(pos, neighbor);

        // Restore Invariants 4 and 6.
        if (valid_ > valid_capacity_) {
            // Invariant 6 held prior to the insertion, so the last element is valid and is
            // the one that must be dropped. Then walk back to the previous valid element.
            assert(valid_ == valid_capacity_ + 1);
            assert(back().valid());
            candidates_.resize(walk_back(candidates_.size() - 2) + 1);
            //                                                ^
            //                                                |
            //                                    Not a typo. It should be "two".
            valid_ -= 1;
        } else if (valid_ == valid_capacity_) {
            candidates_.resize(walk_back(candidates_.size() - 1) + 1);
        }
    }

    bool can_skip(float distance) const {
        // If not full - we cannot skip appending this item.
        // If we are full, then rely on invariant 6 to compare with the last valid element.
        return full() && compare_(back().distance(), distance);
    }

    /// Insert the given neighbor into the search buffer.
    size_t insert(value_type neighbor) {
        count_candidate();
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
        if (i < best_unvisited_) {
            // Maintain Invariant 5: the elements in `[i, best_unvisited_)` will no longer
            // precede the best unvisited candidate.
            for (size_t j = i; j < best_unvisited_; ++j) {
                valid_visited_ -= static_cast<size_t>(candidates_[j].valid());
            }
            best_unvisited_ = i;
        }
        valid_ += static_cast<size_t>(neighbor.valid());
        unsafe_insert(neighbor, pos);
        return i;
    }

    // Sort all stored elements in the buffer.
    void sort() {
        assert(valid() <= valid_capacity_);

        // Put the elements in ascending order.
        std::sort(begin(), end(), compare_);

        // Check if invariant 6 is active.
        // If so, drop invalid elements off the end until a valid element is found.
        if (valid_ == valid_capacity_) {
            assert(!candidates_.empty());
            while (!back().valid()) {
                candidates_.pop_back();
                assert(!candidates_.empty());
            }
        }

        // Sorting may move visited candidates anywhere, so recompute Invariants 3 and 5.
        restore_best_unvisited();
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

  protected:
    /// Recompute `best_unvisited_` and `valid_visited_` from scratch.
    void restore_best_unvisited() {
        valid_visited_ = 0;
        const size_t end = size();
        for (best_unvisited_ = 0; best_unvisited_ < end; ++best_unvisited_) {
            const auto& candidate = candidates_[best_unvisited_];
            if (!candidate.visited()) {
                break;
            }
            valid_visited_ += static_cast<size_t>(candidate.valid());
        }
    }

  private:
    // Opt-in: this update is measurable in the hot insertion path.
    void count_candidate() {
        if constexpr (counts_total_candidates_v<Stop>) {
            ++total_candidates_;
        }
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

template <typename Idx, typename Cmp, typename Stop>
std::ostream& operator<<(std::ostream& io, const MutableBuffer<Idx, Cmp, Stop>& buffer) {
    return io << "MutableBuffer<" << datatype_v<Idx> << ">("
              << "target_valid = " << buffer.target_capacity()
              << ", best_unvisited = " << buffer.best_unvisited()
              << ", valid = " << buffer.valid() << ", size = " << buffer.size() << ")";
}

} // namespace svs::index::vamana
