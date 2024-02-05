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

#include "svs/index/vamana/search_buffer.h"
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
    using value_type = SkippableSearchNeighbor<Idx>;
    using reference = value_type&;
    using const_reference = const value_type&;
    using vector_type = std::vector<value_type, threads::CacheAlignedAllocator<value_type>>;
    using iterator = typename vector_type::iterator;
    using const_iterator = typename vector_type::const_iterator;
    using compare_type = Cmp;
    using filter_type = VisitedFilter<Idx, 16>;

  private:
    ///// Invariants:
    // * target_valid_ <= valid_capacity_

    [[no_unique_address]] Cmp compare_{};
    // Equivalent of the `search_window_size_` in the traditional search buffer.
    uint16_t target_valid_ = 0;
    // Number of valid elements can are configured to contain.
    uint16_t valid_capacity_ = 0;
    uint16_t best_unvisited_ = 0;
    uint16_t roi_end_ = 0; // One past the "target_valid_"th entry.
    uint16_t valid_ = 0;   // number of unskipped neighbors.
    vector_type candidates_{};
    std::optional<filter_type> visited_{std::nullopt};

  public:
    MutableBuffer() = default;
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

    explicit MutableBuffer(size_t size, Cmp compare = Cmp{}, bool enable_visited = false)
        : MutableBuffer{SearchBufferConfig{size}, std::move(compare), enable_visited} {}

    ///
    /// Copy the portions of the MutableBuffer that matter for the purposes of scratch
    /// space.
    ///
    /// Perserves the sizes of various containers but not necessarily the values.
    ///
    MutableBuffer shallow_copy() const {
        // We don't care about the contents of the buffer - just its size.
        // Therefore, we can construct a new buffer from scratch.
        return MutableBuffer{config(), compare_, visited_set_enabled()};
    }

    // TODO: Allow this construction to be noexcept.
    SearchBufferConfig config() const {
        return SearchBufferConfig{target_valid_, valid_capacity_};
    }

    void change_maxsize(SearchBufferConfig config) {
        // Use temporary variables to ensure the given configuration is valid before
        // committing.
        uint16_t target_valid_temp = lib::narrow<uint16_t>(config.get_search_window_size());
        uint16_t valid_capacity_temp = lib::narrow<uint16_t>(config.get_total_capacity());

        target_valid_ = target_valid_temp;
        valid_capacity_ = valid_capacity_temp;
        candidates_.resize(valid_capacity_ + 1);
    }

    // Change the maximum number of elements that can be in the search buffer.
    void change_maxsize(size_t new_size) { change_maxsize(SearchBufferConfig{new_size}); }

    void clear() {
        candidates_.clear();
        best_unvisited_ = 0;
        roi_end_ = 0;
        valid_ = 0;
        if (visited_set_enabled()) {
            visited_->reset();
        }
    }

    size_t size() const { return candidates_.size(); }

    size_t capacity() const { return candidates_.capacity(); }

    size_t valid() const { return valid_; }

    size_t target() const { return valid_capacity_; }

    bool full() const { return valid() == target(); }

    reference operator[](size_t i) { return candidates_[i]; }

    const_reference operator[](size_t i) const { return candidates_[i]; }

    reference back() { return candidates_.back(); }

    const_reference back() const { return candidates_.back(); }

    size_t best_unvisited() const { return best_unvisited_; }

    ///
    /// Return `true` if the search buffer has reached its terminating condition.
    ///
    /// N.B.: If `done()` evaluates to `true`, do not try to extract further candidates
    /// from it using `next()`.
    ///
    bool done() const {
        // Until we've reached the target number of valid elements, we have to ignore
        // the state of `roi_end_`.
        return best_unvisited_ == ((valid() < target_valid_) ? size() : roi_end_);
    }

    ///
    /// Return the best unvisited neighbor in the buffer.
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
        reference node = getindex(candidates_, best_unvisited_);
        node.set_visited();

        // Increment `best_unvisited_` until it's equal to the size OR until we encounter
        // an unvisited node.
        while (++best_unvisited_ != roi_end_ && candidates_[best_unvisited_].visited()) {}
        return node;
    }

    ///
    /// Place the neighbor at the end of the search buffer if `full() != true`.
    /// Otherwise, do nothing.
    ///
    void push_back(value_type neighbor) {
        candidates_.push_back(neighbor);
        // If this neighbor has not been skipped, increment the `valid` count.
        if (!neighbor.skipped()) {
            ++valid_;
            roi_end_ = candidates_.size();
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
    void unsafe_insert(value_type neighbor, iterator pos, size_t offset) {
        candidates_.insert(pos, neighbor);

        // Maintain the ROI pointer.
        // Check the common-case where a split buffer is not being used
        bool at_threshold = valid() == target_valid_;

        bool this_skipped = neighbor.skipped();
        bool inserted_below = offset < roi_end_;
        if (valid() < target_valid_) {
            roi_end_ = size();
        } else if (at_threshold) {
            // Several options:
            // (1) This is the entry that causes us to cross the valid threshold. If so,
            //     then we need to move `roi_end_` to the last valid entry;
            // (2) We've previsouly crossed the threshold and this entry is skipped.
            //     If this is the case, then we must bump `roi_end_` if this neighbor
            //     is inserted below.
            if (!this_skipped) {
                // Case 1: This neighbor caused us to cross the treshold.
                while (candidates_.at(roi_end_).skipped()) {
                    --roi_end_;
                }
                ++roi_end_;
            } else if (inserted_below) {
                // Case 2: We've previously crossed the threshold and this neighbor is
                // skipped.
                ++roi_end_;
            }
        } else {
            // In this path - we assume that `roi_end_` previously pointer to one-past a
            // valid neighbor.
            //
            // If insertion happened above, then we don't need to do anything.
            // If insertion happened below, then we must update `roi_end_` accordingly.
            if (inserted_below) {
                assert(!candidates_.at(roi_end_).skipped());
                if (this_skipped) {
                    ++roi_end_;
                } else {
                    // Since we-ve crossed the threshold - we always need to take a step
                    // back first - hence the pre-inrement operator in the "while" loop.
                    while (candidates_.at(--roi_end_).skipped()) {}
                    ++roi_end_;
                }
            }
        }

        // If we're operating in non-split mode - then then `roi_end_` is also the target
        // capacity.
        if (target_valid_ == valid_capacity_) {
            candidates_.resize(roi_end_);
            if (valid() > target_valid_) {
                valid_ -= 1;
            }
        } else if (slack() > 0) {
            // We can only go over by at most one.
            assert(slack() == 1);
            // Roll back until we hit a valid neighbor.
            // This is the neighbor we are going to drop - so do the process again.
            size_t back = candidates_.size();
            while (candidates_[--back].skipped()) {}
            while (candidates_[--back].skipped()) {}
            candidates_.resize(back + 1);
            valid_ -= 1;
        }
    }

    bool can_skip(float distance) const {
        return compare_(back().distance(), distance) && full();
    }

    // size_t insert(Idx id, float distance, bool valid) {
    size_t insert(value_type neighbor) {
        if (can_skip(neighbor.distance())) {
            return size();
        }
        return insert_inner(neighbor);
    }

    // size_t insert_inner(Idx id, float distance, bool valid) {
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

        // Note: Iterators *may* be invalidates as a result of `unsafe_insert`.
        // Hoist out the insertion position as an index before inserting.
        size_t i = pos - start;
        valid_ += static_cast<size_t>(!neighbor.skipped());
        unsafe_insert(neighbor, pos, i);
        best_unvisited_ = std::min(best_unvisited_, lib::narrow_cast<uint16_t>(i));
        return i;
    }

    // Sort all stored elements in the buffer.
    void sort() { std::sort(begin(), end(), compare_); }

    // TODO: Switch over to using iterators for the return values to avoid this.
    void cleanup() {
        auto new_end =
            std::remove_if(begin(), end(), [](const auto& x) { return x.skipped(); });
        candidates_.resize(new_end - begin());
    }

    ///// Visited API
    bool visited_set_enabled() const { return visited_.has_value(); }
    void enable_visited_set() {
        if (!visited_set_enabled()) {
            visited_.emplace();
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
    ///
    /// Return how many more valid candidates exist than required.
    /// If the number of valid candidates is *less* than the target, a negative number
    /// is returned.
    ///
    int64_t slack() const {
        return lib::narrow_cast<int64_t>(valid()) - lib::narrow_cast<int64_t>(target());
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
