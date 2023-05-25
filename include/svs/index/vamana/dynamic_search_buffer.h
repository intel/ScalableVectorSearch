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
    using set_type = std::unordered_set<Idx>;

  private:
    [[no_unique_address]] Cmp compare_{};
    size_t target_valid_ = 0;
    size_t best_unvisited_ = 0;
    size_t valid_ = 0; // number of unskipped neighbors.
    vector_type candidates_{};

  public:
    MutableBuffer() = default;
    explicit MutableBuffer(size_t size, Cmp compare = Cmp{})
        : compare_{std::move(compare)}
        , target_valid_{size}
        , candidates_(size) {
        candidates_.clear();
    }

    ///
    /// Copy the portions of the MutableBuffer that matter for the purposes of scratch
    /// space.
    ///
    /// Perserves the sizes of various containers but not necessarily the values.
    ///
    MutableBuffer shallow_copy() const {
        // We care about the contents of the buffer - just its size.
        // Therefore, we can construct a new buffer from scratch.
        return MutableBuffer{target_valid_, compare_};
    }

    // Change the maximum number of elements that can be in the search buffer.
    void change_maxsize(size_t new_size) {
        target_valid_ = new_size;
        candidates_.resize(new_size + 1);
    }

    void clear() {
        candidates_.clear();
        best_unvisited_ = 0;
        valid_ = 0;
    }

    size_t size() const { return candidates_.size(); }

    size_t capacity() const { return candidates_.capacity(); }

    size_t valid() const { return valid_; }

    size_t target() const { return target_valid_; }

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
    bool done() const { return best_unvisited_ == size(); }

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
        while (++best_unvisited_ != size() && candidates_[best_unvisited_].visited()) {}
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
        candidates_.insert(pos, neighbor);
        // Decide if we want to shrink the vector or not.
        // We shrink safely shrink if:
        //
        // * The number of valid elements is equal to the target number.
        // * Dropping the last element will not cause the number of valid elements to drop
        //   below the target.
        //
        // N.B.: We don't need to explicitly maintain `best_unvisited_` here.
        // A large cascade of dropping many skipped elements off the end *can* happen, but
        // this will only happen when transitioning from "not full" to "full".
        //
        // In this case, the element just inserted *is* valid and thus will not be dropped.
        //
        // Since it was just inserted, it will be an upper-bound for the new value of
        // `best_unvisited_` and thus `best_unvisited_` is guarenteed to be in-bounds after
        // the insertion.
        if (slack() >= 0) {
            for (;;) {
                auto last_element = back();
                bool skipped = last_element.skipped();
                if (skipped || slack() > 0) {
                    candidates_.pop_back();
                    valid_ -= static_cast<size_t>(!skipped);
                } else {
                    break;
                }
            }
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
        unsafe_insert(neighbor, pos);
        best_unvisited_ = std::min(best_unvisited_, i);
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
    static bool visited(Idx /*i*/) { return false; }
    static void set_visited(Idx /*i*/) {}

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
