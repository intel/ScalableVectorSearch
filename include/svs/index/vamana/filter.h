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

#include "svs/lib/misc.h"
#include "svs/lib/prefetch.h"

// stl
#include <bit>
#include <limits>
#include <type_traits>
#include <vector>

namespace svs::index::vamana {

///
/// A small direct-mapped cache to filter visited neighbors.
///
/// As a direct mapped cache, this set is not exact and will yield false negatives (say
/// a neighbor has not been visited when it has).
///
/// This is acceptable as we can simply compute the distance to a candidate and rediscover
/// that it has been visited.
///
/// We keep this small to add a bounded amount of extra memory per-thread.
/// In highly bandwidth constrained environments, this can yield a performance improvement.
///
/// @tparam N The number of bits used to index the data structure.
///
/// The filter works by using the lower N bits of an ID to access a slot.
///
template <std::integral I, size_t N> class VisitedFilter;

// The original implementation of this filter was only designed to work for 32-bit IDs.
template <size_t N> class VisitedFilter<uint32_t, N> {
  public:
    // The integer type compatible with this filter.
    using integer_type = uint32_t;

    // If we are using 16-bits or more, then we can get away with storing only the upper
    // 16-bits of each ID and reconstructing the full ID using the lower N bits as the
    // index and matching the upper 16-bits.
    using value_type = std::conditional_t<(N >= 16), uint16_t, integer_type>;

    // Sentinel for empty values.
    static constexpr value_type sentinel = std::numeric_limits<value_type>::max();

    // Mask to extract the lower N bits from an integer.
    static constexpr uint32_t hash_mask = lib::bitmask<integer_type>(0, N - 1);

    /// @brief The maximum number of entries in the filter.
    static constexpr size_t filter_capacity = size_t(1) << N;

    /// @brief Construct a new visited filter.
    ///
    /// The returned filter will be ready for immediate use.
    VisitedFilter()
        : values_(filter_capacity, sentinel) {}

    /// @brief Reset the filter for another
    void reset() { std::fill(values_.begin(), values_.end(), sentinel); }

    /// @brief Return the maximum number of entries the filter is capable of holding.
    size_t capacity() const {
        assert(values_.size() == filter_capacity);
        return filter_capacity;
    }

    /// @brief Return the index in the container to check.
    constexpr static size_t hash(integer_type key) { return key & hash_mask; }

    /// @brief Return ``true`` if the stored value original cam from the provided key.
    constexpr static bool check(integer_type key, value_type value) {
        if constexpr (N >= 16) {
            // Make sure the upper-bits match.
            return (key >> 16) == value;
        } else {
            return key == value;
        }
    }

    /// @brief Return the value to store in the `values` array for the given key.
    constexpr static value_type value(uint32_t key) {
        if constexpr (N >= 16) {
            return (key >> 16);
        } else {
            return key;
        }
    }

    /// @brief Prefetch the storage that contains the contents corresponding to `key`.
    void prefetch(integer_type key) const { lib::prefetch_l0(&values_[hash(key)]); }

    /// @brief insert ``key`` into the filter. Return ``true`` if it was already present.
    ///
    /// This function may spuriously return ``false``.
    bool emplace(integer_type key) {
        auto i = hash(key);
        auto& v = values_[i];
        bool b = check(key, v);
        v = value(key);
        return b;
    }

    /// @brief Return whether or not ``key`` has present in the filter.
    ///
    /// This function may spuriously return ``false``.
    bool contains(integer_type key) const { return check(key, values_[hash(key)]); }

    // Internel method for testing.
    value_type at(size_t i) const { return values_.at(i); }

  private:
    // The tags
    std::vector<value_type> values_;
};

} // namespace svs::index::vamana
