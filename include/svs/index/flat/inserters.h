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

// stdlib
#include <algorithm>
#include <iterator>
#include <type_traits>

// svs
#include "svs/lib/array.h"
#include "svs/lib/type_traits.h"

namespace svs::index::flat {
template <std::random_access_iterator T, typename Cmp> struct LinearInserter {
    // Members
    T begin_;
    T end_;
    [[no_unique_address]] Cmp compare_;

    // Constructor
    LinearInserter(T begin, T end, Cmp compare)
        : begin_{begin}
        , end_{end}
        , compare_{compare} {}

    // Type aliases.
    using value_type = std::remove_cvref_t<decltype(*begin_)>;

    // Insert
    void insert(value_type x) {
        // Compare with the last element.
        // If the comparison fails - then don't try the insertion procedure.
        if (compare_(*(end_ - 1), x)) {
            return;
        }

        auto pos = std::lower_bound(begin_, end_, x, compare_);
        std::copy_backward(pos, end_ - 1, end_);
        (*pos) = x;
    }

    // Fill the range with sentinel types.
    void prepare() { std::fill(begin_, end_, type_traits::sentinel_v<value_type, Cmp>); }

    // Nothing to do for cleanup.
    void cleanup() {}
};

// Deduction guide
template <typename T, typename Cmp>
LinearInserter(T begin, T end, Cmp compare_) -> LinearInserter<T, Cmp>;

template <std::random_access_iterator T, typename Cmp> struct HeapInserter {
    // Members
    T begin_;
    T end_;
    [[no_unique_address]] Cmp compare_;

    // Constructor
    HeapInserter(T begin, T end, Cmp compare)
        : begin_{begin}
        , end_{end}
        , compare_{compare} {}

    // Type aliases.
    using value_type = std::remove_cvref_t<decltype(*begin_)>;

    // Insert
    void insert(value_type x) {
        // The "largest" element in the heap is the first element.
        // If `x` is "larger" than this element, don't insert it into the heap.
        if (compare_(*begin_, x)) {
            return;
        }

        // `x` is smaller than the largest element.
        // Pop the largest element then insert `x`.
        std::pop_heap(begin_, end_, compare_);
        *(end_ - 1) = x;
        std::push_heap(begin_, end_, compare_);
    }

    // Fill the range with sentinel types.
    void prepare() {
        auto sentinel = type_traits::sentinel_v<value_type, Cmp>;
        std::fill(begin_, end_, sentinel);
    }

    // Nothing to do for cleanup.
    void cleanup() { return std::sort_heap(begin_, end_, compare_); }
};

// Deduction guide
template <typename T, typename Cmp>
HeapInserter(T begin, T end, Cmp compare_) -> HeapInserter<T, Cmp>;

///
/// Bulk inserter managing mulitple sets of nearest neighbors.
///

template <typename T, typename Cmp> class BulkInserter {
  public:
    using value_type = T;

    // Constructor
    BulkInserter(size_t batch_size, size_t num_neighbors, Cmp compare)
        : data_{make_dims(batch_size, num_neighbors)}
        , compare_{compare} {}

    BulkInserter()
        : data_{1, 1}
        , compare_{} {}

    ///
    /// Prepare for bulk insertion.
    ///
    void prepare() {
        for (size_t i = 0; i < getsize<0>(data_); ++i) {
            inserter(i).prepare();
        }
    }

    ///
    /// Insert an element into batch `i`.
    ///
    void insert(size_t i, T x) { inserter(i).insert(x); }

    // TODO: When using the linear inserter - there's nothing to do when cleaning up.
    // When using a heap based inserter, however, we will indeed need to perform a final
    // fix-up before we can begin yielding resulits.
    //
    // We might be able to propagate a trait to avoid doing this loop if no work actually
    // needs to be done (that is, if the compiler is unable to remove the loop itself).
    void cleanup() {
        for (size_t i = 0; i < getsize<0>(data_); ++i) {
            inserter(i).cleanup();
        }
    }

    ///
    /// Return a view of the underlying data.
    ///
    ConstMatrixView<T> view() const { return data_.view(); }

    ///
    /// Return the results for batch `i`.
    ///
    std::span<const T, Dynamic> result(size_t i) const { return data_.slice(i); }

    ///
    /// Return the currently configured batch size.
    ///
    size_t batch_size() const { return getsize<0>(data_); }

    ///
    /// Return the currently configured number of neighbors.
    ///
    size_t num_neighbors() const { return getsize<1>(data_); }

    ///
    /// Resize the underlying data buffer.
    ///
    void resize(size_t new_batch_size, size_t new_num_neighbors) {
        if (batch_size() != new_batch_size || num_neighbors() != new_num_neighbors) {
            data_ = make_dense_array<T>(new_batch_size, new_num_neighbors);
        }
    }

    ///
    /// Change the configured batch size.
    ///
    void resize_batch(size_t new_batch_size) { resize(new_batch_size, num_neighbors()); }

    ///
    /// Change the number of neighbors.
    ///
    void resize_neighbors(size_t new_num_neighbors) {
        resize(batch_size(), new_num_neighbors);
    }

  private:
    // Helper methods
    auto inserter(size_t i) {
        auto slice = data_.slice(i);
        return HeapInserter{slice.begin(), slice.end(), compare_};
    }

    // Members
    Matrix<T> data_;
    [[no_unique_address]] Cmp compare_;
};

} // namespace svs::index::flat
