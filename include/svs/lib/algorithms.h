/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
 */
#pragma once

// tsl
#include "tsl/robin_set.h"

// stl
#include <iterator>
#include <type_traits>

///
/// @defgroup algorithms Library Algorithms.
/// @brief Helper algorihms.
///

namespace svs::lib {

///
/// @ingroup algorithms
/// @brief A simple aggregate containing a minimum and maximum value.
///
template <typename T> struct MinMax {
    /// Members
    T min;
    T max;

    /// A static initializer to keep this class a simple aggregate.
    static MinMax init() {
        return MinMax{
            .min = std::numeric_limits<T>::max(), .max = std::numeric_limits<T>::lowest()};
    }

    void update(T x) {
        min = std::min(min, x);
        max = std::max(max, x);
    }
};

template <typename Begin, typename End>
MinMax<typename std::iterator_traits<Begin>::value_type> extrema(Begin begin, End end) {
    using T = typename std::iterator_traits<Begin>::value_type;
    auto mm = MinMax<T>::init();
    for (auto it = begin; it != end; ++it) {
        mm.update(*it);
    }
    return mm;
}

///
/// @ingroup algorithms
/// @brief Check if all elements in the range ``[begin, end)`` are unique.
///
/// @param begin Forward iterator to the beginning of the range.
/// @param end Forward iterator to the end of the range.
///
template <class Begin, class End> bool all_unique(const Begin& begin, const End& end) {
    auto seen = tsl::robin_set<std::decay_t<decltype(*begin)>>();
    for (auto i = begin; i != end; ++i) {
        auto [_, inserted] = seen.emplace(*i);
        if (!inserted) {
            return false;
        }
    }
    return true;
}

template <typename InputIt1, typename InputIt2, typename OutputIt, typename Compare>
OutputIt bounded_merge(
    InputIt1 first1,
    InputIt1 last1,
    InputIt2 first2,
    InputIt2 last2,
    OutputIt d_first,
    OutputIt d_last,
    Compare cmp
) {
    assert(
        std::distance(d_first, d_last) <=
        std::distance(first1, last1) + std::distance(first2, last2)
    );

    auto append_1 = [&]() {
        *d_first = *first1;
        ++first1;
    };
    auto append_2 = [&]() {
        *d_first = *first2;
        ++first2;
    };
    for (; d_first != d_last; ++d_first) {
        // First iterator is expired
        if (first1 == last1) {
            assert(first2 != last2);
            append_2();
            // Second iterator is expired.
        } else if (first2 == last2) {
            append_1();
            // Select the smaller current value.
        } else {
            if (cmp(*first1, *first2)) {
                append_1();
            } else {
                append_2();
            }
        }
    }
    return d_first;
}

namespace ranges {
template <typename ForwardRange> bool all_unique(const ForwardRange& range) {
    return lib::all_unique(range.begin(), range.end());
}

template <
    typename ForwardRange1,
    typename ForwardRange2,
    typename OutputRange,
    typename Compare>
void bounded_merge(
    const ForwardRange1& input1,
    const ForwardRange2& input2,
    OutputRange& output,
    Compare cmp
) {
    lib::bounded_merge(
        input1.begin(),
        input1.end(),
        input2.begin(),
        input2.end(),
        output.begin(),
        output.end(),
        std::move(cmp)
    );
}
} // namespace ranges
} // namespace svs::lib
