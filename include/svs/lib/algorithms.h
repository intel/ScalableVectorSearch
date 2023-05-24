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

// tsl
#include "tsl/robin_set.h"

// stl
#include <type_traits>

///
/// @defgroup algorithms Library Algorithms.
/// @brief Helper algorihms.
///

namespace svs::lib {

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

namespace ranges {
template <typename ForwardRange> bool all_unique(const ForwardRange& range) {
    return lib::all_unique(range.begin(), range.end());
}
} // namespace ranges

} // namespace svs::lib
