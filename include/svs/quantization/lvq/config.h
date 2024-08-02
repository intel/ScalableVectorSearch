/**
 *    Copyright (C) 2023, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

#pragma once

#include <concepts>
#include <cstdint>

namespace svs::quantization::lvq {

/// The encoding to use for centroid selection.
using selector_t = uint8_t;

// Helper definitions.
namespace detail {
template <std::integral I> struct DivRem {
    I div;
    I rem;
};
template <std::integral I> inline constexpr DivRem<I> divrem(I x, I y) {
    return DivRem<I>{.div = x / y, .rem = x % y};
}
} // namespace detail
} // namespace svs::quantization::lvq
