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
