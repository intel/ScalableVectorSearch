/*
 * Copyright 2026 Intel Corporation
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

///
/// @file
/// @brief Teaches ``svs::quantization::scalar::SQDataset`` that ``SegmentedBlocked`` is a
/// blocked allocator.
///
/// ``SQDataset`` gates ``resize()`` and ``compact()`` -- both of which the concurrent
/// dynamic index calls -- on ``quantization::scalar::is_resizeable``, which is backed by a
/// trait private to that namespace:
///
/// @code{.cpp}
/// template <typename A> inline constexpr bool is_blocked = false;
/// template <typename A> inline constexpr bool is_blocked<data::Blocked<A>> = true;
/// @endcode
///
/// It is a separate trait from ``svs::data::is_blocked_v``, and partial specialization does
/// not match through derivation, so ``SegmentedBlocked<A>`` deriving from ``data::Blocked<A>``
/// is not enough: without the specialization below, a scalar-quantized concurrent index
/// fails to compile the moment it grows or compacts.
///
/// This lives in its own header rather than in ``blocked_data.h`` so that the core concurrent
/// headers do not pull in the quantization stack, mirroring how ``svs/extensions/vamana/``
/// separates the per-dataset vamana extensions from the index itself. Include it alongside
/// ``svs/extensions/vamana/scalar.h``.
///

#include "svs/concurrent/blocked_data.h"
#include "svs/quantization/scalar/scalar.h"

namespace svs::quantization::scalar::detail {

template <typename A>
inline constexpr bool is_blocked<svs::index::vamana::concurrent::SegmentedBlocked<A>> = true;

} // namespace svs::quantization::scalar::detail
