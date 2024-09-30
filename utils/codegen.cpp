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
// svs
#include "svs/quantization/lvq/compressed.h"
#include "svs/quantization/lvq/vectors.h"

// stl
#include <span>

namespace lvq = svs::quantization::lvq;

void unpack_cv(
    std::span<int32_t> dst,
    lvq::CompressedVector<lvq::Unsigned, 8, 768, lvq::Turbo<16, 4>> cv
) {
    lvq::unpack(dst, cv);
}

void unpack_combined(
    std::span<int32_t> dst, lvq::Combined<4, 8, svs::Dynamic, lvq::Turbo<16, 8>> cv
) {
    lvq::unpack(dst, cv);
}

float distance(
    svs::DistanceL2 tag,
    std::span<const float> x,
    const lvq::ScaledBiasedVector<4, svs::Dynamic, lvq::Sequential>& y
) {
    return svs::distance::compute(tag, x, y);
}

float distance(
    lvq::DistanceFastIP tag,
    std::span<const float> x,
    const lvq::ScaledBiasedVector<8, svs::Dynamic, lvq::Turbo<16, 4>>& y
) {
    return svs::distance::compute(tag, x, y);
}
