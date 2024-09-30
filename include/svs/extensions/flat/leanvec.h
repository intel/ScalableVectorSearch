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

#include "svs/index/flat/flat.h"
#include "svs/leanvec/leanvec.h"

namespace svs::leanvec {

template <IsLeanDataset Data, typename Distance>
auto svs_invoke(
    svs::tag_t<svs::index::flat::extensions::distance>,
    const Data& dataset,
    const Distance& distance
) {
    return dataset.adapt(distance);
}

} // namespace svs::leanvec
