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

#include "svs/index/flat/flat.h"
#include "svs/quantization/lvq/lvq.h"

namespace svs::quantization::lvq {

template <IsLVQDataset Data, typename Distance>
biased_distance_t<Distance> svs_invoke(
    svs::tag_t<svs::index::flat::extensions::distance>,
    const Data& dataset,
    const Distance& distance
) {
    return adapt(dataset, distance);
}

} // namespace svs::quantization::lvq
