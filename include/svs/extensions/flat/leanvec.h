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
