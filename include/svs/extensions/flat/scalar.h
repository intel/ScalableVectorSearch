/*
 * Copyright 2025 Intel Corporation
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

#include "svs/index/flat/flat.h"
#include "svs/quantization/scalar/scalar.h"

namespace svs::quantization::scalar {

template <IsSQData Data, typename Distance>
auto svs_invoke(
    svs::tag_t<svs::index::flat::extensions::distance>,
    const Data& data,
    const Distance& SVS_UNUSED(distance)
) {
    return compressed_distance_t<Distance, typename Data::element_type>(
        data.get_scale(), data.get_bias(), data.dimensions()
    );
}

} // namespace svs::quantization::scalar
