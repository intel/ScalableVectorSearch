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

#include "svs/index/ivf/extensions.h"
#include "svs/quantization/scalar/scalar.h"

namespace svs::quantization::scalar {

template <IsSQData Data, typename Distance>
auto svs_invoke(
    svs::tag_t<svs::index::ivf::extensions::per_thread_batch_search_setup>,
    const Data& data,
    const Distance& SVS_UNUSED(distance)
) {
    return compressed_distance_t<Distance, typename Data::element_type>(
        data.get_scale(), data.get_bias(), data.dimensions()
    );
}

template <IsSQData Data, typename Alloc>
auto svs_invoke(
    svs::tag_t<index::ivf::extensions::create_dense_cluster>,
    const Data& original,
    size_t new_size,
    const Alloc& SVS_UNUSED(allocator)
) {
    auto new_sqdata = SQDataset<typename Data::element_type, Data::extent>(
        new_size, original.dimensions()
    );
    new_sqdata.set_scale(original.get_scale());
    new_sqdata.set_bias(original.get_bias());
    return new_sqdata;
}

// Specialization for blocked allocators (Dynamic IVF)
template <IsSQData Data, typename BlockedAlloc>
auto svs_invoke(
    svs::tag_t<index::ivf::extensions::create_dense_cluster>,
    const Data& original,
    size_t new_size,
    const data::Blocked<BlockedAlloc>& SVS_UNUSED(blocked_alloc)
) {
    auto new_sqdata = SQDataset<
        typename Data::element_type,
        Data::extent,
        data::Blocked<BlockedAlloc>>(new_size, original.dimensions());
    new_sqdata.set_scale(original.get_scale());
    new_sqdata.set_bias(original.get_bias());
    return new_sqdata;
}

} // namespace svs::quantization::scalar
