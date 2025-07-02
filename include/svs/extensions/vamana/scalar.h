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

#include "svs/index/vamana/extensions.h"
#include "svs/quantization/scalar/scalar.h"

namespace svs::quantization::scalar {

template <IsSQData Data>
SVS_FORCE_INLINE scalar::DecompressionAccessor svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::reconstruct_accessor> SVS_UNUSED(cpo),
    const Data& data
) {
    return scalar::DecompressionAccessor{data};
}

template <IsSQData Data, typename Distance>
auto svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::single_search_setup>,
    const Data& data,
    const Distance& SVS_UNUSED(distance)
) {
    return compressed_distance_t<Distance, typename Data::element_type>(
        data.get_scale(), data.get_bias(), data.dimensions()
    );
}

/////
///// Vamana Build
/////

template <IsSQData Data, typename Distance> struct VamanaBuildAdaptor {
  public:
    using distance_type =
        DecompressionAdaptor<compressed_distance_t<Distance, typename Data::element_type>>;
    using search_distance_type = distance_type;
    using general_distance_type = distance_type;

    auto access_query_for_graph_search(const Data& data, size_t i) const {
        return data.get_datum(i);
    }

    template <typename Query>
    SVS_FORCE_INLINE const Query& modify_post_search_query(
        const Data& SVS_UNUSED(data), size_t SVS_UNUSED(i), const Query& query
    ) const {
        return query;
    }

    static constexpr bool refix_argument_after_search = false;

    data::GetDatumAccessor graph_search_accessor() const {
        return data::GetDatumAccessor{};
    }
    search_distance_type& graph_search_distance() { return distance_; }
    general_distance_type& general_distance() { return distance_; }
    data::GetDatumAccessor general_accessor() const { return data::GetDatumAccessor{}; }

    template <typename Query, NeighborLike N>
    SVS_FORCE_INLINE Neighbor<typename N::index_type> post_search_modify(
        const Data& SVS_UNUSED(data),
        general_distance_type& SVS_UNUSED(distance),
        const Query& SVS_UNUSED(query),
        const N& n
    ) const {
        return n;
    }

  public:
    distance_type distance_{};
};

template <IsSQData Data, typename Distance>
VamanaBuildAdaptor<Data, Distance> svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::build_adaptor>,
    const Data& data,
    const Distance& distance
) {
    return VamanaBuildAdaptor<Data, Distance>{adapt_for_self(data, distance)};
}

} // namespace svs::quantization::scalar
