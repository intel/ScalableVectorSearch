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

#pragma once

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

template <IsSQData Data, data::ImmutableMemoryDataset Points, IsSQData DataPoints>
class TransactionData {
  public:
    using points_type = Points;
    using const_point_type = typename Points::const_value_type;

    using value_type = typename Data::value_type;
    using const_value_type = typename Data::const_value_type;

    TransactionData(
        const Data& data,
        const Points& points,
        DataPoints data_points,
        std::span<size_t> slots
    )
        : data_(data)
        , points_(points)
        , data_points_(std::move(data_points))
        , slots_(slots.begin(), slots.end()) {
        assert(std::is_sorted(slots_.begin(), slots_.end()) && "Slots must be sorted");
    }

    const Data& get_data() const { return data_; }

    // Get the index of the point corresponding to the given index' id.
    size_t get_point_index(size_t id) const {
        // find id in slots
        auto it = std::lower_bound(slots_.begin(), slots_.end(), id);
        if (it != slots_.end() && *it == id) {
            return std::distance(slots_.begin(), it);
        } else {
            throw std::out_of_range("Index not found in slots");
        }
    }

    const_point_type get_point(size_t id) const {
        return points_.get_datum(get_point_index(id));
    }

    const points_type& get_points() const { return points_; }
    size_t num_points() const { return points_.size(); }
    size_t get_slot(size_t i) const { return slots_[i]; }

    size_t size() const {
        return std::max(data_.size(), slots_.empty() ? 0 : slots_.back() + 1);
    }
    size_t dimensions() const { return data_.dimensions(); }
    const_value_type get_datum(size_t i) const {
        // find id in slots
        auto it = std::lower_bound(slots_.begin(), slots_.end(), i);
        if (it != slots_.end() && *it == i) {
            return data_points_.get_datum(std::distance(slots_.begin(), it));
        } else {
            return data_.get_datum(i);
        }
    }
    void prefetch(size_t) const {} // data_.prefetch(i); }

    template <typename TargetData> void copy_points(TargetData& target_data) const {
        assert(
            &target_data == &data_ && "Target data must be the same as the original data"
        );
        // auto new_size = std::max_element(slots_.begin(), slots_.end()) + 1;
        //  assuming, slots_ is ordered, we can just take the last element and add 1
        auto new_size = slots_.back() + 1;
        if (new_size > target_data.size()) {
            target_data.resize(new_size);
        }
        for (size_t i = 0; i < points_.size(); ++i) {
            target_data.set_datum(slots_[i], points_.get_datum(i));
        }
    }

  private:
    const Data& data_;
    const Points& points_;
    const DataPoints data_points_;
    std::vector<size_t> slots_;
};

template <IsSQData Data, data::ImmutableMemoryDataset Points, threads::ThreadPool Pool>
auto svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::transaction_data_builder>,
    const Data& data,
    const Points& points,
    std::span<size_t> slots,
    Pool& pool
) {
    using element_type = typename Data::element_type;
    using points_type = SQDataset<element_type, Data::extent, lib::Allocator<element_type>>;
    using compressor_type =
        detail::Compressor<element_type, typename points_type::data_type>;

    const auto scale = data.get_scale();
    const auto bias = data.get_bias();
    auto compressor = compressor_type{scale, bias};
    auto compressed = compressor(points, pool, lib::Allocator<element_type>{});
    return TransactionData(
        data, points, points_type{std::move(compressed), scale, bias}, slots
    );
}

} // namespace svs::quantization::scalar
