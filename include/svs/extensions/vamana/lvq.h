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

#include "svs/index/vamana/extensions.h"
#include "svs/quantization/lvq/lvq.h"

namespace svs::quantization::lvq {

/////
///// Entry Point Computation
/////

template <IsLVQDataset Data, threads::ThreadPool Pool, typename Predicate>
size_t svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::compute_entry_point>,
    const Data& data,
    Pool& threadpool,
    Predicate&& predicate
) {
    return utils::find_medioid(data, threadpool, SVS_FWD(predicate), data.decompressor());
}

/////
///// Vamana Build
/////

template <typename Distance, bool TwoLevel> struct VamanaBuildAdaptor {
  public:
    // LVQ can reuse the same distance functor for all combinations of accesses.
    using distance_type = DecompressionAdaptor<biased_distance_t<Distance>>;
    using search_distance_type = distance_type;
    using general_distance_type = distance_type;
    using graph_search_accessor_type =
        std::conditional_t<TwoLevel, PrimaryAccessor, data::GetDatumAccessor>;

    // Use both the primary and residual (if it exists) to fully reconstruct the LHS.
    template <IsLVQDataset Data>
    auto access_query_for_graph_search(const Data& dataset, size_t i) const {
        return dataset.get_datum(i);
    }

    // There is no need to modify the query following graph search as the same object
    // may be reused for general distance computations.
    template <IsLVQDataset Data, typename Query>
    const Query& modify_post_search_query(
        const Data& SVS_UNUSED(data), size_t SVS_UNUSED(i), const Query& query
    ) const {
        return query;
    }

    // The dataset element extracted for the graph search is reassembled from both the
    // primary and residual components.
    //
    // This same query can be reused for both graph search and general distance
    // computations with the same distance functor.
    //
    // As such, there is no-need to call `maybe_fix_argument` following graph search.
    static constexpr bool refix_argument_after_search = false;

    // Search functor used for the graph search portion of index construction.
    search_distance_type& graph_search_distance() { return distance_; }

    // Accessor used for the graph search portion of index construction.
    // Only access the primary dataset.
    graph_search_accessor_type graph_search_accessor() const {
        return graph_search_accessor_type{};
    }

    // If this is a two-level dataset, we can refine the neighbors returned from graph
    // search by recomputing the distance from the LHS using both the primary and residual.
    //
    // If this is just a one-level dataset, then no such refinement can be performed.
    template <IsLVQDataset Data, typename Query, NeighborLike N>
    Neighbor<typename N::index_type> post_search_modify(
        [[maybe_unused]] const Data& dataset,
        [[maybe_unused]] general_distance_type& d,
        [[maybe_unused]] const Query& query,
        const N& n
    ) const {
        if constexpr (Data::residual_bits == 0) {
            return n;
        } else {
            auto id = n.id();
            return Neighbor(id, distance::compute(d, query, dataset.get_datum(id)));
        }
    }

    // General distance computations use the same underlying distance functor as graph
    // search distances.
    general_distance_type& general_distance() { return distance_; }

    // However, general data access should be done using full precision if available.
    data::GetDatumAccessor general_accessor() const { return data::GetDatumAccessor{}; }

  public:
    distance_type distance_;
};

template <IsLVQDataset Dataset, typename Distance>
VamanaBuildAdaptor<Distance, IsTwoLevelDataset<Dataset>> svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::build_adaptor>,
    const Dataset& dataset,
    const Distance& distance
) {
    constexpr bool TwoLevel = IsTwoLevelDataset<Dataset>;
    return VamanaBuildAdaptor<Distance, TwoLevel>{adapt_for_self(dataset, distance)};
}

/////
///// Vamana Search
/////

template <IsLVQDataset Data, typename Distance>
biased_distance_t<Distance> svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::single_search_setup>,
    const Data& data,
    const Distance& distance
) {
    return adapt(data, distance);
}

// Only extend search for two-level dataset.
// One level datasets can use the default implementation directly.
template <
    IsTwoLevelDataset Data,
    typename SearchBuffer,
    typename Distance,
    typename Query,
    typename Search>
void svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::single_search>,
    const Data& dataset,
    SearchBuffer& search_buffer,
    Distance& distance,
    const Query& query,
    const Search& search
) {
    // Perform graph search.
    search(query, PrimaryAccessor(), distance, search_buffer);

    // Rerank the results
    for (size_t j = 0, jmax = search_buffer.size(); j < jmax; ++j) {
        auto& neighbor = search_buffer[j];
        auto id = neighbor.id();
        auto new_distance = distance::compute(distance, query, dataset.get_datum(id));
        neighbor.set_distance(new_distance);
    }
    search_buffer.sort();
}
} // namespace svs::quantization::lvq
