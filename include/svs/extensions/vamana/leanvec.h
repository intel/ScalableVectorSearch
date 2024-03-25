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
#include "svs/leanvec/leanvec.h"
// We need to include the `lvq` extensions because LeanVec is LVQ compatible and we may
// need to access specializations defined by LVQ.
#include "svs/extensions/vamana/lvq.h"

namespace svs::leanvec {

/////
///// Entry Point Computation
/////

// Delegate to the entry-point computation for the primary dataset.
template <IsLeanDataset Data, threads::ThreadPool Pool, typename Predicate>
size_t svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::compute_entry_point>,
    const Data& data,
    Pool& threadpool,
    Predicate&& predicate
) {
    return svs::index::vamana::extensions::compute_entry_point(
        data.view_primary_dataset(), threadpool, SVS_FWD(predicate)
    );
}

template <IsLeanDataset Data>
svs::index::vamana::GreedySearchPrefetchParameters svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::estimate_prefetch_parameters>,
    const Data& SVS_UNUSED(data)
) {
    // Conservative prefetching.
    return svs::index::vamana::GreedySearchPrefetchParameters{1, 1};
}

/////
///// Vamana Build
/////

template <typename Distance> struct VamanaBuildAdaptor {
  public:
    using search_distance_type = Distance;
    using general_distance_type = Distance;

    explicit VamanaBuildAdaptor(Distance distance)
        : distance_{std::move(distance)} {}

    // For graph construction, primary data is used for all purposes
    template <IsLeanDataset Data>
    auto access_query_for_graph_search(const Data& dataset, size_t i) const {
        return dataset.get_datum(i);
    }

    template <IsLeanDataset Data, typename Query>
    const Query& modify_post_search_query(
        const Data& SVS_UNUSED(data), size_t SVS_UNUSED(i), const Query& query
    ) const {
        return query;
    }

    // As such, there is no-need to call `maybe_fix_argument` following graph search.
    static constexpr bool refix_argument_after_search = false;

    // Search functor used for the graph search portion of index construction.
    search_distance_type& graph_search_distance() { return distance_; }

    // Only access the primary data
    data::GetDatumAccessor graph_search_accessor() const {
        return data::GetDatumAccessor{};
    }

    // Using only the primary data for graph construction, no need for reranking
    template <IsLeanDataset Data, typename Query, NeighborLike N>
    Neighbor<typename N::index_type> post_search_modify(
        const Data& SVS_UNUSED(dataset),
        general_distance_type& SVS_UNUSED(d),
        const Query& SVS_UNUSED(query),
        const N& n
    ) const {
        return n;
    }

    // General distance computations use the same underlying distance functor as graph
    // search distances.
    general_distance_type& general_distance() { return distance_; }

    // Use primary data for graph construction in all cases
    data::GetDatumAccessor general_accessor() const { return data::GetDatumAccessor{}; }

  public:
    Distance distance_;
};

template <IsLeanDataset Dataset, typename Distance>
auto svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::build_adaptor>,
    const Dataset& dataset,
    const Distance& distance
) {
    return VamanaBuildAdaptor{dataset.adapt_for_self(distance)};
}

/////
///// Vamana Search
/////

// Returning a tuple consisting of:
//
// * The original abstract distance (to be used in pre-processing)
// * The distance modified for the primary dataset.
// * The distance modified for the secondary dataset.
//
template <IsLeanDataset Data, typename Distance>
auto svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::per_thread_batch_search_setup>,
    const Data& data,
    const Distance& distance
) {
    return std::make_tuple(
        threads::shallow_copy(distance),
        data.adapt(distance),
        data.adapt_secondary(distance)
    );
}

template <
    IsLeanDataset Data,
    typename SearchBuffer,
    typename Scratch,
    typename QueryType,
    std::integral I,
    typename Search>
void svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::per_thread_batch_search>,
    const Data& dataset,
    SearchBuffer& search_buffer,
    Scratch& scratch,
    data::ConstSimpleDataView<QueryType> queries,
    QueryResultView<I>& result,
    threads::UnitRange<size_t> thread_indices,
    const Search& search
) {
    size_t num_neighbors = result.n_neighbors();
    size_t batch_start = thread_indices.start();

    auto& [distance, distance_primary, distance_secondary] = scratch;

    // TODO: Make view of query views less painful
    auto query_batch = data::ConstSimpleDataView<QueryType>(
        &queries.get_datum(thread_indices.front()).front(),
        thread_indices.size(),
        queries.dimensions()
    );

    auto processed_queries = dataset.preprocess_queries(distance, query_batch);

    // Perform graph search.
    for (auto i : thread_indices) {
        const auto& query = queries.get_datum(i);
        const auto& processed_query = processed_queries.get_datum(i - batch_start);

        {
            auto accessor = data::GetDatumAccessor();
            search(processed_query, accessor, distance_primary, search_buffer);
        }

        // For LeanVec, always rerank the result
        distance::maybe_fix_argument(distance_secondary, query);
        for (size_t j = 0, jmax = search_buffer.size(); j < jmax; ++j) {
            auto& neighbor = search_buffer[j];
            auto id = neighbor.id();
            auto new_distance =
                distance::compute(distance_secondary, query, dataset.get_secondary(id));
            neighbor.set_distance(new_distance);
        }
        search_buffer.sort();

        // Copy back results.
        for (size_t j = 0; j < num_neighbors; ++j) {
            result.set(search_buffer[j], i, j);
        }
    }
}

/////
///// Calibration
/////

template <IsLeanDataset Dataset>
constexpr bool svs_invoke(svs::index::vamana::extensions::UsesReranking<Dataset>) {
    return true;
}

/////
///// Reconstruction
/////

namespace detail {
template <IsLeanDataset Data> using secondary_dataset_type = typename Data::secondary_type;

// An auxiliary accessor that accesses the secondary dataset using the nested accessor.
template <typename T> struct SecondaryReconstructor {
    // The return-type dance here basically says that we return whetever the result of
    // invoking the `secondary_accessor_` on the secondary dataset returns.
    template <IsLeanDataset Data>
    std::invoke_result_t<T, const secondary_dataset_type<Data>&, size_t>
    operator()(const Data& data, size_t i) {
        return secondary_accessor_(data.view_secondary_dataset(), i);
    }

    ///// Members
    // Auxiliary accessor for the secondary dataset.
    T secondary_accessor_;
};

// Get the type of the accessor returned by the secondary dataset for this customization
// point object.
template <IsLeanDataset Data>
using secondary_accessor_t = svs::svs_invoke_result_t<
    svs::tag_t<svs::index::vamana::extensions::reconstruct_accessor>,
    const detail::secondary_dataset_type<Data>&>;

} // namespace detail

// Compose the reconstruction accessor for the secondary dataset with an accessor that
// grabs the secondary dataset.
template <IsLeanDataset Dataset>
detail::SecondaryReconstructor<detail::secondary_accessor_t<Dataset>> svs_invoke(
    svs::tag_t<svs::index::vamana::extensions::reconstruct_accessor> cpo,
    const Dataset& data
) {
    using T = detail::secondary_accessor_t<Dataset>;
    return detail::SecondaryReconstructor<T>{cpo(data.view_secondary_dataset())};
}

} // namespace svs::leanvec
