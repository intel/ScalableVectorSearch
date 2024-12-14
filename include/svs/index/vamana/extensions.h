/*
 * Copyright 2023 Intel Corporation
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

#include "svs/concepts/distance.h"
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/core/medioid.h"
#include "svs/core/query_result.h"
#include "svs/index/vamana/greedy_search.h"
#include "svs/lib/invoke.h"
#include "svs/lib/misc.h"
#include "svs/lib/preprocessor.h"
#include "svs/lib/threads.h"

namespace svs::index::vamana::extensions {

/////
///// MISC EXTENSIONS
/////

struct ComputeEntryPoint {
    ///
    /// @brief Compute the graph entry point for the given dataset.
    ///
    /// @param dataset The dataset being processed.
    /// @param threadpool The pool to use for computation.
    /// @param predicate Optional predicate returning `true` if the corresponding index
    ///        in ``dataset`` should be processed.
    ///
    template <
        typename Data,
        threads::ThreadPool Pool,
        typename Predicate = lib::ReturnsTrueType>
    size_t operator()(
        const Data& dataset, Pool& threadpool, Predicate&& predicate = Predicate{}
    ) const {
        return svs::svs_invoke(*this, dataset, threadpool, SVS_FWD(predicate));
    }
};

/// @brief Entry point customization.
inline constexpr ComputeEntryPoint compute_entry_point{};

// Default Implementation
template <typename Data, threads::ThreadPool Pool, typename Predicate>
size_t svs_invoke(
    svs::tag_t<compute_entry_point>,
    const Data& dataset,
    Pool& threadpool,
    Predicate&& predicate
) {
    return utils::find_medioid(dataset, threadpool, predicate);
}

/////
///// PERFORMANCE EXTENSIONS
/////

struct EstimatePrefetchParameters {
    using This = EstimatePrefetchParameters;

    template <data::ImmutableMemoryDataset Data>
    vamana::GreedySearchPrefetchParameters operator()(const Data& data) const {
        // If a specialization exists - call that.
        // Otherwise, use a default approximation based on the size of the data.
        if constexpr (svs::svs_invocable<This, const Data&>) {
            return svs::svs_invoke(*this, data);
        } else {
            using T = typename Data::element_type;
            auto dims = data.dimensions();
            auto bytes_per_entry = sizeof(T) * dims;

            if (bytes_per_entry >= 4096) {
                // No-prefetching
                return vamana::GreedySearchPrefetchParameters{0, 0};
            } else if (bytes_per_entry >= 1024) {
                // Conservative prefetching
                return vamana::GreedySearchPrefetchParameters{1, 1};
            } else if (bytes_per_entry >= 256) {
                // More aggressive prefetching
                return vamana::GreedySearchPrefetchParameters{4, 2};
            } else if (bytes_per_entry > 128) {
                // Aggressive prefetching
                return vamana::GreedySearchPrefetchParameters{8, 1};
            } else {
                // Aggressive prefetching
                return vamana::GreedySearchPrefetchParameters{16, 1};
            }
        }
    }
};

inline constexpr EstimatePrefetchParameters estimate_prefetch_parameters{};

/////
///// BUILDING EXTENSIONS
/////

struct BuildAdaptorType {
    ///
    /// @brief Return a dataset adaptor to assist with index construction.
    ///
    /// @param dataset The dataset to adapt for index building. This is the primary type
    ///        for controlling dispatch to the implementation.
    /// @param distance The base distance functor desired by the index.
    ///
    /// To specialize this extension point, extend the method:
    ///
    /// @code{cpp}
    /// template</*Data Constraint*/ Data, typename Distance>
    /// svs_invoke(
    ///     svs::tag_t<svs::index::vamana::extensions::build_adaptor>,
    ///     const Data&,
    ///     const Distance&
    /// )
    /// @endcode
    ///
    /// Where the contraint on the ``Data`` template argument (and potentially on
    /// ``Distance``) is used for overload resolution.
    ///
    /// See the referenced link for an example of what the expected adaptor API.
    ///
    /// @sa ``svs::index::vamana::extensions::DefaultBuildAdaptor``
    ///
    template <typename Data, typename Distance>
    SVS_FORCE_INLINE auto operator()(const Data& dataset, const Distance& distance) const {
        return svs::svs_invoke(*this, dataset, distance);
    }
};

inline constexpr BuildAdaptorType build_adaptor{};

/// @brief Default reference implementation for dataset/vamana build adaptors.
template <typename Distance> struct DefaultBuildAdaptor {
  public:
    ///
    /// Index construction happens in two phases: A graph search for candidate generation
    /// and a general phase that includes candidate refinement and pruning.
    ///
    /// While in the case of the example, the distance functor for these two phases are the
    /// same, this does not need to hold in general.
    ///
    using search_distance_type = Distance;
    using general_distance_type = Distance;

    // Helper template
    template <typename Data> using const_value_type = typename Data::const_value_type;

    ///
    /// Access the dataset to obtain a left-hand argument (i.e. "query") to use for the
    /// graph search.
    ///
    /// @param data The dataset over which the index is being constructed.
    /// @param i The index of the dataset element being accessed.
    ///
    template <typename Data>
    const_value_type<Data> access_query_for_graph_search(const Data& data, size_t i) const {
        return data.get_datum(i);
    }

    /// The dataset accessor to use during graph search.
    data::GetDatumAccessor graph_search_accessor() const {
        return data::GetDatumAccessor{};
    }

    ///
    /// The distance functor to use for graph search. Must be able to take the returned
    /// object from ``access_query_for_graph_search`` as the left-hand argument and the
    /// result from ``graph_search_accessor()(data, i)`` on the right-hand side.
    ///
    search_distance_type& graph_search_distance() { return distance_; }

    ///
    /// The distance functor to use for all non-graph-search distance computations.
    /// Will be used in conjunction with the elements obtained through the accessor returned
    /// via ``general_accessor()`` as both left-hand and right-hand arguments.
    ///
    general_distance_type& general_distance() { return distance_; }

    ///
    /// The accessor through which the dataset will be accessed for all non-graph-search
    /// accessed.
    ///
    data::GetDatumAccessor general_accessor() const { return data::GetDatumAccessor{}; }

    ///
    /// Since the distance functors for graph search and general operations may be
    /// different, this method provides a hook for converting the left-hand side obtained
    /// through ``access_query_for_graph_search`` to a type appropriate to give as the
    /// left-hand side of the distance functor given by ``general_distance()``.
    ///
    /// Implementations do not need to explicitly modify the query if it is already
    /// compatible.
    ///
    /// @param data The dataset over which the index is being built.
    /// @param i The index of the dataset element being accessed.
    /// @param pre_search_query The result of ``access_query_for_graph_search(data, i)``.
    ///
    template <typename Data>
    SVS_FORCE_INLINE const const_value_type<Data>& modify_post_search_query(
        const Data& SVS_UNUSED(data),
        size_t SVS_UNUSED(i),
        const const_value_type<Data>& pre_search_query
    ) const {
        return pre_search_query;
    }

    ///
    /// Flag to indicate that the distance functors and left-hand sides for general
    /// distance computations and accesses are sufficiently different that
    /// ``svs::distance::maybe_fix_argument`` needs to be re-applied following graph search.
    ///
    /// If this evaluates to ``true``, then the following will be performed.
    ///
    /// @code{cpp}
    /// // Setup for search
    /// // Variable `adaptor` refers to an instance of ``DefaultBuildAdaptor``.
    /// auto graph_search_query = adaptor.access_query_for_graph_search(data, i);
    ///
    /// // perform graph search.
    /// auto post_search_query = adaptor.modify_post_search_query(
    ///     data, i, graph_search_query
    /// );
    /// auto& general_distance = adaptor.general_distance();
    /// distance::maybe_fix_argument(general_distance, post_search_query);
    /// @endcode
    ///
    static constexpr bool refix_argument_after_search = false;

    ///
    /// Following the graph search, candidate distances may be refined/converted into the
    /// ``geneneral_distance()`` domain using this extension point.
    ///
    /// @param data The dataset overwhich the index is being constructed.
    /// @param distance The distance functor used. The object passed for this argument is
    ///        that returned by ``general_distance()``.
    /// @param query The left-hand side query argument obtained from
    ///        ``modify_post_search_query()``.
    /// @param n The neighbor containing (at least) an ID and previously computed distance.
    ///
    /// @returns A new ``svs::Neighbor<typename N::index_type>`` containing a potentially
    ///          modified distance between the query and the candidate. The ID must be
    ///          preserved.
    ///
    template <typename Data, NeighborLike N>
    SVS_FORCE_INLINE Neighbor<typename N::index_type> post_search_modify(
        const Data& SVS_UNUSED(data),
        general_distance_type& SVS_UNUSED(distance),
        const const_value_type<Data>& SVS_UNUSED(query),
        const N& n
    ) const {
        return n;
    }

  public:
    [[no_unique_address]] Distance distance_{};
};

// Default to the ``DefaultBuildAdaptor``.
template <typename Data, typename Distance>
DefaultBuildAdaptor<Distance> svs_invoke(
    svs::tag_t<build_adaptor>,
    const Data& SVS_UNUSED(data),
    const Distance& SVS_UNUSED(distance)
) {
    return DefaultBuildAdaptor<Distance>{};
}

/////
///// SEARCH EXTENSIONS
/////

// Temporary customization to disable single-search for a given dataset type.
template <typename Data>
SVS_TEMPORARY_DISABLE_SINGLE_SEARCH struct TemporaryDisableSingleSearch {
    [[nodiscard]] inline constexpr bool operator()() const {
        if constexpr (svs::svs_invocable<TemporaryDisableSingleSearch>) {
            return svs::svs_invoke(*this);
        } else {
            return false;
        }
    }
};

template <typename Data>
SVS_TEMPORARY_DISABLE_SINGLE_SEARCH inline constexpr TemporaryDisableSingleSearch<Data>
    temporary_disable_single_search{};

/// Preprartion steps for single search.
struct VamanaSingleSearchSetupType {
    template <typename Data, typename Distance>
    using result_t =
        svs::svs_invoke_result_t<VamanaSingleSearchSetupType, const Data&, const Distance&>;

    ///
    /// @brief Allocate scratch space to be used to process a single query.
    ///
    /// @param dataset The dataset being searched over.
    /// @param distance The base distance functor to use when comparing dataset elements.
    ///
    /// The returned object may be reused for multiple queries and will be passed as the
    /// third argument to the ``svs::index::vamana::extensions::single_search``
    /// customization point object.
    ///
    template <typename Data, typename Distance>
    result_t<Data, Distance>
    operator()(const Data& dataset, const Distance& distance) const {
        return svs::svs_invoke(*this, dataset, distance);
    }
};

/// @brief Customization point object for processing a single query.
inline constexpr VamanaSingleSearchSetupType single_search_setup{};

// Default Implementation.
template <typename Data, typename Distance>
Distance svs_invoke(
    svs::tag_t<single_search_setup>,
    const Data& SVS_UNUSED(dataset),
    const Distance& distance
) {
    return threads::shallow_copy(distance);
}

/// Preparation steps for batch search.
namespace detail {
// The result of setup depends on whether the customization point for batch search is
// explicitly extended or if we're using the single-query fallback.
template <bool> struct ResultTypeDetector {
    template <typename CPO, typename Data, typename Distance>
    using type = svs::svs_invoke_result_t<CPO, const Data&, const Distance&>;
};

template <> struct ResultTypeDetector<false> {
    template <typename CPO, typename Data, typename Distance>
    using type = typename VamanaSingleSearchSetupType::template result_t<Data, Distance>;
};
} // namespace detail

struct VamanaPerThreadBatchSearchSetupType {
    using This = VamanaPerThreadBatchSearchSetupType;

    template <typename Data, typename Distance>
    static constexpr bool extension_exists =
        svs::svs_invocable<This, const Data&, const Distance&>;

    // The dance here goes like this:
    // 1. If an explicit overload for `VamanaPerThreadBatchSearchSetup` exists, then we
    //    want to deduce the return-type from that call.
    // 2. If one *does not* exist, then we want to defer to the implementation for
    //    `VamanaSingleSearchSetup`.
    template <typename Data, typename Distance>
    using result_t = typename detail::ResultTypeDetector<
        extension_exists<Data, Distance>>::template type<This, Data, Distance>;

    ///
    /// @brief Pre-allocate scratch space for processing a batch of queries on a thread.
    ///
    /// @param dataset The dataset being searched.
    /// @param distance The base distance functor to compare queries with dataset entries.
    ///
    /// If an explicit specialization of the
    /// ``svs::index::vamana::per_thread_batch_search_setup`` customization point object
    /// exists, that will be called. Otherwise, the fallback implementation of
    /// @code{cpp}
    /// svs::index::vamana::extensions::single_search_setup(dataset, distance)
    /// @endcode
    /// will be called.
    ///
    template <typename Data, typename Distance>
    result_t<Data, Distance>
    operator()(const Data& dataset, const Distance& distance) const {
        if constexpr (extension_exists<Data, Distance>) {
            return svs::svs_invoke(*this, dataset, distance);
        } else {
            return single_search_setup(dataset, distance);
        }
    }
};

inline constexpr VamanaPerThreadBatchSearchSetupType per_thread_batch_search_setup{};

/// Search Implementations
struct VamanaSingleSearchType {
    ///
    /// @brief Dispatch to a search routine for a single thread.
    ///
    /// @param data The dataset being searched over.
    /// @param search_buffer Search resource to be passed to the ``search`` functor.
    ///        Following the invocation of this function, results will be available in the
    ///        search buffer.
    /// @param scratch Mutable scratch space returned by
    ///        ``svs::index::vamana::extensions::single_search_setup``().
    /// @param query The query used for this search.
    /// @param search A search functor. See the extended description.
    ///
    /// API of the ``search`` argument.
    /// This argument is invocable as follows:
    /// @code{cpp}
    /// search(query, accessor, distance, search_buffer)
    /// @endcode
    /// Where
    /// * ``query`` is the query argument.
    /// * ``accessor`` satisfies ``svs::data::AccessorFor<Data>``.
    /// * ``distance`` is a distance functor capable of handling ``query`` as a left-hand
    ///   argument the result of ``accessor(data, i)`` as a right-hand argument.
    /// * ``search_buffer`` is the provided search buffer.
    ///
    template <
        typename Data,
        typename SearchBuffer,
        typename Scratch,
        typename Query,
        typename Search>
    void operator()(
        const Data& data,
        SearchBuffer& search_buffer,
        Scratch& scratch,
        const Query& query,
        const Search& search,
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
    ) const {
        svs::svs_invoke(*this, data, search_buffer, scratch, query, search, cancel);
    }
};

/// Customization point object for processing single queries.
inline constexpr VamanaSingleSearchType single_search{};

/// In this function, cancel does not need to be called since search will call cancel
/// However, lvq requires cancel to be called to skip reranking
/// Here we have cancel to make the interface consistent with lvq
template <
    typename Data,
    typename SearchBuffer,
    typename Distance,
    typename Query,
    typename Search>
SVS_FORCE_INLINE void svs_invoke(
    svs::tag_t<single_search>,
    const Data& SVS_UNUSED(dataset),
    SearchBuffer& search_buffer,
    Distance& distance,
    const Query& query,
    const Search& search,
    const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
) {
    // Check if request to cancel the search
    if (cancel()) {
        return;
    }
    // Perform graph search.
    auto accessor = data::GetDatumAccessor();
    search(query, accessor, distance, search_buffer);
}

///
/// @brief Customization point for working with a batch of threads.
///
/// For datasets that do not need to explicitly handle batches of queries on a single
/// thread, prefer to extend ``VamanaSingleSearchType`` instead.
///
struct VamanaPerThreadBatchSearchType {
    ///
    /// @brief Dispatch to the implementation for processing a batch of queries.
    ///
    /// @param data The dataset being queried.
    /// @param search_buffer A search resource to pass to the ``search`` functor.
    /// @param scratch The scratch space returned by
    ///        ``svs::index::vamana::extensions::per_thread_batch_search_setup()``.
    /// @param queries The global batch of queries.
    /// @param result The global result buffer.
    /// @param thread_indices The entries in ``queries`` and ``result`` that this function
    ///        is responsible for handling.
    /// @param search A functor for processing a single query and storing the results in
    ///        the ``search_buffer`` argument. See the documentation for
    ///        ``svs::index::vamana::extensions::VamanaSingleSearchType`` for details on
    ///        the signature of ``search``.
    ///
    /// This function is expected to process all element queries for the range defined by
    /// the ``thread_indices`` argument and store the results in the corresponding position
    /// of the ``result`` buffer.
    ///
    /// The expected number of neighbors may be obtained through ``result.n_neighbors()``.
    ///
    template <
        data::ImmutableMemoryDataset Data,
        typename SearchBuffer,
        typename Scratch,
        data::ImmutableMemoryDataset Queries,
        std::integral I,
        typename Search>
    SVS_FORCE_INLINE void operator()(
        const Data& data,
        SearchBuffer& search_buffer,
        Scratch& scratch,
        const Queries& queries,
        QueryResultView<I>& result,
        threads::UnitRange<size_t> thread_indices,
        const Search& search,
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
    ) const {
        svs::svs_invoke(
            *this,
            data,
            search_buffer,
            scratch,
            queries,
            result,
            thread_indices,
            search,
            cancel
        );
    }
};

/// @brief Customization point object for batch search.
inline constexpr VamanaPerThreadBatchSearchType per_thread_batch_search{};

// Default Implementation
template <
    typename Data,
    typename SearchBuffer,
    typename Distance,
    typename Queries,
    std::integral I,
    typename Search>
void svs_invoke(
    svs::tag_t<per_thread_batch_search>,
    const Data& dataset,
    SearchBuffer& search_buffer,
    Distance& distance,
    const Queries& queries,
    QueryResultView<I>& result,
    threads::UnitRange<size_t> thread_indices,
    const Search& search,
    const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
) {
    // Fallback implementation
    size_t num_neighbors = result.n_neighbors();
    for (auto i : thread_indices) {
        // Check if request to cancel the search
        if (cancel()) {
            return;
        }
        // Perform search - results will be queued in the search buffer.
        single_search(dataset, search_buffer, distance, queries.get_datum(i), search, cancel);

        // Copy back results.
        for (size_t j = 0; j < num_neighbors; ++j) {
            result.set(search_buffer[j], i, j);
        }
    }
}

/////
///// Calibration Hooks
/////

template <typename Data> struct UsesReranking {
    constexpr bool operator()() const {
        if constexpr (svs::svs_invocable<UsesReranking>) {
            return svs::svs_invoke(*this);
        } else {
            return false;
        }
    }
};

template <typename Data> inline constexpr UsesReranking<Data> calibration_uses_reranking{};

/////
///// Reconstruct Vector
/////

struct Reconstruct {
    template <data::ImmutableMemoryDataset Data> auto operator()(const Data& data) const {
        return svs::svs_invoke(*this, data);
    }
};

// Customization point for reconstructing vectors.
inline constexpr Reconstruct reconstruct_accessor{};

template <typename T, size_t Extent, typename Alloc>
SVS_FORCE_INLINE data::GetDatumAccessor svs_invoke(
    svs::tag_t<reconstruct_accessor> SVS_UNUSED(cpo),
    const data::SimpleData<T, Extent, Alloc>& SVS_UNUSED(dataset)
) {
    return data::GetDatumAccessor();
}

} // namespace svs::index::vamana::extensions
