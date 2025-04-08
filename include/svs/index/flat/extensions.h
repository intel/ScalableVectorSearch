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

namespace svs::index::flat::extensions {
/////
///// SEARCH EXTENSIONS
/////

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
        single_search(
            dataset, search_buffer, distance, queries.get_datum(i), search, cancel
        );

        // Copy back results.
        for (size_t j = 0; j < num_neighbors; ++j) {
            result.set(search_buffer[j], i, j);
        }
    }
}

/////
///// Distance
/////
struct ComputeDistanceType {
    template <typename Data, typename Distance, typename Query>
    double operator()(
        const Data& data, const Distance& distance, size_t internal_id, const Query& query
    ) const {
        return svs_invoke(*this, data, distance, internal_id, query);
    }
};
// CPO for distance computation
inline constexpr ComputeDistanceType get_distance_ext{};
template <typename Data, typename Distance, typename Query>
double svs_invoke(
    svs::tag_t<get_distance_ext>,
    const Data& data,
    const Distance& distance,
    size_t internal_id,
    const Query& query
) {
    // Convert query to span for uniform handling
    auto query_span = lib::as_const_span(query);

    // Get distance
    auto dist_f = single_search_setup(data, distance);
    svs::distance::maybe_fix_argument(dist_f, query_span);

    // Get the vector from the index
    auto indexed_span = data.get_datum(internal_id);

    // Compute the distance using the appropriate distance function
    auto dist = svs::distance::compute(dist_f, query_span, indexed_span);

    return static_cast<double>(dist);
}
} // namespace svs::index::flat::extensions