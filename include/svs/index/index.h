/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
 */
#pragma once

// svs
#include "svs/concepts/data.h"
#include "svs/core/query_result.h"

// stl
#include <concepts>

///
/// @defgroup index Core Index Implementations
/// @brief Low level implementations for indexes.
///

namespace svs::index {

/// Type alias for the search parameters associated with an index.
template <typename Index>
using search_parameters_t = typename Index::search_parameters_type;

/// Type alias for the scratch space associated with an index.
template <typename Index> using scratchspace_t = typename Index::scratchspace_type;

/////
///// Batch Search
/////

template <typename Index, std::integral I, data::ImmutableMemoryDataset Queries>
void search_batch_into_with(
    Index& index,
    svs::QueryResultView<I> result,
    const Queries& queries,
    const search_parameters_t<Index>& search_parameters
) {
    // Assert pre-conditions.
    assert(result.n_queries() == queries.size());
    index.search(result, queries, search_parameters);
}

// Apply default search parameters
template <typename Index, std::integral I, data::ImmutableMemoryDataset Queries>
void search_batch_into(
    Index& index, svs::QueryResultView<I> result, const Queries& queries
) {
    svs::index::search_batch_into_with(
        index, result, queries, index.get_search_parameters()
    );
}

// Allocate the destination result and invoke `search_batch_into`.
template <typename Index, data::ImmutableMemoryDataset Queries>
svs::QueryResult<size_t> search_batch_with(
    Index& index,
    const Queries& queries,
    size_t num_neighbors,
    const search_parameters_t<Index>& search_parameters
) {
    auto result = svs::QueryResult<size_t>{queries.size(), num_neighbors};
    svs::index::search_batch_into_with(index, result.view(), queries, search_parameters);
    return result;
}

// Obtain default search parameters.
template <typename Index, data::ImmutableMemoryDataset Queries>
svs::QueryResult<size_t>
search_batch(Index& index, const Queries& queries, size_t num_neighbors) {
    return svs::index::search_batch_with(
        index, queries, num_neighbors, index.get_search_parameters()
    );
}
} // namespace svs::index
