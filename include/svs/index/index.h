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

// svs
#include "svs/concepts/data.h"
#include "svs/core/logging.h"
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
    const search_parameters_t<Index>& search_parameters,
    svs::logging::logger_ptr logger = svs::logging::get(),
    const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
) {
    // Assert pre-conditions.
    assert(result.n_queries() == queries.size());
    index.search(result, queries, search_parameters, logger, cancel);
}

// Apply default search parameters
template <typename Index, std::integral I, data::ImmutableMemoryDataset Queries>
void search_batch_into(
    Index& index,
    svs::QueryResultView<I> result,
    const Queries& queries,
    svs::logging::logger_ptr logger = svs::logging::get(),
    const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
) {
    svs::index::search_batch_into_with(
        index, result, queries, index.get_search_parameters(), logger, cancel
    );
}

// Allocate the destination result and invoke `search_batch_into`.
template <typename Index, data::ImmutableMemoryDataset Queries>
svs::QueryResult<size_t> search_batch_with(
    Index& index,
    const Queries& queries,
    size_t num_neighbors,
    const search_parameters_t<Index>& search_parameters,
    svs::logging::logger_ptr logger = svs::logging::get(),
    const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
) {
    auto result = svs::QueryResult<size_t>{queries.size(), num_neighbors};
    svs::index::search_batch_into_with(
        index, result.view(), queries, search_parameters, logger, cancel
    );
    return result;
}

// Obtain default search parameters.
template <typename Index, data::ImmutableMemoryDataset Queries>
svs::QueryResult<size_t> search_batch(
    Index& index,
    const Queries& queries,
    size_t num_neighbors,
    svs::logging::logger_ptr logger = svs::logging::get(),
    const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
) {
    return svs::index::search_batch_with(
        index, queries, num_neighbors, index.get_search_parameters(), logger, cancel
    );
}
} // namespace svs::index
