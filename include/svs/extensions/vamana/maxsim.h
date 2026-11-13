/*
 * Copyright 2026 Intel Corporation
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
#include "svs/core/data/with_vector_sets.h"
#include "svs/core/distance/maxsim.h"
#include "svs/index/vamana/extensions.h"
#include "svs/lib/exception.h"
#include "svs/lib/preprocessor.h"

// stl
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <vector>

namespace svs::index::vamana::extensions {

namespace detail {

// The intended caller's graph-search distance already consumes raw query tokens, so the
// primary distance must accept the query set directly against a primary datum.
template <typename Distance, typename Query, typename PrimaryDatum>
concept AcceptsVectorSetQuery =
    svs::data::VectorSet<Query> &&
    requires(Distance& dist, const Query& query, const PrimaryDatum& datum) {
        { svs::distance::compute(dist, query, datum) } -> std::convertible_to<float>;
    };

// Re-sort by MaxSim score (larger is better). The buffer's own comparator is fixed at
// construction to the primary distance's direction, which may disagree with MaxSim's, so
// sort a local copy and write the order back rather than reuse `search_buffer.sort()`.
template <typename SearchBuffer> void resort_by_maxsim(SearchBuffer& search_buffer) {
    size_t valid = search_buffer.size();
    using value_type = typename SearchBuffer::value_type;
    std::vector<value_type> reordered(search_buffer.begin(), search_buffer.end());
    std::sort(
        reordered.begin(),
        reordered.end(),
        [](const value_type& a, const value_type& b) {
            return svs::distance::MaxSim::compare{}(a.distance(), b.distance());
        }
    );
    for (size_t j = 0; j < valid; ++j) {
        search_buffer[j] = reordered[j];
    }
}

} // namespace detail

/// Native re-rank: run the ordinary graph search with the primary distance, then
/// overwrite every candidate still in the pool with its MaxSim score against the query
/// set, and restore best-first order by that score.
template <
    typename Primary,
    svs::data::ImmutableVectorSetDataset Sets,
    typename SearchBuffer,
    typename Distance,
    typename Query,
    typename Search,
    typename Index>
    requires detail::
        AcceptsVectorSetQuery<Distance, Query, typename Primary::const_value_type>
    void svs_invoke(
        svs::tag_t<single_search>,
        const svs::data::WithVectorSets<Primary, Sets>& dataset,
        SearchBuffer& search_buffer,
        Distance& distance,
        const Query& query,
        const Search& search,
        const Index& SVS_UNUSED(index),
        const lib::DefaultPredicate& cancel = lib::Returns(lib::Const<false>())
    ) {
    // `check_and_supplement_search_buffer` inserts candidates scored with the primary
    // distance into a buffer this overload has already rescored to MaxSim; mixing the two
    // would sort as if every entry were a MaxSim score.
    static_assert(
        !Index::needs_id_translation,
        "MaxSim re-rank does not support indexes that need id translation"
    );

    if (cancel()) {
        return;
    }

    if (query.dimensions() != dataset.vector_sets().dimensions()) {
        throw ANNEXCEPTION(
            "Query token dimension {} does not match dataset token dimension {}",
            query.dimensions(),
            dataset.vector_sets().dimensions()
        );
    }

    auto accessor = data::GetDatumAccessor();
    search(query, accessor, distance, search_buffer);

    if (cancel()) {
        return;
    }

    // Packs the query once (must_fix_argument) and reuses that packing for every
    // candidate still in the pool: repacking per candidate would cost as much as
    // scoring them, which is exactly what PackedOperand/MaxSim::fix_argument avoids.
    svs::distance::MaxSim maxsim{};
    svs::distance::maybe_fix_argument(maxsim, query);

    size_t valid = search_buffer.size();
    for (size_t j = 0; j < valid; ++j) {
        auto& neighbor = search_buffer[j];
        float score =
            svs::distance::compute(maxsim, query, dataset.get_vector_set(neighbor.id()));
        neighbor.set_distance(score);
    }

    detail::resort_by_maxsim(search_buffer);
}

// The default `per_thread_batch_search` (extensions.h) dispatches per query through the
// `single_search` customization point above, which already leaves the buffer sorted
// best-first by MaxSim; no override is needed for the batch path.

// `UsesReranking` is left at its default of `false`: the iterator's reranking path
// (include/svs/index/vamana/iterator.h) recomputes distances via `data.get_primary(id)`
// against the base distance, which this dataset does not expose and which would not be a
// MaxSim score regardless.

template <typename Primary, svs::data::ImmutableVectorSetDataset Sets>
auto svs_invoke(
    svs::tag_t<reconstruct_accessor>,
    const svs::data::WithVectorSets<Primary, Sets>& dataset
) {
    return reconstruct_accessor(dataset.primary());
}

} // namespace svs::index::vamana::extensions
