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
#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(SVS_ENABLE_STATS) && SVS_ENABLE_STATS
#include <chrono>
#endif

namespace svs::index::vamana::extensions {

#if defined(SVS_ENABLE_STATS) && SVS_ENABLE_STATS

/// Process-wide re-rank timing accumulated since the last `maxsim_rerank_stats_reset()`.
struct MaxSimRerankStats {
    uint64_t queries;
    uint64_t search_ns;
    uint64_t rerank_ns;
};

namespace stats_detail {
// Process-wide, not per-index: simpler than plumbing counters through every thread.
inline std::atomic<uint64_t> maxsim_queries{0};
inline std::atomic<uint64_t> maxsim_search_ns{0};
inline std::atomic<uint64_t> maxsim_rerank_ns{0};
} // namespace stats_detail

inline MaxSimRerankStats maxsim_rerank_stats_snapshot() {
    return MaxSimRerankStats{
        stats_detail::maxsim_queries.load(std::memory_order_relaxed),
        stats_detail::maxsim_search_ns.load(std::memory_order_relaxed),
        stats_detail::maxsim_rerank_ns.load(std::memory_order_relaxed)};
}

inline void maxsim_rerank_stats_reset() {
    stats_detail::maxsim_queries.store(0, std::memory_order_relaxed);
    stats_detail::maxsim_search_ns.store(0, std::memory_order_relaxed);
    stats_detail::maxsim_rerank_ns.store(0, std::memory_order_relaxed);
}

#endif

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
// `compare` does not depend on MaxSim's document-element template parameter, so the
// default instantiation's comparator applies regardless of which one scored the buffer.
template <typename SearchBuffer> void resort_by_maxsim(SearchBuffer& search_buffer) {
    size_t valid = search_buffer.size();
    using value_type = typename SearchBuffer::value_type;
    std::vector<value_type> reordered(search_buffer.begin(), search_buffer.end());
    std::sort(
        reordered.begin(),
        reordered.end(),
        [](const value_type& a, const value_type& b) {
            return svs::distance::MaxSim<>::compare{}(a.distance(), b.distance());
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
#if defined(SVS_ENABLE_STATS) && SVS_ENABLE_STATS
    auto t_start = std::chrono::steady_clock::now();
    search(query, accessor, distance, search_buffer);
    auto t_search_done = std::chrono::steady_clock::now();
#else
    search(query, accessor, distance, search_buffer);
#endif

    if (cancel()) {
        return;
    }

    // Packs the query once (must_fix_argument) and reuses that packing for every
    // candidate still in the pool: repacking per candidate would cost as much as
    // scoring them, which is exactly what PackedOperand/MaxSim::fix_argument avoids.
    // The document element type is the vector-set dataset's own element type, so a
    // bfloat16 store automatically gets the bf16 (and AMX, where available) kernel path
    // while float/Float16 stores keep today's unrounded behaviour.
    using DocElement = typename Sets::const_value_type::element_type;
    svs::distance::MaxSim<DocElement> maxsim{};
    svs::distance::maybe_fix_argument(maxsim, query);

    size_t valid = search_buffer.size();
    for (size_t j = 0; j < valid; ++j) {
        auto& neighbor = search_buffer[j];
        float score =
            svs::distance::compute(maxsim, query, dataset.get_vector_set(neighbor.id()));
        neighbor.set_distance(score);
    }

    detail::resort_by_maxsim(search_buffer);

#if defined(SVS_ENABLE_STATS) && SVS_ENABLE_STATS
    auto t_rerank_done = std::chrono::steady_clock::now();
    auto search_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_search_done - t_start)
            .count();
    auto rerank_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_rerank_done - t_search_done)
            .count();
    stats_detail::maxsim_queries.fetch_add(1, std::memory_order_relaxed);
    stats_detail::maxsim_search_ns.fetch_add(
        static_cast<uint64_t>(search_ns), std::memory_order_relaxed
    );
    stats_detail::maxsim_rerank_ns.fetch_add(
        static_cast<uint64_t>(rerank_ns), std::memory_order_relaxed
    );
#endif
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
