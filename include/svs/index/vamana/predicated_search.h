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

#include "svs/concepts/data.h"
#include "svs/core/distance.h"
#include "svs/core/query_result.h"
#include "svs/index/vamana/dynamic_search_buffer.h"
#include "svs/index/vamana/search_params.h"
#include "svs/lib/misc.h"
#include "svs/lib/neighbor.h"
#include "svs/lib/threads.h"
#include "svs/lib/type_traits.h"

#include <algorithm>
#include <vector>

namespace svs::index::vamana {

/// @brief Builder marking every candidate valid, for predicated search without a filter.
struct PredicatedNeighborBuilder {
    template <typename I>
    constexpr PredicatedSearchNeighbor<I> operator()(I i, float distance) const {
        return PredicatedSearchNeighbor<I>{i, distance};
    }
};

///
/// @brief Return the search buffer configuration for predicated search.
///
/// A zero total capacity makes the buffer unbounded. Otherwise, the capacity is raised to
/// at least `min_capacity`.
///
inline SearchBufferConfig
predicated_buffer_config(const SearchBufferConfig& config, size_t min_capacity = 0) {
    const size_t window = config.get_search_window_size();
    const size_t capacity = config.get_total_capacity();
    if (capacity == 0) {
        return SearchBufferConfig{window, MutableBuffer<uint32_t>::unbounded_capacity};
    }
    return SearchBufferConfig{window, std::max(capacity, min_capacity)};
}

namespace detail {

// Batch `predicated_search` shared by the Vamana indexes.
// `translate` maps internal IDs of the search buffer to external IDs.
template <
    typename Index,
    typename Pool,
    std::integral I,
    data::ImmutableMemoryDataset Queries,
    typename Stop,
    typename Translate>
void predicated_batch_search(
    const Index& index,
    Pool& threadpool,
    QueryResultView<I> results,
    const Queries& queries,
    VamanaSearchParameters sp,
    const Stop& stop,
    const lib::DefaultPredicate& cancel,
    const Translate& translate
) {
    using compare_type = distance::compare_t<typename Index::distance_type>;
    constexpr auto sentinel = type_traits::sentinel_v<Neighbor<I>, compare_type>;
    const size_t num_neighbors = results.n_neighbors();
    sp.buffer_config(predicated_buffer_config(sp.buffer_config_, num_neighbors));
    threads::parallel_for(
        threadpool,
        threads::StaticPartition{queries.size()},
        [&](const auto is, uint64_t SVS_UNUSED(tid)) {
            auto scratch =
                index.predicated_scratchspace(sp, CancelableStop<Stop>{stop, &cancel});
            bool cancelled = false;
            for (auto i : is) {
                cancelled = cancelled || cancel();
                if (cancelled) {
                    for (size_t j = 0; j < num_neighbors; ++j) {
                        results.set(sentinel, i, j);
                    }
                    continue;
                }
                index.predicated_search(queries.get_datum(i), scratch);

                const auto& buffer = scratch.buffer;
                const size_t found = std::min(buffer.size(), num_neighbors);
                for (size_t j = 0; j < found; ++j) {
                    const auto& neighbor = buffer[j];
                    results.set(
                        Neighbor<I>{
                            lib::narrow_cast<I>(translate(neighbor.id())),
                            neighbor.distance()},
                        i,
                        j
                    );
                }
                for (size_t j = found; j < num_neighbors; ++j) {
                    results.set(sentinel, i, j);
                }
            }
        }
    );
}

// Batch `range_search` shared by the Vamana indexes.
template <
    typename Index,
    typename Pool,
    data::ImmutableMemoryDataset Queries,
    typename Cmp,
    typename Translate>
std::vector<std::vector<Neighbor<size_t>>> range_batch_search(
    const Index& index,
    Pool& threadpool,
    const Queries& queries,
    float radius,
    const VamanaSearchParameters& sp,
    const lib::DefaultPredicate& cancel,
    const Cmp& compare,
    const Translate& translate
) {
    using stop_type = CancelableStop<RangeStop<Cmp>>;
    auto results = std::vector<std::vector<Neighbor<size_t>>>(queries.size());
    threads::parallel_for(
        threadpool,
        threads::StaticPartition{queries.size()},
        [&](const auto is, uint64_t SVS_UNUSED(tid)) {
            auto scratch =
                index.predicated_scratchspace(sp, stop_type{{radius, compare}, &cancel});
            for (auto i : is) {
                if (cancel()) {
                    return;
                }
                index.predicated_search(queries.get_datum(i), scratch);

                auto& result = results[i];
                for (const auto& neighbor : scratch.buffer) {
                    if (!compare(neighbor.distance(), radius)) {
                        break;
                    }
                    result.emplace_back(translate(neighbor.id()), neighbor.distance());
                }
            }
        }
    );
    return results;
}

} // namespace detail
} // namespace svs::index::vamana
