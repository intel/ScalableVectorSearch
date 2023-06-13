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

#include "svs/concepts/data.h"
#include "svs/concepts/distance.h"
#include "svs/lib/neighbor.h"
#include "svs/lib/type_traits.h"

#include <functional>
#include <type_traits>

namespace svs::index::vamana {

///
/// @brief Function to prune neighbors using MRNG rule (extended with alpha as in Vamana).
///
/// @tparam Data The full type of the given dataset.
/// @tparam Dist The distance functor use when comparing vectors.
/// @tparam Neighbors The full neighbor-type of the candidate pool.
/// @tparam I The type of the reusting index for each neighbor.
/// @tparam Alloc Allocator for the result vector.
///
template <
    data::ImmutableMemoryDataset Data,
    distance::Distance<data::const_value_type_t<Data>, data::const_value_type_t<Data>> Dist,
    NeighborLike Neighbors,
    typename I,
    typename Alloc>
void heuristic_prune_neighbors(
    size_t max_result_size,
    float alpha,
    const Data& dataset,
    Dist& distance_function,
    size_t current_node_id,
    const std::span<const Neighbors>& pool,
    std::vector<I, Alloc>& result
) {
    auto cmp = distance::comparator(distance_function);
    assert(std::is_sorted(pool.begin(), pool.end(), cmp));
    if (pool.empty()) {
        return;
    }

    result.clear();
    result.reserve(max_result_size);
    size_t poolsize = pool.size();
    std::vector<bool> pruned(poolsize, false);
    size_t start = 0;

    while (result.size() < max_result_size && start < poolsize) {
        auto id = pool[start].id();
        if (pruned[start] || id == current_node_id) {
            ++start;
            continue;
        }
        pruned[start] = true;

        // Only once we know this item needs to be processed to we retrieve
        // the corresponding data and perform preprocessing.
        const auto& query = dataset.get_datum(id, data::full_access);
        distance::maybe_fix_argument(distance_function, query);
        result.push_back(id);
        for (size_t t = start + 1; t < poolsize; ++t) {
            if (pruned[t]) {
                continue;
            }

            const auto& candidate = pool[t];
            auto djk = distance::compute(
                distance_function,
                query,
                dataset.get_datum(candidate.id(), data::full_access)
            );
            if (cmp(alpha * djk, candidate.distance())) {
                pruned[t] = true;
            }
        }
        ++start;
    }
}
} // namespace svs::index::vamana
