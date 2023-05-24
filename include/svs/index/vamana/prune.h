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
    const Data& data,
    Dist& distance_function,
    size_t current_node_id,
    std::span<const Neighbors> pool,
    std::vector<I, Alloc>& result
) {
    auto cmp = distance::comparator(distance_function);
    assert(std::is_sorted(pool.begin(), pool.end(), cmp));
    if (pool.empty()) {
        return;
    }

    result.clear();
    result.reserve(max_result_size);
    std::vector<bool> pruned(pool.size(), false);
    size_t start = 0;
    while (result.size() < max_result_size && start < pool.size()) {
        const auto& sn = pool[start];
        const auto query = data.get_datum(sn.id());
        distance::maybe_fix_argument(distance_function, query);

        if (pruned[start] || sn.id() == current_node_id) {
            start++;
            continue;
        }

        pruned[start] = true;
        result.push_back(sn.id());
        for (size_t t = start + 1; t < pool.size() && t < max_result_size; t++) {
            if (pruned[t]) {
                continue;
            }

            auto djk =
                distance::compute(distance_function, query, data.get_datum(pool[t].id()));
            if (cmp(alpha * djk, pool[t].distance())) {
                pruned[t] = true;
            }
        }
        start++;
    }
}
} // namespace svs::index::vamana
