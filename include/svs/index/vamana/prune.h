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
#include "svs/core/distance.h"
#include "svs/lib/neighbor.h"
#include "svs/lib/type_traits.h"

#include <functional>
#include <type_traits>

namespace svs::index::vamana {

// Dispatch tags for pruning strategies.
struct IterativePruneStrategy {};
struct ProgressivePruneStrategy {};
struct LegacyPruneStrategy {};

// Default strategy is the iterative strategy.
template <typename Distance> struct PruneStrategy;

// Strategy for L2
template <> struct PruneStrategy<distance::DistanceL2> {
    using type = ProgressivePruneStrategy;
};

// Specialize IP to use the iterative strategy.
template <> struct PruneStrategy<distance::DistanceIP> {
    using type = IterativePruneStrategy;
};
template <> struct PruneStrategy<distance::DistanceCosineSimilarity> {
    using type = IterativePruneStrategy;
};

template <typename T> using prune_strategy_t = typename PruneStrategy<T>::type;
template <typename T> constexpr prune_strategy_t<T> prune_strategy() {
    return prune_strategy_t<T>();
}
template <typename T>
constexpr prune_strategy_t<T> prune_strategy(const T& SVS_UNUSED(dist)) {
    return prune_strategy<T>();
}

/////
///// Iterative Prune Strategy
/////

enum class PruneState : uint8_t { Available, Added, Pruned };

inline PruneState reenable(PruneState state) {
    return (state == PruneState::Pruned) ? PruneState::Available : state;
}

inline bool excluded(PruneState state) { return state != PruneState::Available; }

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
    data::AccessorFor<Data> Accessor,
    distance::Distance<data::const_value_type_t<Data>, data::const_value_type_t<Data>> Dist,
    NeighborLike Neighbors,
    typename I,
    typename Alloc>
void heuristic_prune_neighbors(
    IterativePruneStrategy SVS_UNUSED(dispatch),
    size_t max_result_size,
    float alpha,
    const Data& dataset,
    const Accessor& accessor,
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
    if (poolsize == 0) {
        return;
    }

    auto pruned = std::vector<PruneState>(poolsize, PruneState::Available);
    float current_alpha = 1.0f;
    while (result.size() < max_result_size && !cmp(alpha, current_alpha)) {
        size_t start = 0;
        while (result.size() < max_result_size && start < poolsize) {
            auto id = pool[start].id();
            if (excluded(pruned[start]) || id == current_node_id) {
                ++start;
                continue;
            }
            pruned[start] = PruneState::Added;

            // Only once we know this item needs to be processed to we retrieve
            // the corresponding data and perform preprocessing.
            const auto& query = accessor(dataset, id);
            distance::maybe_fix_argument(distance_function, query);
            result.push_back(id);
            for (size_t t = start + 1; t < poolsize; ++t) {
                if (excluded(pruned[t])) {
                    continue;
                }

                const auto& candidate = pool[t];
                auto djk = distance::compute(
                    distance_function, query, accessor(dataset, candidate.id())
                );
                if (cmp(current_alpha * djk, candidate.distance())) {
                    pruned[t] = PruneState::Pruned;
                }
            }
            ++start;
        }

        if (alpha == 1) {
            break;
        }

        // Reset pruned elements for the next round.
        for (auto& state : pruned) {
            state = reenable(state);
        }
        current_alpha *= alpha;
    }
}

template <
    data::ImmutableMemoryDataset Data,
    data::AccessorFor<Data> Accessor,
    distance::Distance<data::const_value_type_t<Data>, data::const_value_type_t<Data>> Dist,
    NeighborLike Neighbors,
    typename I,
    typename Alloc>
void heuristic_prune_neighbors(
    ProgressivePruneStrategy SVS_UNUSED(dispatch),
    size_t max_result_size,
    float alpha,
    const Data& dataset,
    const Accessor& accessor,
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
    std::vector<float> pruned(poolsize, type_traits::tombstone_v<float, decltype(cmp)>);

    float current_alpha = 1.0f;
    while (result.size() < max_result_size && !cmp(alpha, current_alpha)) {
        size_t start = 0;
        while (result.size() < max_result_size && start < poolsize) {
            auto id = pool[start].id();
            if (cmp(current_alpha, pruned[start]) || id == current_node_id) {
                ++start;
                continue;
            }
            pruned[start] = type_traits::sentinel_v<float, decltype(cmp)>;

            // Only once we know this item needs to be processed to we retrieve
            // the corresponding data and perform preprocessing.
            const auto& query = accessor(dataset, id);
            distance::maybe_fix_argument(distance_function, query);
            result.push_back(id);
            for (size_t t = start + 1; t < poolsize; ++t) {
                if (cmp(current_alpha, pruned[t])) {
                    continue;
                }

                const auto& candidate = pool[t];
                auto djk = distance::compute(
                    distance_function, query, accessor(dataset, candidate.id())
                );
                pruned[t] = std::max(pruned[t], candidate.distance() / djk, cmp);
            }
            ++start;
        }
        if (alpha == 1) {
            break;
        }
        current_alpha *= alpha;
    }
}

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
    data::AccessorFor<Data> Accessor,
    distance::Distance<data::const_value_type_t<Data>, data::const_value_type_t<Data>> Dist,
    NeighborLike Neighbors,
    typename I,
    typename Alloc>
void heuristic_prune_neighbors(
    LegacyPruneStrategy SVS_UNUSED(dispatch),
    size_t max_result_size,
    float alpha,
    const Data& dataset,
    const Accessor& accessor,
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
        const auto& query = accessor(dataset, id);
        distance::maybe_fix_argument(distance_function, query);
        result.push_back(id);
        for (size_t t = start + 1; t < poolsize; ++t) {
            if (pruned[t]) {
                continue;
            }

            const auto& candidate = pool[t];
            auto djk = distance::compute(
                distance_function, query, accessor(dataset, candidate.id())
            );
            if (cmp(alpha * djk, candidate.distance())) {
                pruned[t] = true;
            }
        }
        ++start;
    }
}
} // namespace svs::index::vamana
