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

// index/vamana depends on core/distance, not the reverse, so these overrides live here
// rather than in core/distance/gem.h.
#include "svs/core/distance/gem.h"
#include "svs/index/vamana/extensions.h"
#include "svs/index/vamana/prune.h"

// stl
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <span>
#include <vector>

namespace svs::index::vamana {

// qEMD is "smaller is better" like L2 (std::less<>), so share its MRNG pruning
// bookkeeping rather than IP/cosine's.
template <> struct PruneStrategy<svs::distance::QEmd> {
    using type = ProgressivePruneStrategy;
};

// qCH is also std::less<>; the MVG ablation (--metric qch) builds the graph with QChamfer
// directly, so it needs the same pruning strategy as QEmd.
template <> struct PruneStrategy<svs::distance::QChamfer> {
    using type = ProgressivePruneStrategy;
};

/// Starts from the index's fixed entry points plus, with a GemRouter, the routed
/// representatives for the query's table; falls back to EntryPointInitializer's
/// behavior exactly when there is no router.
template <std::integral I> struct GemRoutedInitializer {
    template <
        typename Buffer,
        typename Computer,
        graphs::ImmutableMemoryGraph Graph,
        typename Builder,
        typename Tracker>
    void operator()(
        Buffer& buffer,
        const Computer& computer,
        const Graph& graph,
        const Builder& builder,
        Tracker& tracker
    ) const {
        using Idx = typename Buffer::index_type;
        buffer.clear();

        auto push = [&](I id) {
            auto dist = computer(id);
            buffer.push_back(builder(id, dist));
            graph.prefetch_node(id);
            tracker.visited(Neighbor<Idx>{id, dist}, 1);
        };

        const auto& router = distance_->router();
        if (!router) {
            for (I id : entry_points_) {
                push(id);
            }
            buffer.sort();
            return;
        }

        // thread_local: no per-query allocation once scratch reaches its steady-state size.
        thread_local svs::distance::GemRouter::QueryScratch route_scratch;
        thread_local std::vector<uint32_t> routed;
        router->route_query(
            distance_->table(), distance_->query_tokens(), routed, route_scratch
        );

        thread_local std::vector<I> merged;
        merged.assign(entry_points_.begin(), entry_points_.end());
        for (uint32_t id : routed) {
            merged.push_back(static_cast<I>(id));
        }
        std::sort(merged.begin(), merged.end());
        merged.erase(std::unique(merged.begin(), merged.end()), merged.end());

        for (I id : merged) {
            push(id);
        }
        buffer.sort();
    }

    ///// Members
    const svs::distance::QChamferExactQuery* distance_;
    std::span<const I> entry_points_;
};

namespace extensions {

// DefaultBuildAdaptor default-constructs its distance_ member, which drops the
// CentroidTable svs::distance::QEmd holds; forward the real functor through aggregate-init
// instead.
inline DefaultBuildAdaptor<svs::distance::QEmd> svs_invoke(
    svs::tag_t<build_adaptor>, const auto& /*data*/, const svs::distance::QEmd& distance
) {
    return DefaultBuildAdaptor<svs::distance::QEmd>{distance};
}

// Same reason as the QEmd overload above: QChamfer also holds a CentroidTable that a
// default-constructed distance_ would drop.
inline DefaultBuildAdaptor<svs::distance::QChamfer> svs_invoke(
    svs::tag_t<build_adaptor>, const auto& /*data*/, const svs::distance::QChamfer& distance
) {
    return DefaultBuildAdaptor<svs::distance::QChamfer>{distance};
}

// Routes a QChamferExactQuery search; no-op to EntryPointInitializer when unrouted.
template <typename Data, std::integral Idx>
GemRoutedInitializer<Idx> svs_invoke(
    svs::tag_t<search_initializer>,
    const Data& /*data*/,
    const svs::distance::QChamferExactQuery& distance,
    std::span<const Idx> entry_points
) {
    return GemRoutedInitializer<Idx>{&distance, entry_points};
}

} // namespace extensions
} // namespace svs::index::vamana
