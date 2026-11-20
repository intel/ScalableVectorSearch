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
#include "svs/concepts/vector_set.h"
#include "svs/core/data/simple.h"
#include "svs/core/distance/gem.h"
#include "svs/index/vamana/extensions.h"
#include "svs/index/vamana/prune.h"
#include "svs/lib/exception.h"

// stl
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>
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
            distance_->table(),
            distance_->query_tokens(),
            routed,
            route_scratch,
            distance_->nprobe_override()
        );

        thread_local std::vector<I> merged;
        merged.assign(entry_points_.begin(), entry_points_.end());
        for (uint32_t id : routed) {
            merged.push_back(static_cast<I>(id));
        }
        std::sort(merged.begin(), merged.end());
        merged.erase(std::unique(merged.begin(), merged.end()), merged.end());

        // push() truncates in id order once the buffer is full; when seeds outnumber the
        // buffer, keep the closest ones by score instead, with entry_points_ unconditional.
        if constexpr (requires(const Buffer& b) { b.capacity(); }) {
            size_t capacity = buffer.capacity();
            if (capacity > 0 && merged.size() > capacity) {
                thread_local std::vector<I> routed_only;
                routed_only.clear();
                for (I id : merged) {
                    if (std::find(entry_points_.begin(), entry_points_.end(), id) ==
                        entry_points_.end()) {
                        routed_only.push_back(id);
                    }
                }
                size_t entry_count = merged.size() - routed_only.size();
                size_t remaining = capacity > entry_count ? capacity - entry_count : 0;

                thread_local std::vector<std::pair<float, I>> scored;
                scored.clear();
                scored.reserve(routed_only.size());
                for (I id : routed_only) {
                    scored.emplace_back(computer(id), id);
                }
                size_t keep = std::min(remaining, scored.size());
                std::partial_sort(
                    scored.begin(),
                    scored.begin() + static_cast<std::ptrdiff_t>(keep),
                    scored.end(),
                    [](const auto& a, const auto& b) {
                        return svs::distance::QChamferExactQuery::compare{}(
                            a.first, b.first
                        );
                    }
                );

                merged.assign(entry_points_.begin(), entry_points_.end());
                for (size_t i = 0; i < keep; ++i) {
                    merged.push_back(scored[i].second);
                }
                std::sort(merged.begin(), merged.end());
                merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
            }
        }

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

// Routes a node's own insert like GemRoutedInitializer routes a query; no-op if unrouted.
template <svs::data::ImmutableVectorSetDataset Data, std::integral Idx>
    requires std::unsigned_integral<typename Data::const_value_type::element_type>
std::span<const Idx> svs_invoke(
    svs::tag_t<build_entry_points>,
    const Data& data,
    const svs::distance::QEmd& distance,
    Idx node_id,
    std::span<const Idx> entry_points,
    std::vector<Idx>& scratch
) {
    const auto& router = distance.router();
    if (!router) {
        return entry_points;
    }

    auto doc = data.get_datum(node_id);
    svs::distance::gem_detail::require_id_dimensions(doc.dimensions());
    thread_local std::vector<uint32_t> id_buf;
    const uint32_t* ids =
        svs::distance::gem_detail::id_span(doc.data(), doc.cardinality(), id_buf);

    thread_local svs::distance::GemRouter::IdScratch route_scratch;
    thread_local std::vector<uint32_t> routed;
    router->route_ids(
        std::span<const uint32_t>(ids, doc.cardinality()),
        router->nprobe,
        routed,
        route_scratch
    );

    scratch.assign(entry_points.begin(), entry_points.end());
    for (uint32_t id : routed) {
        scratch.push_back(static_cast<Idx>(id));
    }
    std::sort(scratch.begin(), scratch.end());
    scratch.erase(std::unique(scratch.begin(), scratch.end()), scratch.end());
    scratch.erase(std::remove(scratch.begin(), scratch.end(), node_id), scratch.end());

    if (scratch.empty()) {
        return entry_points;
    }
    return std::span<const Idx>(scratch);
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

// L2 medoid of each document's pooled fine-centroid mean; a real entry point for raw ids.
template <svs::data::ImmutableVectorSetDataset IdSets, threads::ThreadPool Pool>
    requires std::unsigned_integral<typename IdSets::const_value_type::element_type>
size_t gem_pooled_entry_point(
    const IdSets& docs, const svs::distance::FineCentroids& centroids, Pool& pool
) {
    size_t n = docs.size();
    size_t dim = centroids.dim;
    auto pooled = svs::data::SimpleData<float>(n, dim);

    threads::parallel_for(
        pool,
        threads::StaticPartition{n},
        [&](const auto& ids, uint64_t /*tid*/) {
            std::vector<double> accum(dim);
            for (size_t i : ids) {
                auto doc = docs.get_datum(i);
                if (doc.dimensions() != 1) {
                    throw ANNEXCEPTION(
                        "gem_pooled_entry_point: document {} has dimensions() = {}, "
                        "expected 1 (one centroid id per token)",
                        i,
                        doc.dimensions()
                    );
                }
                size_t card = doc.cardinality();
                std::fill(accum.begin(), accum.end(), 0.0);
                for (size_t t = 0; t < card; ++t) {
                    uint64_t cid = static_cast<uint64_t>(doc.token(t)[0]);
                    if (cid >= centroids.K) {
                        throw ANNEXCEPTION(
                            "gem_pooled_entry_point: document {} has centroid id {}, "
                            "not < K = {}",
                            i,
                            cid,
                            centroids.K
                        );
                    }
                    const svs::BFloat16* row = centroids.data.get() + cid * dim;
                    for (size_t d = 0; d < dim; ++d) {
                        accum[d] += static_cast<double>(row[d]);
                    }
                }
                auto out = pooled.get_datum(i);
                for (size_t d = 0; d < dim; ++d) {
                    out[d] = card > 0
                                 ? static_cast<float>(accum[d] / static_cast<double>(card))
                                 : 0.0F;
                }
            }
        }
    );

    return utils::find_medioid(pooled, pool);
}

// Turns an opaque medioid.h compile failure on centroid-id sets into a clear diagnostic.
template <typename Data, threads::ThreadPool Pool, typename Predicate>
    requires svs::data::ImmutableVectorSetDataset<Data> &&
             std::unsigned_integral<typename Data::const_value_type::element_type>
size_t svs_invoke(
    svs::tag_t<compute_entry_point>,
    const Data& /*dataset*/,
    Pool& /*threadpool*/,
    Predicate&& /*predicate*/
) {
    static_assert(
        !std::is_same_v<Data, Data>,
        "compute_entry_point has no L2 medoid for centroid-id vector sets; build one "
        "with svs::index::vamana::extensions::gem_pooled_entry_point and pass it to the "
        "VamanaIndex constructor directly instead of relying on automatic entry-point "
        "computation"
    );
    return 0;
}

} // namespace extensions
} // namespace svs::index::vamana
