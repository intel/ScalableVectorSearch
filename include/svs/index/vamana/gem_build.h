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

// Mirrors index.h's build constructor, which hard-codes an L2/IP/Cosine check QEmd fails.
#include "svs/core/distance/gem.h"
#include "svs/core/distance/gem_router.h"
#include "svs/core/graph.h"
#include "svs/index/vamana/build_params.h"
#include "svs/index/vamana/connectivity.h"
#include "svs/index/vamana/extensions.h"
#include "svs/index/vamana/vamana_build.h"
#include "svs/lib/threads.h"

// stl
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <utility>
#include <vector>

namespace svs::index::vamana {

namespace gem_build_detail {
// Shared with gem_insert_order so the two never disagree on a valid representative.
template <std::integral Idx>
std::vector<Idx> valid_representatives(const svs::distance::GemRouter& router) {
    std::vector<uint32_t> reps = router.representative;
    std::sort(reps.begin(), reps.end());
    reps.erase(std::unique(reps.begin(), reps.end()), reps.end());

    std::vector<Idx> out;
    out.reserve(reps.size());
    for (uint32_t rep : reps) {
        if (rep != svs::distance::GemRouter::no_representative) {
            out.push_back(static_cast<Idx>(rep));
        }
    }
    return out;
}
} // namespace gem_build_detail

/// Insert order for a GEM build: every valid representative of `router` (deduplicated,
/// ascending), then every other document ascending. Every document in
/// `[0, num_documents)` appears exactly once.
template <std::integral Idx = uint32_t>
std::vector<Idx>
gem_insert_order(const svs::distance::GemRouter& router, size_t num_documents) {
    std::vector<Idx> order;
    order.reserve(num_documents);
    std::vector<bool> inserted(num_documents, false);

    for (Idx rep : gem_build_detail::valid_representatives<Idx>(router)) {
        inserted.at(rep) = true;
        order.push_back(rep);
    }
    for (size_t doc = 0; doc < num_documents; ++doc) {
        if (!inserted[doc]) {
            order.push_back(static_cast<Idx>(doc));
        }
    }
    return order;
}

/// Outcome of `build_gem_graph`. `edges_added` is the safe total; `edges_forced_unsafe`
/// and `stalled` pass through from the connectivity report so a caller can see whether
/// the repair orphaned a node, or gave up early after making no more progress.
struct GemBuildReport {
    size_t documents = 0;
    size_t representatives_inserted_first = 0;
    size_t unreachable_before_repair = 0;
    size_t unreachable_after_repair = 0;
    size_t repair_rounds = 0;
    size_t edges_added = 0;
    size_t edges_forced_unsafe = 0;
    bool stalled = false;
};

/// Builds a Vamana graph over `data` (centroid-id sets) with `distance` (QEmd). With a
/// router, `gem_insert_order` picks the insert order; otherwise the ascending full range.
/// Repair roots are `entry_point` plus every valid representative.
template <
    data::ImmutableMemoryDataset IdSets,
    threads::ThreadPool Pool,
    typename Allocator = HugepageAllocator<uint32_t>>
std::pair<graphs::SimpleGraph<uint32_t, Allocator>, GemBuildReport> build_gem_graph(
    const IdSets& data,
    const svs::distance::QEmd& distance,
    const VamanaBuildParameters& params,
    Pool& threadpool,
    uint32_t entry_point,
    const Allocator& graph_allocator = {},
    svs::logging::logger_ptr logger = svs::logging::get()
) {
    using Idx = uint32_t;
    size_t num_documents = data.size();

    GemBuildReport report{};
    report.documents = num_documents;

    auto graph =
        default_graph<Idx>(num_documents, params.graph_max_degree, graph_allocator);

    auto router = distance.router();
    std::vector<Idx> roots{entry_point};
    if (router) {
        auto reps = gem_build_detail::valid_representatives<Idx>(*router);
        report.representatives_inserted_first = reps.size();
        roots.insert(roots.end(), reps.begin(), reps.end());
        std::sort(roots.begin(), roots.end());
        roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
    }

    auto builder = VamanaBuilder(
        graph,
        data,
        distance,
        params,
        threadpool,
        extensions::estimate_prefetch_parameters(data),
        logger
    );

    if (router) {
        auto order = gem_insert_order<Idx>(*router, num_documents);
        builder.construct(1.0F, entry_point, order, logging::Level::Trace, logger);
        builder.construct(params.alpha, entry_point, order, logging::Level::Trace, logger);
    } else {
        builder.construct(1.0F, entry_point, logging::Level::Trace, logger);
        builder.construct(params.alpha, entry_point, logging::Level::Trace, logger);
    }

    auto connectivity = repair_connectivity(
        graph, data, distance, params, lib::as_const_span(roots), threadpool
    );
    report.unreachable_before_repair = connectivity.unreachable_before;
    report.unreachable_after_repair = connectivity.unreachable_after;
    report.repair_rounds = connectivity.rounds;
    report.edges_added = connectivity.edges_appended +
                         connectivity.edges_accepted_through_prune +
                         connectivity.edges_forced;
    report.edges_forced_unsafe = connectivity.edges_forced_unsafe;
    report.stalled = connectivity.stalled;

    return {std::move(graph), report};
}

} // namespace svs::index::vamana
