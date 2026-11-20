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

// Header under test: the build-time measurement counters.
#include "svs/index/vamana/build_stats.h"

// svs
#include "svs/core/data/simple.h"
#include "svs/core/distance.h"
#include "svs/core/graph.h"
#include "svs/index/vamana/extensions.h"
#include "svs/index/vamana/prune.h"
#include "svs/index/vamana/vamana_build.h"
#include "svs/lib/narrow.h"
#include "svs/lib/threads.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <atomic>
#include <cstdint>
#include <random>
#include <vector>

// Named (not anonymous) so `svs::index::vamana::PruneStrategy` below can name this type.
namespace build_stats_test {

// Static atomic: DefaultBuildAdaptor default-constructs a fresh Distance per thread.
struct CountingTestDistance {
    using compare = std::less<>;
    static constexpr bool implicit_broadcast = true;

    template <typename A, typename B> float compute(A a, B b) const {
        counter().fetch_add(1, std::memory_order_relaxed);
        return svs::distance::compute(svs::distance::DistanceL2{}, a, b);
    }

    static std::atomic<uint64_t>& counter() {
        static std::atomic<uint64_t> count{0};
        return count;
    }
};

svs::data::SimpleData<float> make_random_data(size_t n, size_t dims, uint64_t seed) {
    auto data = svs::data::SimpleData<float>(n, dims);
    auto rng = std::mt19937_64(seed);
    auto distribution = std::uniform_real_distribution<float>(-1.0F, 1.0F);
    auto buffer = std::vector<float>(dims);
    for (size_t i = 0; i < n; ++i) {
        for (float& v : buffer) {
            v = distribution(rng);
        }
        data.set_datum(i, buffer);
    }
    return data;
}

} // namespace build_stats_test

namespace svs::index::vamana {
// Shares QEmd's PruneStrategy (std::less<>), not IP/cosine's (extensions/vamana/gem.h).
template <> struct PruneStrategy<build_stats_test::CountingTestDistance> {
    using type = ProgressivePruneStrategy;
};
} // namespace svs::index::vamana

CATCH_TEST_CASE("Build Stats", "[vamana][build_stats]") {
    namespace vamana = svs::index::vamana;

    const size_t n_points = 300;
    const size_t dims = 16;
    auto data = build_stats_test::make_random_data(n_points, dims, 0xB0105);

    // Small graph_max_degree vs window_size/pool_size forces backedge pruning to occur.
    auto build_parameters = vamana::VamanaBuildParameters{1.2F, 8, 16, 32, 6, true};

    auto entry_point_pool = svs::threads::DefaultThreadPool(2);
    auto entry_point = svs::lib::narrow<uint32_t>(
        vamana::extensions::compute_entry_point(data, entry_point_pool)
    );

    auto graph = svs::default_graph<uint32_t>(n_points, build_parameters.graph_max_degree);
    auto build_pool = svs::threads::DefaultThreadPool(2);

    build_stats_test::CountingTestDistance::counter() = 0;
    vamana::build_stats_reset();

    // Instantiate with CountingBuildStats explicitly, so this test exercises the real
    // collector regardless of the ambient SVS_ENABLE_STATS build flag.
    auto builder = vamana::VamanaBuilder<
        decltype(graph),
        decltype(data),
        build_stats_test::CountingTestDistance,
        decltype(build_pool),
        vamana::CountingBuildStats>(
        graph,
        data,
        build_stats_test::CountingTestDistance{},
        build_parameters,
        build_pool,
        vamana::extensions::estimate_prefetch_parameters(data)
    );
    // Two full passes, mirroring VamanaIndex: alpha pinned to 1.0, then the real alpha.
    builder.construct(1.0F, entry_point);
    builder.construct(build_parameters.alpha, entry_point);

    auto stats = vamana::build_stats_snapshot();
    uint64_t total_compute_calls = build_stats_test::CountingTestDistance::counter().load();

    CATCH_REQUIRE(stats.inserts == 2 * n_points);
    CATCH_REQUIRE(
        stats.search_distances + stats.pool_distances + stats.prune_distances +
            stats.backedge_prune_distances ==
        total_compute_calls
    );

    CATCH_REQUIRE(stats.search_distances > 0);
    CATCH_REQUIRE(stats.pool_distances > 0);
    CATCH_REQUIRE(stats.prune_distances > 0);
    CATCH_REQUIRE(stats.backedge_prune_distances > 0);
    CATCH_REQUIRE(stats.search_ns > 0);
    CATCH_REQUIRE(stats.prune_ns > 0);
    CATCH_REQUIRE(stats.backedge_prune_ns > 0);
    CATCH_REQUIRE(stats.max_search_distances_per_insert > 0);
}
