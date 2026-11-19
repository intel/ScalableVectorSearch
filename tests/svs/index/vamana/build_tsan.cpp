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

// A small graph_max_degree and several threads over few points make it likely that two
// threads touch the same node's list in one add_reverse_edges round.

// header under test
#include "svs/index/vamana/vamana_build.h"

#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/core/graph.h"
#include "svs/lib/threads.h"

// test utils
#include "tests/utils/test_dataset.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <cstddef>
#include <cstdint>

namespace {
constexpr size_t kPoints = 2000;
constexpr size_t kMaxDegree = 4;
constexpr size_t kWindowSize = 16;
constexpr size_t kMaxCandidatePoolSize = 32;
constexpr size_t kThreads = 8;
} // namespace

CATCH_TEST_CASE("Static Vamana build under TSan", "[vamana][build][tsan]") {
    auto full_data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());
    const size_t dim = full_data.dimensions();

    auto data = svs::data::SimpleData<float>(kPoints, dim);
    for (size_t i = 0; i < kPoints; ++i) {
        data.set_datum(i, full_data.get_datum(i));
    }

    auto graph = svs::graphs::SimpleGraph<uint32_t>(kPoints, kMaxDegree);
    svs::distance::DistanceL2 distance_function;
    auto threadpool = svs::threads::DefaultThreadPool(kThreads);

    svs::index::vamana::VamanaBuildParameters build_params(
        1.2f, kMaxDegree, kWindowSize, kMaxCandidatePoolSize, kMaxDegree, true
    );

    auto builder = svs::index::vamana::VamanaBuilder(
        graph, data, distance_function, build_params, threadpool
    );

    uint32_t entry_point = 0;
    builder.construct(1.0f, entry_point);
    builder.construct(build_params.alpha, entry_point);

    size_t total_edges = 0;
    for (size_t i = 0; i < kPoints; ++i) {
        total_edges += graph.get_node_degree(i);
    }
    CATCH_REQUIRE(total_edges > 0);
}
