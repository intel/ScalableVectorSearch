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

// This file reproduces how an out-of-tree consumer reaches the concurrent index, and exists
// because two defects were only observable from that vantage point. Both are compile-time,
// so most of the value here is in the include order at the top of the file.
//
// (1) The upstream vamana headers come FIRST. `svs::index::vamana::concurrent` is nested
//     inside `svs::index::vamana`, so anything the enclosing namespace declares
//     participates in unqualified and relative lookup from inside the concurrent headers.
//     Whether it has been declared *yet* depends on the translation unit's include order,
//     and the concurrent tests happen to include the concurrent headers first, which hid
//     the problem: within this order `detail::pause()` resolved to
//     `svs::index::vamana::detail` (no `pause`) and `name<datatype_v<T>>()` resolved to
//     `svs::index::vamana::name(SlotMetadata)`.
//
// (2) A scalar-quantized dataset with the concurrent storage tag. `SQDataset` gates
//     `resize()`/`compact()` on a trait private to `svs::quantization::scalar`, which knows
//     only about `data::Blocked` -- deriving from it is not enough. Any growth or
//     compaction of a quantized concurrent index failed to compile until
//     `svs/concurrent/extensions/scalar.h` supplied the specialization.
#include "svs/index/vamana/dynamic_index.h"
#include "svs/index/vamana/multi.h"

// headers under test
#include "svs/concurrent/concurrent.h"
#include "svs/concurrent/extensions/scalar.h"

// svs
#include "svs/extensions/vamana/scalar.h"

// stl
#include <cstdint>
#include <numeric>
#include <vector>

// catch2
#include "catch2/catch_test_macros.hpp"

// tests
#include "tests/utils/test_dataset.h"

namespace cc = svs::index::vamana::concurrent;

namespace {

const size_t num_threads = 2;

// A single-element dataset holding a copy of row `i`, for exercising add_points().
template <typename Data> auto one_point_from(const Data& data, size_t i) {
    auto point = svs::data::SimpleData<typename Data::element_type, svs::Dynamic>(
        1, data.dimensions()
    );
    point.set_datum(0, data.get_datum(i));
    return point;
}

} // namespace

CATCH_TEST_CASE(
    "Concurrent index with upstream vamana headers included first",
    "[concurrent][downstream]"
) {
    // `has_edge` and the translator's `validate` are the two entities that failed to
    // compile in this include order, so both are called here rather than merely
    // instantiated.
    auto base = cc::SegmentedBlockedData<float>::load(test_dataset::data_svs_file());
    std::vector<size_t> ids(base.size());
    std::iota(ids.begin(), ids.end(), 0);

    svs::index::vamana::VamanaBuildParameters parameters{1.2, 64, 10, 20, 10, true};
    auto index = cc::MutableVamanaIndex(
        parameters, std::move(base), ids, svs::distance::DistanceL2(), num_threads
    );

    CATCH_REQUIRE(index.size() == ids.size());

    // Reaches SimpleGraph::has_edge, i.e. svs::detail::pause.
    const auto& graph = index.view_graph();
    bool any_edge = false;
    for (size_t i = 0, imax = std::min<size_t>(graph.n_nodes(), 16); i < imax; ++i) {
        for (auto neighbor : graph.get_node(i)) {
            any_edge = any_edge || graph.has_edge(i, neighbor);
        }
    }
    CATCH_REQUIRE(any_edge);

    // Reaches IDTranslator::validate through the save/load round trip.
    svs_test::prepare_temp_directory();
    auto dir = svs_test::temp_directory();
    index.save(dir / "config", dir / "graph", dir / "data");
    auto reloaded = cc::auto_dynamic_assemble(
        dir / "config",
        SVS_LAZY(cc::graphs::SimpleBlockedGraph<uint32_t>::load(dir / "graph")),
        SVS_LAZY(cc::SegmentedBlockedData<float>::load(dir / "data")),
        svs::distance::DistanceL2(),
        num_threads
    );
    CATCH_REQUIRE(reloaded.size() == ids.size());
}

CATCH_TEST_CASE(
    "Concurrent index over a scalar-quantized dataset grows and compacts",
    "[concurrent][downstream]"
) {
    using SQAllocator = cc::SegmentedBlocked<svs::lib::Allocator<std::int8_t>>;
    using SQData =
        svs::quantization::scalar::SQDataset<std::int8_t, svs::Dynamic, SQAllocator>;

    auto base = test_dataset::data_f32();
    auto threadpool = svs::threads::DefaultThreadPool(num_threads);
    auto compressed = SQData::compress(base, threadpool, SQAllocator{});

    const size_t n = compressed.size();
    std::vector<size_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);

    svs::index::vamana::VamanaBuildParameters parameters{1.2, 64, 10, 20, 10, true};
    auto index = cc::MutableVamanaIndex(
        parameters, std::move(compressed), ids, svs::distance::DistanceL2(), num_threads
    );
    CATCH_REQUIRE(index.size() == n);

    // Growth: no free slot exists yet, so this goes through data_.resize().
    auto point = one_point_from(base, 0);
    index.add_points(point, std::vector<size_t>{n + 100});
    CATCH_REQUIRE(index.has_id(n + 100));
    CATCH_REQUIRE(index.size() == n + 1);

    // Compaction: delete_entries + consolidate leaves holes, compact() closes them via
    // data_.compact() and then shrinks with data_.resize().
    std::vector<size_t> to_delete(n / 10);
    std::iota(to_delete.begin(), to_delete.end(), 0);
    CATCH_REQUIRE(index.delete_entries(to_delete) == to_delete.size());
    index.consolidate(to_delete);
    index.compact();
    CATCH_REQUIRE(index.size() == n + 1 - to_delete.size());
    index.debug_check_invariants(false);

    // The index still answers queries after all of that.
    const size_t num_neighbors = 10;
    auto queries = test_dataset::queries();
    auto search_params = svs::index::vamana::VamanaSearchParameters{};
    search_params.buffer_config_ = svs::index::vamana::SearchBufferConfig{num_neighbors};
    auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
    index.search(results.view(), queries.cview(), search_params);
    CATCH_REQUIRE(results.n_queries() == queries.size());
}
