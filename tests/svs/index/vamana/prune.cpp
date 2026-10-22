/*
 * Copyright 2023 Intel Corporation
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

// header under test
#include "svs/index/vamana/prune.h"

// svs
#include "svs/core/data/simple.h"
#include "svs/core/distance.h"
#include "svs/lib/float16.h"
#include "svs/lib/neighbor.h"

// catch2
#include "catch2/catch_template_test_macros.hpp"
#include "catch2/catch_test_macros.hpp"

// tests
#include "tests/utils/generators.h"
#include "tests/utils/preconversion.h"
#include "tests/utils/utils.h"

// stl
#include <algorithm>
#include <array>
#include <span>
#include <string_view>
#include <vector>

namespace {

namespace v = svs::index::vamana;

template <size_t Extent>
svs::data::SimpleData<svs::Float16, Extent> random_dataset(size_t size, size_t dims) {
    auto data = svs::data::SimpleData<svs::Float16, Extent>(size, dims);
    auto generator = svs_test::make_generator<svs::Float16>(-1, 1);
    auto datum = std::vector<svs::Float16>{};
    for (size_t i = 0; i < size; ++i) {
        svs_test::populate(datum, generator, dims);
        svs_test::insert_subnormals(datum);
        data.set_datum(i, datum);
    }
    return data;
}

// A `float` copy holding exactly what the kernels at the dispatched instruction-set level
// compute with when they convert `data`.
template <size_t Extent>
svs::data::SimpleData<float, Extent>
to_float(const svs::data::SimpleData<svs::Float16, Extent>& data) {
    auto copy = svs::data::SimpleData<float, Extent>(data.size(), data.dimensions());
    for (size_t i = 0; i < data.size(); ++i) {
        copy.set_datum(i, svs_test::convert_like_kernels(data.get_datum(i)));
    }
    return copy;
}

// Every vector of `data`, ordered by its distance to `node` as a build orders its
// candidate pool. The pool includes `node` itself, which the prune must skip.
template <typename Distance, typename Data>
std::vector<svs::Neighbor<uint32_t>>
sorted_pool(Distance distance, const Data& data, size_t node) {
    auto query = data.get_datum(node);
    svs::distance::maybe_fix_argument(distance, query);
    auto pool = std::vector<svs::Neighbor<uint32_t>>{};
    for (size_t i = 0; i < data.size(); ++i) {
        pool.emplace_back(i, svs::distance::compute(distance, query, data.get_datum(i)));
    }
    std::sort(
        pool.begin(), pool.end(), svs::TotalOrder(svs::distance::comparator(distance))
    );
    return pool;
}

template <typename Data, typename Distance>
std::vector<uint32_t> prune(
    const Data& data,
    Distance distance,
    float alpha,
    size_t max_result_size,
    size_t node,
    const std::vector<svs::Neighbor<uint32_t>>& pool
) {
    auto result = std::vector<uint32_t>{};
    v::heuristic_prune_neighbors(
        v::prune_strategy(distance),
        max_result_size,
        alpha,
        data,
        svs::data::GetDatumAccessor{},
        distance,
        node,
        svs::lib::as_const_span(pool),
        result
    );
    return result;
}

// The query of every prune is converted to `float` once and compared against `Float16`
// candidates. Pruning a `float` copy converted the way the kernels convert runs the
// `float x float` kernels, which compute bit-identical distances at the same
// instruction-set level, so the pruned neighbors must be identical too.
template <typename Distance, size_t Extent>
void require_float16_prune_matches_float(size_t dims) {
    constexpr size_t num_vectors = 300;
    // `alpha < 1` reruns the pruning rounds with re-enabled candidates.
    constexpr auto alphas = std::to_array<float>({1.0F, 0.95F});
    // Fewer slots than survivors stops a round early; more lets every round run out.
    constexpr auto result_sizes = std::to_array<size_t>({8, 64});
    constexpr auto nodes = std::to_array<size_t>({0, 17, 299});

    auto half_data = random_dataset<Extent>(num_vectors, dims);
    using accessor_type = svs::data::GetDatumAccessor;

    svs_test::for_each_supported_isa([&](std::string_view isa) {
        CATCH_INFO("ISA: " << isa);
        auto float_data = to_float(half_data);
        auto storage =
            v::prune_query_storage_t<decltype(half_data), accessor_type, Distance>(
                half_data.dimensions()
            );

        for (float alpha : alphas) {
            for (size_t max_result_size : result_sizes) {
                for (size_t node : nodes) {
                    CATCH_INFO(
                        "alpha: " << alpha << ", max_result_size: " << max_result_size
                                  << ", node: " << node
                    );
                    auto pool = sorted_pool(Distance(), float_data, node);
                    auto expected =
                        prune(float_data, Distance(), alpha, max_result_size, node, pool);
                    CATCH_REQUIRE(!expected.empty());

                    // Through the signature that allocates its own storage.
                    CATCH_REQUIRE(
                        prune(half_data, Distance(), alpha, max_result_size, node, pool) ==
                        expected
                    );

                    // Through caller-held storage, reused across every prune of this
                    // loop: nothing a previous query left in it may change the result.
                    auto distance = Distance();
                    auto result = std::vector<uint32_t>{};
                    v::heuristic_prune_neighbors(
                        v::IterativePruneStrategy{},
                        max_result_size,
                        alpha,
                        half_data,
                        accessor_type{},
                        distance,
                        node,
                        svs::lib::as_const_span(pool),
                        result,
                        storage
                    );
                    CATCH_REQUIRE(result == expected);
                }
            }
        }
    });
}

} // namespace

CATCH_TEST_CASE("Pruning", "[index][vamana]") {
    namespace v = svs::index::vamana;
    // Protect against changes to the default strategies getting merged.
    static_assert(std::is_same_v<
                  v::prune_strategy_t<svs::distance::DistanceL2>,
                  v::ProgressivePruneStrategy>);
    static_assert(std::is_same_v<
                  v::prune_strategy_t<svs::distance::DistanceIP>,
                  v::IterativePruneStrategy>);
    static_assert(std::is_same_v<
                  v::prune_strategy_t<svs::distance::DistanceCosineSimilarity>,
                  v::IterativePruneStrategy>);

    CATCH_SECTION("Iterative Strategy") {
        CATCH_SECTION("Prune State") {
            CATCH_REQUIRE(
                v::reenable(v::PruneState::Available) == v::PruneState::Available
            );
            CATCH_REQUIRE(v::reenable(v::PruneState::Added) == v::PruneState::Added);
            CATCH_REQUIRE(v::reenable(v::PruneState::Pruned) == v::PruneState::Available);

            CATCH_REQUIRE(v::excluded(v::PruneState::Available) == false);
            CATCH_REQUIRE(v::excluded(v::PruneState::Added) == true);
            CATCH_REQUIRE(v::excluded(v::PruneState::Pruned) == true);
        }
    }
}

CATCH_TEMPLATE_TEST_CASE(
    "Iterative pruning with a pre-converted query",
    "[index][vamana][conversion]",
    svs::distance::DistanceIP,
    svs::distance::DistanceCosineSimilarity
) {
    using Distance = TestType;
    static_assert(std::is_same_v<v::prune_strategy_t<Distance>, v::IterativePruneStrategy>);

    // Ragged tail, the specialized static extent, and a high dimensionality.
    CATCH_SECTION("Dynamic extent, 100") {
        require_float16_prune_matches_float<Distance, svs::Dynamic>(100);
    }
    CATCH_SECTION("Static extent, 128") {
        require_float16_prune_matches_float<Distance, 128>(128);
    }
    CATCH_SECTION("Dynamic extent, 1000") {
        require_float16_prune_matches_float<Distance, svs::Dynamic>(1000);
    }
}

CATCH_TEST_CASE("Pruning with query storage it does not use", "[index][vamana]") {
    // Callers pass storage regardless of the strategy their distance selects. The
    // strategies that do not convert their query must ignore it and prune as before.
    using Distance = svs::distance::DistanceL2;
    static_assert(std::is_same_v<
                  v::prune_strategy_t<Distance>,
                  v::ProgressivePruneStrategy>);

    auto data = random_dataset<svs::Dynamic>(300, 100);
    using accessor_type = svs::data::GetDatumAccessor;
    using storage_type = v::prune_query_storage_t<decltype(data), accessor_type, Distance>;
    // L2 does not convert `Float16`, so the storage is the forwarding kind.
    static_assert(std::is_same_v<
                  svs::distance::
                      distance_argument_t<Distance, std::span<const svs::Float16>>,
                  std::span<const svs::Float16>>);
    auto storage = storage_type(data.dimensions());

    constexpr size_t node = 0;
    constexpr size_t max_result_size = 16;
    constexpr float alpha = 1.2F;
    auto pool = sorted_pool(Distance(), data, node);
    auto require_storage_ignored = [&](auto dispatch) {
        auto distance = Distance();
        auto expected = std::vector<uint32_t>{};
        v::heuristic_prune_neighbors(
            dispatch,
            max_result_size,
            alpha,
            data,
            accessor_type{},
            distance,
            node,
            svs::lib::as_const_span(pool),
            expected
        );
        auto result = std::vector<uint32_t>{};
        v::heuristic_prune_neighbors(
            dispatch,
            max_result_size,
            alpha,
            data,
            accessor_type{},
            distance,
            node,
            svs::lib::as_const_span(pool),
            result,
            storage
        );
        CATCH_REQUIRE(!expected.empty());
        CATCH_REQUIRE(result == expected);
    };

    CATCH_SECTION("Progressive") { require_storage_ignored(v::ProgressivePruneStrategy{}); }
    CATCH_SECTION("Legacy") { require_storage_ignored(v::LegacyPruneStrategy{}); }
}
