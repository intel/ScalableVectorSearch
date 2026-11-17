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

// Proves a VamanaIndex builds, searches, and saves/reloads over an owning
// VectorSetData<uint32_t, 1> of centroid-id vector sets, using QChamfer throughout.
#include "svs/core/distance/gem.h"
#include "svs/extensions/vamana/gem.h"

// svs
#include "svs/core/data/vector_set.h"
#include "svs/core/data/vector_set_view.h"
#include "svs/core/graph.h"
#include "svs/index/vamana/index.h"
#include "svs/lib/threads.h"

// tests
#include "tests/svs/index/vamana/gem_test_utils.h"
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

// stl
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace {

using svs_test::gem::make_centroid_table;
using svs_test::gem::make_centroids;
using svs_test::gem::make_doc_corpus;

svs::index::vamana::VamanaSearchParameters search_params(size_t window, size_t capacity) {
    return svs::index::vamana::VamanaSearchParameters{{window, capacity}, false, 0, 0};
}

// Builds a graph+index over `docs` through VamanaBuilder directly (not the
// VamanaIndex build-parameters constructor, which hard-codes L2/IP/Cosine and throws for
// qCH, same as qEMD).
template <typename Docs>
auto build_index(
    Docs docs,
    svs::distance::QChamfer dist,
    const svs::index::vamana::VamanaBuildParameters& build_parameters,
    uint32_t entry_point
) {
    auto graph =
        svs::default_graph<uint32_t>(docs.size(), build_parameters.graph_max_degree);
    auto build_pool = svs::threads::DefaultThreadPool(1);
    auto builder = svs::index::vamana::VamanaBuilder(
        graph,
        docs,
        dist,
        build_parameters,
        build_pool,
        svs::index::vamana::extensions::estimate_prefetch_parameters(docs)
    );
    builder.construct(build_parameters.alpha, entry_point);

    auto pool = svs::threads::DefaultThreadPool(1);
    return svs::index::vamana::VamanaIndex(
        std::move(graph), std::move(docs), entry_point, dist, std::move(pool)
    );
}

} // namespace

CATCH_TEST_CASE(
    "Vamana build, search and save/reload over a vector-set dataset", "[vamana][gem]"
) {
    auto table = make_centroid_table(make_centroids(0x5EED));
    svs::distance::QChamfer dist{table};

    const size_t n_docs = 200;
    auto corpus = make_doc_corpus(n_docs, 0xDEADBEEF);
    auto queries = make_doc_corpus(10, 0xC0FFEE);
    auto query_view = queries.view();
    auto owning_docs = svs::data::VectorSetData<uint32_t, 1>::copy_from(corpus.view());

    auto build_parameters =
        svs::index::vamana::VamanaBuildParameters{1.2F, 16, 32, 64, 16, true};
    const uint32_t entry_point = 0;

    auto index = build_index(std::move(owning_docs), dist, build_parameters, entry_point);
    const auto& docs = index.view_data();

    CATCH_SECTION("Small-window search returns a best-first, internally consistent pool") {
        const size_t capacity = 20;
        auto sp = search_params(capacity, capacity);
        auto scratch = index.scratchspace(sp);

        for (size_t q = 0; q < query_view.size(); ++q) {
            auto query = query_view.get_datum(q);
            index.search(query, scratch);

            size_t valid = scratch.buffer.size();
            CATCH_REQUIRE(valid > 0);
            float previous = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < valid; ++j) {
                const auto& neighbor = scratch.buffer[j];
                float direct =
                    svs::distance::compute(dist, query, docs.get_datum(neighbor.id()));
                CATCH_REQUIRE(
                    static_cast<double>(neighbor.distance()) ==
                    Catch::Approx(direct).epsilon(1e-5).margin(1e-5)
                );
                CATCH_REQUIRE(neighbor.distance() >= previous);
                previous = neighbor.distance();
            }
        }
    }

    CATCH_SECTION("Exhaustive search's top-1 matches brute force") {
        const size_t capacity = n_docs;
        auto sp = search_params(capacity, capacity);
        auto scratch = index.scratchspace(sp);

        for (size_t q = 0; q < query_view.size(); ++q) {
            auto query = query_view.get_datum(q);
            index.search(query, scratch);
            auto oracle = svs_test::brute_force_top_k(dist, query, docs);

            // Centroid ids repeat, so several documents can tie for the best score; check
            // the id against every document tied for it, not just oracle[0].first.
            const auto& top1 = scratch.buffer[0];
            CATCH_REQUIRE(
                static_cast<double>(top1.distance()) ==
                Catch::Approx(oracle[0].second).epsilon(1e-5).margin(1e-5)
            );
            bool id_ties_for_best =
                std::any_of(oracle.begin(), oracle.end(), [&](const auto& entry) {
                    return entry.first == top1.id() &&
                           std::abs(entry.second - oracle[0].second) < 1e-5F;
                });
            CATCH_REQUIRE(id_ties_for_best);
        }
    }

    CATCH_SECTION("Save and reload preserve every query's result") {
        // `apply` gives the index real VamanaBuildParameters (build_index's lower
        // VamanaBuilder path left them at sentinels that overflow TOML save) without
        // re-running the L2/IP/Cosine distance-type check.
        index.apply(svs::index::vamana::VamanaIndexParameters{
            entry_point, build_parameters, index.get_search_parameters()});

        auto temp_directory = svs_test::prepare_temp_directory_v2();
        auto config_dir = temp_directory / "config";
        auto graph_dir = temp_directory / "graph";
        auto data_dir = temp_directory / "data";
        index.save(config_dir, graph_dir, data_dir);

        auto reload_pool = svs::threads::DefaultThreadPool(1);
        auto reloaded = svs::index::vamana::auto_assemble(
            config_dir,
            svs::GraphLoader<uint32_t>(graph_dir),
            svs::VectorSetDataLoader<uint32_t, 1>(data_dir),
            dist,
            std::move(reload_pool)
        );

        const size_t capacity = n_docs;
        auto sp = search_params(capacity, capacity);
        auto scratch_before = index.scratchspace(sp);
        auto scratch_after = reloaded.scratchspace(sp);

        for (size_t q = 0; q < query_view.size(); ++q) {
            auto query = query_view.get_datum(q);
            index.search(query, scratch_before);
            reloaded.search(query, scratch_after);

            CATCH_REQUIRE(scratch_before.buffer.size() == scratch_after.buffer.size());
            for (size_t j = 0; j < scratch_before.buffer.size(); ++j) {
                CATCH_REQUIRE(
                    scratch_before.buffer[j].id() == scratch_after.buffer[j].id()
                );
                CATCH_REQUIRE(
                    scratch_before.buffer[j].distance() ==
                    scratch_after.buffer[j].distance()
                );
            }
        }
    }
}
