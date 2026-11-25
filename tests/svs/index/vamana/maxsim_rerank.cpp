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

// Header under test: the Vamana re-rank extension for vector-set (MaxSim) datasets.
#include "svs/extensions/vamana/maxsim.h"

// svs
#include "svs/concepts/vector_set.h"
#include "svs/core/data.h"
#include "svs/core/data/vector_set.h"
#include "svs/core/data/vector_set_view.h"
#include "svs/core/data/with_vector_sets.h"
#include "svs/core/distance/maxsim.h"
#include "svs/core/graph.h"
#include "svs/core/query_result.h"
#include "svs/index/vamana/index.h"
#include "svs/lib/misc.h"
#include "svs/lib/narrow.h"
#include "svs/lib/saveload.h"
#include "svs/lib/threads.h"

// tests
#include "tests/svs/index/vamana/gem_test_utils.h"
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

// stl
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <limits>
#include <random>
#include <set>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr size_t DIM = 8;
constexpr double EPS = 1e-4;

// `compare = std::less<>` is the opposite direction from MaxSim's `std::greater<>`, to
// exercise the re-rank extension's comparator-flip handling.
struct MeanNegativeInnerProduct {
    using compare = std::less<>;
    static constexpr bool implicit_broadcast = true;
    static constexpr std::string_view name = "test_mean_negative_ip";

    template <svs::data::VectorSet Q>
    float compute(const Q& query, std::span<const float> primary_row) const {
        size_t dim = primary_row.size();
        size_t cardinality = query.cardinality();
        std::vector<double> mean(dim, 0.0);
        for (size_t t = 0; t < cardinality; ++t) {
            auto token = query.token(t);
            for (size_t d = 0; d < dim; ++d) {
                mean[d] += token[d];
            }
        }
        double dot = 0.0;
        for (size_t d = 0; d < dim; ++d) {
            dot += (mean[d] / static_cast<double>(cardinality)) * primary_row[d];
        }
        return static_cast<float>(-dot);
    }

    // Build compares two primary rows directly, not a query set against a row, so this
    // overload serves vamana_build.h's candidate-pool and reverse-edge loops.
    float compute(std::span<const float> a, std::span<const float> b) const {
        double dot = 0.0;
        for (size_t d = 0; d < a.size(); ++d) {
            dot += static_cast<double>(a[d]) * static_cast<double>(b[d]);
        }
        return static_cast<float>(-dot);
    }
};

} // namespace

// IP-flavored (no triangle inequality), so pruning needs IterativePruneStrategy like
// DistanceIP, not vamana_build.h's default L2 one.
namespace svs::index::vamana {
template <> struct PruneStrategy<::MeanNegativeInnerProduct> {
    using type = IterativePruneStrategy;
};
} // namespace svs::index::vamana

namespace {

using Dist = MeanNegativeInnerProduct;

void normalize(std::span<float> token) {
    double norm = 0.0;
    for (float v : token) {
        norm += static_cast<double>(v) * static_cast<double>(v);
    }
    norm = std::sqrt(norm);
    for (float& v : token) {
        v = static_cast<float>(static_cast<double>(v) / norm);
    }
}

// A deterministic corpus of vector sets: cardinality cycles 1..6, tokens are unit-norm
// (MaxSim assumes unit-norm tokens), and a document's primary row is the mean of its
// tokens.
struct Corpus {
    std::vector<float> tokens;
    std::vector<uint64_t> offsets;
    svs::data::SimpleData<float> primary;

    size_t size() const { return offsets.size() - 1; }

    svs::data::ConstVectorSetView<float> vector_sets() const {
        return svs::data::ConstVectorSetView<float>(
            std::span<const float>(tokens), std::span<const uint64_t>(offsets), DIM
        );
    }
};

Corpus make_corpus(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

    Corpus corpus;
    corpus.offsets.resize(n + 1, 0);
    corpus.primary = svs::data::SimpleData<float>(n, DIM);

    std::vector<float> mean(DIM);
    std::vector<float> token(DIM);
    for (size_t i = 0; i < n; ++i) {
        size_t cardinality = 1 + (i % 6);
        corpus.offsets[i + 1] = corpus.offsets[i] + cardinality;
        std::fill(mean.begin(), mean.end(), 0.0F);
        for (size_t t = 0; t < cardinality; ++t) {
            for (size_t d = 0; d < DIM; ++d) {
                token[d] = dist(rng);
            }
            normalize(token);
            for (size_t d = 0; d < DIM; ++d) {
                mean[d] += token[d];
            }
            corpus.tokens.insert(corpus.tokens.end(), token.begin(), token.end());
        }
        for (size_t d = 0; d < DIM; ++d) {
            mean[d] /= static_cast<float>(cardinality);
        }
        corpus.primary.set_datum(i, mean);
    }
    return corpus;
}

svs::index::vamana::VamanaSearchParameters search_params(size_t window, size_t capacity) {
    return svs::index::vamana::VamanaSearchParameters{{window, capacity}, false, 0, 0};
}

uint32_t local_entry_point(const svs::data::SimpleData<float>& primary) {
    auto pool = svs::threads::DefaultThreadPool(1);
    return svs::lib::narrow<uint32_t>(
        svs::index::vamana::extensions::compute_entry_point(primary, pool)
    );
}

// Builds a graph and VamanaIndex over `data` through VamanaBuilder directly (not the
// VamanaIndex build-parameters constructor, which hard-codes L2/IP/Cosine and throws for
// this test-local distance), with the two-phase construct VamanaIndex's own constructor
// runs internally (alpha pinned to 1.0, then the real alpha).
template <typename Data>
auto build_maxsim_index(
    Data data,
    const svs::index::vamana::VamanaBuildParameters& build_parameters,
    uint32_t entry_point
) {
    auto graph =
        svs::default_graph<uint32_t>(data.size(), build_parameters.graph_max_degree);
    auto build_pool = svs::threads::DefaultThreadPool(1);
    auto builder = svs::index::vamana::VamanaBuilder(
        graph,
        data,
        Dist{},
        build_parameters,
        build_pool,
        svs::index::vamana::extensions::estimate_prefetch_parameters(data)
    );
    builder.construct(1.0F, entry_point);
    builder.construct(build_parameters.alpha, entry_point);

    auto pool = svs::threads::DefaultThreadPool(1);
    return svs::index::vamana::VamanaIndex(
        std::move(graph), std::move(data), entry_point, Dist{}, std::move(pool)
    );
}

// Overwrites docs 198/199 (cardinality 1 and 2) and query 1 (cardinality 2) so doc 199's
// second token wins against the query's second token (better MaxSim score) while its
// primary mean-dot score is worse than doc 198's: a MaxSim/primary-order inversion
// witness, independent of the random seed.
void make_inversion_witness(Corpus& corpus, Corpus& query_corpus) {
    constexpr size_t DOC_A = 198;
    constexpr size_t DOC_B = 199;
    constexpr size_t QUERY = 1;

    std::vector<float> e1(DIM, 0.0F);
    e1[0] = 1.0F;
    std::vector<float> e2(DIM, 0.0F);
    e2[1] = 1.0F;
    std::vector<float> q2 = {0.3F, 0.95F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    normalize(q2);

    auto set_token = [](Corpus& c, size_t doc, size_t t, const std::vector<float>& v) {
        size_t base = (c.offsets[doc] + t) * DIM;
        std::copy(v.begin(), v.end(), c.tokens.data() + base);
    };

    set_token(corpus, DOC_A, 0, e1);
    corpus.primary.set_datum(DOC_A, e1);

    set_token(corpus, DOC_B, 0, e1);
    set_token(corpus, DOC_B, 1, e2);
    std::vector<float> mean_b(DIM, 0.0F);
    for (size_t d = 0; d < DIM; ++d) {
        mean_b[d] = (e1[d] + e2[d]) / 2.0F;
    }
    corpus.primary.set_datum(DOC_B, mean_b);

    set_token(query_corpus, QUERY, 0, e1);
    set_token(query_corpus, QUERY, 1, q2);
}

// Checks the MaxSim<ElemT> rerank contract against the whole candidate pool a
// single-query search leaves in `scratch`: every score matches the oracle, the pool is
// sorted best-first, and its first `num_neighbors` entries are the brute-force top-k
// restricted to the pool's own ids.
template <
    typename ElemT,
    svs::data::VectorSet Query,
    typename Index,
    typename Scratch,
    typename DocsView>
void check_rerank_pool(
    Index& index,
    Scratch& scratch,
    const Query& query,
    const DocsView& doc_sets,
    size_t num_neighbors
) {
    index.search(query, scratch);
    const auto& buffer = scratch.buffer;

    svs::distance::MaxSim<ElemT> maxsim{};
    svs::distance::maybe_fix_argument(maxsim, query);

    std::vector<uint32_t> pool_ids;
    pool_ids.reserve(buffer.size());
    float previous_score = std::numeric_limits<float>::infinity();
    for (const auto& neighbor : buffer) {
        float expected =
            svs::distance::compute(maxsim, query, doc_sets.get_datum(neighbor.id()));
        CATCH_REQUIRE(
            static_cast<double>(neighbor.distance()) ==
            Catch::Approx(expected).epsilon(EPS).margin(EPS)
        );
        CATCH_REQUIRE(neighbor.distance() <= previous_score);
        previous_score = neighbor.distance();
        pool_ids.push_back(neighbor.id());
    }
    // Pool strictly larger than num_neighbors: the re-rank had more than one candidate.
    CATCH_REQUIRE(pool_ids.size() > num_neighbors);

    std::set<uint32_t> pool_set(pool_ids.begin(), pool_ids.end());
    auto oracle = svs_test::brute_force_top_k(maxsim, query, doc_sets);
    std::vector<uint32_t> restricted_oracle;
    for (const auto& scored : oracle) {
        if (pool_set.contains(scored.first)) {
            restricted_oracle.push_back(scored.first);
            if (restricted_oracle.size() == num_neighbors) {
                break;
            }
        }
    }
    for (size_t k = 0; k < num_neighbors; ++k) {
        CATCH_REQUIRE(buffer[k].id() == restricted_oracle[k]);
    }
}

// `Data` has no `allocator_type` of its own to build a `VectorDataLoader`-style loader
// around, so a `svs::lib::Lazy` callable reloads it whole instead.
template <typename Index>
void check_save_reload_round_trip(
    Index& index,
    const svs::index::vamana::VamanaBuildParameters& build_parameters,
    uint32_t entry_point,
    const svs::data::ConstVectorSetView<float>& queries
) {
    using Data = typename Index::data_type;
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
        SVS_LAZY(svs::lib::load_from_disk<Data>(data_dir)),
        Dist{},
        std::move(reload_pool)
    );

    const size_t capacity = index.size();
    auto sp = search_params(capacity, capacity);
    auto scratch_before = index.scratchspace(sp);
    auto scratch_after = reloaded.scratchspace(sp);

    for (size_t q = 0; q < queries.size(); ++q) {
        auto query = queries.get_datum(q);
        index.search(query, scratch_before);
        reloaded.search(query, scratch_after);

        CATCH_REQUIRE(scratch_before.buffer.size() == scratch_after.buffer.size());
        for (size_t j = 0; j < scratch_before.buffer.size(); ++j) {
            CATCH_REQUIRE(scratch_before.buffer[j].id() == scratch_after.buffer[j].id());
            CATCH_REQUIRE(
                scratch_before.buffer[j].distance() == scratch_after.buffer[j].distance()
            );
        }
    }
}

} // namespace

// Builds one owning WithVectorSets index and runs every behavior below as a section on
// it (Catch re-runs this setup per leaf section).
CATCH_TEST_CASE("MaxSim re-rank Vamana extension", "[vamana][maxsim_rerank]") {
    using Data = svs::data::
        WithVectorSets<svs::data::SimpleData<float>, svs::data::VectorSetData<float>>;

    CATCH_SECTION("Mismatched primary/sets sizes throw") {
        auto primary_corpus = make_corpus(5, 1);
        auto sets_corpus = make_corpus(4, 2);
        CATCH_REQUIRE_THROWS(Data(
            primary_corpus.primary,
            svs::data::VectorSetData<float>::copy_from(sets_corpus.vector_sets())
        ));
    }

    const size_t n_docs = 200;
    auto corpus = make_corpus(n_docs, 0xDEADBEEF);
    auto query_corpus = make_corpus(5, 0xC0FFEE);
    make_inversion_witness(corpus, query_corpus);

    auto build_parameters =
        svs::index::vamana::VamanaBuildParameters{1.2F, 16, 32, 64, 16, true};
    auto entry_point = local_entry_point(corpus.primary);

    auto full_index = build_maxsim_index(
        Data(
            corpus.primary, svs::data::VectorSetData<float>::copy_from(corpus.vector_sets())
        ),
        build_parameters,
        entry_point
    );
    const auto& doc_sets = full_index.view_data().vector_sets();
    auto queries = query_corpus.vector_sets();

    CATCH_SECTION("Query token dimension mismatch throws before the graph search") {
        std::vector<float> bad_tokens(DIM - 1, 0.1F);
        std::vector<uint64_t> bad_offsets{0, 1};
        auto bad_query_set = svs::data::ConstVectorSetView<float>(
            std::span<const float>(bad_tokens),
            std::span<const uint64_t>(bad_offsets),
            DIM - 1
        );
        auto bad_query = bad_query_set.get_datum(0);

        auto sp = search_params(40, 40);
        auto full_scratch = full_index.scratchspace(sp);
        CATCH_REQUIRE_THROWS(full_index.search(bad_query, full_scratch));
    }

    // Capacity == n_docs leaves every document in the pool: the strongest check against
    // the full-corpus oracle. A tighter window only re-exercises the generic search
    // buffer's eviction path, which search_buffer.cpp already owns.
    CATCH_SECTION("Candidate pool is scored and ordered by MaxSim") {
        const size_t num_neighbors = 10;
        auto sp = search_params(n_docs, n_docs);
        auto full_scratch = full_index.scratchspace(sp);

        for (size_t q = 0; q < queries.size(); ++q) {
            auto query = queries.get_datum(q);
            check_rerank_pool<float>(
                full_index, full_scratch, query, doc_sets, num_neighbors
            );
        }
    }

    CATCH_SECTION("MaxSim order of the pool can differ from the primary-distance order") {
        constexpr uint32_t DOC_A = 198;
        constexpr uint32_t DOC_B = 199;
        auto sp = search_params(n_docs, n_docs);
        auto full_scratch = full_index.scratchspace(sp);

        auto query = queries.get_datum(1);
        full_index.search(query, full_scratch);
        const auto& buffer = full_scratch.buffer;

        auto it_a = std::find_if(buffer.begin(), buffer.end(), [](const auto& n) {
            return n.id() == DOC_A;
        });
        auto it_b = std::find_if(buffer.begin(), buffer.end(), [](const auto& n) {
            return n.id() == DOC_B;
        });
        CATCH_REQUIRE(it_a != buffer.end());
        CATCH_REQUIRE(it_b != buffer.end());
        CATCH_REQUIRE(
            std::distance(buffer.begin(), it_b) < std::distance(buffer.begin(), it_a)
        );

        // The generic `distance::compute` dispatch takes the functor by non-const
        // reference, so a named lvalue is required (a temporary will not bind).
        Dist dist{};
        float primary_a =
            svs::distance::compute(dist, query, corpus.primary.get_datum(DOC_A));
        float primary_b =
            svs::distance::compute(dist, query, corpus.primary.get_datum(DOC_B));
        CATCH_REQUIRE(Dist::compare{}(primary_a, primary_b));
    }

    CATCH_SECTION("Batch search matches the first k entries of the single-query path") {
        const size_t capacity = n_docs;
        const size_t num_neighbors = 8;
        auto sp = search_params(capacity, capacity);
        const size_t num_queries = queries.size();

        auto full_scratch = full_index.scratchspace(sp);
        using PoolEntry = typename std::decay_t<decltype(full_scratch.buffer)>::value_type;
        std::vector<std::vector<PoolEntry>> single_query_results;
        single_query_results.reserve(num_queries);
        for (size_t q = 0; q < num_queries; ++q) {
            auto query = queries.get_datum(q);
            full_index.search(query, full_scratch);
            std::vector<PoolEntry> top_k(
                full_scratch.buffer.begin(), full_scratch.buffer.begin() + num_neighbors
            );
            single_query_results.push_back(std::move(top_k));
        }

        svs::QueryResult<uint32_t> result(num_queries, num_neighbors);
        full_index.search(result.view(), queries, sp);

        for (size_t q = 0; q < num_queries; ++q) {
            for (size_t k = 0; k < num_neighbors; ++k) {
                uint32_t id = result.index(q, k);
                float score = result.distance(q, k);
                CATCH_REQUIRE(id == single_query_results[q][k].id());
                CATCH_REQUIRE(
                    static_cast<double>(score) ==
                    Catch::Approx(single_query_results[q][k].distance())
                        .epsilon(EPS)
                        .margin(EPS)
                );
            }
        }
    }

    CATCH_SECTION("Save and reload preserve every query's result") {
        check_save_reload_round_trip(full_index, build_parameters, entry_point, queries);
    }
}

namespace {

// Rounds `corpus`'s float tokens to bfloat16 into a fresh owning `VectorSetData<BFloat16>`
// with the same CSR offsets, so the bf16 store's documents are exactly what the float
// corpus's documents round to.
svs::data::VectorSetData<svs::BFloat16> round_tokens_to_bf16(const Corpus& corpus) {
    size_t total_tokens = corpus.tokens.size() / DIM;
    svs::data::SimpleData<svs::BFloat16> tokens(total_tokens, DIM);
    std::vector<svs::BFloat16> row(DIM);
    for (size_t i = 0; i < total_tokens; ++i) {
        for (size_t d = 0; d < DIM; ++d) {
            row[d] = svs::round_to_bfloat16(corpus.tokens[i * DIM + d]);
        }
        tokens.set_datum(i, row);
    }
    return svs::data::VectorSetData<svs::BFloat16>(std::move(tokens), corpus.offsets);
}

} // namespace

// A BFloat16 element type makes the re-rank extension pick MaxSim<svs::BFloat16>
// (svs/extensions/vamana/maxsim.h selects on Sets::const_value_type::element_type),
// reaching the bf16 InnerProductMatrix::column_max overload rather than the float one.
CATCH_TEST_CASE(
    "MaxSim re-rank over an owning bfloat16 vector-set store", "[vamana][maxsim_rerank]"
) {
    using Data = svs::data::WithVectorSets<
        svs::data::SimpleData<float>,
        svs::data::VectorSetData<svs::BFloat16>>;

    const size_t n_docs = 80;
    auto corpus = make_corpus(n_docs, 0x5EED5EED);
    auto query_corpus = make_corpus(5, 0xFACE5EED);
    auto data = Data(corpus.primary, round_tokens_to_bf16(corpus));

    auto build_parameters =
        svs::index::vamana::VamanaBuildParameters{1.2F, 16, 32, 64, 16, true};
    auto entry_point = local_entry_point(corpus.primary);
    auto index = build_maxsim_index(std::move(data), build_parameters, entry_point);

    auto queries = query_corpus.vector_sets();
    const size_t capacity = n_docs;
    const size_t num_neighbors = 10;
    auto sp = search_params(capacity, capacity);
    auto scratch = index.scratchspace(sp);

    for (size_t q = 0; q < queries.size(); ++q) {
        auto query = queries.get_datum(q);
        check_rerank_pool<svs::BFloat16>(
            index, scratch, query, index.view_data().vector_sets(), num_neighbors
        );
    }
}
