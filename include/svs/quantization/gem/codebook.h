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

#if SVS_ENABLE_MULTI_VECTOR

// svs
#include "svs/concepts/data.h"
#include "svs/concepts/vector_set.h"
#include "svs/core/data/simple.h"
#include "svs/core/data/vector_set.h"
#include "svs/core/distance/gem.h"
#include "svs/core/distance/inner_product_matrix.h"
#include "svs/core/kmeans_training.h"
#include "svs/lib/bfloat16.h"
#include "svs/lib/exception.h"
#include "svs/lib/float16.h"
#include "svs/lib/misc.h"
#include "svs/lib/narrow.h"
#include "svs/lib/threads.h"
#include "svs/lib/timing.h"

// stl
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

// Trains the codebook GEM assigns tokens to, matching faiss's ColBERT k-means indexer.
namespace svs::quantization::gem {

namespace detail {

// Exact floor(sqrt(x)): std::sqrt seeds it, then local search corrects any libm rounding.
inline size_t isqrt_floor(size_t x) {
    if (x == 0) {
        return 0;
    }
    size_t r = static_cast<size_t>(std::sqrt(static_cast<double>(x)));
    while (r > 0 && r > x / r) {
        --r;
    }
    while ((r + 1) <= x / (r + 1)) {
        ++r;
    }
    return r;
}

} // namespace detail

/// K = 2^floor(log2(16 * sqrt(num_tokens))), ColBERT's indexer rule. Uses std::frexp's
/// exact mantissa/exponent split rather than std::log2, since log2 of an exact power of
/// two is not guaranteed correctly rounded on every libm.
inline size_t colbert_num_partitions(size_t num_tokens) {
    if (num_tokens == 0) {
        throw ANNEXCEPTION("colbert_num_partitions: num_tokens must be positive, got 0");
    }
    double raw = 16.0 * std::sqrt(static_cast<double>(num_tokens));
    int exponent = 0;
    std::frexp(raw, &exponent);
    // mantissa in [0.5,1) means floor(log2(raw)) == exponent - 1 exactly.
    if (exponent < 1) {
        throw ANNEXCEPTION(
            "colbert_num_partitions: num_tokens={} is too small to produce a positive "
            "centroid count",
            num_tokens
        );
    }
    return size_t{1} << (exponent - 1);
}

/// num_sampled_passages = min(1 + floor(16 * sqrt(120 * num_passages)), num_passages).
/// 16 * sqrt(120 * n) == sqrt(30720 * n) exactly, so the floor uses integer sqrt
/// (detail::isqrt_floor) instead of floating log2/sqrt, avoiding rounding ambiguity.
inline size_t colbert_sample_size(size_t num_passages) {
    if (num_passages == 0) {
        throw ANNEXCEPTION("colbert_sample_size: num_passages must be positive, got 0");
    }
    size_t raw = detail::isqrt_floor(30720ULL * num_passages);
    return std::min(size_t{1} + raw, num_passages);
}

/// Which kernel assign_tokens/quantize use to find each token's nearest centroid.
enum class Assigner { ExactGemm, InnerProductMatrix };

/// Parameters for codebook training, assignment and the coarse-cluster side table.
struct CodebookParameters {
    /// Number of fine centroids K; 0 selects colbert_num_partitions from the token count.
    size_t num_centroids = 0;
    size_t kmeans_iterations = 4;
    /// GEMM chunk size for training's shared buffer; assign_tokens sizes its own instead.
    size_t minibatch_size = 10'000;
    uint32_t seed = 42;
    double heldout_fraction = 0.05;
    size_t heldout_cap = 50'000;
    /// Number of coarse clusters. 0 selects max(1, num_centroids * 5 / 32).
    size_t num_coarse = 0;
    size_t coarse_iterations = 4;
    /// Default InnerProductMatrix measured faster than ExactGemm and matched it exactly.
    Assigner assigner = Assigner::InnerProductMatrix;
    size_t assign_candidates = 8;
    svs::distance::Ground ground = svs::distance::Ground::Cos;
    /// Per-thread score-buffer budget; fixed-row sizing once hit 587 GB at K=131'072.
    size_t assign_buffer_bytes = 32 * 1024 * 1024;
};

/// Any contiguous vector-set dataset of float or Float16 tokens: VectorSetData<T> and
/// ConstVectorSetView<T> both satisfy this without copying their token storage.
template <typename Data>
concept TokenVectorStore =
    svs::data::ImmutableVectorSetDataset<Data> &&
    svs::data::ContiguousVectorSet<typename Data::const_value_type> &&
    (std::is_same_v<typename Data::element_type, float> ||
     std::is_same_v<typename Data::element_type, svs::Float16>) &&
    requires(const Data& d) {
        { d.total_tokens() } -> std::convertible_to<size_t>;
        { d.cardinality(size_t{0}) } -> std::convertible_to<size_t>;
    };

namespace detail {

// Sum of squared L2 distances to assigned centroids, accumulated in double; O(n*dim).
template <svs::data::ImmutableMemoryDataset Data>
double compute_objective(
    const Data& data,
    const svs::data::SimpleData<float>& centroids,
    const std::vector<size_t>& assignments
) {
    double total = 0.0;
    for (auto i : data.eachindex()) {
        auto datum = data.get_datum(i);
        auto c = centroids.get_datum(assignments[i]);
        double sq = 0.0;
        for (size_t d = 0, dmax = datum.size(); d < dmax; ++d) {
            double diff = static_cast<double>(datum[d]) - static_cast<double>(c[d]);
            sq += diff * diff;
        }
        total += sq;
    }
    return total;
}

// Plain mean, matching faiss, not svs::kmeans::centroid_adjustment's IVF-owned formula.
template <svs::data::ImmutableMemoryDataset Data, threads::ThreadPool Pool>
void plain_mean_update(
    const Data& data,
    svs::data::SimpleData<float>& centroids,
    const std::vector<size_t>& assignments,
    std::vector<size_t>& counts,
    Pool& threadpool
) {
    size_t n_threads = threadpool.size();
    size_t n_centroids = centroids.size();
    std::fill(counts.begin(), counts.end(), size_t{0});

    threads::parallel_for(
        threadpool,
        threads::StaticPartition{n_centroids},
        [&](auto indices, auto /*tid*/) {
            for (auto i : indices) {
                auto c = centroids.get_datum(i);
                std::fill(c.begin(), c.end(), 0.0F);
            }
        }
    );

    threads::parallel_for(
        threadpool,
        threads::StaticPartition{n_threads},
        [&](auto /*indices*/, auto tid) {
            size_t centroid_start = (n_centroids * tid) / n_threads;
            size_t centroid_end = (n_centroids * (tid + 1)) / n_threads;
            for (auto i : data.eachindex()) {
                auto assignment = assignments[i];
                if (assignment >= centroid_start && assignment < centroid_end) {
                    counts.at(assignment)++;
                    auto datum = data.get_datum(i);
                    auto c = centroids.get_datum(assignment);
                    for (size_t p = 0, pmax = c.size(); p < pmax; ++p) {
                        c[p] += static_cast<float>(datum[p]);
                    }
                }
            }
        }
    );

    threads::parallel_for(
        threadpool,
        threads::StaticPartition{n_centroids},
        [&](auto indices, auto /*tid*/) {
            for (auto i : indices) {
                if (counts.at(i) != 0) {
                    auto c = centroids.get_datum(i);
                    float norm = 1.0F / static_cast<float>(counts.at(i));
                    for (size_t p = 0, pmax = c.size(); p < pmax; ++p) {
                        c[p] *= norm;
                    }
                }
            }
        }
    );
}

} // namespace detail

/// Centroids and the k-means objective after train_centroids; the observer (if any) owns
/// per-iteration diagnostics instead, so this result does not carry them.
struct L2KMeansResult {
    svs::data::SimpleData<float> centroids;
    double objective = 0.0;
};

/// Default k-means observer: both hooks are empty, so an optimizer can elide the calls
/// entirely and a caller pays nothing for not timing iterations.
struct NoOpKMeansObserver {
    void before_iteration(size_t /*iter*/) {}
    void after_iteration(size_t /*iter*/) {}
};

/// Satisfied by NoOpKMeansObserver and by any type exposing the same two hooks, e.g. one
/// that times the interval between them.
template <typename T>
concept KMeansObserver = requires(T observer, size_t iter) {
                             observer.before_iteration(iter);
                             observer.after_iteration(iter);
                         };

/// Shared L2/plain-mean trainer; takes Data by non-const ref, so callers pass a local
/// copy. `observer`'s before_/after_iteration are called once per iteration, with no
/// other effect on the result, so a caller can measure iterations without this function
/// depending on any timing facility itself.
template <
    svs::data::ImmutableMemoryDataset Data,
    threads::ThreadPool Pool,
    KMeansObserver Observer = NoOpKMeansObserver>
L2KMeansResult train_centroids(
    Data& data,
    size_t num_clusters,
    size_t iterations,
    size_t minibatch_size,
    std::mt19937& rng,
    Pool& threadpool,
    Observer observer = Observer{}
) {
    size_t n = data.size();
    if (n < num_clusters) {
        throw ANNEXCEPTION(
            "train_centroids: only {} points available, need at least num_clusters={}",
            n,
            num_clusters
        );
    }
    // faiss subsamples past 256/centroid; throws instead of silently diverging from it.
    if (n > num_clusters * 256) {
        throw ANNEXCEPTION(
            "train_centroids: {} points exceed faiss's 256-per-centroid training bound "
            "for num_clusters={} (max {}); subsampling instead of throwing would silently "
            "differ from the reference training procedure",
            n,
            num_clusters,
            num_clusters * 256
        );
    }

    std::vector<size_t> id_scratch(num_clusters);
    auto centroids =
        svs::kmeans::init_centroids<float>(data, id_scratch, num_clusters, rng, threadpool);

    size_t batch_cap = std::min(minibatch_size, n);
    auto matmul_results = svs::data::SimpleData<float>(batch_cap, num_clusters);
    std::vector<float> data_norm;
    svs::kmeans::generate_norms(data, data_norm, threadpool);
    std::vector<float> centroids_norm;
    std::vector<size_t> assignments(n);
    std::vector<size_t> counts(num_clusters);
    auto distance = svs::DistanceL2();
    svs::lib::Timer scratch_timer;

    double objective = 0.0;
    for (size_t iter = 0; iter < iterations; ++iter) {
        observer.before_iteration(iter);
        svs::kmeans::generate_norms(centroids, centroids_norm, threadpool);

        size_t num_batches = svs::lib::div_round_up(n, batch_cap);
        for (size_t batch = 0; batch < num_batches; ++batch) {
            auto this_batch = threads::UnitRange<uint64_t>{
                batch * batch_cap, std::min((batch + 1) * batch_cap, n)};
            auto data_batch = data::make_view(data, this_batch);
            svs::kmeans::centroid_assignment(
                data_batch,
                data_norm,
                this_batch,
                distance,
                centroids,
                centroids_norm,
                assignments,
                matmul_results,
                threadpool,
                scratch_timer
            );
        }

        // Objective is defined on the last pass, against the centroids before its update.
        if (iter + 1 == iterations) {
            objective = detail::compute_objective(data, centroids, assignments);
        }

        detail::plain_mean_update(data, centroids, assignments, counts, threadpool);
        svs::kmeans::centroid_split(
            data, centroids, counts, rng, threadpool, scratch_timer
        );
        observer.after_iteration(iter);
    }

    return L2KMeansResult{std::move(centroids), objective};
}

/// sample_training_tokens's output: the float32 tokens train_centroids trains on, plus
/// the resolved centroid count and sampling sizes train_codebook reports.
struct SampledTrainingTokens {
    svs::data::SimpleData<float> trainset;
    size_t K = 0;
    size_t num_sampled_passages = 0;
    size_t num_training_tokens = 0;
};

/// Steps 1-3: resolve K, sample passages, then gather/shuffle/hold out their tokens into a
/// float32 trainset. Only sampled tokens convert to float32, so a memory-mapped Float16
/// store is never copied in full. Advances `rng`, so a caller chaining into
/// train_centroids with the same `rng` continues its stream rather than restarting it.
template <TokenVectorStore Data, threads::ThreadPool Pool>
SampledTrainingTokens sample_training_tokens(
    const Data& tokens,
    const CodebookParameters& parameters,
    std::mt19937& rng,
    Pool& threadpool
) {
    size_t dim = tokens.dimensions();
    size_t num_passages = tokens.size();
    size_t num_total_tokens = tokens.total_tokens();
    if (num_passages == 0 || num_total_tokens == 0) {
        throw ANNEXCEPTION(
            "sample_training_tokens: tokens dataset is empty ({} passages, {} tokens)",
            num_passages,
            num_total_tokens
        );
    }

    size_t K = parameters.num_centroids == 0 ? colbert_num_partitions(num_total_tokens)
                                             : parameters.num_centroids;
    constexpr size_t max_id = static_cast<size_t>(std::numeric_limits<uint32_t>::max());
    if (K > max_id) {
        throw ANNEXCEPTION(
            "sample_training_tokens: K={} exceeds the uint32_t centroid-id range (max {})",
            K,
            max_id
        );
    }

    size_t sample_size = colbert_sample_size(num_passages);

    std::vector<size_t> passage_ids(sample_size);
    svs::kmeans::generate_unique_ids(passage_ids, num_passages, rng);
    std::sort(passage_ids.begin(), passage_ids.end());

    size_t total_sample_tokens = 0;
    for (size_t pid : passage_ids) {
        total_sample_tokens += tokens.cardinality(pid);
    }

    struct TokenRef {
        uint32_t sample_idx;
        uint32_t token_idx;
    };
    std::vector<TokenRef> refs;
    refs.reserve(total_sample_tokens);
    for (size_t s = 0; s < sample_size; ++s) {
        size_t card = tokens.cardinality(passage_ids[s]);
        for (size_t t = 0; t < card; ++t) {
            refs.push_back(
                {svs::lib::narrow_cast<uint32_t>(s), svs::lib::narrow_cast<uint32_t>(t)}
            );
        }
    }
    std::shuffle(refs.begin(), refs.end(), rng);

    size_t heldout = std::min(
        parameters.heldout_cap,
        static_cast<size_t>(
            parameters.heldout_fraction * static_cast<double>(total_sample_tokens)
        )
    );
    size_t num_training = total_sample_tokens - heldout;

    if (num_training < K) {
        throw ANNEXCEPTION(
            "sample_training_tokens: only {} training tokens sampled (after a {} token "
            "held-out split), need at least K={}",
            num_training,
            heldout,
            K
        );
    }

    auto trainset = svs::data::SimpleData<float>(num_training, dim);
    threads::parallel_for(
        threadpool,
        threads::StaticPartition{num_training},
        [&](auto indices, auto /*tid*/) {
            std::vector<float> row(dim);
            for (auto i : indices) {
                auto ref = refs[i];
                size_t pid = passage_ids[ref.sample_idx];
                auto token = tokens.get_datum(pid).token(ref.token_idx);
                for (size_t d = 0; d < dim; ++d) {
                    row[d] = static_cast<float>(token[d]);
                }
                trainset.set_datum(i, row);
            }
        }
    );

    return SampledTrainingTokens{std::move(trainset), K, sample_size, num_training};
}

/// Result of train_codebook: the normalized float32 centroids plus training diagnostics.
struct TrainedCodebook {
    svs::data::SimpleData<float> centroids;
    double objective = 0.0;
    size_t num_sampled_passages = 0;
    size_t num_training_tokens = 0;
};

/// Composes sample_training_tokens and train_centroids, then normalizes once: the
/// pipeline a caller wanting plain results (no phase timing) runs end to end.
template <TokenVectorStore Data, threads::ThreadPool Pool>
TrainedCodebook
train_codebook(const Data& tokens, const CodebookParameters& parameters, Pool& threadpool) {
    std::mt19937 rng(parameters.seed);
    auto sampled = sample_training_tokens(tokens, parameters, rng, threadpool);

    auto kmeans_result = train_centroids(
        sampled.trainset,
        sampled.K,
        parameters.kmeans_iterations,
        parameters.minibatch_size,
        rng,
        threadpool
    );

    svs::lib::Timer scratch_timer;
    svs::kmeans::normalize_centroids(kmeans_result.centroids, threadpool, scratch_timer);

    return TrainedCodebook{
        std::move(kmeans_result.centroids),
        kmeans_result.objective,
        sampled.num_sampled_passages,
        sampled.num_training_tokens};
}

/// Step 5: assign every document token to its nearest (argmax inner product) normalized
/// centroid, lowest id breaking ties. Threads own disjoint passage ranges and write only
/// their own tokens' ids, so the result never depends on thread count.
template <TokenVectorStore Data, threads::ThreadPool Pool>
svs::data::VectorSetData<uint32_t> assign_tokens(
    const Data& tokens,
    const svs::data::SimpleData<float>& centroids,
    const CodebookParameters& parameters,
    Pool& threadpool
) {
    size_t dim = tokens.dimensions();
    if (centroids.dimensions() != dim) {
        throw ANNEXCEPTION(
            "assign_tokens: token dimension {} does not match centroid dimension {}",
            dim,
            centroids.dimensions()
        );
    }
    size_t K = centroids.size();
    constexpr size_t max_id = static_cast<size_t>(std::numeric_limits<uint32_t>::max());
    if (K > max_id) {
        throw ANNEXCEPTION(
            "assign_tokens: K={} exceeds the uint32_t centroid-id range (max {})", K, max_id
        );
    }

    size_t num_passages = tokens.size();
    std::vector<uint64_t> offsets(num_passages + 1, 0);
    for (size_t p = 0; p < num_passages; ++p) {
        offsets[p + 1] = offsets[p] + tokens.cardinality(p);
    }
    size_t total_tokens = lib::narrow<size_t>(offsets.back());

    auto id_storage = svs::data::SimpleData<uint32_t>(total_tokens, 1);

    svs::distance::PackedOperand packed_centroids;
    if (parameters.assigner == Assigner::InnerProductMatrix) {
        packed_centroids.pack_rounded_to_bf16(centroids.data(), K, dim);
    }

    size_t candidates = std::max(size_t{1}, std::min(parameters.assign_candidates, K));

    // scores is batch_cap * K floats; derive batch_cap from assign_buffer_bytes, not K.
    constexpr size_t max_assign_batch_rows = 4096;
    size_t bytes_per_row = K * sizeof(float);
    size_t rows_by_budget =
        std::max(size_t{1}, parameters.assign_buffer_bytes / bytes_per_row);
    size_t batch_cap = std::min(rows_by_budget, max_assign_batch_rows);
    batch_cap = std::min(batch_cap, std::max(size_t{1}, total_tokens));

    threads::parallel_for(
        threadpool,
        threads::StaticPartition{num_passages},
        [&](auto indices, auto /*tid*/) {
            std::vector<float> batch_floats(batch_cap * dim);
            std::vector<uint64_t> batch_global_idx(batch_cap);
            std::vector<float> scores(batch_cap * K);
            std::vector<svs::BFloat16> batch_bf16;
            std::vector<uint32_t> shortlist(K);
            if (parameters.assigner == Assigner::InnerProductMatrix) {
                batch_bf16.resize(batch_cap * dim);
            }
            size_t count = 0;

            auto flush = [&]() {
                if (count == 0) {
                    return;
                }
                if (parameters.assigner == Assigner::ExactGemm) {
                    svs::kmeans::compute_matmul(
                        batch_floats.data(), centroids.data(), scores.data(), count, K, dim
                    );
                    for (size_t r = 0; r < count; ++r) {
                        const float* row = scores.data() + r * K;
                        uint32_t best = 0;
                        float best_score = row[0];
                        for (size_t j = 1; j < K; ++j) {
                            if (row[j] > best_score) {
                                best_score = row[j];
                                best = svs::lib::narrow_cast<uint32_t>(j);
                            }
                        }
                        id_storage.get_datum(batch_global_idx[r])[0] = best;
                    }
                } else {
                    for (size_t r = 0; r < count; ++r) {
                        const float* src = batch_floats.data() + r * dim;
                        svs::BFloat16* dst = batch_bf16.data() + r * dim;
                        for (size_t d = 0; d < dim; ++d) {
                            dst[d] = svs::round_to_bfloat16(src[d]);
                        }
                    }
                    svs::distance::InnerProductMatrix::compute_packed(
                        batch_bf16.data(), count, packed_centroids, scores.data()
                    );
                    for (size_t r = 0; r < count; ++r) {
                        const float* row = scores.data() + r * K;
                        for (size_t j = 0; j < K; ++j) {
                            shortlist[j] = svs::lib::narrow_cast<uint32_t>(j);
                        }
                        std::partial_sort(
                            shortlist.begin(),
                            shortlist.begin() + static_cast<ptrdiff_t>(candidates),
                            shortlist.end(),
                            [&](uint32_t a, uint32_t b) { return row[a] > row[b]; }
                        );
                        // Ascending by id so the rescore below ties like ExactGemm does.
                        std::sort(
                            shortlist.begin(),
                            shortlist.begin() + static_cast<ptrdiff_t>(candidates)
                        );

                        const float* token_row = batch_floats.data() + r * dim;
                        uint32_t best = shortlist[0];
                        float best_dot = -std::numeric_limits<float>::infinity();
                        for (size_t c = 0; c < candidates; ++c) {
                            uint32_t cid = shortlist[c];
                            auto crow = centroids.get_datum(cid);
                            float dot = 0.0F;
                            for (size_t d = 0; d < dim; ++d) {
                                dot += token_row[d] * crow[d];
                            }
                            if (dot > best_dot) {
                                best_dot = dot;
                                best = cid;
                            }
                        }
                        id_storage.get_datum(batch_global_idx[r])[0] = best;
                    }
                }
                count = 0;
            };

            for (auto p : indices) {
                auto doc = tokens.get_datum(p);
                size_t card = doc.cardinality();
                for (size_t t = 0; t < card; ++t) {
                    auto token = doc.token(t);
                    float* dst = batch_floats.data() + count * dim;
                    for (size_t d = 0; d < dim; ++d) {
                        dst[d] = static_cast<float>(token[d]);
                    }
                    batch_global_idx[count] = offsets[p] + t;
                    ++count;
                    if (count == batch_cap) {
                        flush();
                    }
                }
            }
            flush();
        }
    );

    return svs::data::VectorSetData<uint32_t>(std::move(id_storage), offsets);
}

/// Mismatches between the stored ids and a brute-force exact float32 argmax, on a random
/// sample of tokens. Uses plain loops only, sharing no kernel with either assigner, so it
/// serves as an independent check on both.
struct MismatchReport {
    size_t mismatches = 0;
    size_t sample_size = 0;
};

template <TokenVectorStore Data>
MismatchReport count_assignment_mismatches(
    const Data& tokens,
    const svs::data::SimpleData<float>& centroids,
    const svs::data::VectorSetData<uint32_t>& ids,
    size_t sample_size,
    uint32_t seed
) {
    size_t dim = tokens.dimensions();
    size_t K = centroids.size();
    auto offsets = ids.offsets();
    size_t total_tokens = lib::narrow<size_t>(offsets.back());
    sample_size = std::min(sample_size, total_tokens);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, total_tokens == 0 ? 0 : total_tokens - 1);

    size_t mismatches = 0;
    for (size_t s = 0; s < sample_size; ++s) {
        size_t global = dist(rng);
        size_t passage = static_cast<size_t>(
            std::upper_bound(offsets.begin(), offsets.end(), global) - offsets.begin() - 1
        );
        size_t local = global - lib::narrow<size_t>(offsets[passage]);

        auto token = tokens.get_datum(passage).token(local);
        uint32_t best = 0;
        float best_dot = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < K; ++j) {
            auto c = centroids.get_datum(j);
            float dot = 0.0F;
            for (size_t d = 0; d < dim; ++d) {
                dot += static_cast<float>(token[d]) * c[d];
            }
            if (dot > best_dot) {
                best_dot = dot;
                best = svs::lib::narrow_cast<uint32_t>(j);
            }
        }

        uint32_t stored = ids.get_datum(passage).token(local)[0];
        if (stored != best) {
            ++mismatches;
        }
    }
    return MismatchReport{mismatches, sample_size};
}

/// Clusters fine centroids into coarse groups via the same L2/plain-mean trainer, then
/// an exact nearest-coarse assignment. `centroids` is read-only; a local copy is
/// clustered since train_centroids takes Data by non-const ref.
struct CoarseClusters {
    std::vector<uint32_t> coarse_of_fine;
    size_t num_coarse = 0;
};

template <threads::ThreadPool Pool>
CoarseClusters coarse_clusters(
    const svs::data::SimpleData<float>& centroids,
    const CodebookParameters& parameters,
    Pool& threadpool
) {
    size_t K = centroids.size();
    size_t dim = centroids.dimensions();
    size_t num_coarse = parameters.num_coarse == 0 ? std::max<size_t>(1, K * 5 / 32)
                                                   : parameters.num_coarse;
    if (num_coarse > K) {
        throw ANNEXCEPTION(
            "coarse_clusters: num_coarse={} exceeds the number of fine centroids {}",
            num_coarse,
            K
        );
    }

    // Offset from the fine-training seed so the two never share a random stream.
    std::mt19937 rng(parameters.seed + 1);
    auto working_copy = centroids;
    auto kmeans_result = train_centroids(
        working_copy,
        num_coarse,
        parameters.coarse_iterations,
        parameters.minibatch_size,
        rng,
        threadpool
    );
    svs::lib::Timer scratch_timer;
    svs::kmeans::normalize_centroids(kmeans_result.centroids, threadpool, scratch_timer);

    std::vector<uint32_t> coarse_of_fine(K);
    threads::parallel_for(
        threadpool,
        threads::StaticPartition{K},
        [&](auto indices, auto /*tid*/) {
            for (auto i : indices) {
                auto c = centroids.get_datum(i);
                uint32_t best = 0;
                float best_dot = -std::numeric_limits<float>::infinity();
                for (size_t j = 0; j < num_coarse; ++j) {
                    auto cc = kmeans_result.centroids.get_datum(j);
                    float dot = 0.0F;
                    for (size_t d = 0; d < dim; ++d) {
                        dot += c[d] * cc[d];
                    }
                    if (dot > best_dot) {
                        best_dot = dot;
                        best = svs::lib::narrow_cast<uint32_t>(j);
                    }
                }
                coarse_of_fine[i] = best;
            }
        }
    );

    return CoarseClusters{std::move(coarse_of_fine), num_coarse};
}

/// build_ground_structures's output: the bf16 FineCentroids and pairwise CentroidTable
/// GEM's distances need, built from the same trained centroids.
struct GroundStructures {
    std::shared_ptr<const svs::distance::FineCentroids> fine_centroids;
    std::shared_ptr<const svs::distance::CentroidTable> centroid_table;
};

/// Builds the bf16 FineCentroids and the pairwise CentroidTable from trained centroids;
/// the pair quantize's distances need but neither centroid_table nor FineCentroids'
/// constructor derives from the other, so they are built together here once.
inline GroundStructures build_ground_structures(
    const svs::data::SimpleData<float>& centroids,
    const CodebookParameters& parameters,
    size_t num_threads
) {
    size_t K = centroids.size();
    size_t dim = centroids.dimensions();
    auto fine_centroids = std::make_shared<const svs::distance::FineCentroids>(
        K, dim, centroids.data(), num_threads
    );
    auto centroid_table = std::make_shared<const svs::distance::CentroidTable>(
        svs::distance::CentroidTable::from_centroids(
            centroids.data(), K, dim, parameters.ground, num_threads
        )
    );
    return GroundStructures{std::move(fine_centroids), std::move(centroid_table)};
}

/// Full pipeline: train, assign, build the bf16 FineCentroids and the pairwise
/// CentroidTable, and cluster the fine centroids themselves. The single entry point a
/// driver calls to turn a raw token store into everything GEM's distances need.
struct Quantized {
    svs::data::SimpleData<float> centroids;
    std::shared_ptr<const svs::distance::FineCentroids> fine_centroids;
    std::shared_ptr<const svs::distance::CentroidTable> centroid_table;
    svs::data::VectorSetData<uint32_t> ids;
    std::vector<uint32_t> coarse_of_fine;
    size_t num_coarse = 0;
    size_t K = 0;
    double objective = 0.0;
    size_t num_sampled_passages = 0;
    size_t num_training_tokens = 0;
};

/// Composes train_codebook, assign_tokens, build_ground_structures and coarse_clusters:
/// the pipeline a caller wanting plain results (no phase timing) runs end to end.
template <TokenVectorStore Data, threads::ThreadPool Pool>
Quantized
quantize(const Data& tokens, const CodebookParameters& parameters, Pool& threadpool) {
    auto trained = train_codebook(tokens, parameters, threadpool);
    auto ids = assign_tokens(tokens, trained.centroids, parameters, threadpool);

    size_t K = trained.centroids.size();
    size_t num_threads = threadpool.size();
    auto ground = build_ground_structures(trained.centroids, parameters, num_threads);
    auto coarse = coarse_clusters(trained.centroids, parameters, threadpool);

    return Quantized{
        std::move(trained.centroids),
        std::move(ground.fine_centroids),
        std::move(ground.centroid_table),
        std::move(ids),
        std::move(coarse.coarse_of_fine),
        coarse.num_coarse,
        K,
        trained.objective,
        trained.num_sampled_passages,
        trained.num_training_tokens};
}

} // namespace svs::quantization::gem

#else

#error \
    "svs/quantization/gem/codebook.h needs SVS_ENABLE_MULTI_VECTOR=1 (cmake: SVS_EXPERIMENTAL_ENABLE_MULTI_VECTOR=YES); it pulls in Intel MKL through kmeans_training.h."

#endif // SVS_ENABLE_MULTI_VECTOR
