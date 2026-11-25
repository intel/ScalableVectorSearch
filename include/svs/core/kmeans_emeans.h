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

// svs
#include "svs/core/kmeans_training.h"
#include "svs/lib/exception.h"
#include "svs/lib/threads/threadpool.h"
#include "svs/lib/timing.h"

// stl
#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

// e-means: mini-batch k-means with EMA centroid updates, a cosine-annealed step size, and
// dead-cluster revival, following Algorithm 1 of Bessa et al., "e-means: Billion-scale
// mini-batch k-means with low-cost approximations", arXiv:2611.09501v1. Shares its data,
// rng, thread pool and result shape with kmeans_training.h's Lloyd path and
// quantization/gem/codebook.h's train_centroids.
namespace svs::kmeans {

/// @brief Parameters for e-means training (arXiv:2611.09501v1, Algorithm 1).
struct EMeansParameters {
    /// Number of epochs T: full passes over a fresh shuffle of the data. 2 is an SVS
    /// default for GEM-scale codebooks, not a value the paper recommends.
    size_t epochs = 2;
    /// Minibatch size B. Unset selects the batch scaling law (Eq. 7, Table 2), rounded to
    /// the nearest power of two and capped at the dataset size.
    std::optional<size_t> batch_size = std::nullopt;
    /// Initial EMA step alpha0. Unset selects the step scaling law (Eq. 6, Table 2),
    /// evaluated at the resolved batch size and capped at 1.
    std::optional<double> alpha0 = std::nullopt;
    /// Cosine-annealing floor: alpha decays from alpha0 to epsilon * alpha0 (Eq. 5). Must
    /// lie in [0, 1].
    double epsilon = 0.1;
    /// Revival threshold: a cluster is revived once its EMA count drops below tau times
    /// the mean EMA count over all clusters (Appendix H.1). Must be non-negative.
    double tau = 0.1;
    /// Batches between dead-cluster revival checks. Must be positive (used as a modulus).
    size_t revival_interval = 5;
    /// When true, run the closing pass that replaces each centroid with the plain mean of
    /// the points assigned to it during the last epoch (Fig. 22's "closing mean").
    bool final_mean_pass = true;
    /// Caps the per-chunk assignment score buffer at assignment_chunk * k floats, the same
    /// role kmeans_training.h's minibatch_size plays for the Lloyd path. Must be positive.
    size_t assignment_chunk = 10'000;
};

/// @brief Centroids and the k-means objective (mean squared error) after emeans_train;
/// same shape as quantization::gem::L2KMeansResult, the GEM Lloyd path's result.
struct EMeansResult {
    data::SimpleData<float> centroids;
    double objective = 0.0;
};

namespace detail {

// Table 2 coefficients (arXiv:2611.09501v1 Sec. 4), fit by ordinary least squares in log
// space on validation MSE from DEEP/L2, 256 <= k <= 65536; not validated for other
// metrics, other datasets, or very low dimension.
inline constexpr double EMEANS_BATCH_COEFF_SMALL_N = 0.035;
// Table 2 prints this branch's coefficient as 0.0044; kept as 0.035/8 (0.004375) to match
// the paper's stated derivation ("divide by 8") rather than its rounded printed value.
inline constexpr double EMEANS_BATCH_COEFF_LARGE_N = 0.035 / 8.0; // n > 2e6 (Eq. 7 note)
inline constexpr size_t EMEANS_BATCH_LARGE_N_THRESHOLD = 2'000'000;
inline constexpr double EMEANS_BATCH_EXP_K = 0.13;
inline constexpr double EMEANS_BATCH_EXP_N = 0.88;
inline constexpr double EMEANS_BATCH_EXP_T = 0.81;

inline constexpr double EMEANS_ALPHA_COEFF = 0.824;
inline constexpr double EMEANS_ALPHA_EXP_K = -0.46;
inline constexpr double EMEANS_ALPHA_EXP_B = 0.90;
inline constexpr double EMEANS_ALPHA_EXP_N = -0.45;
inline constexpr double EMEANS_ALPHA_EXP_T = -0.11;

// Appendix H.1's split delta: one reading of "BF16 machine epsilon" as 2^-7 (bf16's 7
// explicit mantissa bits); the paper does not spell out the exact bit count it means.
inline constexpr double EMEANS_REVIVAL_DELTA = 1.0 / 128.0;

// EMA count/sum initializer (Algorithm 1, step 2): N_j starts near zero so the first
// batch's statistics dominate gamma*N_j + alpha*n_j immediately.
inline constexpr double EMEANS_INITIAL_COUNT = 1e-10;

/// Eq. 7: batch-size scaling law, rounded to the nearest power of two and capped at n.
// "Nearest power of two" rounds log2(raw) to the nearest integer; the paper does not say
// which of the two bracketing powers wins exactly at a midpoint, this does round-half-up.
inline size_t resolve_batch_size(size_t n, size_t k, size_t epochs) {
    double b_c = n > EMEANS_BATCH_LARGE_N_THRESHOLD ? EMEANS_BATCH_COEFF_LARGE_N
                                                    : EMEANS_BATCH_COEFF_SMALL_N;
    double raw = b_c * std::pow(static_cast<double>(k), EMEANS_BATCH_EXP_K) *
                 std::pow(static_cast<double>(n), EMEANS_BATCH_EXP_N) *
                 std::pow(static_cast<double>(epochs), EMEANS_BATCH_EXP_T);
    double pow2 = std::pow(2.0, std::round(std::log2(std::max(raw, 1.0))));
    size_t b = static_cast<size_t>(std::min(pow2, static_cast<double>(n)));
    return std::clamp(b, size_t{1}, n);
}

/// Eq. 6: step scaling law, evaluated at the resolved batch size, capped at 1.
inline double resolve_alpha0(size_t n, size_t k, size_t batch_size, size_t epochs) {
    double raw = EMEANS_ALPHA_COEFF * std::pow(static_cast<double>(k), EMEANS_ALPHA_EXP_K) *
                 std::pow(static_cast<double>(batch_size), EMEANS_ALPHA_EXP_B) *
                 std::pow(static_cast<double>(n), EMEANS_ALPHA_EXP_N) *
                 std::pow(static_cast<double>(epochs), EMEANS_ALPHA_EXP_T);
    return std::min(raw, 1.0);
}

// Eq. 5: alpha(t) = alpha0 * (eps + (1-eps) * 0.5 * (1 + cos(pi*t/M))), t = 1..M, matching
// Algorithm 1's own 1-indexed t (also used, unchanged, for the "t mod 5" revival check).
inline double cosine_step(double alpha0, double epsilon, size_t t, size_t total_batches) {
    double phase =
        std::numbers::pi * static_cast<double>(t) / static_cast<double>(total_batches);
    return alpha0 * (epsilon + (1.0 - epsilon) * 0.5 * (1.0 + std::cos(phase)));
}

// Appendix H.1: every `revival_interval` batches, a cluster with EMA count below
// tau * mean(N) is revived by splitting a donor drawn with probability proportional to
// the donor's own EMA count, without replacement among this round's revivals. Mutates N/S
// in place. `dead_scratch`/`excluded_scratch`/`cum_scratch` are caller-owned, sized k, so
// a revival round allocates nothing of its own inside the batch loop.
//
// O(k) to build the prefix sum once, then O(log k) expected per donor draw (binary search
// plus a bounded number of redraws on a collision), with an O(k) linear-scan fallback only
// once the redraw budget is exhausted. The paper draws donors "among all clusters"; this
// also excludes this round's own dead clusters from the donor pool, which differs only
// when a dead cluster would otherwise have been drawn as its own donor (negligible: a
// dead cluster's weight N_j is already far below the mean by construction).
inline void revive_dead_clusters(
    std::vector<double>& N,
    std::vector<double>& S,
    size_t dims,
    double tau,
    std::mt19937& rng,
    std::vector<size_t>& dead_scratch,
    std::vector<bool>& excluded_scratch,
    std::vector<double>& cum_scratch
) {
    size_t k = N.size();
    double mean_n = std::accumulate(N.begin(), N.end(), 0.0) / static_cast<double>(k);
    double threshold = tau * mean_n;

    auto& dead = dead_scratch;
    auto& excluded = excluded_scratch;
    auto& cum = cum_scratch;
    dead.clear();
    std::fill(excluded.begin(), excluded.end(), false);
    for (size_t j = 0; j < k; ++j) {
        if (N[j] < threshold) {
            dead.push_back(j);
            excluded[j] = true;
        }
    }
    if (dead.empty()) {
        return;
    }

    // Prefix sum of donor weight; excluded (dead) clusters contribute 0, so a draw never
    // lands on one without needing a separate filter in the binary search below.
    double running = 0.0;
    for (size_t c = 0; c < k; ++c) {
        running += excluded[c] ? 0.0 : N[c];
        cum[c] = running;
    }
    double total_weight = running;
    if (total_weight <= 0.0) {
        return; // No eligible donor at all this round; leave every cluster dead for now.
    }

    auto distribution = std::uniform_real_distribution<double>(0.0, total_weight);
    constexpr int max_redraws = 8;
    for (size_t j : dead) {
        size_t donor = k;
        for (int attempt = 0; attempt < max_redraws; ++attempt) {
            double r = distribution(rng);
            auto it = std::lower_bound(cum.begin(), cum.end(), r);
            size_t candidate = std::min(static_cast<size_t>(it - cum.begin()), k - 1);
            if (!excluded[candidate]) {
                donor = candidate;
                break;
            }
        }
        if (donor == k) {
            // Redraw budget exhausted (heavy exclusion this round): fall back to a
            // guaranteed-correct linear scan for the first eligible cluster.
            for (size_t c = 0; c < k; ++c) {
                if (!excluded[c]) {
                    donor = c;
                    break;
                }
            }
        }
        if (donor == k) {
            continue; // No eligible donor left this round; leave the cluster dead for now.
        }
        excluded[donor] = true;

        // Split the donor along a hyperplane through its centroid: opposite-sign
        // perturbation on alternating dimensions, then halve its count between the two.
        double donor_n_half = N[donor] / 2.0;
        for (size_t d = 0; d < dims; ++d) {
            double c_val = S[donor * dims + d] / N[donor];
            double sign = (d % 2 == 0) ? 1.0 : -1.0;
            S[j * dims + d] = c_val * (1.0 + sign * EMEANS_REVIVAL_DELTA) * donor_n_half;
            S[donor * dims + d] =
                c_val * (1.0 - sign * EMEANS_REVIVAL_DELTA) * donor_n_half;
        }
        N[j] = donor_n_half;
        N[donor] = donor_n_half;
    }
}

// Accumulates this batch's per-cluster count n_j and sum s_j (Algorithm 1's per-batch
// statistics), ahead of the EMA update. Range-partitions by centroid id so each thread
// owns disjoint output, the same determinism-by-construction pattern kmeans_training.h's
// centroid_adjustment uses; `batch_counts`/`batch_sums` must be zeroed by the caller.
template <threads::ThreadPool Pool>
void accumulate_batch_stats(
    const data::SimpleData<float>& gathered,
    size_t count,
    const std::vector<size_t>& local_assignments,
    size_t num_clusters,
    std::vector<double>& batch_counts,
    std::vector<double>& batch_sums,
    Pool& threadpool
) {
    size_t n_threads = threadpool.size();
    size_t dims = gathered.dimensions();
    threads::parallel_for(
        threadpool,
        threads::StaticPartition{n_threads},
        [&](auto /*indices*/, auto tid) {
            size_t centroid_start = (num_clusters * tid) / n_threads;
            size_t centroid_end = (num_clusters * (tid + 1)) / n_threads;
            for (size_t i = 0; i < count; ++i) {
                size_t assignment = local_assignments[i];
                if (assignment >= centroid_start && assignment < centroid_end) {
                    batch_counts[assignment] += 1.0;
                    auto datum = gathered.get_datum(i);
                    double* sum_row = batch_sums.data() + assignment * dims;
                    for (size_t p = 0; p < dims; ++p) {
                        sum_row[p] += static_cast<double>(datum[p]);
                    }
                }
            }
        }
    );
}

// Closing pass (Fig. 22): replace each centroid with the plain mean of the points
// assigned to it over the whole last epoch. A cluster with no last-epoch assignment keeps
// its EMA readout rather than being zeroed. Allocates once, after the batch loop ends, not
// inside it.
template <data::ImmutableMemoryDataset Data, threads::ThreadPool Pool>
void closing_mean_pass(
    const Data& data,
    data::SimpleData<float>& centroids,
    const std::vector<size_t>& assignments,
    Pool& threadpool
) {
    size_t n_threads = threadpool.size();
    size_t k = centroids.size();
    size_t dims = centroids.dimensions();
    std::vector<double> sums(k * dims, 0.0);
    std::vector<size_t> counts(k, 0);

    threads::parallel_for(
        threadpool,
        threads::StaticPartition{n_threads},
        [&](auto /*indices*/, auto tid) {
            size_t centroid_start = (k * tid) / n_threads;
            size_t centroid_end = (k * (tid + 1)) / n_threads;
            for (auto i : data.eachindex()) {
                size_t a = assignments[i];
                if (a >= centroid_start && a < centroid_end) {
                    counts[a]++;
                    auto datum = data.get_datum(i);
                    double* row = sums.data() + a * dims;
                    for (size_t d = 0; d < dims; ++d) {
                        row[d] += static_cast<double>(datum[d]);
                    }
                }
            }
        }
    );

    threads::parallel_for(
        threadpool,
        threads::StaticPartition{k},
        [&](auto indices, auto /*tid*/) {
            for (auto j : indices) {
                if (counts[j] != 0) {
                    auto c = centroids.get_datum(j);
                    double inv = 1.0 / static_cast<double>(counts[j]);
                    const double* row = sums.data() + j * dims;
                    for (size_t d = 0; d < dims; ++d) {
                        c[d] = static_cast<float>(row[d] * inv);
                    }
                }
            }
        }
    );
}

// Sum of squared L2 distances from `data_batch[0..count)` to its assigned centroid,
// computed from the raw vectors (not the matmul scores). Per-thread partials over a
// StaticPartition, summed in tid order, so the result is independent of scheduling.
template <data::ImmutableMemoryDataset Data, threads::ThreadPool Pool>
double accumulate_squared_error(
    const Data& data_batch,
    size_t count,
    const std::vector<size_t>& local_assignments,
    const data::SimpleData<float>& centroids,
    Pool& threadpool
) {
    size_t n_threads = threadpool.size();
    size_t dims = centroids.dimensions();
    std::vector<double> partials(n_threads, 0.0);
    threads::parallel_for(
        threadpool,
        threads::StaticPartition{count},
        [&](auto indices, auto tid) {
            double sum = 0.0;
            for (auto i : indices) {
                auto x = data_batch.get_datum(i);
                auto c = centroids.get_datum(local_assignments[i]);
                double sq = 0.0;
                for (size_t d = 0; d < dims; ++d) {
                    double diff = static_cast<double>(x[d]) - static_cast<double>(c[d]);
                    sq += diff * diff;
                }
                sum += sq;
            }
            partials[tid] = sum;
        }
    );
    double total = 0.0;
    for (size_t tid = 0; tid < n_threads; ++tid) {
        total += partials[tid];
    }
    return total;
}

// Assigns data[window_start..window_start+window_size) to their nearest centroid, into
// chunk_assignments[0..window_size); window_size must not exceed chunk_assignments' size.
// Caller chunks the batch so chunk_matmul_results never grows past chunk_size * k floats.
template <data::ImmutableMemoryDataset Data, typename Distance, threads::ThreadPool Pool>
void assign_chunk(
    const Data& data,
    size_t window_start,
    size_t window_size,
    Distance& distance,
    data::SimpleData<float>& centroids,
    std::vector<float>& centroids_norm,
    std::vector<size_t>& chunk_assignments,
    std::vector<float>& chunk_norm_scratch,
    data::SimpleData<float>& chunk_matmul_results,
    Pool& threadpool,
    lib::Timer& scratch_timer
) {
    auto window_view = data::make_view(
        data, threads::UnitRange<uint64_t>{window_start, window_start + window_size}
    );
    generate_norms(window_view, chunk_norm_scratch, threadpool);
    centroid_assignment(
        window_view,
        chunk_norm_scratch,
        threads::UnitRange<uint64_t>{0, window_size},
        distance,
        centroids,
        centroids_norm,
        chunk_assignments,
        chunk_matmul_results,
        threadpool,
        scratch_timer
    );
}

// Final objective (Algorithm 1's MSE): one pass over `data` in assignment_chunk-sized
// windows, reusing the Lloyd assignment kernel, not a per-point brute-force scan over k.
template <data::ImmutableMemoryDataset Data, threads::ThreadPool Pool>
double compute_objective(
    Data& data,
    data::SimpleData<float>& centroids,
    size_t chunk_size,
    data::SimpleData<float>& chunk_matmul_results,
    std::vector<float>& chunk_norm_scratch,
    std::vector<float>& centroids_norm,
    std::vector<size_t>& chunk_assignments,
    Pool& threadpool,
    lib::Timer& scratch_timer
) {
    size_t n = data.size();
    generate_norms(centroids, centroids_norm, threadpool);
    auto distance = svs::DistanceL2{};

    double total = 0.0;
    for (size_t start = 0; start < n; start += chunk_size) {
        size_t this_chunk = std::min(chunk_size, n - start);
        assign_chunk(
            data,
            start,
            this_chunk,
            distance,
            centroids,
            centroids_norm,
            chunk_assignments,
            chunk_norm_scratch,
            chunk_matmul_results,
            threadpool,
            scratch_timer
        );
        auto window_view =
            data::make_view(data, threads::UnitRange<uint64_t>{start, start + this_chunk});
        total += accumulate_squared_error(
            window_view, this_chunk, chunk_assignments, centroids, threadpool
        );
    }
    return total / static_cast<double>(n);
}

} // namespace detail

/// @brief The batch size and alpha0 emeans_train actually uses, after applying Table 2's
/// scaling laws to any unset EMeansParameters field. Pure function of its inputs, so a
/// caller can recompute the same values for logging without re-running training.
struct ResolvedEMeansParameters {
    size_t batch_size = 0;
    double alpha0 = 0.0;
};

/// @brief Resolve EMeansParameters::batch_size/alpha0 against n and k, applying Table 2's
/// scaling laws (Eq. 6, Eq. 7) to whichever field is unset. The sole place that combines a
/// user override with the scaling-law default, so emeans_train and a caller logging the
/// effective parameters agree by construction.
inline ResolvedEMeansParameters
resolve_emeans_parameters(size_t n, size_t k, const EMeansParameters& parameters) {
    size_t batch_size = std::clamp(
        parameters.batch_size.value_or(detail::resolve_batch_size(n, k, parameters.epochs)),
        size_t{1},
        n
    );
    double alpha0 = parameters.alpha0.value_or(
        detail::resolve_alpha0(n, k, batch_size, parameters.epochs)
    );
    return ResolvedEMeansParameters{batch_size, alpha0};
}

/// @brief e-means training (arXiv:2611.09501v1, Algorithm 1), starting from caller-given
/// initial centroids.
///
/// @param parameters e-means hyperparameters; see EMeansParameters.
/// @param data The training set. Not modified.
/// @param initial_centroids The k starting centroids (Algorithm 1, step 1). Its size fixes
///        the number of clusters k; copied into the EMA state and not aliased afterward.
/// @param rng Advanced once per epoch shuffle, and once per revival-round donor draw; a
///        fixed seed and thread count reproduce the same centroids run to run.
/// @param threadpool Thread pool used for every parallel step.
/// @param timer Receives per-epoch timing entries, in kmeans_training.h's style.
///
/// @throws ANNEXCEPTION if initial_centroids has more clusters than `data` has points; if
///         a user-supplied batch_size is 0 or alpha0 is outside (0, 1]; or if epochs,
///         assignment_chunk, or revival_interval is 0, epsilon is outside [0, 1], or tau
///         is negative.
template <data::ImmutableMemoryDataset Data, threads::ThreadPool Pool>
EMeansResult emeans_train(
    const EMeansParameters& parameters,
    Data& data,
    data::SimpleData<float> initial_centroids,
    std::mt19937& rng,
    Pool& threadpool,
    lib::Timer& timer
) {
    size_t n = data.size();
    size_t k = initial_centroids.size();
    size_t dims = data.dimensions();

    if (k > n) {
        throw ANNEXCEPTION(
            "emeans_train: num_clusters={} exceeds the number of available points n={}",
            k,
            n
        );
    }
    if (parameters.epochs == 0) {
        throw ANNEXCEPTION("emeans_train: epochs must be positive, got 0");
    }
    if (parameters.batch_size.has_value() && *parameters.batch_size == 0) {
        throw ANNEXCEPTION(
            "emeans_train: batch_size must be positive, got {}", *parameters.batch_size
        );
    }
    if (parameters.alpha0.has_value() &&
        (*parameters.alpha0 <= 0.0 || *parameters.alpha0 > 1.0)) {
        throw ANNEXCEPTION(
            "emeans_train: alpha0 must lie in (0, 1], got {}", *parameters.alpha0
        );
    }
    if (parameters.assignment_chunk == 0) {
        throw ANNEXCEPTION("emeans_train: assignment_chunk must be positive, got 0");
    }
    if (parameters.revival_interval == 0) {
        throw ANNEXCEPTION("emeans_train: revival_interval must be positive, got 0");
    }
    if (parameters.epsilon < 0.0 || parameters.epsilon > 1.0) {
        throw ANNEXCEPTION(
            "emeans_train: epsilon must lie in [0, 1], got {}", parameters.epsilon
        );
    }
    if (parameters.tau < 0.0) {
        throw ANNEXCEPTION(
            "emeans_train: tau must be non-negative, got {}", parameters.tau
        );
    }

    auto training_timer = timer.push_back("EMeans training");

    auto resolved = resolve_emeans_parameters(n, k, parameters);
    size_t batch_size = resolved.batch_size;
    double alpha0 = resolved.alpha0;
    size_t chunk_size = std::min(parameters.assignment_chunk, n);

    // M, the cosine schedule's span: the paper writes M = T*floor(n/B), but a tail batch
    // (n not a multiple of B) is still trained here, so M is the true batch count instead.
    size_t batches_per_epoch = lib::div_round_up(n, batch_size);
    size_t total_batches = parameters.epochs * batches_per_epoch;

    // State (Algorithm 1, step 2): double counts/sums, k + k*d doubles, allocated once.
    std::vector<double> N(k, detail::EMEANS_INITIAL_COUNT);
    std::vector<double> S(k * dims, 0.0);
    for (size_t j = 0; j < k; ++j) {
        auto c = initial_centroids.get_datum(j);
        for (size_t d = 0; d < dims; ++d) {
            S[j * dims + d] = N[j] * static_cast<double>(c[d]);
        }
    }

    // Scratch buffers sized once and reused for every batch; chunk_matmul_results is
    // capped at chunk_size rows, not batch_size (B from the scaling law can be huge).
    auto centroids_fp32 = data::SimpleData<float>(k, dims);
    auto gathered = data::SimpleData<float>(batch_size, dims);
    auto chunk_matmul_results = data::SimpleData<float>(chunk_size, k);
    std::vector<float> chunk_norm_scratch(chunk_size);
    std::vector<size_t> chunk_assignments(chunk_size);
    std::vector<float> centroids_norm(k);
    std::vector<size_t> local_assignments(batch_size);
    std::vector<double> batch_counts(k);
    std::vector<double> batch_sums(k * dims);
    std::vector<size_t> perm(n);
    for (size_t i = 0; i < n; ++i) {
        perm[i] = i;
    }
    std::vector<size_t> last_epoch_assignments(n);
    std::vector<size_t> revival_dead_scratch;
    revival_dead_scratch.reserve(k);
    std::vector<bool> revival_excluded_scratch(k, false);
    std::vector<double> revival_cum_scratch(k);
    // Per-batch kernel timing is uninteresting at hundreds of batches; keep it off `timer`.
    lib::Timer scratch_timer;

    auto distance = svs::DistanceL2{};
    // Algorithm 1 is 1-indexed ("t <- t+1, gamma <- gamma(t)"): t ranges 1..M, so
    // alpha(t) < alpha0 strictly even on the very first batch (gamma never starts at 0).
    size_t t = 0;
    for (size_t epoch = 0; epoch < parameters.epochs; ++epoch) {
        auto epoch_timer = timer.push_back("epoch");
        std::shuffle(perm.begin(), perm.end(), rng);
        bool last_epoch = (epoch + 1 == parameters.epochs);

        for (size_t batch = 0; batch < batches_per_epoch; ++batch) {
            size_t start = batch * batch_size;
            size_t end = std::min(start + batch_size, n);
            size_t this_batch_size = end - start;
            ++t;

            // Gather this batch's points into the contiguous scratch buffer the reused
            // Lloyd-path assignment kernel (centroid_assignment) requires.
            threads::parallel_for(
                threadpool,
                threads::StaticPartition{this_batch_size},
                [&](auto indices, auto /*tid*/) {
                    for (auto i : indices) {
                        auto dst = gathered.get_datum(i);
                        auto src = data.get_datum(perm[start + i]);
                        for (size_t d = 0; d < dims; ++d) {
                            dst[d] = static_cast<float>(src[d]);
                        }
                    }
                }
            );

            double alpha =
                detail::cosine_step(alpha0, parameters.epsilon, t, total_batches);
            double gamma = 1.0 - alpha;

            // Eq. 4 readout: c_j = S_j / N_j, into the fp32 buffer centroid_assignment
            // uses. A cluster with N_j == 0 (EMA-decayed to nothing, never revived) keeps
            // its previous centroid instead of dividing by zero.
            threads::parallel_for(
                threadpool,
                threads::StaticPartition{k},
                [&](auto indices, auto /*tid*/) {
                    for (auto j : indices) {
                        if (N[j] == 0.0) {
                            continue;
                        }
                        auto c = centroids_fp32.get_datum(j);
                        for (size_t d = 0; d < dims; ++d) {
                            c[d] = static_cast<float>(S[j * dims + d] / N[j]);
                        }
                    }
                }
            );

            generate_norms(centroids_fp32, centroids_norm, threadpool);
            for (size_t offset = 0; offset < this_batch_size; offset += chunk_size) {
                size_t this_chunk = std::min(chunk_size, this_batch_size - offset);
                detail::assign_chunk(
                    gathered,
                    offset,
                    this_chunk,
                    distance,
                    centroids_fp32,
                    centroids_norm,
                    chunk_assignments,
                    chunk_norm_scratch,
                    chunk_matmul_results,
                    threadpool,
                    scratch_timer
                );
                for (size_t i = 0; i < this_chunk; ++i) {
                    local_assignments[offset + i] = chunk_assignments[i];
                }
            }

            std::fill(batch_counts.begin(), batch_counts.end(), 0.0);
            std::fill(batch_sums.begin(), batch_sums.end(), 0.0);
            detail::accumulate_batch_stats(
                gathered,
                this_batch_size,
                local_assignments,
                k,
                batch_counts,
                batch_sums,
                threadpool
            );

            // Eq. 2: N_j <- gamma*N_j + alpha*n_j, S_j <- gamma*S_j + alpha*s_j.
            threads::parallel_for(
                threadpool,
                threads::StaticPartition{k},
                [&](auto indices, auto /*tid*/) {
                    for (auto j : indices) {
                        N[j] = gamma * N[j] + alpha * batch_counts[j];
                        double* s_row = S.data() + j * dims;
                        const double* bs_row = batch_sums.data() + j * dims;
                        for (size_t d = 0; d < dims; ++d) {
                            s_row[d] = gamma * s_row[d] + alpha * bs_row[d];
                        }
                    }
                }
            );

            if (last_epoch) {
                for (size_t i = 0; i < this_batch_size; ++i) {
                    last_epoch_assignments[perm[start + i]] = local_assignments[i];
                }
            }

            if (t % parameters.revival_interval == 0) {
                detail::revive_dead_clusters(
                    N,
                    S,
                    dims,
                    parameters.tau,
                    rng,
                    revival_dead_scratch,
                    revival_excluded_scratch,
                    revival_cum_scratch
                );
            }
        }
    }

    // Final readout (same zero-count guard as the per-batch readout above), then the
    // optional closing-mean pass (Fig. 22).
    threads::parallel_for(
        threadpool,
        threads::StaticPartition{k},
        [&](auto indices, auto /*tid*/) {
            for (auto j : indices) {
                if (N[j] == 0.0) {
                    continue;
                }
                auto c = centroids_fp32.get_datum(j);
                for (size_t d = 0; d < dims; ++d) {
                    c[d] = static_cast<float>(S[j * dims + d] / N[j]);
                }
            }
        }
    );
    if (parameters.final_mean_pass) {
        detail::closing_mean_pass(data, centroids_fp32, last_epoch_assignments, threadpool);
    }

    double objective = detail::compute_objective(
        data,
        centroids_fp32,
        chunk_size,
        chunk_matmul_results,
        chunk_norm_scratch,
        centroids_norm,
        chunk_assignments,
        threadpool,
        scratch_timer
    );
    training_timer.finish();
    return EMeansResult{std::move(centroids_fp32), objective};
}

/// @brief e-means training (arXiv:2611.09501v1, Algorithm 1), initializing centroids by
/// random sampling without replacement from `data` (Algorithm 1, step 1).
///
/// @param parameters e-means hyperparameters; see EMeansParameters.
/// @param data The training set. Not modified.
/// @param num_clusters k, the target number of clusters.
/// @param rng Advanced by the initial sample, then as emeans_train's other overload does.
/// @param threadpool Thread pool used for every parallel step.
/// @param timer Receives per-epoch timing entries, in kmeans_training.h's style.
///
/// @throws ANNEXCEPTION if num_clusters exceeds the number of available points, or for the
///         same invalid-parameter cases as the initial-centroids overload.
template <data::ImmutableMemoryDataset Data, threads::ThreadPool Pool>
EMeansResult emeans_train(
    const EMeansParameters& parameters,
    Data& data,
    size_t num_clusters,
    std::mt19937& rng,
    Pool& threadpool,
    lib::Timer& timer
) {
    size_t n = data.size();
    if (num_clusters > n) {
        throw ANNEXCEPTION(
            "emeans_train: num_clusters={} exceeds the number of available points n={}",
            num_clusters,
            n
        );
    }
    std::vector<size_t> id_scratch(num_clusters);
    auto initial = init_centroids<float>(data, id_scratch, num_clusters, rng, threadpool);
    return emeans_train(parameters, data, std::move(initial), rng, threadpool, timer);
}

} // namespace svs::kmeans
