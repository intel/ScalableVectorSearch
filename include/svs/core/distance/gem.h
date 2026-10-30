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

// Vendored POT 0.9.5 network simplex solver (include/svs/third-party/pot/); not
// warning-clean under -Werror, so only the warnings it actually trips are silenced.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "svs/third-party/pot/network_simplex_simple.h"
#pragma GCC diagnostic pop

#include "svs/core/distance/distance_core.h"
#include "svs/lib/avx_detection.h"
#include "svs/lib/preprocessor.h"
#include "svs/lib/threads.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace svs::distance {

// Same node-index type POT's EMD_wrap instantiates the solver with (ot/lp/EMD.h).
using gem_node_id_type = unsigned int;

/// Ground distance between two unit centroids the from_centroids() table is built from:
/// Cos is `max(0, 1 - <ci,cj>)` (GEM Eq. 11), L2 is `sqrt(max(0, 2 - 2<ci,cj>))`.
enum class Ground { Cos, L2 };

inline Ground parse_ground(std::string_view s) {
    if (s == "cos") {
        return Ground::Cos;
    }
    if (s == "l2") {
        return Ground::L2;
    }
    throw std::invalid_argument(
        "parse_ground: unsupported ground distance '" + std::string(s) +
        "'; expected cos or l2"
    );
}

/// Owns a dense K x K matrix of pairwise centroid distances, row-major. Built either by
/// loading a precomputed file or in-process from centroids (see from_centroids()).
struct CentroidTable {
    size_t K;
    std::unique_ptr<float[]> data;

    CentroidTable(size_t k, std::vector<float> table)
        : K{k}
        , data{std::make_unique<float[]>(k * k)} {
        if (table.size() != K * K) {
            throw std::invalid_argument("CentroidTable: data size does not match K*K");
        }
        std::copy(table.begin(), table.end(), data.get());
    }

    // The file has no header, so K comes from table.json via the caller.
    static CentroidTable load(const std::filesystem::path& path, size_t k) {
        auto expected_bytes = k * k * sizeof(float);
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("CentroidTable::load: cannot open " + path.string());
        }
        in.seekg(0, std::ios::end);
        auto actual_bytes = static_cast<size_t>(in.tellg());
        if (actual_bytes != expected_bytes) {
            throw std::runtime_error(
                "CentroidTable::load: " + path.string() + " has " +
                std::to_string(actual_bytes) + " bytes, expected " +
                std::to_string(expected_bytes) + " for K=" + std::to_string(k)
            );
        }
        in.seekg(0, std::ios::beg);
        // Uninitialized: the read below fills every element, so zero-filling first is
        // wasted bandwidth on a buffer that can be hundreds of GiB.
        CentroidTable result{k, uninitialized_tag{}};
        in.read(
            reinterpret_cast<char*>(result.data.get()),
            static_cast<std::streamsize>(expected_bytes)
        );
        if (!in) {
            throw std::runtime_error("CentroidTable::load: short read on " + path.string());
        }
        return result;
    }

    /// Computes T[i][j] = ground(c_i, c_j) from K row-major `dim`-wide centroids,
    /// T[i][i] = 0. Parallelized over row blocks on `num_threads` threads; each thread's
    /// first write to its row range is also its NUMA first-touch.
    static CentroidTable from_centroids(
        const float* centroids, size_t k, size_t dim, Ground ground, size_t num_threads
    ) {
        CentroidTable result{k, uninitialized_tag{}};
        float* out = result.data.get();
        svs::threads::DefaultThreadPool pool(num_threads);

        // Column block width: a block's centroid rows (col_block * dim * 4 bytes, 128 KiB
        // at dim=128) stay resident in L2 while every row a thread owns sweeps over it.
        constexpr size_t col_block = 256;

        svs::threads::parallel_for(
            pool,
            svs::threads::StaticPartition(k),
            [&](const auto& row_ids, uint64_t /*tid*/) {
                for (size_t i : row_ids) {
                    const float* ci = centroids + i * dim;
                    for (size_t jb = 0; jb < k; jb += col_block) {
                        size_t jmax = std::min(jb + col_block, k);
                        for (size_t j = jb; j < jmax; ++j) {
                            if (i == j) {
                                out[i * k + j] = 0.0F;
                                continue;
                            }
                            const float* cj = centroids + j * dim;
                            float dot = 0.0F;
                            for (size_t d = 0; d < dim; ++d) {
                                dot += ci[d] * cj[d];
                            }
                            out[i * k + j] =
                                ground == Ground::Cos
                                    ? std::max(0.0F, 1.0F - dot)
                                    : std::sqrt(std::max(0.0F, 2.0F - 2.0F * dot));
                        }
                    }
                }
            }
        );
        return result;
    }

    float operator()(uint32_t i, uint32_t j) const {
        return data[static_cast<size_t>(i) * K + j];
    }

  private:
    struct uninitialized_tag {};

    // new[] with no parens default-initializes float (indeterminate value, no zero
    // fill); ownership passes to unique_ptr immediately, so this stays exception-safe.
    CentroidTable(size_t k, uninitialized_tag)
        : K{k}
        , data{std::make_unique_for_overwrite<float[]>(k * k)} {}
};

/// Owns the raw K x dim fine-centroid vectors (row-major, not a pairwise table). Used by
/// QChamferExactQuery to build a per-query (K x query_tokens) dot table against the exact
/// float query, mirroring GEM's score_fine_centers (src/search/routing.cpp:55-61), which
/// never quantizes the query side.
struct FineCentroids {
    size_t K;
    size_t dim;
    std::unique_ptr<float[]> data;

    FineCentroids(size_t k, size_t dim_, const float* flat)
        : K{k}
        , dim{dim_}
        , data{std::make_unique<float[]>(k * dim_)} {
        std::copy(flat, flat + k * dim_, data.get());
    }

    // Raw file of K rows of `dim` floats with no header, so K comes from the file size.
    static FineCentroids load(const std::filesystem::path& path, size_t dim) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("FineCentroids::load: cannot open " + path.string());
        }
        in.seekg(0, std::ios::end);
        auto bytes = static_cast<size_t>(in.tellg());
        auto row_bytes = dim * sizeof(float);
        if (bytes % row_bytes != 0) {
            throw std::runtime_error(
                "FineCentroids::load: " + path.string() + " has " + std::to_string(bytes) +
                " bytes, not a multiple of dim=" + std::to_string(dim) + " * 4 bytes"
            );
        }
        auto k = bytes / row_bytes;
        std::vector<float> flat(k * dim);
        in.seekg(0, std::ios::beg);
        in.read(reinterpret_cast<char*>(flat.data()), static_cast<std::streamsize>(bytes));
        if (!in) {
            throw std::runtime_error("FineCentroids::load: short read on " + path.string());
        }
        return FineCentroids{k, dim, flat.data()};
    }
};

namespace gem_detail {

// Token-set span layout: [0] = token count n, [1..n] = centroid ids, rest unused padding.
inline size_t set_size(std::span<const float> s) { return static_cast<size_t>(s[0]); }
inline const float* set_ids(std::span<const float> s) { return s.data() + 1; }

// Raw-query span layout (QChamferExactQuery): [0] = token count n, [1..1+n*dim) = n
// raw fp32 token embeddings row-major, rest unused padding.
inline size_t vector_set_size(std::span<const float> s) {
    return static_cast<size_t>(s[0]);
}
inline const float* vector_set_tokens(std::span<const float> s) { return s.data() + 1; }

// Thread-local because the functors are shared across threads (implicit_broadcast).
struct MergeScratch {
    std::vector<uint32_t> ids;
    std::vector<double> weights;
};

// Per-token mass is 1/n (GEM Eq. 12); tokens sharing a centroid merge into one weighted
// point.
inline size_t merge_duplicates(const float* ids_f, size_t n, MergeScratch& scratch) {
    scratch.ids.resize(n);
    for (size_t i = 0; i < n; ++i) {
        scratch.ids[i] = static_cast<uint32_t>(ids_f[i]);
    }
    std::sort(scratch.ids.begin(), scratch.ids.end());
    scratch.weights.clear();
    size_t unique_count = 0;
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        while (j < n && scratch.ids[j] == scratch.ids[i]) {
            ++j;
        }
        scratch.ids[unique_count] = scratch.ids[i];
        scratch.weights.push_back(static_cast<double>(j - i) / static_cast<double>(n));
        ++unique_count;
        i = j;
    }
    scratch.ids.resize(unique_count);
    return unique_count;
}

// Shared by QChamferExactImpl's generic and per-ISA compute_decoded specializations
// (GEM's code_distance, src/distance/maxsim.cpp:9-45) so the reduction is written once.
inline float qchamfer_exact_reduce(
    const float* table, size_t n_query, const uint32_t* doc_ids, size_t n_doc
) {
    thread_local std::vector<float> maxima;
    maxima.assign(n_query, -std::numeric_limits<float>::infinity());
    for (size_t j = 0; j < n_doc; ++j) {
        const float* row = table + static_cast<size_t>(doc_ids[j]) * n_query;
        for (size_t i = 0; i < n_query; ++i) {
            maxima[i] = std::max(maxima[i], row[i]);
        }
    }
    double total = 0.0;
    for (size_t i = 0; i < n_query; ++i) {
        total += 1.0 - static_cast<double>(maxima[i]);
    }
    return static_cast<float>(total / static_cast<double>(n_query));
}

// Casts a token-set's float centroid ids to uint32_t with no merge; qCH counts duplicates.
inline void decode_ids(const float* ids_f, size_t n, std::vector<uint32_t>& out) {
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<uint32_t>(ids_f[i]);
    }
}

} // namespace gem_detail

// Instantiated per ISA in svs/multi-arch/x86; without SVS_NOINLINE the caller inlines the
// body at baseline flags. The bodies are scalar, so only compiler auto-vectorization
// differs. compute_decoded takes already-decoded (deduplicated, weighted) centroid ids so
// gem_time can benchmark the solve separately from the span decode.
template <AVX_AVAILABILITY Avx> struct QEmdImpl {
    SVS_NOINLINE static float compute_decoded(
        const CentroidTable& table,
        const uint32_t* ids_a,
        const double* w_a,
        size_t ua,
        const uint32_t* ids_b,
        const double* w_b,
        size_t ub
    ) {
        if (ua == 1 && ub == 1) {
            return table(ids_a[0], ids_b[0]);
        }

        // Mirrors POT's EMD_wrap (ot/lp/EMD_wrapper.cpp); all weights are already nonzero.
        using Digraph = lemon::FullBipartiteDigraph;
        Digraph di(static_cast<int>(ua), static_cast<int>(ub));
        lemon::NetworkSimplexSimple<Digraph, double, double, gem_node_id_type> net(
            di,
            true,
            static_cast<int>(ua + ub),
            static_cast<int64_t>(ua) * static_cast<int64_t>(ub),
            std::numeric_limits<uint64_t>::max()
        );

        thread_local std::vector<double> neg_wb;
        neg_wb.resize(ub);
        for (size_t j = 0; j < ub; ++j) {
            neg_wb[j] = -w_b[j];
        }
        net.supplyMap(w_a, static_cast<int>(ua), neg_wb.data(), static_cast<int>(ub));

        int64_t idarc = 0;
        for (size_t i = 0; i < ua; ++i) {
            for (size_t j = 0; j < ub; ++j) {
                net.setCost(di.arcFromId(idarc), table(ids_a[i], ids_b[j]));
                ++idarc;
            }
        }

        if (net.run() != decltype(net)::OPTIMAL) {
            throw std::runtime_error(
                "QEmd: network simplex did not reach an optimal solution"
            );
        }

        double cost = 0.0;
        Digraph::Arc arc;
        for (di.first(arc); arc != INVALID; di.next(arc)) {
            auto src = static_cast<size_t>(di.source(arc));
            auto dst = static_cast<size_t>(di.target(arc)) - ua;
            cost += net.flow(arc) * table(ids_a[src], ids_b[dst]);
        }
        return static_cast<float>(cost);
    }

    SVS_NOINLINE static float compute(
        const CentroidTable& table, std::span<const float> a, std::span<const float> b
    ) {
        thread_local gem_detail::MergeScratch scratch_a;
        thread_local gem_detail::MergeScratch scratch_b;
        size_t ua = gem_detail::merge_duplicates(
            gem_detail::set_ids(a), gem_detail::set_size(a), scratch_a
        );
        size_t ub = gem_detail::merge_duplicates(
            gem_detail::set_ids(b), gem_detail::set_size(b), scratch_b
        );
        return compute_decoded(
            table,
            scratch_a.ids.data(),
            scratch_a.weights.data(),
            ua,
            scratch_b.ids.data(),
            scratch_b.weights.data(),
            ub
        );
    }
};

// Dispatched like QEmdImpl. compute_decoded takes already-decoded (cast, unmerged) centroid
// id arrays so gem_time can benchmark the min-sum loop separately from the span decode.
template <AVX_AVAILABILITY Avx> struct QChamferImpl {
    SVS_NOINLINE static float compute_decoded(
        const CentroidTable& table,
        const uint32_t* q,
        size_t nq,
        const uint32_t* d,
        size_t nd
    ) {
        double total = 0.0;
        for (size_t i = 0; i < nq; ++i) {
            float best = std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < nd; ++j) {
                best = std::min(best, table(q[i], d[j]));
            }
            total += best;
        }
        return static_cast<float>(total);
    }

    SVS_NOINLINE static float compute(
        const CentroidTable& table, std::span<const float> query, std::span<const float> doc
    ) {
        thread_local std::vector<uint32_t> qbuf;
        thread_local std::vector<uint32_t> dbuf;
        gem_detail::decode_ids(
            gem_detail::set_ids(query), gem_detail::set_size(query), qbuf
        );
        gem_detail::decode_ids(gem_detail::set_ids(doc), gem_detail::set_size(doc), dbuf);
        return compute_decoded(table, qbuf.data(), qbuf.size(), dbuf.data(), dbuf.size());
    }
};

// Dispatched like QChamferImpl. GEM never quantizes the query token side (src/search/
// routing.cpp:55-61, src/distance/maxsim.cpp:9-45): build_table scores the raw float
// query tokens against every fine centroid (score_fine_centers), and compute_decoded
// reduces that table against the document's quantized centroid ids (code_distance).
template <AVX_AVAILABILITY Avx> struct QChamferExactImpl {
    // table[f * n_query + i] = dot(centroid_f, query_token_i), matching maxsim.cpp's
    // Eigen::Map<MatrixXf>(table, query_tokens, fine_centers) column-major read of the
    // same buffer (scores.col(f)[i] == table[f*n_query+i]).
    SVS_NOINLINE static void build_table(
        const float* centroids,
        size_t fine_centers,
        size_t dim,
        const float* query_tokens,
        size_t n_query,
        float* table
    ) {
        for (size_t f = 0; f < fine_centers; ++f) {
            const float* c = centroids + f * dim;
            float* row = table + f * n_query;
            for (size_t i = 0; i < n_query; ++i) {
                const float* t = query_tokens + i * dim;
                float dot = 0.0F;
                for (size_t d = 0; d < dim; ++d) {
                    dot += c[d] * t[d];
                }
                row[i] = dot;
            }
        }
    }

    // GEM's code_distance (src/distance/maxsim.cpp:9-45): for each query token, the max
    // dot product to any of the document's (possibly repeated) centroid ids; final score
    // is `1 - mean_q max_doc dot`, i.e. 1 minus the mean of those per-token maxima.
    SVS_NOINLINE static float compute_decoded(
        const float* table, size_t n_query, const uint32_t* doc_ids, size_t n_doc
    ) {
        return gem_detail::qchamfer_exact_reduce(table, n_query, doc_ids, n_doc);
    }
};

#if defined(__x86_64__)

// build_table vectorizes over query tokens, not over `dim`: per output lane i,
// the d=0..dim-1 FMA order matches the scalar version exactly, so only the
// single-rounding FMA (vs. separate mul+add) can move the result off the scalar
// value, well under the 1e-6 relative tolerance this table is checked against.
// Transposing query_tokens into [dim][n_query] turns each centroid row into
// contiguous per-token accumulators with no horizontal reduction, at the cost of
// a one-time O(dim * n_query) transpose (query is a handful of KiB; centroids
// are tens to hundreds of MiB, so the transpose is noise against the table build
// it replaces the scalar dot loop in).
// Declared here, defined in avx512.cpp/avx2.cpp: the bodies use __m512/__m256 intrinsics,
// so each must only be compiled under its own -march flags, not wherever gem.h is included
// (e.g. gem_time.cpp, which carries neither). Defining them in-class would make them
// implicitly inline, and an explicit instantiation of an already-specialized class is a
// no-op in GCC, so no other TU would get a symbol to link against.
SVS_VALIDATE_BOOL_ENV(SVS_AVX512_F)
#if SVS_AVX512_F
template <> struct QChamferExactImpl<AVX_AVAILABILITY::AVX512> {
    SVS_NOINLINE static void build_table(
        const float* centroids,
        size_t fine_centers,
        size_t dim,
        const float* query_tokens,
        size_t n_query,
        float* table
    );
    SVS_NOINLINE static float compute_decoded(
        const float* table, size_t n_query, const uint32_t* doc_ids, size_t n_doc
    );
};
#endif

SVS_VALIDATE_BOOL_ENV(SVS_AVX512_F)
SVS_VALIDATE_BOOL_ENV(SVS_AVX2)
#if !SVS_AVX512_F && SVS_AVX2
template <> struct QChamferExactImpl<AVX_AVAILABILITY::AVX2> {
    SVS_NOINLINE static void build_table(
        const float* centroids,
        size_t fine_centers,
        size_t dim,
        const float* query_tokens,
        size_t n_query,
        float* table
    );
    SVS_NOINLINE static float compute_decoded(
        const float* table, size_t n_query, const uint32_t* doc_ids, size_t n_doc
    );
};
#endif

#endif // defined(__x86_64__)

#if defined(__x86_64__)
extern template struct QEmdImpl<AVX_AVAILABILITY::AVX512>;
extern template struct QEmdImpl<AVX_AVAILABILITY::AVX2>;
extern template struct QChamferImpl<AVX_AVAILABILITY::AVX512>;
extern template struct QChamferImpl<AVX_AVAILABILITY::AVX2>;
extern template struct QChamferExactImpl<AVX_AVAILABILITY::AVX512>;
extern template struct QChamferExactImpl<AVX_AVAILABILITY::AVX2>;
#endif

/// Reports which ISA path QEmd/QChamfer dispatch to on this process, for diagnostics.
inline std::string_view gem_dispatch_isa() {
    if (svs::detail::avx_runtime_flags.is_avx512f_supported()) {
        return "avx512";
    }
    if (svs::detail::avx_runtime_flags.is_avx2_supported()) {
        return "avx2";
    }
    return "none";
}

/// Build-time distance: quantized Earth Mover's Distance (qEMD) between two token sets,
/// solved exactly as a balanced transportation problem over merged centroid weights.
/// Smaller is better.
class QEmd {
  public:
    using compare = std::less<>;

    static constexpr bool implicit_broadcast = true;
    static constexpr std::string_view name = "gem_qemd";

    explicit QEmd(std::shared_ptr<const CentroidTable> table)
        : table_{std::move(table)} {}

    template <size_t Ea, size_t Eb>
    float compute(std::span<const float, Ea> a, std::span<const float, Eb> b) const {
        std::span<const float> da{a.data(), a.size()};
        std::span<const float> db{b.data(), b.size()};
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx512f_supported(), 1)) {
            return QEmdImpl<AVX_AVAILABILITY::AVX512>::compute(*table_, da, db);
        }
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx2_supported(), 1)) {
            return QEmdImpl<AVX_AVAILABILITY::AVX2>::compute(*table_, da, db);
        }
        return QEmdImpl<AVX_AVAILABILITY::NONE>::compute(*table_, da, db);
    }

  private:
    std::shared_ptr<const CentroidTable> table_;
};

inline constexpr bool operator==(const QEmd&, const QEmd&) { return true; }

/// Search-time distance: quantized Chamfer distance (qCH, GEM Eq. 16). For every query
/// token (duplicates counted), takes the min centroid distance to any doc token, and
/// sums. Smaller is better. The query is the left-hand argument.
class QChamfer {
  public:
    using compare = std::less<>;
    static constexpr bool implicit_broadcast = true;
    static constexpr std::string_view name = "gem_qchamfer";

    explicit QChamfer(std::shared_ptr<const CentroidTable> table)
        : table_{std::move(table)} {}

    template <size_t Eq, size_t Ed>
    float compute(std::span<const float, Eq> query, std::span<const float, Ed> doc) const {
        std::span<const float> dq{query.data(), query.size()};
        std::span<const float> dd{doc.data(), doc.size()};
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx512f_supported(), 1)) {
            return QChamferImpl<AVX_AVAILABILITY::AVX512>::compute(*table_, dq, dd);
        }
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx2_supported(), 1)) {
            return QChamferImpl<AVX_AVAILABILITY::AVX2>::compute(*table_, dq, dd);
        }
        return QChamferImpl<AVX_AVAILABILITY::NONE>::compute(*table_, dq, dd);
    }

  private:
    std::shared_ptr<const CentroidTable> table_;
};

inline constexpr bool operator==(const QChamfer&, const QChamfer&) { return true; }

/// Search-time distance: GEM's actual in-graph qCH (GEM_REFERENCE.md "search-time distance:
/// is the query quantized?" row). Unlike QChamfer, the query is never quantized: it stays
/// as raw float tokens, and fix_argument() builds a dense (K x query_tokens) dot table
/// against every fine centroid once per query (score_fine_centers, src/search/
/// routing.cpp:55-61); compute() then reduces that table against the document's quantized
/// centroid ids (code_distance, src/distance/maxsim.cpp:9-45). Smaller is better.
///
/// Holds a per-query table as ordinary (non-shared) state, so each search thread's
/// shallow_copy of this functor gets its own table; implicit_broadcast is false and
/// must_fix_argument is true to force a fix_argument() call before any compute().
class QChamferExactQuery {
  public:
    using compare = std::less<>;
    static constexpr bool implicit_broadcast = false;
    static constexpr bool must_fix_argument = true;
    static constexpr std::string_view name = "gem_qchamfer_exact_query";

    explicit QChamferExactQuery(std::shared_ptr<const FineCentroids> centroids)
        : centroids_{std::move(centroids)} {}

    // Raw-query span layout (gem_detail::vector_set_size/vector_set_tokens): [0] = token
    // count n, [1..1+n*dim) = n raw fp32 tokens row-major, rest padding.
    template <size_t E> void fix_argument(std::span<const float, E> query) {
        std::span<const float> dq{query.data(), query.size()};
        n_query_ = gem_detail::vector_set_size(dq);
        const float* tokens = gem_detail::vector_set_tokens(dq);
        table_.resize(centroids_->K * n_query_);
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx512f_supported(), 1)) {
            QChamferExactImpl<AVX_AVAILABILITY::AVX512>::build_table(
                centroids_->data.get(),
                centroids_->K,
                centroids_->dim,
                tokens,
                n_query_,
                table_.data()
            );
        } else if (__builtin_expect(
                       svs::detail::avx_runtime_flags.is_avx2_supported(), 1
                   )) {
            QChamferExactImpl<AVX_AVAILABILITY::AVX2>::build_table(
                centroids_->data.get(),
                centroids_->K,
                centroids_->dim,
                tokens,
                n_query_,
                table_.data()
            );
        } else {
            QChamferExactImpl<AVX_AVAILABILITY::NONE>::build_table(
                centroids_->data.get(),
                centroids_->K,
                centroids_->dim,
                tokens,
                n_query_,
                table_.data()
            );
        }
    }

    // `query` is unused: fix_argument() already consumed it into table_/n_query_, and
    // every compute() call in a search shares that one query's table.
    template <size_t Eq, size_t Ed>
    float
    compute(std::span<const float, Eq> /*query*/, std::span<const float, Ed> doc) const {
        std::span<const float> dd{doc.data(), doc.size()};
        thread_local std::vector<uint32_t> dbuf;
        gem_detail::decode_ids(gem_detail::set_ids(dd), gem_detail::set_size(dd), dbuf);
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx512f_supported(), 1)) {
            return QChamferExactImpl<AVX_AVAILABILITY::AVX512>::compute_decoded(
                table_.data(), n_query_, dbuf.data(), dbuf.size()
            );
        }
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx2_supported(), 1)) {
            return QChamferExactImpl<AVX_AVAILABILITY::AVX2>::compute_decoded(
                table_.data(), n_query_, dbuf.data(), dbuf.size()
            );
        }
        return QChamferExactImpl<AVX_AVAILABILITY::NONE>::compute_decoded(
            table_.data(), n_query_, dbuf.data(), dbuf.size()
        );
    }

  private:
    std::shared_ptr<const FineCentroids> centroids_;
    std::vector<float> table_;
    size_t n_query_ = 0;
};

inline constexpr bool operator==(const QChamferExactQuery&, const QChamferExactQuery&) {
    return true;
}

} // namespace svs::distance
