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

#include "svs/concepts/vector_set.h"
#include "svs/core/data/simple.h"
#include "svs/core/distance/distance_core.h"
#include "svs/core/distance/gem_router.h"
#include "svs/core/distance/inner_product_matrix.h"
#include "svs/core/distance/transport_simplex.h"
#include "svs/lib/avx_detection.h"
#include "svs/lib/bfloat16.h"
#include "svs/lib/misc.h"
#include "svs/lib/preprocessor.h"
#include "svs/lib/threads.h"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace svs::distance {

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
/// Backed by the standard allocator (`svs::lib::Allocator`): a GEM runtime process mmaps
/// many segments concurrently, and `svs::HugepageAllocator::deallocate` throws on munmap
/// failure (e.g. ENOMEM against vm.max_map_count) from inside a destructor, which
/// terminates the process. Random lookups into a large table are still TLB/cache-miss
/// bound without hugepages, but that cost is preferred over a process abort.
struct CentroidTable {
    size_t K;
    svs::data::SimpleData<float, svs::Dynamic, svs::lib::Allocator<float>> data;
    // True only for a table from_centroids() built; see the comment where it is set.
    bool symmetric = false;

    CentroidTable(size_t k, std::vector<float> table)
        : K{k}
        , data{k, k, svs::lib::Allocator<float>()} {
        if (table.size() != K * K) {
            throw std::invalid_argument("CentroidTable: data size does not match K*K");
        }
        std::copy(table.begin(), table.end(), data.data());
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
            reinterpret_cast<char*>(result.data.data()),
            static_cast<std::streamsize>(expected_bytes)
        );
        if (!in) {
            throw std::runtime_error("CentroidTable::load: short read on " + path.string());
        }
        return result;
    }

    /// Computes T[i][j] = ground(bf16(c_i), bf16(c_j)) from K row-major `dim`-wide
    /// centroids, T[i][i] = 0. `centroids` is float32 (the on-disk/in-memory format is
    /// unchanged); each tile rounds its own row and column block to bfloat16 (round to
    /// nearest, ties to even, matching FineCentroids) before computing, so this table and
    /// FineCentroids agree on the same rounded centroid values. Symmetric by
    /// construction: each unordered pair {i, j} is computed once, by the tile that owns
    /// it, and the same float is stored at T[i][j] and T[j][i] (never two
    /// independently-computed dot products); that holds regardless of which operand role
    /// bf16-rounding was applied to, since both row and column blocks are rounded
    /// identically before either is used. Parallelized over tiles on `pool`, dynamically
    /// scheduled so no two tiles ever write the same cell; the result is bit-for-bit
    /// identical regardless of the pool's size. Uses an uninitialized allocation (see the
    /// private constructor below) and writes every cell exactly once (off-diagonal tiles)
    /// or twice with the same value (diagonal tiles' upper/lower
    /// halves), so no thread does a sequential full-matrix pass.
    static CentroidTable from_centroids(
        const float* centroids, size_t k, size_t dim, Ground ground, size_t num_threads
    ) {
        svs::threads::DefaultThreadPool pool(num_threads);
        return from_centroids(centroids, k, dim, ground, pool);
    }

    template <svs::threads::ThreadPool Pool>
    static CentroidTable from_centroids(
        const float* centroids, size_t k, size_t dim, Ground ground, Pool& pool
    ) {
        CentroidTable result{k, uninitialized_tag{}};
        float* out = result.data.data();

        // Tile edge: an operand block (tile * dim * 4 bytes, 128 KiB at dim=128) and the
        // tile buffer (tile * tile * 4 bytes, 256 KiB) together stay well inside a
        // typical >=1 MiB per-core L2.
        constexpr size_t tile = 256;
        // Tiles per dynamic-partition steal: a few tiles amortize the atomic-counter
        // fetch_add without starving the small tiles near the far column blocks.
        constexpr size_t grain = 4;

        size_t num_blocks = svs::lib::div_round_up(k, tile);

        // Upper triangle of blocks, diagonal included: tile (bi, bj), bi <= bj, owns
        // every unordered pair whose row falls in block bi and column in block bj. Built
        // once so the uneven per-row-block tile count (row block 0 has num_blocks tiles,
        // the last has one) can be load-balanced by a dynamic partition over this list
        // rather than a static partition over rows.
        struct BlockPair {
            uint32_t bi;
            uint32_t bj;
        };
        std::vector<BlockPair> tiles;
        tiles.reserve(num_blocks * (num_blocks + 1) / 2);
        for (uint32_t bi = 0; bi < num_blocks; ++bi) {
            for (uint32_t bj = bi; bj < num_blocks; ++bj) {
                tiles.push_back({bi, bj});
            }
        }

        auto ground_cost = [ground](float dot) {
            return ground == Ground::Cos ? std::max(0.0F, 1.0F - dot)
                                         : std::sqrt(std::max(0.0F, 2.0F - 2.0F * dot));
        };

        svs::threads::parallel_for(
            pool,
            svs::threads::DynamicPartition{tiles.size(), grain},
            [&](const auto& tile_ids, uint64_t /*tid*/) {
                // Reused across this thread's tiles: `buf` holds the tile output,
                // `rows_bf16` the row block rounded to bfloat16, `cols_packed` the column
                // block packed and rounded by the same bfloat16 rounding. Both operands end
                // up rounded to the same precision this table is meant to agree with
                // FineCentroids on, so the whole method sees one (bf16-rounded) set of
                // centroid values.
                thread_local std::vector<float> buf;
                thread_local std::vector<svs::BFloat16> rows_bf16;
                thread_local PackedOperand cols_packed;
                buf.resize(tile * tile);
                for (size_t t : tile_ids) {
                    auto [bi, bj] = tiles[t];
                    size_t row0 = static_cast<size_t>(bi) * tile;
                    size_t col0 = static_cast<size_t>(bj) * tile;
                    size_t rows = std::min(tile, k - row0);
                    size_t cols = std::min(tile, k - col0);
                    rows_bf16.resize(rows * dim);
                    for (size_t idx = 0; idx < rows * dim; ++idx) {
                        rows_bf16[idx] =
                            svs::round_to_bfloat16(centroids[row0 * dim + idx]);
                    }
                    cols_packed.pack_rounded_to_bf16(centroids + col0 * dim, cols, dim);
                    InnerProductMatrix::compute_packed(
                        rows_bf16.data(), rows, cols_packed, buf.data()
                    );
                    bool diag = (bi == bj);
                    for (size_t r = 0; r < rows; ++r) {
                        size_t i = row0 + r;
                        // Off-diagonal tiles (row and column blocks disjoint) own every
                        // (r, c); diagonal tiles start at c == r to skip the lower half,
                        // already owned by this same tile's (r, c) with roles swapped.
                        size_t c_start = diag ? r : 0;
                        for (size_t c = c_start; c < cols; ++c) {
                            size_t j = col0 + c;
                            if (diag && c == r) {
                                out[i * k + j] = 0.0F;
                                continue;
                            }
                            float value = ground_cost(buf[r * cols + c]);
                            out[i * k + j] = value;
                            out[j * k + i] = value;
                        }
                    }
                }
            }
        );
        // Every unordered pair is written once and copied to both halves above, so
        // table(i,j) == table(j,i) bit-for-bit; QEmdImpl's swapped transposed-solve
        // lookup (gem.h) is only exact because of that, not a general table property.
        result.symmetric = true;
        return result;
    }

    float operator()(uint32_t i, uint32_t j) const {
        return data.data()[static_cast<size_t>(i) * K + j];
    }

  private:
    struct uninitialized_tag {};

    // SimpleData's allocator-backed ctor default-constructs each element in place; for
    // a trivial float that writes nothing, matching make_unique_for_overwrite's intent.
    CentroidTable(size_t k, uninitialized_tag)
        : K{k}
        , data{k, k, svs::lib::Allocator<float>()} {}
};

/// Owns the raw K x dim fine-centroid vectors (row-major, not a pairwise table), as
/// bfloat16: every value is the float32 input rounded to nearest, ties to even (matching
/// `svs::round_to_bfloat16`, not `BFloat16`'s own truncating conversion). One
/// representation, no float32 copy kept once construction/`load` returns, so every ISA
/// level (including a future one that multiplies bf16 natively) computes on identical
/// values. Used by QChamferExactQuery to build a per-query (K x query_tokens) dot table
/// against the exact float query, mirroring GEM's score_fine_centers
/// (src/search/routing.cpp:55-61), which never quantizes the query side.
struct FineCentroids {
    size_t K;
    size_t dim;
    std::unique_ptr<svs::BFloat16[]> data;

    /// Rounds `flat` (K rows of `dim_` float32 each, row-major; the on-disk/in-memory
    /// input format is unchanged) to bfloat16, in parallel over row blocks on
    /// `num_threads` threads. Peak memory is `flat`'s own K * dim_ * 4 bytes plus this
    /// object's K * dim_ * 2 bytes: no second float32 buffer is allocated here.
    FineCentroids(size_t k, size_t dim_, const float* flat, size_t num_threads = 1)
        : FineCentroids{k, dim_, uninitialized_tag{}} {
        svs::threads::DefaultThreadPool pool(num_threads);
        round_rows(flat, data.get(), k * dim_, pool);
    }

    template <svs::threads::ThreadPool Pool>
    FineCentroids(size_t k, size_t dim_, const float* flat, Pool& pool)
        : FineCentroids{k, dim_, uninitialized_tag{}} {
        round_rows(flat, data.get(), k * dim_, pool);
    }

    // Raw file of K rows of `dim` floats with no header, so K comes from the file size.
    // Reads and rounds in row-block chunks rather than slurping the whole file into a
    // second float32 buffer: peak memory is this object's K * dim * 2 bytes (the bf16
    // result, allocated once, uninitialized) plus one chunk's
    // min(K, kLoadChunkRows) * dim * 4 bytes (4 MiB at dim=128), not the full K * dim
    // float32 matrix held alongside the bf16 one.
    static FineCentroids
    load(const std::filesystem::path& path, size_t dim, size_t num_threads = 1) {
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
        in.seekg(0, std::ios::beg);

        FineCentroids result{k, dim, uninitialized_tag{}};
        constexpr size_t kLoadChunkRows = 8192;
        std::vector<float> chunk(std::min(k, kLoadChunkRows) * dim);
        for (size_t row0 = 0; row0 < k; row0 += kLoadChunkRows) {
            size_t rows = std::min(kLoadChunkRows, k - row0);
            size_t count = rows * dim;
            in.read(
                reinterpret_cast<char*>(chunk.data()),
                static_cast<std::streamsize>(count * sizeof(float))
            );
            if (!in) {
                throw std::runtime_error(
                    "FineCentroids::load: short read on " + path.string()
                );
            }
            round_rows(chunk.data(), result.data.get() + row0 * dim, count, num_threads);
        }
        return result;
    }

  private:
    struct uninitialized_tag {};

    // new[] with no parens default-initializes BFloat16 (indeterminate value, matching
    // CentroidTable's uninitialized constructor); every element is overwritten by
    // round_rows before this object becomes visible to a caller.
    FineCentroids(size_t k, size_t dim_, uninitialized_tag)
        : K{k}
        , dim{dim_}
        , data{std::make_unique_for_overwrite<svs::BFloat16[]>(k * dim_)} {}

    // Rounds `count` contiguous float32 values to bfloat16, statically partitioned
    // across the threads of `pool`; shared by the constructors and the per-chunk path in
    // `load` so there is exactly one place that performs this conversion.
    template <svs::threads::ThreadPool Pool>
    static void
    round_rows(const float* flat, svs::BFloat16* out, size_t count, Pool& pool) {
        svs::threads::parallel_for(
            pool,
            svs::threads::StaticPartition{count},
            [&](const auto& ids, uint64_t /*tid*/) {
                for (size_t i : ids) {
                    out[i] = svs::round_to_bfloat16(flat[i]);
                }
            }
        );
    }

    static void
    round_rows(const float* flat, svs::BFloat16* out, size_t count, size_t num_threads) {
        svs::threads::DefaultThreadPool pool(num_threads);
        round_rows(flat, out, count, pool);
    }
};

namespace gem_detail {

// Thread-local because the functors are shared across threads (implicit_broadcast).
struct MergeScratch {
    std::vector<uint32_t> ids;
    std::vector<double> weights;
};

// Casts a vector set's centroid ids to uint32_t with no merge; qCH counts duplicates.
// Only instantiated for element types other than uint32_t (id_span's cast-copy branch).
template <typename T>
inline void decode_ids(const T* ids, size_t n, std::vector<uint32_t>& out) {
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<uint32_t>(ids[i]);
    }
}

// Returns a contiguous uint32_t id span for a 1-D id vector set with element type T: its
// own storage when T is already uint32_t (no copy), else a cast copy into `scratch`.
template <typename T>
inline const uint32_t* id_span(const T* ids, size_t n, std::vector<uint32_t>& scratch) {
    if constexpr (std::is_same_v<T, uint32_t>) {
        return ids;
    } else {
        decode_ids(ids, n, scratch);
        return scratch.data();
    }
}

// An id vector set stores one centroid id per token, so dimensions() must be 1. A
// concept-constrained vector set has no static Extent to check at compile time, so this
// is the vector-set path's only dimension check.
inline void require_id_dimensions(size_t dims) {
    if (dims != 1) {
        throw std::invalid_argument(
            "id vector set must have dimensions() == 1 (one centroid id per "
            "token), got " +
            std::to_string(dims)
        );
    }
}

} // namespace gem_detail

// Instantiated per ISA in svs/multi-arch/x86; without SVS_NOINLINE the caller inlines the
// body at baseline flags. compute_decoded's pricing loop is explicitly vectorized per Avx
// (TransportSimplex); compute_decoded takes already-decoded (deduplicated, weighted)
// centroid ids, keeping the merge and the transportation solve as separately testable
// steps.
template <AVX_AVAILABILITY Avx> struct QEmdImpl {
    SVS_NOINLINE static float compute_decoded(
        const CentroidTable& table,
        const uint32_t* ids_a,
        const double* w_a,
        size_t ua,
        const uint32_t* ids_b,
        const double* w_b,
        size_t ub,
        TransportStart start
    ) {
        if (ua == 1 && ub == 1) {
            return table(ids_a[0], ids_b[0]);
        }

        // merge_sorted_duplicates divides integer counts by n in double, so w_a/w_b each
        // sum to 1.0 up to a few ULPs rather than exactly; TransportSimplex::solve folds
        // that residual into the last demand entry instead of requiring exact balance.
        if (table.symmetric && ua > ub) {
            // TransportSimplex transposes and walks ids_a in its inner loop when ua >
            // ub; swapping keeps the row (ids_b) fixed there. If that transpose rule
            // changes, results stay correct but this locality win is lost.
            auto cost_fn = [&](size_t i, size_t j) { return table(ids_b[j], ids_a[i]); };
            double cost = TransportSimplex<Avx>::solve(ua, w_a, ub, w_b, cost_fn, start);
            return static_cast<float>(cost);
        }
        auto cost_fn = [&](size_t i, size_t j) { return table(ids_a[i], ids_b[j]); };
        double cost = TransportSimplex<Avx>::solve(ua, w_a, ub, w_b, cost_fn, start);
        return static_cast<float>(cost);
    }

    // Decodes both ids-sets' own uint32_t storage, merges each set's duplicates into
    // weighted unique ids, then solves the transportation problem over the result.
    SVS_NOINLINE static float compute(
        const CentroidTable& table,
        const uint32_t* ids_a,
        size_t na,
        const uint32_t* ids_b,
        size_t nb,
        TransportStart start
    ) {
        thread_local gem_detail::MergeScratch scratch_a;
        thread_local gem_detail::MergeScratch scratch_b;
        size_t ua = merge_duplicates(ids_a, na, scratch_a);
        size_t ub = merge_duplicates(ids_b, nb, scratch_b);
        return compute_decoded(
            table,
            scratch_a.ids.data(),
            scratch_a.weights.data(),
            ua,
            scratch_b.ids.data(),
            scratch_b.weights.data(),
            ub,
            start
        );
    }

  private:
    // Overloads, not a free function template: QEmdImpl<Avx> is explicitly instantiated
    // per ISA in avx2.cpp/avx512.cpp, and a free template's inline symbol is shared
    // (and ODR-violated under different -m flags) across those translation units.
    static size_t merge_sorted_duplicates(size_t n, gem_detail::MergeScratch& scratch) {
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

    static size_t
    merge_duplicates(const uint32_t* ids, size_t n, gem_detail::MergeScratch& scratch) {
        scratch.ids.assign(ids, ids + n);
        return merge_sorted_duplicates(n, scratch);
    }
};

// Dispatched like QEmdImpl. compute_decoded takes centroid id arrays already cast to
// uint32_t from the vector set's own storage by QChamfer::compute.
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
};

// GEM never quantizes query tokens: QChamferExactQuery scores them via InnerProductMatrix.
// compute_decoded reduces that table; no intrinsics, but still instantiated per ISA (not
// specialized) like QEmdImpl, so no weak inline symbol is shared across -m flags.
template <AVX_AVAILABILITY Avx> struct QChamferExactImpl {
    // GEM's code_distance (src/distance/maxsim.cpp:9-45): for each query token, the max
    // dot product to any of the document's (possibly repeated) centroid ids; final score
    // is `1 - mean_q max_doc dot`, i.e. 1 minus the mean of those per-token maxima.
    SVS_NOINLINE static float compute_decoded(
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
};

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
///
/// Both arguments are id vector sets (`svs::data::ContiguousVectorSet` with an unsigned
/// integer element type and `dimensions() == 1`): one centroid id per token, each id in
/// `[0, table.K)`. Holds a shared, read-only `CentroidTable` the caller owns and keeps
/// alive for the lifetime of this object.
class QEmd {
  public:
    using compare = std::less<>;

    static constexpr bool implicit_broadcast = true;
    static constexpr std::string_view name = "gem_qemd";

    explicit QEmd(
        std::shared_ptr<const CentroidTable> table,
        std::shared_ptr<const GemRouter> router = nullptr,
        TransportStart start = TransportStart::NorthWest
    )
        : table_{std::move(table)}
        , router_{std::move(router)}
        , start_{start} {}

    const std::shared_ptr<const GemRouter>& router() const { return router_; }

    /// Transport-simplex start used by `compute`. Not saved with an index: it changes the
    /// pivot count, not the optimum.
    TransportStart start() const { return start_; }

    // Vector-set path: ids come straight from a contiguous vector set's own storage; T ==
    // uint32_t skips the cast copy (gem_detail::id_span), any other unsigned type costs it.
    template <svs::data::ContiguousVectorSet Sa, svs::data::ContiguousVectorSet Sb>
        requires std::unsigned_integral<typename Sa::element_type> &&
                 std::unsigned_integral<typename Sb::element_type>
    float compute(const Sa& a, const Sb& b) const {
        gem_detail::require_id_dimensions(a.dimensions());
        gem_detail::require_id_dimensions(b.dimensions());
        thread_local std::vector<uint32_t> buf_a;
        thread_local std::vector<uint32_t> buf_b;
        const uint32_t* ids_a = gem_detail::id_span(a.data(), a.cardinality(), buf_a);
        const uint32_t* ids_b = gem_detail::id_span(b.data(), b.cardinality(), buf_b);
        size_t na = a.cardinality();
        size_t nb = b.cardinality();
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx512f_supported(), 1)) {
            return QEmdImpl<AVX_AVAILABILITY::AVX512>::compute(
                *table_, ids_a, na, ids_b, nb, start_
            );
        }
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx2_supported(), 1)) {
            return QEmdImpl<AVX_AVAILABILITY::AVX2>::compute(
                *table_, ids_a, na, ids_b, nb, start_
            );
        }
        return QEmdImpl<AVX_AVAILABILITY::NONE>::compute(
            *table_, ids_a, na, ids_b, nb, start_
        );
    }

  private:
    std::shared_ptr<const CentroidTable> table_;
    std::shared_ptr<const GemRouter> router_;
    TransportStart start_;
};

inline constexpr bool operator==(const QEmd&, const QEmd&) { return true; }

/// Search-time distance: quantized Chamfer distance (qCH, GEM Eq. 16). For every query
/// token (duplicates counted), takes the min centroid distance to any doc token, and
/// sums. Smaller is better. The query is the left-hand argument.
///
/// Both arguments are id vector sets (`svs::data::ContiguousVectorSet` with an unsigned
/// integer element type and `dimensions() == 1`): one centroid id per token, each id in
/// `[0, table.K)`. Holds a shared, read-only `CentroidTable` the caller owns and keeps
/// alive for the lifetime of this object.
class QChamfer {
  public:
    using compare = std::less<>;
    static constexpr bool implicit_broadcast = true;
    static constexpr std::string_view name = "gem_qchamfer";

    explicit QChamfer(std::shared_ptr<const CentroidTable> table)
        : table_{std::move(table)} {}

    // Vector-set path: compute_decoded's signature already matches an id vector set's
    // own (ids pointer, cardinality) storage, so T == uint32_t needs no decode copy.
    template <svs::data::ContiguousVectorSet Sq, svs::data::ContiguousVectorSet Sd>
        requires std::unsigned_integral<typename Sq::element_type> &&
                 std::unsigned_integral<typename Sd::element_type>
    float compute(const Sq& query, const Sd& doc) const {
        gem_detail::require_id_dimensions(query.dimensions());
        gem_detail::require_id_dimensions(doc.dimensions());
        thread_local std::vector<uint32_t> buf_q;
        thread_local std::vector<uint32_t> buf_d;
        const uint32_t* q_ids =
            gem_detail::id_span(query.data(), query.cardinality(), buf_q);
        const uint32_t* d_ids = gem_detail::id_span(doc.data(), doc.cardinality(), buf_d);
        size_t nq = query.cardinality();
        size_t nd = doc.cardinality();
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx512f_supported(), 1)) {
            return QChamferImpl<AVX_AVAILABILITY::AVX512>::compute_decoded(
                *table_, q_ids, nq, d_ids, nd
            );
        }
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx2_supported(), 1)) {
            return QChamferImpl<AVX_AVAILABILITY::AVX2>::compute_decoded(
                *table_, q_ids, nq, d_ids, nd
            );
        }
        return QChamferImpl<AVX_AVAILABILITY::NONE>::compute_decoded(
            *table_, q_ids, nq, d_ids, nd
        );
    }

  private:
    std::shared_ptr<const CentroidTable> table_;
};

inline constexpr bool operator==(const QChamfer&, const QChamfer&) { return true; }

/// Search-time distance: GEM's actual in-graph qCH (GEM_REFERENCE.md "search-time distance:
/// is the query quantized?" row). Unlike QChamfer, the query is never id-quantized: it
/// stays as raw float tokens, and fix_argument() builds a dense (K x query_tokens) dot
/// table against every fine centroid once per query (score_fine_centers, src/search/
/// routing.cpp:55-61), both sides rounded to bfloat16 so the table's values do not depend
/// on which ISA level computed them (see FineCentroids); compute() then reduces that
/// table against the document's quantized centroid ids (code_distance,
/// src/distance/maxsim.cpp:9-45). Smaller is better.
///
/// Holds a per-query table as ordinary (non-shared) state, so each search thread's
/// shallow_copy of this functor gets its own table; implicit_broadcast is false and
/// must_fix_argument is true to force a fix_argument() call before any compute().
///
/// The query (fix_argument's argument) is a vector set of raw (unquantized) fp32 tokens
/// with dimensions() equal to the centroid dimension. The doc (compute's second argument)
/// is an id vector set (unsigned integer element type, dimensions() == 1): one centroid
/// id per token, each id in [0, K). compute's query argument is unused; fix_argument
/// already consumed the query into this object's table.
class QChamferExactQuery {
  public:
    using compare = std::less<>;
    static constexpr bool implicit_broadcast = false;
    static constexpr bool must_fix_argument = true;
    static constexpr std::string_view name = "gem_qchamfer_exact_query";

    explicit QChamferExactQuery(
        std::shared_ptr<const FineCentroids> centroids,
        std::shared_ptr<const GemRouter> router = nullptr
    )
        : centroids_{std::move(centroids)}
        , router_{std::move(router)} {
        if (router_ && router_->num_fine() != centroids_->K) {
            throw std::invalid_argument(
                "QChamferExactQuery: router has " + std::to_string(router_->num_fine()) +
                " fine centroids, centroids has K = " + std::to_string(centroids_->K)
            );
        }
    }

    const std::shared_ptr<const GemRouter>& router() const { return router_; }

    /// Sets the per-search `nprobe` override on this object. Call this only on the
    /// shallow_copy a search already owns (e.g. scratchspace's distance copy): every
    /// concurrent search gets its own copy, so this is never shared mutable state.
    void set_nprobe_override(size_t nprobe) { nprobe_override_ = nprobe; }

    /// Current per-search `nprobe` override passed to `GemRouter::route_query` by
    /// `GemRoutedInitializer`; `GemRouter::kUseStoredNprobe` means "use the router's own".
    size_t nprobe_override() const { return nprobe_override_; }

    /// The per-query (K x n_query) dot table built by the last fix_argument() call.
    const float* table() const { return table_.data(); }

    /// Number of query tokens the last fix_argument() call built table() for.
    size_t query_tokens() const { return n_query_; }

    // Vector-set path: a query set of raw (unquantized) fp32 token vectors, contiguous
    // in the set's own storage, with dimensions() equal to the centroid dimension.
    template <svs::data::ContiguousVectorSet S>
        requires std::same_as<typename S::element_type, float>
    void fix_argument(const S& query) {
        if (query.dimensions() != centroids_->dim) {
            throw std::invalid_argument(
                "QChamferExactQuery::fix_argument: query dimension " +
                std::to_string(query.dimensions()) + " does not match centroid dimension " +
                std::to_string(centroids_->dim)
            );
        }
        fix_argument_decoded(query.data(), query.cardinality());
    }

    // `query` is unused: fix_argument() already consumed it into table_/n_query_, and
    // every compute() call in a search shares that one query's table. `doc` is an id
    // vector set, decoded with no copy when its element type is uint32_t.
    template <typename Query, svs::data::ContiguousVectorSet Sd>
        requires std::unsigned_integral<typename Sd::element_type>
    float compute(const Query& /*query*/, const Sd& doc) const {
        gem_detail::require_id_dimensions(doc.dimensions());
        thread_local std::vector<uint32_t> buf_d;
        const uint32_t* doc_ids = gem_detail::id_span(doc.data(), doc.cardinality(), buf_d);
        return compute_decoded(doc_ids, doc.cardinality());
    }

  private:
    // Builds the (K x n_query) dot table once per query: table[f * n_query + i] =
    // dot(bf16(centroid_f), bf16(query_token_i)), matching maxsim.cpp's
    // Eigen::Map<MatrixXf>(table, query_tokens, fine_centers) column-major read of the
    // same buffer (scores.col(f)[i] == table[f*n_query+i]). That is exactly
    // InnerProductMatrix::compute_packed's A x B^T with A = the (already bf16) centroids
    // (K rows) and B = query tokens (n_query rows), rounded to bf16 by this call rather
    // than the kernel, so every ISA level sees the same rounded values on both operands.
    void fix_argument_decoded(const float* tokens, size_t n_query) {
        n_query_ = n_query;
        table_.resize(centroids_->K * n_query_);
        packed_query_.pack_rounded_to_bf16(tokens, n_query_, centroids_->dim);
        InnerProductMatrix::compute_packed(
            centroids_->data.get(), centroids_->K, packed_query_, table_.data()
        );
    }

    // Reduces the per-query table against a doc's decoded ids, dispatched to the ISA the
    // per-ISA compute_decoded was built for.
    float compute_decoded(const uint32_t* doc_ids, size_t n_doc) const {
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx512f_supported(), 1)) {
            return QChamferExactImpl<AVX_AVAILABILITY::AVX512>::compute_decoded(
                table_.data(), n_query_, doc_ids, n_doc
            );
        }
        if (__builtin_expect(svs::detail::avx_runtime_flags.is_avx2_supported(), 1)) {
            return QChamferExactImpl<AVX_AVAILABILITY::AVX2>::compute_decoded(
                table_.data(), n_query_, doc_ids, n_doc
            );
        }
        return QChamferExactImpl<AVX_AVAILABILITY::NONE>::compute_decoded(
            table_.data(), n_query_, doc_ids, n_doc
        );
    }

    std::shared_ptr<const FineCentroids> centroids_;
    std::shared_ptr<const GemRouter> router_;
    size_t nprobe_override_ = GemRouter::kUseStoredNprobe;
    PackedOperand packed_query_;
    std::vector<float> table_;
    size_t n_query_ = 0;
};

inline constexpr bool operator==(const QChamferExactQuery&, const QChamferExactQuery&) {
    return true;
}

} // namespace svs::distance
