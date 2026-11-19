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

#include <svs/runtime/api_defs.h>

#include <cstddef>
#include <cstdint>
#include <iosfwd>

namespace svs {
namespace runtime {
SVS_DECLARE_NAMESPACE_VERSION(0) {

/// Transport-simplex pivot start `QEmd` uses while building the graph. Both choices find
/// the same optimum; they differ only in pivot count, not graph quality.
enum class GemTransportStart { NorthWest, CheapestRow };

/// Parameters for `GemIndex::build`. See `GEM_MILVUS_DESIGN.md`'s "Layer 1" for the
/// pipeline each field feeds: codebook training, graph construction, and the optional
/// router.
struct GemBuildParameters {
    size_t num_centroids = 0;             // 0: auto, 2^floor(log2(16*sqrt(tokens)))
    size_t graph_max_degree = 24;         // R
    size_t construction_window_size = 80; // L
    float alpha = 1.2f;
    size_t prune_to = 0;                // 0: R - 4
    size_t max_candidate_pool_size = 0; // 0: 3L
    GemTransportStart start = GemTransportStart::NorthWest;
    bool build_router = true;   // train and save a GemRouter for search-time routing
    bool route_inserts = false; // route graph inserts too (GEM-style); off keeps the
                                // reference graph
    uint32_t seed = 42;
    bool keep_tokens_for_rerank = false; // opt-in: keeps fp16 tokens for native re-rank
    size_t num_threads = 0; // 0: omp_get_max_threads(), as the IVF runtime does
};

/// Parameters for `GemIndex::search`. `native_rerank` is rejected with
/// `INVALID_ARGUMENT` unless the index was built with `keep_tokens_for_rerank`.
struct GemSearchParameters {
    size_t search_window_size = 1000; // W. Clamped up to k
    size_t router_nprobe = 4;         // coarse clusters per query token; 0: fixed entry
                                      // point only
    bool native_rerank = false;
};

/// Runtime GEM (quantized multi-vector / late-interaction) index: a Vamana graph over
/// per-document token sets, searched with the quantized Chamfer distance (qCH). See
/// `GEM_MILVUS_DESIGN.md`'s "Layer 1" for the serialization layout and build pipeline.
struct SVS_RUNTIME_API GemIndex {
    /// `tokens`: fp32, row-major, `offsets[n_docs]` tokens of `dim` floats; `offsets` has
    /// `n_docs + 1` entries, `offsets[0] == 0`.
    static Status build(
        GemIndex** index,
        size_t dim,
        size_t n_docs,
        const float* tokens,
        const size_t* offsets,
        const GemBuildParameters& = {},
        const GemSearchParameters& default_search = {}
    ) noexcept;

    static Status destroy(GemIndex* index) noexcept;

    /// Scores are qChamfer distances, smaller is closer. `labels` are doc ordinals.
    /// Padding when fewer than `k` results exist: `label = size_t(-1)`,
    /// `distance = +inf`.
    virtual Status search(
        size_t n_queries,
        const float* query_tokens,
        const size_t* query_offsets,
        size_t k,
        float* distances,
        size_t* labels,
        const GemSearchParameters* = nullptr,
        IDFilter* = nullptr
    ) const noexcept = 0;

    virtual Status save(std::ostream& out) const noexcept = 0;
    static Status load(GemIndex** index, std::istream& in) noexcept;

    virtual size_t size() const noexcept = 0; // docs
    virtual size_t dimensions() const noexcept = 0;
    virtual size_t num_centroids() const noexcept = 0; // K, for the report
    virtual size_t get_memory_usage() const noexcept = 0;

    virtual ~GemIndex();
};

} // SVS_DECLARE_NAMESPACE_VERSION(0)
} // namespace runtime
} // namespace svs
