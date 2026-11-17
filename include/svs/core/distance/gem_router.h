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
#include "svs/lib/exception.h"
#include "svs/lib/threads.h"

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace svs::distance {

/// Groups GEM fine centroids into coarse clusters, each with a representative document,
/// so a Vamana search can start from multiple entry points chosen by query-token score.
/// See `svs::index::vamana::extensions::GemRoutedInitializer` (extensions/vamana/gem.h).
struct GemRouter {
    static constexpr uint32_t no_representative = std::numeric_limits<uint32_t>::max();

    // coarse_of_fine[f] is the coarse cluster of fine centroid f, < num_coarse().
    std::vector<uint32_t> coarse_of_fine;
    // representative[c] is a document id, or no_representative if no document touches c.
    std::vector<uint32_t> representative;
    // Coarse clusters probed per query token in route_query.
    size_t nprobe = 4;

    size_t num_coarse() const { return representative.size(); }
    size_t num_fine() const { return coarse_of_fine.size(); }

    /// Scratch for route_query, reused across calls so a search thread allocates nothing
    /// once every buffer has grown to its steady-state size.
    struct QueryScratch {
        std::vector<float> coarse_score;
        std::vector<std::pair<float, uint32_t>> candidates;
        std::vector<uint32_t> picked;
    };

    /// Scratch for route_ids, reused the same way as QueryScratch.
    struct IdScratch {
        std::vector<std::pair<uint32_t, uint32_t>> counts;
    };

    /// Builds representatives from `docs` (one centroid id per token, dimensions() == 1)
    /// given a fine-to-coarse assignment. Deterministic regardless of `pool` size: shares
    /// are reduced in ascending document-id order, so ties always pick the lowest id.
    template <svs::data::ImmutableVectorSetDataset IdSets, threads::ThreadPool Pool>
        requires std::unsigned_integral<typename IdSets::const_value_type::element_type>
    static GemRouter build(
        std::vector<uint32_t> coarse_of_fine,
        size_t num_coarse,
        const IdSets& docs,
        Pool& pool,
        size_t nprobe = 4
    ) {
        size_t k = coarse_of_fine.size();
        for (size_t f = 0; f < k; ++f) {
            if (coarse_of_fine[f] >= num_coarse) {
                throw ANNEXCEPTION(
                    "GemRouter::build: coarse_of_fine[{}] = {} is not < num_coarse = {}",
                    f,
                    coarse_of_fine[f],
                    num_coarse
                );
            }
        }

        // (doc id, cluster, share) claims; disjoint per thread, so concatenated not merged.
        struct Claim {
            uint32_t doc_id;
            uint32_t cluster;
            double share;
        };
        threads::SequentialTLS<std::vector<Claim>> claims(pool.size());

        // Errors are recorded, not thrown: parallel_for doesn't guarantee propagation.
        std::atomic<bool> bad_id{false};
        std::atomic<uint64_t> bad_value{0};
        std::atomic<bool> bad_dims{false};
        std::atomic<uint64_t> bad_dims_doc{0};
        std::atomic<uint64_t> bad_dims_value{0};

        threads::parallel_for(
            pool,
            threads::StaticPartition{docs.size()},
            [&](const auto& ids, uint64_t tid) {
                auto& out = claims.at(tid);
                std::vector<std::pair<uint32_t, uint32_t>> local_counts;
                for (size_t i : ids) {
                    auto doc = docs.get_datum(i);
                    if (doc.dimensions() != 1) {
                        bad_dims.store(true, std::memory_order_relaxed);
                        bad_dims_doc.store(i, std::memory_order_relaxed);
                        bad_dims_value.store(doc.dimensions(), std::memory_order_relaxed);
                        continue;
                    }
                    size_t card = doc.cardinality();
                    if (card == 0) {
                        continue;
                    }
                    local_counts.clear();
                    for (size_t t = 0; t < card; ++t) {
                        uint64_t fine_id = static_cast<uint64_t>(doc.token(t)[0]);
                        if (fine_id >= k) {
                            bad_id.store(true, std::memory_order_relaxed);
                            bad_value.store(fine_id, std::memory_order_relaxed);
                            continue;
                        }
                        local_counts.emplace_back(coarse_of_fine[fine_id], uint32_t{1});
                    }
                    std::sort(local_counts.begin(), local_counts.end());
                    size_t w = 0;
                    size_t n = local_counts.size();
                    for (size_t j = 0; j < n;) {
                        size_t j2 = j;
                        uint32_t total = 0;
                        while (j2 < n && local_counts[j2].first == local_counts[j].first) {
                            total += local_counts[j2].second;
                            ++j2;
                        }
                        local_counts[w++] = {local_counts[j].first, total};
                        j = j2;
                    }
                    for (size_t j = 0; j < w; ++j) {
                        out.push_back(
                            {static_cast<uint32_t>(i),
                             local_counts[j].first,
                             static_cast<double>(local_counts[j].second) /
                                 static_cast<double>(card)}
                        );
                    }
                }
            }
        );

        if (bad_dims.load()) {
            throw ANNEXCEPTION(
                "GemRouter::build: document {} has dimensions() = {}, expected 1 (one "
                "centroid id per token)",
                bad_dims_doc.load(),
                bad_dims_value.load()
            );
        }
        if (bad_id.load()) {
            throw ANNEXCEPTION(
                "GemRouter::build: document centroid id {} is not < K = {}",
                bad_value.load(),
                k
            );
        }

        std::vector<Claim> all;
        claims.visit([&](const std::vector<Claim>& part) {
            all.insert(all.end(), part.begin(), part.end());
        });
        std::sort(all.begin(), all.end(), [](const Claim& a, const Claim& b) {
            return a.doc_id < b.doc_id;
        });

        GemRouter result;
        result.coarse_of_fine = std::move(coarse_of_fine);
        result.representative.assign(num_coarse, no_representative);
        result.nprobe = nprobe;
        std::vector<double> best_share(num_coarse, -1.0);
        for (const auto& claim : all) {
            if (claim.share > best_share[claim.cluster]) {
                best_share[claim.cluster] = claim.share;
                result.representative[claim.cluster] = claim.doc_id;
            }
        }
        return result;
    }

    /// Thread-count convenience overload: builds a `DefaultThreadPool` of
    /// `num_threads` threads for the duration of the call.
    template <svs::data::ImmutableVectorSetDataset IdSets>
        requires std::unsigned_integral<typename IdSets::const_value_type::element_type>
    static GemRouter build(
        std::vector<uint32_t> coarse_of_fine,
        size_t num_coarse,
        const IdSets& docs,
        size_t num_threads,
        size_t nprobe = 4
    ) {
        auto pool = threads::DefaultThreadPool(num_threads);
        return build(std::move(coarse_of_fine), num_coarse, docs, pool, nprobe);
    }

    /// Scores cluster c for query token i as the max of `table[f * n_query + i]` over its
    /// fine centroids f, picks the `nprobe` best clusters per token (ties to lower id),
    /// and writes their representatives to `out`, deduplicated and sorted ascending.
    void route_query(
        const float* table,
        size_t n_query,
        std::vector<uint32_t>& out,
        QueryScratch& scratch
    ) const {
        size_t nc = num_coarse();
        scratch.coarse_score.assign(nc * n_query, -std::numeric_limits<float>::infinity());
        size_t k = num_fine();
        for (size_t f = 0; f < k; ++f) {
            uint32_t c = coarse_of_fine[f];
            const float* row = table + f * n_query;
            float* dst = scratch.coarse_score.data() + static_cast<size_t>(c) * n_query;
            for (size_t i = 0; i < n_query; ++i) {
                dst[i] = std::max(dst[i], row[i]);
            }
        }

        scratch.picked.clear();
        size_t take = std::min(nprobe, nc);
        for (size_t i = 0; i < n_query; ++i) {
            scratch.candidates.clear();
            scratch.candidates.reserve(nc);
            for (size_t c = 0; c < nc; ++c) {
                scratch.candidates.emplace_back(
                    scratch.coarse_score[c * n_query + i], static_cast<uint32_t>(c)
                );
            }
            std::partial_sort(
                scratch.candidates.begin(),
                scratch.candidates.begin() + static_cast<std::ptrdiff_t>(take),
                scratch.candidates.end(),
                [](const auto& a, const auto& b) {
                    if (a.first != b.first) {
                        return a.first > b.first;
                    }
                    return a.second < b.second;
                }
            );
            for (size_t j = 0; j < take; ++j) {
                uint32_t rep = representative[scratch.candidates[j].second];
                if (rep != no_representative) {
                    scratch.picked.push_back(rep);
                }
            }
        }

        out.assign(scratch.picked.begin(), scratch.picked.end());
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }

    /// Ranks `ids`' coarse clusters by token count (ties to the lower cluster id), takes
    /// the top `max_clusters`, and writes their representatives (deduplicated, sorted
    /// ascending, `no_representative` skipped) to `out`. Lookups only, no scoring.
    void route_ids(
        std::span<const uint32_t> ids,
        size_t max_clusters,
        std::vector<uint32_t>& out,
        IdScratch& scratch
    ) const {
        scratch.counts.clear();
        for (uint32_t id : ids) {
            if (id >= num_fine()) {
                throw ANNEXCEPTION(
                    "GemRouter::route_ids: centroid id {} is not < num_fine() = {}",
                    id,
                    num_fine()
                );
            }
            scratch.counts.emplace_back(coarse_of_fine[id], uint32_t{1});
        }
        std::sort(scratch.counts.begin(), scratch.counts.end());
        size_t w = 0;
        size_t n = scratch.counts.size();
        for (size_t i = 0; i < n;) {
            size_t j = i;
            uint32_t total = 0;
            while (j < n && scratch.counts[j].first == scratch.counts[i].first) {
                total += scratch.counts[j].second;
                ++j;
            }
            scratch.counts[w++] = {scratch.counts[i].first, total};
            i = j;
        }
        scratch.counts.resize(w);

        size_t take = std::min(max_clusters, scratch.counts.size());
        std::partial_sort(
            scratch.counts.begin(),
            scratch.counts.begin() + static_cast<std::ptrdiff_t>(take),
            scratch.counts.end(),
            [](const auto& a, const auto& b) {
                if (a.second != b.second) {
                    return a.second > b.second;
                }
                return a.first < b.first;
            }
        );

        out.clear();
        for (size_t i = 0; i < take; ++i) {
            uint32_t rep = representative[scratch.counts[i].first];
            if (rep != no_representative) {
                out.push_back(rep);
            }
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }

    void save(const std::filesystem::path& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw ANNEXCEPTION("GemRouter::save: cannot open {}", path.string());
        }
        Header header{};
        header.k = num_fine();
        header.num_coarse_val = num_coarse();
        header.nprobe_val = nprobe;
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));
        out.write(
            reinterpret_cast<const char*>(coarse_of_fine.data()),
            static_cast<std::streamsize>(coarse_of_fine.size() * sizeof(uint32_t))
        );
        out.write(
            reinterpret_cast<const char*>(representative.data()),
            static_cast<std::streamsize>(representative.size() * sizeof(uint32_t))
        );
        if (!out) {
            throw ANNEXCEPTION("GemRouter::save: short write to {}", path.string());
        }
    }

    static GemRouter load(const std::filesystem::path& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw ANNEXCEPTION("GemRouter::load: cannot open {}", path.string());
        }
        Header header{};
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!in) {
            throw ANNEXCEPTION(
                "GemRouter::load: {} is too short to hold a GemRouter header", path.string()
            );
        }
        if (header.magic != kMagic) {
            throw ANNEXCEPTION(
                "GemRouter::load: {} has magic {:#x}, expected {:#x}",
                path.string(),
                header.magic,
                kMagic
            );
        }
        if (header.version != kVersion) {
            throw ANNEXCEPTION(
                "GemRouter::load: {} has version {}, expected {}",
                path.string(),
                header.version,
                kVersion
            );
        }

        GemRouter result;
        result.coarse_of_fine.resize(header.k);
        result.representative.resize(header.num_coarse_val);
        result.nprobe = header.nprobe_val;
        in.read(
            reinterpret_cast<char*>(result.coarse_of_fine.data()),
            static_cast<std::streamsize>(header.k * sizeof(uint32_t))
        );
        in.read(
            reinterpret_cast<char*>(result.representative.data()),
            static_cast<std::streamsize>(header.num_coarse_val * sizeof(uint32_t))
        );
        if (!in) {
            throw ANNEXCEPTION("GemRouter::load: {} is truncated", path.string());
        }
        return result;
    }

  private:
    static constexpr uint64_t kMagic = 0x474d52545231ULL; // "GMRTR1" in ASCII hex-ish.
    static constexpr uint32_t kVersion = 1;

    // Trivially copyable on-disk header; field order is the file layout.
    struct Header {
        uint64_t magic = kMagic;
        uint32_t version = kVersion;
        uint64_t k = 0;
        uint64_t num_coarse_val = 0;
        uint64_t nprobe_val = 0;
    };
};

} // namespace svs::distance
