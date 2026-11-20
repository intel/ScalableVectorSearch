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

#include "svs/runtime/gem_index.h"

#include "svs_runtime_utils.h"

#include <svs/core/data/simple.h>
#include <svs/core/data/vector_set.h>
#include <svs/core/data/vector_set_view.h>
#include <svs/core/distance/gem.h>
#include <svs/core/distance/gem_router.h>
#include <svs/core/distance/maxsim.h>
#include <svs/core/graph/graph.h>
#include <svs/extensions/vamana/gem.h>
#include <svs/index/vamana/gem_build.h>
#include <svs/index/vamana/index.h>
#include <svs/lib/float16.h>
#include <svs/lib/narrow.h>
#include <svs/lib/saveload.h>
#include <svs/lib/threads.h>
#include <svs/quantization/gem/codebook.h>
#include <svs/quantization/gem/codebook_timed.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <istream>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <span>
#include <vector>

namespace svs {
namespace data {

// VectorSetData has no element_size(); VamanaIndex::get_memory_breakdown() calls
// dataset_allocated_bytes(data_) on the id-set dataset, so this overload (found via ADL,
// preferred over the data.h template) supplies the byte count directly instead.
inline size_t dataset_allocated_bytes(const VectorSetData<uint32_t>& dataset) {
    return dataset.tokens().capacity() * dataset.tokens().element_size() +
           dataset.offsets().size() * sizeof(uint64_t);
}

} // namespace data
} // namespace svs

namespace svs {
namespace runtime {

// GEM runtime implementation: a Vamana graph over per-document centroid-id sets, served
// with QChamferExactQuery (qCH, raw float query against bf16 fine centroids). Unlike
// VamanaIndexImpl, there is only one Data/Distance instantiation: GEM has no storage-kind
// dispatch, so this class holds a single `VIndex` directly rather than dispatching through
// a type-erased orchestrator.
class GemIndexImpl {
  public:
    using IdSets = svs::data::VectorSetData<uint32_t>;
    // Not svs::HugepageAllocator: this process mmaps many files concurrently (many
    // segment builds plus a load), and GenericHugepageAllocator::deallocate throws on
    // munmap failure from inside a destructor, which aborts the process.
    using Graph = svs::graphs::SimpleGraph<uint32_t, svs::lib::Allocator<uint32_t>>;
    using Dist = svs::distance::QChamferExactQuery;
    using VIndex = svs::index::vamana::VamanaIndex<Graph, IdSets, Dist>;

    GemIndexImpl(
        size_t dim,
        size_t n_docs,
        const float* tokens,
        const size_t* offsets,
        svs::threads::ThreadPoolHandle& pool,
        const GemBuildParameters& build_params,
        const GemSearchParameters& default_search
    )
        : dim_{dim}
        , default_search_{default_search}
        , keep_tokens_{build_params.keep_tokens_for_rerank} {
        if (dim == 0) {
            throw StatusException{ErrorCode::INVALID_ARGUMENT, "dim must be non-zero"};
        }
        if (n_docs == 0) {
            throw StatusException{ErrorCode::INVALID_ARGUMENT, "n_docs must be non-zero"};
        }

        auto build_start = std::chrono::steady_clock::now();
        auto elapsed_s = [](std::chrono::steady_clock::time_point since) {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - since)
                .count();
        };

        std::vector<uint64_t> offsets64(offsets, offsets + n_docs + 1);
        size_t total_tokens = static_cast<size_t>(offsets64.back());
        auto token_view = svs::data::ConstVectorSetView<float>(
            std::span<const float>(tokens, total_tokens * dim),
            std::span<const uint64_t>(offsets64),
            dim
        );

        svs::quantization::gem::CodebookParameters codebook_params{};
        codebook_params.num_centroids = build_params.num_centroids;
        codebook_params.seed = build_params.seed;

        auto [quantized, quantize_timings] =
            svs::quantization::gem::quantize_timed(token_view, codebook_params, pool);
        double train_s =
            quantize_timings.sample + quantize_timings.train + quantize_timings.normalize;
        double assign_s = quantize_timings.assign;
        double ground_s = quantize_timings.ground_table + quantize_timings.coarse;
        num_centroids_ = quantized.K;
        fine_centroids_ = quantized.fine_centroids;

        auto entry_start = std::chrono::steady_clock::now();
        uint32_t entry_point = svs::lib::narrow_cast<uint32_t>(
            svs::index::vamana::extensions::gem_pooled_entry_point(
                quantized.ids.view(), *fine_centroids_, pool
            )
        );
        double entry_s = elapsed_s(entry_start);

        auto router_start = std::chrono::steady_clock::now();
        if (build_params.build_router) {
            auto built = svs::distance::GemRouter::build(
                quantized.coarse_of_fine,
                quantized.num_coarse,
                quantized.ids.view(),
                pool,
                default_search.router_nprobe
            );
            router_ = std::make_shared<const svs::distance::GemRouter>(std::move(built));
        }
        double router_s = elapsed_s(router_start);

        auto start = build_params.start == GemTransportStart::NorthWest
                         ? svs::distance::TransportStart::NorthWest
                         : svs::distance::TransportStart::CheapestRow;
        svs::distance::QEmd build_distance{
            quantized.centroid_table,
            build_params.route_inserts ? router_ : nullptr,
            start};

        size_t prune_to =
            build_params.prune_to != 0
                ? build_params.prune_to
                : (build_params.graph_max_degree > 4 ? build_params.graph_max_degree - 4
                                                     : build_params.graph_max_degree);
        size_t max_pool = build_params.max_candidate_pool_size != 0
                              ? build_params.max_candidate_pool_size
                              : 3 * build_params.construction_window_size;
        svs::index::vamana::VamanaBuildParameters vamana_params{
            build_params.alpha,
            build_params.graph_max_degree,
            build_params.construction_window_size,
            max_pool,
            prune_to,
            false};

        // Explicit allocator argument (rather than build_gem_graph's hugepage default)
        // deduces its Allocator template parameter to match Graph above.
        auto graph_start = std::chrono::steady_clock::now();
        auto [graph, report] = svs::index::vamana::build_gem_graph(
            quantized.ids,
            build_distance,
            vamana_params,
            pool,
            entry_point,
            svs::lib::Allocator<uint32_t>()
        );
        (void)report;
        double graph_s = elapsed_s(graph_start);

        auto tokens_start = std::chrono::steady_clock::now();
        if (keep_tokens_) {
            tokens_ = make_fp16_tokens(tokens, offsets64, dim, pool);
        }
        double tokens_s = elapsed_s(tokens_start);

        auto assemble_start = std::chrono::steady_clock::now();
        // The K*K centroid_table (build-only) is dropped here: `quantized` and
        // `build_distance` (its last owners) go out of scope at the end of this
        // constructor.
        Dist serve_distance{fine_centroids_, router_};
        index_ = std::make_unique<VIndex>(
            std::move(graph),
            std::move(quantized.ids),
            entry_point,
            std::move(serve_distance),
            serving_pool()
        );
        index_->set_search_parameters(make_vamana_search_parameters(default_search_));
        double assemble_s = elapsed_s(assemble_start);
        double total_s = elapsed_s(build_start);

        std::fprintf(
            stderr,
            "SVS_GEM_BUILD_PHASES n_docs=%zu n_tokens=%zu K=%zu threads=%zu "
            "train_s=%.3f assign_s=%.3f ground_s=%.3f entry_s=%.3f router_s=%.3f "
            "graph_s=%.3f tokens_s=%.3f assemble_s=%.3f total_s=%.3f\n",
            n_docs,
            total_tokens,
            num_centroids_,
            pool.size(),
            train_s,
            assign_s,
            ground_s,
            entry_s,
            router_s,
            graph_s,
            tokens_s,
            assemble_s,
            total_s
        );
        std::fflush(stderr);
    }

    size_t size() const { return index_->size(); }
    size_t dimensions() const { return dim_; }
    size_t num_centroids() const { return num_centroids_; }

    size_t get_memory_usage() const {
        size_t total = index_->get_memory_breakdown().total();
        total += num_centroids_ * dim_ * sizeof(svs::BFloat16);
        if (router_) {
            total += (router_->num_fine() + router_->num_coarse()) * sizeof(uint32_t);
        }
        if (tokens_) {
            total +=
                tokens_->tokens().size() * tokens_->dimensions() * sizeof(svs::Float16);
        }
        return total;
    }

    void search(
        size_t n_queries,
        const float* query_tokens,
        const size_t* query_offsets,
        size_t k,
        float* distances,
        size_t* labels,
        const GemSearchParameters* params,
        IDFilter* filter
    ) const {
        if (k == 0) {
            throw StatusException{ErrorCode::INVALID_ARGUMENT, "k must be greater than 0"};
        }
        GemSearchParameters sp = params ? *params : default_search_;
        if (sp.native_rerank && !tokens_) {
            throw StatusException{
                ErrorCode::INVALID_ARGUMENT,
                "native_rerank was requested but the index was not built with "
                "keep_tokens_for_rerank"};
        }
        if (n_queries == 0) {
            return;
        }

        std::vector<uint64_t> offsets64(query_offsets, query_offsets + n_queries + 1);
        size_t total_tokens = static_cast<size_t>(offsets64.back());
        auto query_view = svs::data::ConstVectorSetView<float>(
            std::span<const float>(query_tokens, total_tokens * dim_),
            std::span<const uint64_t>(offsets64),
            dim_
        );

        size_t window = std::max(sp.search_window_size, k);
        auto search_params = index_->get_search_parameters();
        search_params.buffer_config({window, window});
        // `scratch` (and its distance copy, `scratch.scratch`) is local to this call, so
        // setting the override here is per-search state, never shared across concurrent
        // searches on the same index.
        auto scratch = index_->scratchspace(search_params);
        scratch.scratch.set_nprobe_override(sp.router_nprobe);

        for (size_t q = 0; q < n_queries; ++q) {
            auto query = query_view.get_datum(q);
            run_one_query(query, scratch, k, filter, sp, distances + q * k, labels + q * k);
        }
    }

    void save(std::ostream& out) const {
        svs::lib::begin_serialization(out);

        ConfigHeader header{};
        header.dim = dim_;
        header.num_centroids = num_centroids_;
        header.entry_point = index_->parameters().entry_point;
        header.default_search_window = default_search_.search_window_size;
        header.default_router_nprobe = default_search_.router_nprobe;
        header.has_router = router_ ? 1 : 0;
        header.has_tokens = keep_tokens_ ? 1 : 0;
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));

        const IdSets& ids = index_->view_data();
        svs::lib::save_to_stream(ids.tokens(), out);
        save_offsets(ids.offsets(), out);

        svs::lib::save_to_stream(index_->view_graph(), out);

        svs::lib::save_to_stream(reconstruct_fp32_centroids(), out);

        if (router_) {
            router_->save(out);
        }

        if (keep_tokens_) {
            svs::lib::save_to_stream(tokens_->tokens(), out);
            save_offsets(tokens_->offsets(), out);
        }

        if (!out) {
            throw StatusException{ErrorCode::RUNTIME_ERROR, "GemIndex::save: short write"};
        }
    }

    static std::unique_ptr<GemIndexImpl> load(std::istream& in) {
        // Consumes the single stream-wide magic `save()` wrote via begin_serialization;
        // every component after this point is its own metadata-table-plus-payload unit.
        svs::lib::detail::Deserializer::build(in);

        ConfigHeader header{};
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!in) {
            throw StatusException{
                ErrorCode::RUNTIME_ERROR, "GemIndex::load: stream too short for a header"};
        }

        auto token_ids = svs::lib::load_from_stream<svs::data::SimpleData<uint32_t>>(in);
        auto id_offsets = load_offsets(in);
        IdSets ids(std::move(token_ids), std::move(id_offsets));

        auto graph = Graph::load(in);

        auto centroids_fp32 = svs::lib::load_from_stream<svs::data::SimpleData<float>>(in);
        auto fine_centroids = std::make_shared<const svs::distance::FineCentroids>(
            centroids_fp32.size(), centroids_fp32.dimensions(), centroids_fp32.data(), 1
        );

        std::shared_ptr<const svs::distance::GemRouter> router;
        if (header.has_router != 0) {
            router = std::make_shared<const svs::distance::GemRouter>(
                svs::distance::GemRouter::load(in)
            );
        }

        std::optional<svs::data::VectorSetData<svs::Float16>> tokens;
        if (header.has_tokens != 0) {
            auto token_matrix =
                svs::lib::load_from_stream<svs::data::SimpleData<svs::Float16>>(in);
            auto token_offsets = load_offsets(in);
            tokens.emplace(std::move(token_matrix), std::move(token_offsets));
        }

        auto result = std::make_unique<GemIndexImpl>(PrivateLoadTag{});
        result->dim_ = header.dim;
        result->num_centroids_ = header.num_centroids;
        result->default_search_ = GemSearchParameters{
            header.default_search_window, header.default_router_nprobe, false};
        result->keep_tokens_ = header.has_tokens != 0;
        result->fine_centroids_ = fine_centroids;
        result->router_ = router;
        result->tokens_ = std::move(tokens);

        Dist serve_distance{result->fine_centroids_, result->router_};

        result->index_ = std::make_unique<VIndex>(
            std::move(graph),
            std::move(ids),
            svs::lib::narrow_cast<uint32_t>(header.entry_point),
            std::move(serve_distance),
            serving_pool()
        );
        result->index_->set_search_parameters(
            make_vamana_search_parameters(result->default_search_)
        );
        return result;
    }

  private:
    // A host such as Knowhere loads one index per segment and searches from its own pool; a
    // per-index pool of omp_get_max_threads() workers exhausted threads and memory maps.
    static svs::threads::DefaultThreadPool serving_pool() {
        return svs::threads::DefaultThreadPool(1);
    }

    // Tag-dispatched constructor `load()` uses to build an otherwise-empty instance
    // before filling in every member from the stream.
    struct PrivateLoadTag {};

  public:
    // Public so make_unique can call it; PrivateLoadTag stays private, so only this
    // class can name the tag and thus construct an empty instance.
    explicit GemIndexImpl(PrivateLoadTag) {}

  private:
    // On-disk layout: a fixed POD header (not the TOML SaveTable the rest of the library
    // uses), matching GemRouter's own header convention (gem_router.h) rather than
    // pulling lib::SaveTable machinery in for a handful of scalars.
    struct ConfigHeader {
        uint64_t dim = 0;
        uint64_t num_centroids = 0;
        uint64_t entry_point = 0;
        uint64_t default_search_window = 0;
        uint64_t default_router_nprobe = 0;
        uint8_t has_router = 0;
        uint8_t has_tokens = 0;
    };

    static void save_offsets(std::span<const uint64_t> offsets, std::ostream& out) {
        uint64_t count = offsets.size();
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        out.write(
            reinterpret_cast<const char*>(offsets.data()),
            static_cast<std::streamsize>(offsets.size() * sizeof(uint64_t))
        );
    }

    static std::vector<uint64_t> load_offsets(std::istream& in) {
        uint64_t count = 0;
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        std::vector<uint64_t> offsets(count);
        in.read(
            reinterpret_cast<char*>(offsets.data()),
            static_cast<std::streamsize>(count * sizeof(uint64_t))
        );
        if (!in) {
            throw StatusException{
                ErrorCode::RUNTIME_ERROR, "GemIndex::load: truncated offsets"};
        }
        return offsets;
    }

    static svs::index::vamana::VamanaSearchParameters
    make_vamana_search_parameters(const GemSearchParameters& sp) {
        size_t window = std::max(sp.search_window_size, size_t{1});
        return svs::index::vamana::VamanaSearchParameters{}.buffer_config({window, window});
    }

    // The flat input buffer is already laid out exactly as the CSR token matrix wants it
    // (same order, same row width), so this is a per-element cast, not a per-document copy.
    static svs::data::VectorSetData<svs::Float16> make_fp16_tokens(
        const float* tokens,
        const std::vector<uint64_t>& offsets64,
        size_t dim,
        svs::threads::ThreadPoolHandle& pool
    ) {
        size_t total = static_cast<size_t>(offsets64.back());
        auto matrix = svs::data::SimpleData<svs::Float16>(total, dim);
        svs::Float16* dst = matrix.data();
        svs::threads::parallel_for(
            pool,
            svs::threads::StaticPartition{total * dim},
            [&](const auto& range, uint64_t /*tid*/) {
                for (size_t i : range) {
                    dst[i] = svs::Float16(tokens[i]);
                }
            }
        );
        return svs::data::VectorSetData<svs::Float16>(std::move(matrix), offsets64);
    }

    // Upcasts the bf16 fine centroids this index already holds, rather than keeping a
    // second fp32 copy from build time alive just for save(): rounding an already-bf16
    // value to bf16 again on reload is a no-op, so this round-trips exactly.
    svs::data::SimpleData<float> reconstruct_fp32_centroids() const {
        size_t k = fine_centroids_->K;
        size_t dim = fine_centroids_->dim;
        auto out = svs::data::SimpleData<float>(k, dim);
        float* dst = out.data();
        const svs::BFloat16* src = fine_centroids_->data.get();
        for (size_t i = 0, n = k * dim; i < n; ++i) {
            dst[i] = static_cast<float>(src[i]);
        }
        return out;
    }

    // One query's qCH search plus, with a filter, growing the window until `k` filtered
    // matches are found or the whole index has been scanned; with native_rerank, every
    // surviving candidate in the final window is rescored by exact MaxSim over the kept
    // fp16 tokens and re-sorted, matching the design note's "exact MaxSim over the search
    // buffer". The sign is flipped (-score) to keep this API's "smaller is closer"
    // contract for every search, routed or reranked.
    void run_one_query(
        const svs::data::ConstVectorSet<float>& query,
        VIndex::scratchspace_type& scratch,
        size_t k,
        IDFilter* filter,
        const GemSearchParameters& sp,
        float* out_distances,
        size_t* out_labels
    ) const {
        const size_t max_window = index_->size();
        size_t window = scratch.buffer.config().get_search_window_size();
        auto current_params = index_->get_search_parameters();
        current_params.buffer_config({window, window});

        for (;;) {
            index_->search(query, scratch);
            if (filter == nullptr) {
                break;
            }
            size_t found = 0;
            for (const auto& neighbor : scratch.buffer) {
                if (filter->is_member(neighbor.id())) {
                    ++found;
                }
            }
            if (found >= k || window >= max_window) {
                break;
            }
            window = std::min(window * 2, max_window);
            current_params.buffer_config({window, window});
            scratch.apply(current_params);
        }

        struct Candidate {
            uint32_t id;
            float distance;
        };
        std::vector<Candidate> picked;
        picked.reserve(scratch.buffer.size());
        for (const auto& neighbor : scratch.buffer) {
            if (filter != nullptr && !filter->is_member(neighbor.id())) {
                continue;
            }
            picked.push_back({neighbor.id(), neighbor.distance()});
        }

        if (sp.native_rerank) {
            svs::distance::MaxSim<svs::Float16> maxsim{};
            maxsim.fix_argument(query);
            for (auto& candidate : picked) {
                auto doc = tokens_->get_datum(candidate.id);
                candidate.distance = -maxsim.compute(query, doc);
            }
            std::sort(
                picked.begin(),
                picked.end(),
                [](const Candidate& a, const Candidate& b) {
                    return a.distance < b.distance;
                }
            );
        }

        size_t count = std::min(picked.size(), k);
        for (size_t i = 0; i < count; ++i) {
            out_distances[i] = picked[i].distance;
            out_labels[i] = picked[i].id;
        }
        for (size_t i = count; i < k; ++i) {
            out_distances[i] = Unspecify<float>();
            out_labels[i] = Unspecify<size_t>();
        }
    }

    size_t dim_ = 0;
    size_t num_centroids_ = 0;
    GemSearchParameters default_search_{};
    bool keep_tokens_ = false;
    std::shared_ptr<const svs::distance::FineCentroids> fine_centroids_;
    std::shared_ptr<const svs::distance::GemRouter> router_;
    std::optional<svs::data::VectorSetData<svs::Float16>> tokens_;
    std::unique_ptr<VIndex> index_;
};

} // namespace runtime
} // namespace svs
