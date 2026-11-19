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

#include "svs/runtime/gem_index.h"

#include "gem_index_impl.h"
#include "svs_runtime_utils.h"

#include <cassert>
#include <memory>

namespace svs {
namespace runtime {

namespace {
struct GemIndexManager : public GemIndex {
    std::unique_ptr<GemIndexImpl> impl_;

    explicit GemIndexManager(std::unique_ptr<GemIndexImpl> impl)
        : impl_{std::move(impl)} {
        assert(impl_ != nullptr);
    }

    GemIndexManager(const GemIndexManager&) = delete;
    GemIndexManager& operator=(const GemIndexManager&) = delete;
    GemIndexManager(GemIndexManager&&) = default;
    GemIndexManager& operator=(GemIndexManager&&) = default;
    ~GemIndexManager() override = default;

    Status search(
        size_t n_queries,
        const float* query_tokens,
        const size_t* query_offsets,
        size_t k,
        float* distances,
        size_t* labels,
        const GemSearchParameters* params,
        IDFilter* filter
    ) const noexcept override {
        return runtime_error_wrapper([&] {
            impl_->search(
                n_queries, query_tokens, query_offsets, k, distances, labels, params, filter
            );
        });
    }

    Status save(std::ostream& out) const noexcept override {
        return runtime_error_wrapper([&] { impl_->save(out); });
    }

    size_t size() const noexcept override { return impl_->size(); }
    size_t dimensions() const noexcept override { return impl_->dimensions(); }
    size_t num_centroids() const noexcept override { return impl_->num_centroids(); }
    size_t get_memory_usage() const noexcept override { return impl_->get_memory_usage(); }
};
} // namespace

GemIndex::~GemIndex() = default;

Status GemIndex::build(
    GemIndex** index,
    size_t dim,
    size_t n_docs,
    const float* tokens,
    const size_t* offsets,
    const GemBuildParameters& build_params,
    const GemSearchParameters& default_search
) noexcept {
    *index = nullptr;
    return runtime_error_wrapper([&] {
        auto impl = std::make_unique<GemIndexImpl>(
            dim, n_docs, tokens, offsets, build_params, default_search
        );
        *index = new GemIndexManager{std::move(impl)};
    });
}

Status GemIndex::destroy(GemIndex* index) noexcept {
    return runtime_error_wrapper([&] { delete index; });
}

Status GemIndex::load(GemIndex** index, std::istream& in) noexcept {
    *index = nullptr;
    return runtime_error_wrapper([&] {
        auto impl = GemIndexImpl::load(in);
        *index = new GemIndexManager{std::move(impl)};
    });
}

} // namespace runtime
} // namespace svs
