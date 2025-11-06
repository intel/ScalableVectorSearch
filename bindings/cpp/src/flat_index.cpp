/*
 * Copyright 2025 Intel Corporation
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

#include "flat_index.h"
#include "flat_index_impl.h"
#include "svs_runtime_utils.h"

#include <algorithm>
#include <memory>
#include <span>
#include <variant>

#include <svs/core/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/cpuid.h>
#include <svs/extensions/vamana/scalar.h>

namespace svs {
namespace runtime {

namespace {
struct FlatIndexManager : public FlatIndex {
    std::unique_ptr<FlatIndexImpl> impl_;

    FlatIndexManager(std::unique_ptr<FlatIndexImpl> impl)
        : impl_{std::move(impl)} {
        assert(impl_ != nullptr);
    }

    ~FlatIndexManager() override = default;

    Status add(size_t n, const float* x) noexcept override {
        SVS_RUNTIME_TRY_BEGIN
        svs::data::ConstSimpleDataView<float> data{x, n, impl_->dimensions()};
        impl_->add(data);
        return Status_Ok;
        SVS_RUNTIME_TRY_END
    }

    Status search(size_t n, const float* x, size_t k, float* distances, size_t* labels)
        const noexcept override {
        SVS_RUNTIME_TRY_BEGIN
        // TODO wrap arguments into proper data structures in FlatIndexImpl and
        // here
        impl_->search(n, x, k, distances, labels);
        return Status_Ok;
        SVS_RUNTIME_TRY_END
    }

    Status reset() noexcept override {
        SVS_RUNTIME_TRY_BEGIN
        impl_->reset();
        return Status_Ok;
        SVS_RUNTIME_TRY_END
    }

    Status save(std::ostream& out) const noexcept override {
        SVS_RUNTIME_TRY_BEGIN
        impl_->save(out);
        return Status_Ok;
        SVS_RUNTIME_TRY_END
    }
};
} // namespace

// FlatIndex interface implementation
FlatIndex::~FlatIndex() = default;

Status FlatIndex::build(FlatIndex** index, size_t dim, MetricType metric) noexcept {
    *index = nullptr;
    SVS_RUNTIME_TRY_BEGIN
    auto impl = std::make_unique<FlatIndexImpl>(dim, metric);
    *index = new FlatIndexManager{std::move(impl)};
    return Status_Ok;
    SVS_RUNTIME_TRY_END
}

Status FlatIndex::destroy(FlatIndex* index) noexcept {
    SVS_RUNTIME_TRY_BEGIN
    delete index;
    return Status_Ok;
    SVS_RUNTIME_TRY_END
}

Status FlatIndex::load(FlatIndex** index, std::istream& in, MetricType metric) noexcept {
    *index = nullptr;
    SVS_RUNTIME_TRY_BEGIN
    std::unique_ptr<FlatIndexImpl> impl{FlatIndexImpl::load(in, metric)};
    *index = new FlatIndexManager{std::move(impl)};
    return Status_Ok;
    SVS_RUNTIME_TRY_END
}
} // namespace runtime
} // namespace svs
