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

#include "dynamic_vamana_index.h"
#include "dynamic_vamana_index_impl.h"
#include "svs_runtime_utils.h"

#include <algorithm>
#include <memory>
#include <variant>
#include <vector>

#include <svs/core/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/cpuid.h>
#include <svs/extensions/vamana/scalar.h>
#include <svs/lib/float16.h>
#include <svs/orchestrators/dynamic_vamana.h>
#include <svs/quantization/scalar/scalar.h>

#include SVS_LVQ_HEADER
#include SVS_LEANVEC_HEADER

namespace svs {
namespace runtime {

namespace {
template <typename Impl = DynamicVamanaIndexImpl>
struct DynamicVamanaIndexManagerBase : public DynamicVamanaIndex {
    std::unique_ptr<Impl> impl_;

    DynamicVamanaIndexManagerBase(std::unique_ptr<Impl> impl)
        : impl_{std::move(impl)} {
        assert(impl_ != nullptr);
    }

    DynamicVamanaIndexManagerBase(const DynamicVamanaIndexManagerBase&) = delete;
    DynamicVamanaIndexManagerBase& operator=(const DynamicVamanaIndexManagerBase&) = delete;
    DynamicVamanaIndexManagerBase(DynamicVamanaIndexManagerBase&&) = default;
    DynamicVamanaIndexManagerBase& operator=(DynamicVamanaIndexManagerBase&&) = default;
    ~DynamicVamanaIndexManagerBase() override = default;

    Status add(size_t n, const size_t* labels, const float* x) noexcept override {
        SVS_RUNTIME_TRY_BEGIN
        svs::data::ConstSimpleDataView<float> data{x, n, impl_->dimensions()};
        std::span<const size_t> lbls(labels, n);
        impl_->add(data, lbls);
        return Status_Ok;
        SVS_RUNTIME_TRY_END
    }

    Status
    remove_selected(size_t* num_removed, const IDFilter& selector) noexcept override {
        SVS_RUNTIME_TRY_BEGIN
        *num_removed = impl_->remove_selected(selector);
        return Status_Ok;
        SVS_RUNTIME_TRY_END
    }

    Status remove(size_t n, const size_t* labels) noexcept override {
        SVS_RUNTIME_TRY_BEGIN
        std::span<const size_t> lbls(labels, n);
        impl_->remove(lbls);
        return Status_Ok;
        SVS_RUNTIME_TRY_END
    }

    Status search(
        size_t n,
        const float* x,
        size_t k,
        float* distances,
        size_t* labels,
        const SearchParams* params = nullptr,
        IDFilter* filter = nullptr
    ) const noexcept override {
        SVS_RUNTIME_TRY_BEGIN
        // TODO wrap arguments into proper data structures in DynamicVamanaIndexImpl and
        // here
        impl_->search(n, x, k, distances, labels, params, filter);
        return Status_Ok;
        SVS_RUNTIME_TRY_END
    }

    Status range_search(
        size_t n,
        const float* x,
        float radius,
        const ResultsAllocator& results,
        const SearchParams* params = nullptr,
        IDFilter* filter = nullptr
    ) const noexcept override {
        SVS_RUNTIME_TRY_BEGIN
        // TODO wrap arguments into proper data structures in DynamicVamanaIndexImpl and
        // here
        impl_->range_search(n, x, radius, results, params, filter);
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

using DynamicVamanaIndexManager = DynamicVamanaIndexManagerBase<DynamicVamanaIndexImpl>;
using DynamicVamanaIndexLeanVecImplManager =
    DynamicVamanaIndexManagerBase<DynamicVamanaIndexLeanVecImpl>;

} // namespace

Status DynamicVamanaIndex::build(
    DynamicVamanaIndex** index,
    size_t dim,
    MetricType metric,
    StorageKind storage_kind,
    const DynamicVamanaIndex::BuildParams& params,
    const DynamicVamanaIndex::SearchParams& default_search_params
) noexcept {
    *index = nullptr;
    SVS_RUNTIME_TRY_BEGIN
    auto impl = std::make_unique<DynamicVamanaIndexImpl>(
        dim, metric, storage_kind, params, default_search_params
    );
    *index = new DynamicVamanaIndexManager{std::move(impl)};
    return Status_Ok;
    SVS_RUNTIME_TRY_END
}

Status DynamicVamanaIndex::destroy(DynamicVamanaIndex* index) noexcept {
    SVS_RUNTIME_TRY_BEGIN
    delete index;
    return Status_Ok;
    SVS_RUNTIME_TRY_END
}

Status DynamicVamanaIndex::load(
    DynamicVamanaIndex** index,
    std::istream& in,
    MetricType metric,
    StorageKind storage_kind
) noexcept {
    *index = nullptr;
    SVS_RUNTIME_TRY_BEGIN
    std::unique_ptr<DynamicVamanaIndexImpl> impl{
        DynamicVamanaIndexImpl::load(in, metric, storage_kind)};
    *index = new DynamicVamanaIndexManager{std::move(impl)};
    return Status_Ok;
    SVS_RUNTIME_TRY_END
}

// Specialization to build LeanVec-based Vamana index with specified leanvec dims
Status DynamicVamanaIndexLeanVec::build(
    DynamicVamanaIndex** index,
    size_t dim,
    MetricType metric,
    StorageKind storage_kind,
    size_t leanvec_dims,
    const DynamicVamanaIndex::BuildParams& params,
    const DynamicVamanaIndex::SearchParams& default_search_params
) noexcept {
    *index = nullptr;
    SVS_RUNTIME_TRY_BEGIN
    auto impl = std::make_unique<DynamicVamanaIndexLeanVecImpl>(
        dim, metric, storage_kind, leanvec_dims, params, default_search_params
    );
    *index = new DynamicVamanaIndexLeanVecImplManager{std::move(impl)};
    return Status_Ok;
    SVS_RUNTIME_TRY_END
}

// Specialization to build LeanVec-based Vamana index with provided training data
Status DynamicVamanaIndexLeanVec::build(
    DynamicVamanaIndex** index,
    size_t dim,
    MetricType metric,
    StorageKind storage_kind,
    const LeanVecTrainingData* training_data,
    const DynamicVamanaIndex::BuildParams& params,
    const DynamicVamanaIndex::SearchParams& default_search_params
) noexcept {
    *index = nullptr;
    SVS_RUNTIME_TRY_BEGIN
    auto training_data_impl =
        static_cast<const LeanVecTrainingDataManager*>(training_data)->impl_;
    auto impl = std::make_unique<DynamicVamanaIndexLeanVecImpl>(
        dim, metric, storage_kind, training_data_impl, params, default_search_params
    );
    *index = new DynamicVamanaIndexLeanVecImplManager{std::move(impl)};
    return Status_Ok;
    SVS_RUNTIME_TRY_END
}

} // namespace runtime
} // namespace svs
