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

#include <svs/runtime/dynamic_vamana_index.h>

#include "dynamic_vamana_index_impl.h"
#include "svs_runtime_utils.h"

#include <algorithm>
#include <memory>
#include <variant>
#include <vector>

#include <svs/core/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/extensions/vamana/scalar.h>
#include <svs/lib/float16.h>
#include <svs/orchestrators/dynamic_vamana.h>
#include <svs/quantization/scalar/scalar.h>

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
        return runtime_error_wrapper([&] {
            svs::data::ConstSimpleDataView<float> data{x, n, impl_->dimensions()};
            std::span<const size_t> lbls(labels, n);
            impl_->add(data, lbls);
        });
    }

    Status
    remove_selected(size_t* num_removed, const IDFilter& selector) noexcept override {
        return runtime_error_wrapper([&] {
            *num_removed = impl_->remove_selected(selector);
        });
    }

    Status remove(size_t n, const size_t* labels) noexcept override {
        return runtime_error_wrapper([&] {
            std::span<const size_t> lbls(labels, n);
            impl_->remove(lbls);
        });
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
        return runtime_error_wrapper([&] {
            auto result = svs::QueryResultView<size_t>{
                svs::MatrixView<size_t>{svs::make_dims(n, k), labels},
                svs::MatrixView<float>{svs::make_dims(n, k), distances}};
            auto queries = svs::data::ConstSimpleDataView<float>(x, n, impl_->dimensions());
            impl_->search(result, queries, params, filter);
        });
    }

    Status range_search(
        size_t n,
        const float* x,
        float radius,
        const ResultsAllocator& results,
        const SearchParams* params = nullptr,
        IDFilter* filter = nullptr
    ) const noexcept override {
        return runtime_error_wrapper([&] {
            auto queries = svs::data::ConstSimpleDataView<float>(x, n, impl_->dimensions());
            impl_->range_search(queries, radius, results, params, filter);
        });
    }

    Status reset() noexcept override {
        return runtime_error_wrapper([&] { impl_->reset(); });
    }

    Status save(std::ostream& out) const noexcept override {
        return runtime_error_wrapper([&] { impl_->save(out); });
    }
};

using DynamicVamanaIndexManager = DynamicVamanaIndexManagerBase<DynamicVamanaIndexImpl>;
#ifdef SVS_RUNTIME_ENABLE_LVQ_LEANVEC
using DynamicVamanaIndexLeanVecImplManager =
    DynamicVamanaIndexManagerBase<DynamicVamanaIndexLeanVecImpl>;
#endif

} // namespace

// DynamicVamanaIndex interface implementation
Status DynamicVamanaIndex::check_storage_kind(StorageKind storage_kind) noexcept {
    return runtime_error_wrapper([&] {
        return storage::is_supported_storage_kind(storage_kind)
                   ? Status_Ok
                   : Status(
                         ErrorCode::INVALID_ARGUMENT,
                         "The specified storage kind is not compatible with the "
                         "DynamicVamanaIndex"
                     );
    });
}

Status DynamicVamanaIndex::build(
    DynamicVamanaIndex** index,
    size_t dim,
    MetricType metric,
    StorageKind storage_kind,
    const DynamicVamanaIndex::BuildParams& params,
    const DynamicVamanaIndex::SearchParams& default_search_params
) noexcept {
    *index = nullptr;
    return runtime_error_wrapper([&] {
        auto impl = std::make_unique<DynamicVamanaIndexImpl>(
            dim, metric, storage_kind, params, default_search_params
        );
        *index = new DynamicVamanaIndexManager{std::move(impl)};
    });
}

Status DynamicVamanaIndex::destroy(DynamicVamanaIndex* index) noexcept {
    return runtime_error_wrapper([&] { delete index; });
}

Status DynamicVamanaIndex::load(
    DynamicVamanaIndex** index,
    std::istream& in,
    MetricType metric,
    StorageKind storage_kind
) noexcept {
    *index = nullptr;
    return runtime_error_wrapper([&] {
        std::unique_ptr<DynamicVamanaIndexImpl> impl{
            DynamicVamanaIndexImpl::load(in, metric, storage_kind)};
        *index = new DynamicVamanaIndexManager{std::move(impl)};
    });
}

#ifdef SVS_RUNTIME_ENABLE_LVQ_LEANVEC
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
    return runtime_error_wrapper([&] {
        auto impl = std::make_unique<DynamicVamanaIndexLeanVecImpl>(
            dim, metric, storage_kind, leanvec_dims, params, default_search_params
        );
        *index = new DynamicVamanaIndexLeanVecImplManager{std::move(impl)};
    });
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
    return runtime_error_wrapper([&] {
        auto training_data_impl =
            static_cast<const LeanVecTrainingDataManager*>(training_data)->impl_;
        auto impl = std::make_unique<DynamicVamanaIndexLeanVecImpl>(
            dim, metric, storage_kind, training_data_impl, params, default_search_params
        );
        *index = new DynamicVamanaIndexLeanVecImplManager{std::move(impl)};
    });
}
#endif

} // namespace runtime
} // namespace svs
