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

#include "svs/runtime/dynamic_ivf_index.h"

#include "dynamic_ivf_index_impl.h"
#include "svs_runtime_utils.h"

#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
#include "training_impl.h"
#endif

#include <svs/core/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>

#include <algorithm>
#include <memory>
#include <span>
#include <variant>

namespace svs {
namespace runtime {

namespace {

// Manager class for Dynamic IVF Index
struct DynamicIVFIndexManager : public DynamicIVFIndex {
    std::unique_ptr<DynamicIVFIndexImpl> impl_;

    DynamicIVFIndexManager(std::unique_ptr<DynamicIVFIndexImpl> impl)
        : impl_{std::move(impl)} {
        assert(impl_ != nullptr);
    }

    ~DynamicIVFIndexManager() override = default;

    Status search(
        size_t n,
        const float* x,
        size_t k,
        float* distances,
        size_t* labels,
        const SearchParams* params = nullptr
    ) const noexcept override {
        return runtime_error_wrapper([&] {
            auto result = svs::QueryResultView<size_t>{
                svs::MatrixView<size_t>{svs::make_dims(n, k), labels},
                svs::MatrixView<float>{svs::make_dims(n, k), distances}};
            auto queries = svs::data::ConstSimpleDataView<float>(x, n, impl_->dimensions());
            impl_->search(result, queries, params);
        });
    }

    Status
    add(size_t n, const size_t* labels, const float* x, bool reuse_empty
    ) noexcept override {
        return runtime_error_wrapper([&] {
            svs::data::ConstSimpleDataView<float> data{x, n, impl_->dimensions()};
            std::span<const size_t> lbls(labels, n);
            impl_->add(data, lbls, reuse_empty);
        });
    }

    Status remove(size_t n, const size_t* labels) noexcept override {
        return runtime_error_wrapper([&] {
            std::span<const size_t> lbls(labels, n);
            impl_->remove(lbls);
        });
    }

    Status
    remove_selected(size_t* num_removed, const IDFilter& selector) noexcept override {
        return runtime_error_wrapper([&] {
            *num_removed = impl_->remove_selected(selector);
        });
    }

    Status has_id(bool* exists, size_t id) const noexcept override {
        return runtime_error_wrapper([&] { *exists = impl_->has_id(id); });
    }

    Status consolidate() noexcept override {
        return runtime_error_wrapper([&] { impl_->consolidate(); });
    }

    Status compact(size_t batchsize) noexcept override {
        return runtime_error_wrapper([&] { impl_->compact(batchsize); });
    }

    Status save(std::ostream& out) const noexcept override {
        return runtime_error_wrapper([&] { impl_->save(out); });
    }
};

} // namespace

// DynamicIVFIndex interface implementation
Status DynamicIVFIndex::check_storage_kind(StorageKind storage_kind) noexcept {
    if (ivf_storage::is_supported_storage_kind(storage_kind)) {
        return Status_Ok;
    } else {
        return Status{
            ErrorCode::INVALID_ARGUMENT,
            "DynamicIVFIndex only supports FP32, FP16, SQI8, and LVQ storage kinds"};
    }
}

Status DynamicIVFIndex::build(
    DynamicIVFIndex** index,
    size_t dim,
    MetricType metric,
    StorageKind storage_kind,
    size_t n,
    const float* data,
    const size_t* labels,
    const IVFIndex::BuildParams& params,
    const IVFIndex::SearchParams& default_search_params,
    size_t num_threads,
    size_t intra_query_threads
) noexcept {
    *index = nullptr;
    return runtime_error_wrapper([&] {
        auto impl = std::make_unique<DynamicIVFIndexImpl>(
            dim,
            metric,
            storage_kind,
            params,
            default_search_params,
            num_threads,
            intra_query_threads
        );

        // Build with provided data if any
        if (n > 0 && data != nullptr && labels != nullptr) {
            svs::data::ConstSimpleDataView<float> data_view{data, n, dim};
            std::span<const size_t> labels_span{labels, n};
            impl->build(data_view, labels_span);
        }

        *index = new DynamicIVFIndexManager{std::move(impl)};
    });
}

Status DynamicIVFIndex::destroy(DynamicIVFIndex* index) noexcept {
    return runtime_error_wrapper([&] { delete index; });
}

Status DynamicIVFIndex::load(
    DynamicIVFIndex** index,
    std::istream& in,
    MetricType metric,
    StorageKind storage_kind,
    size_t num_threads,
    size_t intra_query_threads
) noexcept {
    *index = nullptr;
    return runtime_error_wrapper([&] {
        std::unique_ptr<DynamicIVFIndexImpl> impl;
#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
        if (ivf_storage::is_leanvec_storage_kind(storage_kind)) {
            impl.reset(DynamicIVFIndexLeanVecImpl::load(
                in, metric, storage_kind, num_threads, intra_query_threads
            ));
        } else
#endif
        {
            impl.reset(DynamicIVFIndexImpl::load(
                in, metric, storage_kind, num_threads, intra_query_threads
            ));
        }
        *index = new DynamicIVFIndexManager{std::move(impl)};
    });
}

// DynamicIVFIndexLeanVec implementations
#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
Status DynamicIVFIndexLeanVec::build(
    DynamicIVFIndex** index,
    size_t dim,
    MetricType metric,
    StorageKind storage_kind,
    size_t n,
    const float* data,
    const size_t* labels,
    size_t leanvec_dims,
    const IVFIndex::BuildParams& params,
    const IVFIndex::SearchParams& default_search_params,
    size_t num_threads,
    size_t intra_query_threads
) noexcept {
    *index = nullptr;
    return runtime_error_wrapper([&] {
        auto impl = std::make_unique<DynamicIVFIndexLeanVecImpl>(
            dim,
            metric,
            storage_kind,
            leanvec_dims,
            params,
            default_search_params,
            num_threads,
            intra_query_threads
        );

        if (n > 0 && data != nullptr && labels != nullptr) {
            svs::data::ConstSimpleDataView<float> data_view{data, n, dim};
            std::span<const size_t> labels_span{labels, n};
            impl->build(data_view, labels_span);
        }

        *index = new DynamicIVFIndexManager{std::move(impl)};
    });
}

Status DynamicIVFIndexLeanVec::build(
    DynamicIVFIndex** index,
    size_t dim,
    MetricType metric,
    StorageKind storage_kind,
    size_t n,
    const float* data,
    const size_t* labels,
    const LeanVecTrainingData* training_data,
    const IVFIndex::BuildParams& params,
    const IVFIndex::SearchParams& default_search_params,
    size_t num_threads,
    size_t intra_query_threads
) noexcept {
    *index = nullptr;
    return runtime_error_wrapper([&] {
        auto training_data_impl =
            static_cast<const LeanVecTrainingDataManager*>(training_data)->impl_;
        auto impl = std::make_unique<DynamicIVFIndexLeanVecImpl>(
            dim,
            metric,
            storage_kind,
            training_data_impl,
            params,
            default_search_params,
            num_threads,
            intra_query_threads
        );

        if (n > 0 && data != nullptr && labels != nullptr) {
            svs::data::ConstSimpleDataView<float> data_view{data, n, dim};
            std::span<const size_t> labels_span{labels, n};
            impl->build(data_view, labels_span);
        }

        *index = new DynamicIVFIndexManager{std::move(impl)};
    });
}
#else  // SVS_RUNTIME_HAVE_LVQ_LEANVEC
Status DynamicIVFIndexLeanVec::build(
    DynamicIVFIndex**,
    size_t,
    MetricType,
    StorageKind,
    size_t,
    const float*,
    const size_t*,
    size_t,
    const IVFIndex::BuildParams&,
    const IVFIndex::SearchParams&,
    size_t,
    size_t
) noexcept {
    return Status(
        ErrorCode::NOT_IMPLEMENTED,
        "DynamicIVFIndexLeanVec is not supported in this build configuration."
    );
}

Status DynamicIVFIndexLeanVec::build(
    DynamicIVFIndex**,
    size_t,
    MetricType,
    StorageKind,
    size_t,
    const float*,
    const size_t*,
    const LeanVecTrainingData*,
    const IVFIndex::BuildParams&,
    const IVFIndex::SearchParams&,
    size_t,
    size_t
) noexcept {
    return Status(
        ErrorCode::NOT_IMPLEMENTED,
        "DynamicIVFIndexLeanVec is not supported in this build configuration."
    );
}
#endif // SVS_RUNTIME_HAVE_LVQ_LEANVEC

} // namespace runtime
} // namespace svs
