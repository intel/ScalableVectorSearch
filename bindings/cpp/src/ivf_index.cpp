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

#include "svs/runtime/ivf_index.h"

#include "ivf_index_impl.h"
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

// Manager class for IVF Index
struct IVFIndexManager : public IVFIndex {
    std::unique_ptr<IVFIndexImpl> impl_;

    IVFIndexManager(std::unique_ptr<IVFIndexImpl> impl)
        : impl_{std::move(impl)} {
        assert(impl_ != nullptr);
    }

    ~IVFIndexManager() override = default;

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

    Status save(std::ostream& out) const noexcept override {
        return runtime_error_wrapper([&] { impl_->save(out); });
    }

    Status set_num_threads(size_t num_threads) noexcept override {
        return runtime_error_wrapper([&] { impl_->set_num_threads(num_threads); });
    }

    Status get_num_threads(size_t* num_threads) const noexcept override {
        return runtime_error_wrapper([&] { *num_threads = impl_->get_num_threads(); });
    }

    Status set_intra_query_threads(size_t intra_query_threads) noexcept override {
        return runtime_error_wrapper([&] {
            impl_->set_intra_query_threads(intra_query_threads);
        });
    }

    Status get_intra_query_threads(size_t* intra_query_threads) const noexcept override {
        return runtime_error_wrapper([&] {
            *intra_query_threads = impl_->get_intra_query_threads();
        });
    }
};

} // namespace

// IVFIndex interface implementation
IVFIndex::~IVFIndex() = default;

// IVFIndex interface implementation
Status IVFIndex::check_storage_kind(StorageKind storage_kind) noexcept {
    if (ivf_storage::is_supported_storage_kind(storage_kind)) {
        return Status_Ok;
    } else {
        return Status{
            ErrorCode::INVALID_ARGUMENT,
            "IVFIndex only supports FP32, FP16, SQI8, and LVQ storage kinds"};
    }
}

Status IVFIndex::build(
    IVFIndex** index,
    size_t dim,
    MetricType metric,
    StorageKind storage_kind,
    size_t n,
    const float* data,
    const IVFIndex::BuildParams& params,
    const IVFIndex::SearchParams& default_search_params,
    size_t num_threads,
    size_t intra_query_threads
) noexcept {
    *index = nullptr;
    return runtime_error_wrapper([&] {
        auto impl = std::make_unique<IVFIndexImpl>(
            dim,
            metric,
            storage_kind,
            params,
            default_search_params,
            num_threads,
            intra_query_threads
        );

        // Build with provided data
        svs::data::ConstSimpleDataView<float> data_view{data, n, dim};
        impl->build(data_view);

        *index = new IVFIndexManager{std::move(impl)};
    });
}

Status IVFIndex::build(
    IVFIndex** index,
    size_t dim,
    MetricType metric,
    StorageKind storage_kind,
    size_t n,
    const float* data,
    const IVFIndex::BuildParams& params,
    size_t num_threads,
    size_t intra_query_threads
) noexcept {
    SearchParams default_search_params;
    return build(
        index,
        dim,
        metric,
        storage_kind,
        n,
        data,
        params,
        default_search_params,
        num_threads,
        intra_query_threads
    );
}

Status IVFIndex::destroy(IVFIndex* index) noexcept {
    return runtime_error_wrapper([&] { delete index; });
}

Status IVFIndex::load(
    IVFIndex** index,
    std::istream& in,
    MetricType metric,
    StorageKind storage_kind,
    size_t num_threads,
    size_t intra_query_threads
) noexcept {
    *index = nullptr;
    return runtime_error_wrapper([&] {
        std::unique_ptr<IVFIndexImpl> impl;
#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
        if (ivf_storage::is_leanvec_storage_kind(storage_kind)) {
            impl.reset(IVFIndexLeanVecImpl::load(
                in, metric, storage_kind, num_threads, intra_query_threads
            ));
        } else
#endif
        {
            impl.reset(IVFIndexImpl::load(
                in, metric, storage_kind, num_threads, intra_query_threads
            ));
        }
        *index = new IVFIndexManager{std::move(impl)};
    });
}

// IVFIndexLeanVec implementations
#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
Status IVFIndexLeanVec::build(
    IVFIndex** index,
    size_t dim,
    MetricType metric,
    StorageKind storage_kind,
    size_t n,
    const float* data,
    size_t leanvec_dims,
    const IVFIndex::BuildParams& params,
    const IVFIndex::SearchParams& default_search_params,
    size_t num_threads,
    size_t intra_query_threads
) noexcept {
    *index = nullptr;
    return runtime_error_wrapper([&] {
        auto impl = std::make_unique<IVFIndexLeanVecImpl>(
            dim,
            metric,
            storage_kind,
            leanvec_dims,
            params,
            default_search_params,
            num_threads,
            intra_query_threads
        );

        svs::data::ConstSimpleDataView<float> data_view{data, n, dim};
        impl->build(data_view);

        *index = new IVFIndexManager{std::move(impl)};
    });
}

Status IVFIndexLeanVec::build(
    IVFIndex** index,
    size_t dim,
    MetricType metric,
    StorageKind storage_kind,
    size_t n,
    const float* data,
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
        auto impl = std::make_unique<IVFIndexLeanVecImpl>(
            dim,
            metric,
            storage_kind,
            training_data_impl,
            params,
            default_search_params,
            num_threads,
            intra_query_threads
        );

        svs::data::ConstSimpleDataView<float> data_view{data, n, dim};
        impl->build(data_view);

        *index = new IVFIndexManager{std::move(impl)};
    });
}
#else  // SVS_RUNTIME_HAVE_LVQ_LEANVEC
Status IVFIndexLeanVec::build(
    IVFIndex**,
    size_t,
    MetricType,
    StorageKind,
    size_t,
    const float*,
    size_t,
    const IVFIndex::BuildParams&,
    const IVFIndex::SearchParams&,
    size_t,
    size_t
) noexcept {
    return Status(
        ErrorCode::NOT_IMPLEMENTED,
        "IVFIndexLeanVec is not supported in this build configuration."
    );
}

Status IVFIndexLeanVec::build(
    IVFIndex**,
    size_t,
    MetricType,
    StorageKind,
    size_t,
    const float*,
    const LeanVecTrainingData*,
    const IVFIndex::BuildParams&,
    const IVFIndex::SearchParams&,
    size_t,
    size_t
) noexcept {
    return Status(
        ErrorCode::NOT_IMPLEMENTED,
        "IVFIndexLeanVec is not supported in this build configuration."
    );
}
#endif // SVS_RUNTIME_HAVE_LVQ_LEANVEC

} // namespace runtime
} // namespace svs
