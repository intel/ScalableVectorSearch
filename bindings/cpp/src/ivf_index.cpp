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

// Manager class for Static IVF Index
struct StaticIVFIndexManager : public StaticIVFIndex {
    std::unique_ptr<StaticIVFIndexImpl> impl_;

    StaticIVFIndexManager(std::unique_ptr<StaticIVFIndexImpl> impl)
        : impl_{std::move(impl)} {
        assert(impl_ != nullptr);
    }

    ~StaticIVFIndexManager() override = default;

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

    Status save(std::ostream& out) const noexcept override {
        return runtime_error_wrapper([&] { impl_->save(out); });
    }
};

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

// IVFIndex interface implementation
IVFIndex::~IVFIndex() = default;

// StaticIVFIndex interface implementation
Status StaticIVFIndex::check_storage_kind(StorageKind storage_kind) noexcept {
    if (ivf_storage::is_supported_storage_kind(storage_kind)) {
        return Status_Ok;
    } else {
        return Status{
            ErrorCode::INVALID_ARGUMENT,
            "StaticIVFIndex only supports FP32 and FP16 storage kinds"};
    }
}

Status StaticIVFIndex::build(
    StaticIVFIndex** index,
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
        auto impl = std::make_unique<StaticIVFIndexImpl>(
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

        *index = new StaticIVFIndexManager{std::move(impl)};
    });
}

Status StaticIVFIndex::destroy(StaticIVFIndex* index) noexcept {
    return runtime_error_wrapper([&] { delete index; });
}

Status StaticIVFIndex::load(
    StaticIVFIndex** index,
    std::istream& in,
    MetricType metric,
    StorageKind storage_kind,
    size_t num_threads,
    size_t intra_query_threads
) noexcept {
    *index = nullptr;
    return runtime_error_wrapper([&] {
        std::unique_ptr<StaticIVFIndexImpl> impl{StaticIVFIndexImpl::load(
            in, metric, storage_kind, num_threads, intra_query_threads
        )};
        *index = new StaticIVFIndexManager{std::move(impl)};
    });
}

// DynamicIVFIndex interface implementation
Status DynamicIVFIndex::check_storage_kind(StorageKind storage_kind) noexcept {
    if (ivf_storage::is_supported_storage_kind(storage_kind)) {
        return Status_Ok;
    } else {
        return Status{
            ErrorCode::INVALID_ARGUMENT,
            "DynamicIVFIndex only supports FP32 and FP16 storage kinds"};
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
        std::unique_ptr<DynamicIVFIndexImpl> impl{DynamicIVFIndexImpl::load(
            in, metric, storage_kind, num_threads, intra_query_threads
        )};
        *index = new DynamicIVFIndexManager{std::move(impl)};
    });
}

} // namespace runtime
} // namespace svs
