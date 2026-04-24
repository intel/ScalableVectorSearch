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

#include "svs/runtime/vamana_index.h"

#include "svs_runtime_utils.h"
#include "vamana_index_impl.h"

namespace svs {
namespace runtime {

namespace {
template <typename Impl = VamanaIndexImpl>
struct VamanaIndexManagerBase : public VamanaIndex {
    std::unique_ptr<Impl> impl_;

    VamanaIndexManagerBase(std::unique_ptr<Impl> impl)
        : impl_{std::move(impl)} {
        assert(impl_ != nullptr);
    }

    VamanaIndexManagerBase(const VamanaIndexManagerBase&) = delete;
    VamanaIndexManagerBase& operator=(const VamanaIndexManagerBase&) = delete;
    VamanaIndexManagerBase(VamanaIndexManagerBase&&) = default;
    VamanaIndexManagerBase& operator=(VamanaIndexManagerBase&&) = default;
    ~VamanaIndexManagerBase() override = default;

    Status add(size_t n, const float* x) noexcept override {
        return runtime_error_wrapper([&] {
            svs::data::ConstSimpleDataView<float> data{x, n, impl_->dimensions()};
            impl_->add(data);
            return Status_Ok;
        });
    }

    Status reset() noexcept override {
        return runtime_error_wrapper([&] {
            impl_->reset();
            return Status_Ok;
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

    Status save(std::ostream& out) const noexcept override {
        return runtime_error_wrapper([&] { impl_->save(out); });
    }
};
} // namespace

// VamanaIndex interface implementation
VamanaIndex::~VamanaIndex() = default;

Status VamanaIndex::check_storage_kind(StorageKind storage_kind) noexcept {
    bool supported = false;
    auto status = runtime_error_wrapper([&] {
        supported = storage::is_supported_storage_kind(storage_kind);
    });
    if (!status.ok()) {
        return status;
    }
    return supported
               ? Status_Ok
               : Status(
                     ErrorCode::INVALID_ARGUMENT,
                     "The specified storage kind is not compatible with the VamanaIndex"
                 );
}

Status VamanaIndex::build(
    VamanaIndex** index,
    size_t dim,
    MetricType metric,
    StorageKind storage_kind,
    const VamanaIndex::BuildParams& params,
    const VamanaIndex::SearchParams& default_search_params
) noexcept {
    using Impl = VamanaIndexImpl;
    *index = nullptr;

    return runtime_error_wrapper([&] {
        auto impl = std::make_unique<Impl>(
            dim, metric, storage_kind, params, default_search_params
        );
        *index = new VamanaIndexManagerBase<Impl>{std::move(impl)};
    });
}

Status VamanaIndex::destroy(VamanaIndex* index) noexcept {
    return runtime_error_wrapper([&] { delete index; });
}

Status VamanaIndex::load(
    VamanaIndex** index, std::istream& in, MetricType metric, StorageKind storage_kind
) noexcept {
    using Impl = VamanaIndexImpl;
    *index = nullptr;
    return runtime_error_wrapper([&] {
        std::unique_ptr<Impl> impl{Impl::load(in, metric, storage_kind)};
        *index = new VamanaIndexManagerBase<Impl>{std::move(impl)};
    });
}

Status VamanaIndex::map_to_file(
    VamanaIndex** index, const char* path, MetricType metric, StorageKind storage_kind
) noexcept {
    using Impl = VamanaIndexImpl;
    *index = nullptr;
    return runtime_error_wrapper([&] {
        std::filesystem::path fs_path(path);
        auto is = std::make_unique<svs::io::mmstream>(fs_path);
        std::unique_ptr<Impl> impl{
            Impl::map_to_stream(std::move(is), metric, storage_kind)};
        *index = new VamanaIndexManagerBase<Impl>{std::move(impl)};
    });
}

Status VamanaIndex::map_to_memory(
    VamanaIndex** index,
    void* data,
    size_t size,
    MetricType metric,
    StorageKind storage_kind
) noexcept {
    using Impl = VamanaIndexImpl;
    *index = nullptr;
    return runtime_error_wrapper([&] {
        auto sp = std::span(reinterpret_cast<char*>(data), size);
        auto is = std::make_unique<svs::io::ispanstream>(sp);
        std::unique_ptr<Impl> impl{
            Impl::map_to_stream(std::move(is), metric, storage_kind)};
        *index = new VamanaIndexManagerBase<Impl>{std::move(impl)};
    });
}

#ifdef SVS_RUNTIME_HAVE_LVQ_LEANVEC
// Specialization to build LeanVec-based Vamana index with specified leanvec dims
Status VamanaIndexLeanVec::build(
    VamanaIndex** index,
    size_t dim,
    MetricType metric,
    StorageKind storage_kind,
    size_t leanvec_dims,
    const VamanaIndex::BuildParams& params,
    const VamanaIndex::SearchParams& default_search_params
) noexcept {
    using Impl = VamanaIndexLeanVecImpl;
    *index = nullptr;

    return runtime_error_wrapper([&] {
        auto impl = std::make_unique<Impl>(
            dim, metric, storage_kind, leanvec_dims, params, default_search_params
        );
        *index = new VamanaIndexManagerBase<Impl>{std::move(impl)};
    });
}

// Specialization to build LeanVec-based Vamana index with provided training data
Status VamanaIndexLeanVec::build(
    VamanaIndex** index,
    size_t dim,
    MetricType metric,
    StorageKind storage_kind,
    const LeanVecTrainingData* training_data,
    const VamanaIndex::BuildParams& params,
    const VamanaIndex::SearchParams& default_search_params
) noexcept {
    using Impl = VamanaIndexLeanVecImpl;
    *index = nullptr;

    return runtime_error_wrapper([&] {
        if (training_data == nullptr) {
            throw StatusException{
                ErrorCode::INVALID_ARGUMENT, "Training data must not be null"};
        }
        auto training_data_impl =
            static_cast<const LeanVecTrainingDataManager*>(training_data)->impl_;
        auto impl = std::make_unique<Impl>(
            dim, metric, storage_kind, training_data_impl, params, default_search_params
        );
        *index = new VamanaIndexManagerBase<Impl>{std::move(impl)};
    });
}

#else  // SVS_RUNTIME_HAVE_LVQ_LEANVEC
// LeanVec storage kind is not supported in this build configuration
Status VamanaIndexLeanVec::
    build(VamanaIndex**, size_t, MetricType, StorageKind, size_t, const VamanaIndex::BuildParams&, const VamanaIndex::SearchParams&) noexcept {
    return Status(
        ErrorCode::NOT_IMPLEMENTED,
        "VamanaIndexLeanVec is not supported in this build configuration."
    );
}

Status VamanaIndexLeanVec::
    build(VamanaIndex**, size_t, MetricType, StorageKind, const LeanVecTrainingData*, const VamanaIndex::BuildParams&, const VamanaIndex::SearchParams&) noexcept {
    return Status(
        ErrorCode::NOT_IMPLEMENTED,
        "VamanaIndexLeanVec is not supported in this build configuration."
    );
}
#endif // SVS_RUNTIME_HAVE_LVQ_LEANVEC
} // namespace runtime
} // namespace svs
