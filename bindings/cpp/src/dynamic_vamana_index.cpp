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

#include "svs/runtime/dynamic_vamana_index.h"

#include "dynamic_vamana_index_impl.h"
#include "svs_runtime_utils.h"

#ifdef SVS_LEANVEC_HEADER
#include "dynamic_vamana_index_leanvec_impl.h"
#endif

#include <svs/core/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/extensions/vamana/scalar.h>
#include <svs/lib/float16.h>
#include <svs/orchestrators/dynamic_vamana.h>
#include <svs/quantization/scalar/scalar.h>

#include <algorithm>
#include <memory>
#include <variant>
#include <vector>

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

    size_t blocksize_bytes() const noexcept { return impl_->blocksize_bytes(); }

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
} // namespace

// DynamicVamanaIndex interface implementation
Status DynamicVamanaIndex::check_storage_kind(StorageKind storage_kind) noexcept {
    bool supported = false;
    auto status = runtime_error_wrapper([&] {
        supported = storage::is_supported_storage_kind(storage_kind);
    });
    if (!status.ok()) {
        return status;
    }
    return supported ? Status_Ok
                     : Status(
                           ErrorCode::INVALID_ARGUMENT,
                           "The specified storage kind is not compatible with the "
                           "DynamicVamanaIndex"
                       );
}

Status DynamicVamanaIndex::check_params(
    const DynamicVamanaIndex::DynamicIndexParams& dynamic_index_params
) noexcept {
    constexpr static size_t kMaxBlockSizeExp = 30; // 1GB
    constexpr static size_t kMinBlockSizeExp = 12; // 4KB

    if (dynamic_index_params.blocksize_exp > kMaxBlockSizeExp)
        return Status(ErrorCode::INVALID_ARGUMENT, "Blocksize is too large");

    if (dynamic_index_params.blocksize_exp < kMinBlockSizeExp)
        return Status(ErrorCode::INVALID_ARGUMENT, "Blocksize is too small");

    return Status_Ok;
}

// ABI backward compatibility
Status DynamicVamanaIndex::build(
    DynamicVamanaIndex** index,
    size_t dim,
    MetricType metric,
    StorageKind storage_kind,
    const DynamicVamanaIndex::BuildParams& params,
    const DynamicVamanaIndex::SearchParams& default_search_params
) noexcept {
    return build(
        index,
        dim,
        metric,
        storage_kind,
        params,
        default_search_params,
        DynamicVamanaIndex::DynamicIndexParams{}
    );
}

Status DynamicVamanaIndex::build(
    DynamicVamanaIndex** index,
    size_t dim,
    MetricType metric,
    StorageKind storage_kind,
    const DynamicVamanaIndex::BuildParams& params,
    const DynamicVamanaIndex::SearchParams& default_search_params,
    const DynamicVamanaIndex::DynamicIndexParams& dynamic_index_params
) noexcept {
    using Impl = DynamicVamanaIndexImpl;
    *index = nullptr;

    auto status = DynamicVamanaIndex::check_params(dynamic_index_params);
    if (!status.ok())
        return status;

    return runtime_error_wrapper([&] {
        auto impl = std::make_unique<Impl>(
            dim, metric, storage_kind, params, default_search_params, dynamic_index_params
        );
        *index = new DynamicVamanaIndexManagerBase<Impl>{std::move(impl)};
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
    using Impl = DynamicVamanaIndexImpl;
    *index = nullptr;
    return runtime_error_wrapper([&] {
        std::unique_ptr<Impl> impl{Impl::load(in, metric, storage_kind)};
        *index = new DynamicVamanaIndexManagerBase<Impl>{std::move(impl)};
    });
}

// Specialization to build LeanVec-based Vamana index with specified leanvec dims
// ABI backward compatibility
Status DynamicVamanaIndexLeanVec::build(
    DynamicVamanaIndex** index,
    size_t dim,
    MetricType metric,
    StorageKind storage_kind,
    size_t leanvec_dims,
    const DynamicVamanaIndex::BuildParams& params,
    const DynamicVamanaIndex::SearchParams& default_search_params
) noexcept {
    return build(
        index,
        dim,
        metric,
        storage_kind,
        leanvec_dims,
        params,
        default_search_params,
        DynamicVamanaIndex::DynamicIndexParams{}
    );
}

// Specialization to build LeanVec-based Vamana index with provided training data
// ABI backward compatibility
Status DynamicVamanaIndexLeanVec::build(
    DynamicVamanaIndex** index,
    size_t dim,
    MetricType metric,
    StorageKind storage_kind,
    const LeanVecTrainingData* training_data,
    const DynamicVamanaIndex::BuildParams& params,
    const DynamicVamanaIndex::SearchParams& default_search_params
) noexcept {
    return build(
        index,
        dim,
        metric,
        storage_kind,
        training_data,
        params,
        default_search_params,
        DynamicVamanaIndex::DynamicIndexParams{}
    );
}

#ifdef SVS_LEANVEC_HEADER
// Specialization to build LeanVec-based Vamana index with specified leanvec dims
Status DynamicVamanaIndexLeanVec::build(
    DynamicVamanaIndex** index,
    size_t dim,
    MetricType metric,
    StorageKind storage_kind,
    size_t leanvec_dims,
    const DynamicVamanaIndex::BuildParams& params,
    const DynamicVamanaIndex::SearchParams& default_search_params,
    const DynamicVamanaIndex::DynamicIndexParams& dynamic_index_params
) noexcept {
    using Impl = DynamicVamanaIndexLeanVecImpl;
    *index = nullptr;

    auto status = DynamicVamanaIndex::check_params(dynamic_index_params);
    if (!status.ok())
        return status;

    return runtime_error_wrapper([&] {
        auto impl = std::make_unique<Impl>(
            dim,
            metric,
            storage_kind,
            leanvec_dims,
            params,
            default_search_params,
            dynamic_index_params
        );
        *index = new DynamicVamanaIndexManagerBase<Impl>{std::move(impl)};
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
    const DynamicVamanaIndex::SearchParams& default_search_params,
    const DynamicVamanaIndex::DynamicIndexParams& dynamic_index_params
) noexcept {
    using Impl = DynamicVamanaIndexLeanVecImpl;
    *index = nullptr;

    auto status = DynamicVamanaIndex::check_params(dynamic_index_params);
    if (!status.ok())
        return status;

    return runtime_error_wrapper([&] {
        auto training_data_impl =
            static_cast<const LeanVecTrainingDataManager*>(training_data)->impl_;
        std::cout << "Building DynamicVamanaIndexLeanVec with training data..."
                  << std::endl;

        auto matrices = training_data_impl.get_leanvec_matrices();

        auto print10 = [](const float* vec) {
            for (size_t i = 0; i < 10; ++i) {
                printf("   %+.6f", static_cast<float>(vec[i]));
            }
        };

        std::cout << "data_matrix[0][:10]    =";
        print10(matrices.view_data_matrix().get_datum(0).data());
        std::cout << std::endl;
        std::cout << "query_matrix[0][:10]   =";
        print10(matrices.view_query_matrix().get_datum(0).data());
        std::cout << std::endl;
        auto impl = std::make_unique<Impl>(
            dim,
            metric,
            storage_kind,
            training_data_impl,
            params,
            default_search_params,
            dynamic_index_params
        );
        *index = new DynamicVamanaIndexManagerBase<Impl>{std::move(impl)};
    });
}

#else  // SVS_LEANVEC_HEADER
// LeanVec storage kind is not supported in this build configuration
Status DynamicVamanaIndexLeanVec::
    build(DynamicVamanaIndex**, size_t, MetricType, StorageKind, size_t, const DynamicVamanaIndex::BuildParams&, const DynamicVamanaIndex::SearchParams&, const DynamicVamanaIndex::DynamicIndexParams&) noexcept {
    return Status(
        ErrorCode::NOT_IMPLEMENTED,
        "DynamicVamanaIndexLeanVec is not supported in this build configuration."
    );
}

Status DynamicVamanaIndexLeanVec::
    build(DynamicVamanaIndex**, size_t, MetricType, StorageKind, const LeanVecTrainingData*, const DynamicVamanaIndex::BuildParams&, const DynamicVamanaIndex::SearchParams&, const DynamicVamanaIndex::DynamicIndexParams&) noexcept {
    return Status(
        ErrorCode::NOT_IMPLEMENTED,
        "DynamicVamanaIndexLeanVec is not supported in this build configuration."
    );
}
#endif // SVS_LEANVEC_HEADER
} // namespace runtime
} // namespace svs
