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

#pragma once

#include "svs/runtime/vamana_index.h"

#include "dynamic_vamana_index_impl.h"
#include "training_impl.h"

#include <svs/core/data.h>
#include <svs/orchestrators/dynamic_vamana.h>

#include <cassert>
#include <optional>
#include <span>

namespace svs {
namespace runtime {

// Vamana index implementation for LeanVec storage kinds
struct DynamicVamanaIndexLeanVecImpl : public DynamicVamanaIndexImpl {
    using LeanVecMatricesType = LeanVecTrainingDataImpl::LeanVecMatricesType;
    using allocator_type = svs::data::Blocked<svs::lib::Allocator<std::byte>>;

    DynamicVamanaIndexLeanVecImpl(
        std::unique_ptr<svs::DynamicVamana>&& impl,
        MetricType metric,
        StorageKind storage_kind
    )
        : DynamicVamanaIndexImpl{std::move(impl), metric, storage_kind}
        , leanvec_dims_{0}
        , leanvec_matrices_{std::nullopt} {
        check_storage_kind(storage_kind);
    }

    DynamicVamanaIndexLeanVecImpl(
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        const LeanVecTrainingDataImpl& training_data,
        const VamanaIndex::BuildParams& params,
        const VamanaIndex::SearchParams& default_search_params,
        const VamanaIndex::DynamicIndexParams& dynamic_index_params
    )
        : DynamicVamanaIndexImpl{dim, metric, storage_kind, params, default_search_params, dynamic_index_params}
        , leanvec_dims_{training_data.get_leanvec_dims()}
        , leanvec_matrices_{training_data.get_leanvec_matrices()} {
        check_storage_kind(storage_kind);
    }

    DynamicVamanaIndexLeanVecImpl(
        size_t dim,
        MetricType metric,
        StorageKind storage_kind,
        size_t leanvec_dims,
        const VamanaIndex::BuildParams& params,
        const VamanaIndex::SearchParams& default_search_params,
        const VamanaIndex::DynamicIndexParams& dynamic_index_params
    )
        : DynamicVamanaIndexImpl{dim, metric, storage_kind, params, default_search_params, dynamic_index_params}
        , leanvec_dims_{leanvec_dims}
        , leanvec_matrices_{std::nullopt} {
        check_storage_kind(storage_kind);
    }

    template <typename F, typename... Args>
    static auto dispatch_leanvec_storage_kind(StorageKind kind, F&& f, Args&&... args) {
        switch (kind) {
            case StorageKind::LeanVec4x4:
                return f(
                    storage::StorageType<StorageKind::LeanVec4x4, allocator_type>{},
                    std::forward<Args>(args)...
                );
            case StorageKind::LeanVec4x8:
                return f(
                    storage::StorageType<StorageKind::LeanVec4x8, allocator_type>{},
                    std::forward<Args>(args)...
                );
            case StorageKind::LeanVec8x8:
                return f(
                    storage::StorageType<StorageKind::LeanVec8x8, allocator_type>{},
                    std::forward<Args>(args)...
                );
            default:
                throw StatusException{
                    ErrorCode::INVALID_ARGUMENT, "SVS LeanVec storage kind required"};
        }
    }

    void init_impl(
        data::ConstSimpleDataView<float> data,
        std::span<const size_t> labels,
        lib::PowerOfTwo blocksize_bytes
    ) override {
        assert(storage::is_leanvec_storage(this->storage_kind_));
        if (this->deferred_compression_enabled() &&
            data.size() <
                this->dynamic_index_params_.deferred_compression_threshold) {
            // Delegate the build to the base class (which builds with the
            // uncompressed `initial_storage_kind`) and let our overridden
            // `setup_deferred_compression_swap` install a LeanVec-aware swap closure.
            DynamicVamanaIndexImpl::init_impl(data, labels, blocksize_bytes);
            return;
        }
        // Eager path (also taken when the very first add already meets the deferred
        // threshold): build the LeanVec backend directly with the configured
        // training data (matrices / leanvec_dims).
        if (this->deferred_compression_enabled()) {
            this->current_storage_kind_ = this->storage_kind_;
        }
        impl_.reset(dispatch_leanvec_storage_kind(
            this->storage_kind_,
            [this](
                auto&& tag,
                data::ConstSimpleDataView<float> data,
                std::span<const size_t> labels,
                lib::PowerOfTwo blocksize_bytes
            ) {
                using Tag = std::decay_t<decltype(tag)>;
                return DynamicVamanaIndexImpl::build_impl(
                    std::forward<Tag>(tag),
                    this->metric_type_,
                    this->vamana_build_parameters(),
                    data,
                    labels,
                    blocksize_bytes,
                    this->leanvec_dims_,
                    this->leanvec_matrices_
                );
            },
            data,
            labels,
            blocksize_bytes
        ));
    }

    void setup_deferred_compression_swap(
        StorageKind initial_kind, lib::PowerOfTwo blocksize_bytes
    ) override {
        // Capture LeanVec training data by value so the closure does not depend on
        // the lifetime of `*this` for those parameters.
        LeanVecTrainer trainer{leanvec_dims_, leanvec_matrices_};
        storage::dispatch_storage_kind<allocator_type>(
            initial_kind,
            [&](auto&& tag) {
                using Tag = std::decay_t<decltype(tag)>;
                this->install_swap_closure_with_trainer<Tag>(
                    blocksize_bytes, trainer
                );
            }
        );
    }

  protected:
    size_t leanvec_dims_;
    std::optional<LeanVecMatricesType> leanvec_matrices_;

    /// @brief Trainer used by the deferred-compression swap when the target is a
    /// LeanVec storage kind. Reuses pre-trained matrices when supplied; otherwise
    /// trains PCA matrices from the accumulated source dataset (the same path the
    /// eager builder uses when `leanvec_matrices_ == std::nullopt`).
    struct LeanVecTrainer {
        size_t leanvec_dims;
        std::optional<LeanVecMatricesType> leanvec_matrices;

        // Only LeanVec target storage kinds are supported.
        template <typename TargetTag>
        static constexpr bool supports =
            svs::leanvec::IsLeanDataset<typename TargetTag::type>;

        template <typename TargetTag, typename Source, typename Pool, typename Alloc>
        auto operator()(
            TargetTag, const Source& source, Pool& pool, const Alloc& allocator
        ) const {
            using TargetData = typename TargetTag::type;
            if constexpr (svs::leanvec::IsLeanDataset<TargetData>) {
                size_t d = leanvec_dims;
                if (d == 0) {
                    d = (source.dimensions() + 1) / 2;
                }
                return TargetData::reduce(
                    source,
                    leanvec_matrices,
                    pool,
                    0,
                    svs::lib::MaybeStatic{d},
                    allocator
                );
            } else {
                static_assert(
                    !sizeof(TargetData*),
                    "LeanVecTrainer instantiated for a non-LeanVec target type"
                );
            }
        }
    };

    StorageKind check_storage_kind(StorageKind kind) {
        if (!storage::is_leanvec_storage(kind)) {
            throw StatusException(
                ErrorCode::INVALID_ARGUMENT, "SVS LeanVec storage kind required"
            );
        }
        if (!svs::detail::lvq_leanvec_enabled()) {
            throw StatusException(
                ErrorCode::NOT_IMPLEMENTED,
                "LeanVec storage kind requested but not supported by CPU"
            );
        }
        return kind;
    }
};

} // namespace runtime
} // namespace svs
