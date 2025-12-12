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

#include SVS_LEANVEC_HEADER

#include <cassert>
#include <optional>
#include <span>

namespace svs {
namespace runtime {

// Vamana index implementation for LeanVec storage kinds
struct DynamicVamanaIndexLeanVecImpl : public DynamicVamanaIndexImpl {
    using LeanVecMatricesType = LeanVecTrainingDataImpl::LeanVecMatricesType;

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
        const VamanaIndex::SearchParams& default_search_params
    )
        : DynamicVamanaIndexImpl{dim, metric, storage_kind, params, default_search_params}
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
        const VamanaIndex::SearchParams& default_search_params
    )
        : DynamicVamanaIndexImpl{dim, metric, storage_kind, params, default_search_params}
        , leanvec_dims_{leanvec_dims}
        , leanvec_matrices_{std::nullopt} {
        check_storage_kind(storage_kind);
    }

    template <typename F, typename... Args>
    static auto dispatch_leanvec_storage_kind(StorageKind kind, F&& f, Args&&... args) {
        switch (kind) {
            case StorageKind::LeanVec4x4:
                return f(storage::LeanVec4x4Tag{}, std::forward<Args>(args)...);
            case StorageKind::LeanVec4x8:
                return f(storage::LeanVec4x8Tag{}, std::forward<Args>(args)...);
            case StorageKind::LeanVec8x8:
                return f(storage::LeanVec8x8Tag{}, std::forward<Args>(args)...);
            default:
                throw StatusException{
                    ErrorCode::INVALID_ARGUMENT, "SVS LeanVec storage kind required"};
        }
    }

    void init_impl(data::ConstSimpleDataView<float> data, std::span<const size_t> labels)
        override {
        assert(storage::is_leanvec_storage(this->storage_kind_));
        impl_.reset(dispatch_leanvec_storage_kind(
            this->storage_kind_,
            [this](
                auto&& tag,
                data::ConstSimpleDataView<float> data,
                std::span<const size_t> labels
            ) {
                using Tag = std::decay_t<decltype(tag)>;
                return DynamicVamanaIndexImpl::build_impl(
                    std::forward<Tag>(tag),
                    this->metric_type_,
                    this->vamana_build_parameters(),
                    data,
                    labels,
                    this->leanvec_dims_,
                    this->leanvec_matrices_
                );
            },
            data,
            labels
        ));
    }

  protected:
    size_t leanvec_dims_;
    std::optional<LeanVecMatricesType> leanvec_matrices_;

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
