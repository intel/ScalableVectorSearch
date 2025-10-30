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
#include "IndexSVSImplDefs.h"
#include "IndexSVSVamanaImpl.h"

#include <memory>
#include <span>

namespace svs {
namespace leanvec {
template <size_t Extent> struct LeanVecMatrices;
} // namespace leanvec

namespace runtime {

struct SVS_RUNTIME_API IndexSVSVamanaLeanVecImpl : IndexSVSVamanaImpl {
    enum LeanVecLevel { LeanVec4x4, LeanVec4x8, LeanVec8x8 };

    static IndexSVSVamanaLeanVecImpl* build(
        size_t dim,
        MetricType metric,
        const BuildParams& params,
        size_t leanvec_dims,
        LeanVecLevel leanvec_level
    ) noexcept;

    void reset() noexcept override;

    Status train(size_t n, const float* x) noexcept;

    Status deserialize_impl(std::istream& in) noexcept override;

    bool is_trained() const noexcept { return trained; }

  protected:
    IndexSVSVamanaLeanVecImpl();

    IndexSVSVamanaLeanVecImpl(
        size_t d,
        size_t degree,
        MetricType metric = MetricType::L2,
        size_t leanvec_dims = 0,
        LeanVecLevel leanvec_level = LeanVecLevel::LeanVec4x4
    );

    ~IndexSVSVamanaLeanVecImpl() override;

    Status init_impl(size_t n, const float* x) noexcept override;

    size_t leanvec_d;
    LeanVecLevel leanvec_level;
    std::unique_ptr<svs::leanvec::LeanVecMatrices<std::dynamic_extent>> leanvec_matrix;
    bool trained = false;
};

} // namespace runtime
} // namespace svs
