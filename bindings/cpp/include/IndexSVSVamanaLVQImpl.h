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
#include "IndexSVSVamanaImpl.h"
#include "version.h"

namespace svs::runtime::v0 {

struct SVS_RUNTIME_API IndexSVSVamanaLVQImpl : IndexSVSVamanaImpl {
    enum LVQLevel { LVQ4x0, LVQ4x4, LVQ4x8 };
    static IndexSVSVamanaLVQImpl*
    build(size_t dim, MetricType metric, const BuildParams& params, LVQLevel lvq) noexcept;

    Status deserialize_impl(std::istream& in) noexcept override;

  protected:
    IndexSVSVamanaLVQImpl();
    IndexSVSVamanaLVQImpl(size_t d, size_t degree, MetricType metric, LVQLevel lvq_level);

    ~IndexSVSVamanaLVQImpl() override;

    Status init_impl(size_t n, const float* x) noexcept override;

    LVQLevel lvq_level;
};

} // namespace svs::runtime::v0

// Bring current version APIs to parent namespace
namespace svs::runtime {
using v0::IndexSVSVamanaLVQImpl;
}
