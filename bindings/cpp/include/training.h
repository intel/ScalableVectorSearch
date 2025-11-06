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

#include <cstddef>
#include <istream>
#include <ostream>

namespace svs {
namespace runtime {
struct SVS_RUNTIME_API LeanVecTrainingData {
    virtual ~LeanVecTrainingData() = 0;
    static Status build(
        LeanVecTrainingData** training_data,
        size_t dim,
        size_t n,
        const float* x,
        size_t leanvec_dims
    ) noexcept;

    static Status destroy(LeanVecTrainingData* training_data) noexcept;

    virtual Status save(std::ostream& out) const noexcept;
    static Status load(LeanVecTrainingData** training_data, std::istream& in) noexcept;
};
} // namespace runtime
} // namespace svs
