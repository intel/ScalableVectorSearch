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

#include "svs/runtime/training.h"

#include "svs_runtime_utils.h"

#ifdef SVS_LEANVEC_HEADER
#include "training_impl.h"

namespace svs {
namespace runtime {

LeanVecTrainingData::~LeanVecTrainingData() = default;

Status LeanVecTrainingData::build(
    LeanVecTrainingData** training_data,
    size_t dim,
    size_t n,
    const float* x,
    size_t n_train,
    const float* q,
    size_t leanvec_dims
) noexcept {
    return runtime_error_wrapper([&] {
        const auto data = svs::data::ConstSimpleDataView<float>(x, n, dim);
        if (!q) {
            // ID training
            *training_data =
                new LeanVecTrainingDataManager{LeanVecTrainingDataImpl{data, leanvec_dims}};
            return;
        } else {
            // OOD training
            const auto queries = svs::data::ConstSimpleDataView<float>(q, n_train, dim);
            *training_data = new LeanVecTrainingDataManager{
                LeanVecTrainingDataImpl{data, queries, leanvec_dims}};
            return;
        }
    });
}

Status LeanVecTrainingData::destroy(LeanVecTrainingData* training_data) noexcept {
    return runtime_error_wrapper([&] { delete training_data; });
}

Status
LeanVecTrainingData::load(LeanVecTrainingData** training_data, std::istream& in) noexcept {
    return runtime_error_wrapper([&] {
        *training_data = new LeanVecTrainingDataManager{LeanVecTrainingDataImpl::load(in)};
    });
}
} // namespace runtime
} // namespace svs

#else  // SVS_LEANVEC_HEADER
namespace svs {
namespace runtime {
LeanVecTrainingData::~LeanVecTrainingData() = default;
Status LeanVecTrainingData::build(
    LeanVecTrainingData** SVS_UNUSED(training_data),
    size_t SVS_UNUSED(dim),
    size_t SVS_UNUSED(n),
    const float* SVS_UNUSED(x),
    size_t SVS_UNUSED(leanvec_dims)
) noexcept {
    return Status(
        ErrorCode::NOT_IMPLEMENTED,
        "LeanVecTrainingData is not supported in this build configuration."
    );
}
Status LeanVecTrainingData::destroy(LeanVecTrainingData* SVS_UNUSED(training_data)
) noexcept {
    return Status(
        ErrorCode::NOT_IMPLEMENTED,
        "LeanVecTrainingData is not supported in this build configuration."
    );
}
Status LeanVecTrainingData::load(
    LeanVecTrainingData** SVS_UNUSED(training_data), std::istream& SVS_UNUSED(in)
) noexcept {
    return Status(
        ErrorCode::NOT_IMPLEMENTED,
        "LeanVecTrainingData is not supported in this build configuration."
    );
}
} // namespace runtime
} // namespace svs
#endif // SVS_LEANVEC_HEADER
