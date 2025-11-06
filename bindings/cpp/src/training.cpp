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

#include "training.h"
#include "svs_runtime_utils.h"
#include "training_impl.h"

namespace svs {
namespace runtime {

LeanVecTrainingData::~LeanVecTrainingData() = default;

Status LeanVecTrainingData::build(
    LeanVecTrainingData** training_data,
    size_t dim,
    size_t n,
    const float* x,
    size_t leanvec_dims
) noexcept {
    SVS_RUNTIME_TRY_BEGIN
    const auto data = svs::data::ConstSimpleDataView<float>(x, n, dim);
    *training_data =
        new LeanVecTrainingDataManager{LeanVecTrainingDataImpl{data, leanvec_dims}};
    return Status_Ok;
    SVS_RUNTIME_TRY_END
}

Status LeanVecTrainingData::destroy(LeanVecTrainingData* training_data) noexcept {
    SVS_RUNTIME_TRY_BEGIN
    delete training_data;
    return Status_Ok;
    SVS_RUNTIME_TRY_END
}

Status LeanVecTrainingData::save(std::ostream& /*out*/) const noexcept {
    // providing an implementation of a virtual function to anchor vtable
    return {ErrorCode::NOT_IMPLEMENTED, "Not implemented"};
}

Status
LeanVecTrainingData::load(LeanVecTrainingData** training_data, std::istream& in) noexcept {
    SVS_RUNTIME_TRY_BEGIN
    *training_data = new LeanVecTrainingDataManager{LeanVecTrainingDataImpl::load(in)};
    return Status_Ok;
    SVS_RUNTIME_TRY_END
}
} // namespace runtime
} // namespace svs
