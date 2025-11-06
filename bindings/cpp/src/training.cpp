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

Status LeanVecTrainingData::build(
    LeanVecTrainingData** training_data,
    size_t dim,
    size_t n,
    const float* x,
    size_t leanvec_dims
) noexcept {
    return safe_runtime_call([&] {
        const auto data = svs::data::ConstSimpleDataView<float>(x, n, dim);
        *training_data =
            new LeanVecTrainingDataManager{LeanVecTrainingDataImpl{data, leanvec_dims}};
    });
}

Status LeanVecTrainingData::destroy(LeanVecTrainingData* training_data) noexcept {
    return safe_runtime_call([&] { delete training_data; });
}

Status
LeanVecTrainingData::load(LeanVecTrainingData** training_data, std::istream& in) noexcept {
    return safe_runtime_call([&] {
        *training_data = new LeanVecTrainingDataManager{LeanVecTrainingDataImpl::load(in)};
    });
}
} // namespace runtime
} // namespace svs
