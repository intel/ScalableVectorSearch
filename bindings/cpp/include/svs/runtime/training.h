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
#include <svs/runtime/api_defs.h>

#include <cstddef>
#include <istream>
#include <ostream>

namespace svs {
namespace runtime {
namespace v0 {

struct SVS_RUNTIME_API LeanVecTrainingData {
    virtual ~LeanVecTrainingData();

    /* Build LeanVec training data (compression matrices) from the provided
     * data.
     * @param training_data Output parameter to the created training data object
     * @param dim Dimensionality of the input data and queries
     * @param n Number of data points and queries
     * @param x Pointer to the input data
     * @param leanvec_dims Number of dimensions in the resulting LeanVec data
     */
    static Status build(
        LeanVecTrainingData** training_data,
        size_t dim,
        size_t n,
        const float* x,
        size_t leanvec_dims
    ) noexcept;

    static Status destroy(LeanVecTrainingData* training_data) noexcept;

    virtual Status save(std::ostream& out) const noexcept = 0;
    static Status load(LeanVecTrainingData** training_data, std::istream& in) noexcept;
};

} // namespace v0

namespace v1 {

struct SVS_RUNTIME_API LeanVecTrainingData : public v0::LeanVecTrainingData {
    using v0::LeanVecTrainingData::destroy;
    using v0::LeanVecTrainingData::save;

    /* Build LeanVec training data (compression matrices) from the provided
     * data.
     * Accepts optional training queries for out-of-distribution training.
     * @param training_data Output parameter to the created training data object
     * @param dim Dimensionality of the input data and queries
     * @param n Number of data points and queries
     * @param x Pointer to the input data
     * @param n_train Number of training queries (can be 0)
     * @param q Pointer to the training queries (can be nullptr)
     * @param leanvec_dims Number of dimensions in the resulting LeanVec data
     */
    static Status build(
        LeanVecTrainingData** training_data,
        size_t dim,
        size_t n,
        const float* x,
        size_t n_train,
        const float* q,
        size_t leanvec_dims
    ) noexcept;

    static Status load(LeanVecTrainingData** training_data, std::istream& in) noexcept;
};

} // namespace v1
} // namespace runtime
} // namespace svs
