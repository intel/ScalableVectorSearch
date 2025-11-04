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

#include <svs/leanvec/leanvec.h>

#include <iostream>
#include <memory>
#include <span>

namespace svs {
namespace runtime {
namespace detail {

struct TrainingInfoImpl {
    virtual ~TrainingInfoImpl() = default;

    virtual void serialize(std::ostream& out) const = 0;
    virtual void deserialize(std::istream& in) = 0;
};

// TrainingInfo wrapper around pre-computed leanvec matrix
struct LeanVecTrainingInfoImpl : public TrainingInfoImpl {
    LeanVecTrainingInfoImpl(svs::leanvec::LeanVecMatrices<svs::Dynamic> matrix);

    void serialize(std::ostream& out) const override;
    void deserialize(std::istream& in) override;

    svs::leanvec::LeanVecMatrices<svs::Dynamic> leanvec_matrix;
};

} // namespace detail
} // namespace runtime
} // namespace svs
