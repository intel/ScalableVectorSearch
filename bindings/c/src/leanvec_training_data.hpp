/*
 * Copyright 2026 Intel Corporation
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

#ifdef SVS_RUNTIME_ENABLE_LVQ_LEANVEC

#include "svs/c/svs_c.h"

#include <svs/core/data/simple.h>
#include <svs/cpuid.h>
#include <svs/lib/threads/threadpool.h>

#ifdef SVS_LEANVEC_HEADER
#include SVS_LEANVEC_HEADER
#else
#include <svs/leanvec/leanvec.h>
#endif

#include <cstddef>

namespace svs::c_runtime {

// Holds LeanVec dimensionality-reduction matrices trained from a data sample.
// Mirrors the runtime bindings' LeanVecTrainingData: matrices are computed once
// and later handed to LeanVecDataBuilder to reduce the dataset. When training
// queries are supplied the matrices are learned out-of-distribution (OOD),
// otherwise in-distribution (PCA) matrices are used for both data and queries.
class LeanVecTrainingData {
  public:
    using matrices_type = svs::leanvec::LeanVecMatrices<svs::Dynamic>;

    LeanVecTrainingData(
        svs::data::ConstSimpleDataView<float> data,
        svs::data::ConstSimpleDataView<float> queries,
        size_t leanvec_dims,
        svs::threads::ThreadPoolHandle& pool
    )
        : matrices_{compute_matrices(data, queries, leanvec_dims, pool)} {}

    size_t leanvec_dims() const { return matrices_.num_cols(); }
    const matrices_type& matrices() const { return matrices_; }

    static bool enabled() { return svs::detail::intel_enabled(); }

  private:
    matrices_type matrices_;

    static matrices_type compute_matrices(
        svs::data::ConstSimpleDataView<float> data,
        svs::data::ConstSimpleDataView<float> queries,
        size_t leanvec_dims,
        svs::threads::ThreadPoolHandle& pool
    );
};

} // namespace svs::c_runtime

#else // SVS_RUNTIME_ENABLE_LVQ_LEANVEC
namespace svs::c_runtime {
class LeanVecTrainingData {
  public:
    template <typename... Args> LeanVecTrainingData(Args&&...) {
        throw svs::c_runtime::not_implemented(
            "LeanVec training is not implemented in this build"
        );
    }

    static bool enabled() { return false; }
};
} // namespace svs::c_runtime

#endif // SVS_RUNTIME_ENABLE_LVQ_LEANVEC
