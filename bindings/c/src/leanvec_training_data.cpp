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

#ifdef SVS_RUNTIME_ENABLE_LVQ_LEANVEC

#include "leanvec_training_data.hpp"

#include "error.hpp"

#include <svs/core/medioid.h>
#include <svs/lib/static.h>

namespace svs::c_runtime {

LeanVecTrainingData::matrices_type LeanVecTrainingData::compute_matrices(
    svs::data::ConstSimpleDataView<float> data,
    svs::data::ConstSimpleDataView<float> queries,
    size_t leanvec_dims,
    svs::threads::ThreadPoolHandle& pool
) {
    if (!enabled()) {
        throw svs::c_runtime::unsupported_hw(
            "LeanVec training is not supported on this hardware"
        );
    }

    // If queries are provided, compute out-of-distribution (OOD) matrices.
    if (queries.size() > 0) {
        return svs::leanvec::compute_leanvec_matrices_ood<svs::Dynamic>(
            data, queries, pool, svs::lib::MaybeStatic{leanvec_dims}
        );
    }

    // Else: PCA path (in-distribution)
    auto means = svs::utils::compute_medioid(data, pool);
    auto matrix = svs::leanvec::compute_leanvec_matrix<svs::Dynamic, svs::Dynamic>(
        data, means, pool, svs::lib::MaybeStatic{leanvec_dims}
    );
    // A copy is used for the query matrix: in PCA mode data and query
    // transforms are identical, and passing the same object twice trips
    // use-after-move warnings and DenseArray double-free issues.
    auto query_matrix = matrix;
    return matrices_type{std::move(matrix), std::move(query_matrix)};
}

} // namespace svs::c_runtime

#endif // SVS_RUNTIME_ENABLE_LVQ_LEANVEC
