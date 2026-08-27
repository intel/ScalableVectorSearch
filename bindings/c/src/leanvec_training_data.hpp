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
#include <svs/core/medioid.h>
#include <svs/lib/static.h>
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
        : leanvec_dims_{leanvec_dims}
        , matrices_{
              queries.size() == 0 ? compute_pca(data, leanvec_dims, pool)
                                  : compute_ood(data, queries, leanvec_dims, pool)} {}

    size_t leanvec_dims() const { return leanvec_dims_; }
    const matrices_type& matrices() const { return matrices_; }

  private:
    size_t leanvec_dims_;
    matrices_type matrices_;

    static matrices_type compute_pca(
        svs::data::ConstSimpleDataView<float> data,
        size_t leanvec_dims,
        svs::threads::ThreadPoolHandle& pool
    ) {
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

    static matrices_type compute_ood(
        svs::data::ConstSimpleDataView<float> data,
        svs::data::ConstSimpleDataView<float> queries,
        size_t leanvec_dims,
        svs::threads::ThreadPoolHandle& pool
    ) {
        return svs::leanvec::compute_leanvec_matrices_ood<svs::Dynamic>(
            data, queries, pool, svs::lib::MaybeStatic{leanvec_dims}
        );
    }
};

} // namespace svs::c_runtime

#endif // SVS_RUNTIME_ENABLE_LVQ_LEANVEC
