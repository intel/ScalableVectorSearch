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

#include "svs_runtime_utils.h"

#include <algorithm>
#include <memory>
#include <variant>
#include <vector>

#include <svs/core/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/cpuid.h>
#include <svs/extensions/vamana/scalar.h>
#include <svs/lib/float16.h>
#include <svs/orchestrators/dynamic_vamana.h>
#include <svs/quantization/scalar/scalar.h>

#include SVS_LVQ_HEADER
#include SVS_LEANVEC_HEADER

namespace svs {
namespace runtime {

struct LeanVecTrainingDataImpl {
    LeanVecTrainingDataImpl(LeanVecMatricesType&& matrices)
        : leanvec_matrices{std::move(matrices)} {}

    LeanVecTrainingDataImpl(
        const svs::data::ConstSimpleDataView<float>& data, size_t leanvec_dims
    )
        : LeanVecTrainingDataImpl{compute_leanvec_matrices(data, leanvec_dims)} {}

    const LeanVecMatricesType& get_leanvec_matrices() const { return leanvec_matrices; }

    void save(std::ostream& out) const {
        lib::UniqueTempDirectory tempdir{"svs_leanvec_matrix_save"};
        svs::lib::save_to_disk(leanvec_matrices, tempdir);
        lib::DirectoryArchiver::pack(tempdir, out);
    }

    static LeanVecTrainingDataImpl load(std::istream& in) {
        lib::UniqueTempDirectory tempdir{"svs_leanvec_matrix_load"};
        lib::DirectoryArchiver::unpack(in, tempdir);
        return LeanVecTrainingDataImpl{
            svs::lib::load_from_disk<LeanVecMatricesType>(tempdir)};
    }

  private:
    LeanVecMatricesType leanvec_matrices;

    static LeanVecMatricesType compute_leanvec_matrices(
        const svs::data::ConstSimpleDataView<float>& data, size_t leanvec_dims
    ) {
        auto threadpool =
            svs::threads::ThreadPoolHandle(svs::threads::OMPThreadPool(omp_get_max_threads()
            ));

        auto means = svs::utils::compute_medioid(data, threadpool);
        auto matrix = svs::leanvec::compute_leanvec_matrix<svs::Dynamic, svs::Dynamic>(
            data, means, threadpool, svs::lib::MaybeStatic<svs::Dynamic>{leanvec_dims}
        );
        return LeanVecMatricesType{matrix, matrix};
    }
};

struct LeanVecTrainingDataManager : public svs::runtime::LeanVecTrainingData {
    LeanVecTrainingDataImpl impl_;

    LeanVecTrainingDataManager(LeanVecTrainingDataImpl impl)
        : impl_{std::move(impl)} {}

    Status save(std::ostream& out) const noexcept override {
        SVS_RUNTIME_TRY_BEGIN
        impl_.save(out);
        return Status_Ok;
        SVS_RUNTIME_TRY_END
    }
};

} // namespace runtime
} // namespace svs
