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

#include <cstddef>
#include <memory>

#include <svs/core/data.h>
#include <svs/core/medioid.h>
#include <svs/lib/saveload.h>
#include <svs/lib/threads.h>

#include SVS_LVQ_HEADER
#include SVS_LEANVEC_HEADER

namespace svs {
namespace runtime {

struct LeanVecTrainingDataImpl {
    LeanVecTrainingDataImpl(LeanVecMatricesType&& matrices)
        : leanvec_dims_{matrices.view_data_matrix().dimensions()}
        , leanvec_matrices_{std::move(matrices)} {}

    LeanVecTrainingDataImpl(
        const svs::data::ConstSimpleDataView<float>& data, size_t leanvec_dims
    )
        : leanvec_dims_{leanvec_dims}
        , leanvec_matrices_{compute_leanvec_matrices(data, leanvec_dims)} {}

    size_t get_leanvec_dims() const { return leanvec_dims_; }
    const LeanVecMatricesType& get_leanvec_matrices() const { return leanvec_matrices_; }

    void save(std::ostream& out) const {
        lib::UniqueTempDirectory tempdir{"svs_leanvec_matrix_save"};
        svs::lib::save_to_disk(leanvec_matrices_, tempdir);
        lib::DirectoryArchiver::pack(tempdir, out);
    }

    static LeanVecTrainingDataImpl load(std::istream& in) {
        lib::UniqueTempDirectory tempdir{"svs_leanvec_matrix_load"};
        lib::DirectoryArchiver::unpack(in, tempdir);
        return LeanVecTrainingDataImpl{
            svs::lib::load_from_disk<LeanVecMatricesType>(tempdir)};
    }

  private:
    size_t leanvec_dims_;
    LeanVecMatricesType leanvec_matrices_;

    static LeanVecMatricesType compute_leanvec_matrices(
        const svs::data::ConstSimpleDataView<float>& data, size_t leanvec_dims
    ) {
        auto threadpool = default_threadpool();

        auto means = svs::utils::compute_medioid(data, threadpool);
        auto matrix = svs::leanvec::compute_leanvec_matrix<svs::Dynamic, svs::Dynamic>(
            data, means, threadpool, svs::lib::MaybeStatic<svs::Dynamic>{leanvec_dims}
        );
        // Intentionally using the same matrix for both elements of LeanVecMatricesType.
        // This may be required by downstream code expecting two matrices, even if they are
        // identical.
        return LeanVecMatricesType{matrix, matrix};
    }
};

struct LeanVecTrainingDataManager : public svs::runtime::LeanVecTrainingData {
    LeanVecTrainingDataManager(LeanVecTrainingDataImpl impl)
        : impl_{std::move(impl)} {}

    Status save(std::ostream& out) const noexcept override {
        SVS_RUNTIME_TRY_BEGIN
        impl_.save(out);
        return Status_Ok;
        SVS_RUNTIME_TRY_END
    }

    LeanVecTrainingDataImpl impl_;
};

} // namespace runtime
} // namespace svs
