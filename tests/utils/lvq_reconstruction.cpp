/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
 */
// Associated header
#include "tests/utils/lvq_reconstruction.h"

// svs
#include "svs/core/data.h"
#include "svs/core/medioid.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <cmath>
#include <limits>

namespace svs_test {
namespace {

// A small margin to add to the LVQ error term to help account for rounding inaccuracies.
const double LVQ_MARGIN = 1.0 / 8192.0;

} // namespace

void check_lvq_reconstruction(
    svs::data::ConstSimpleDataView<float> original,
    svs::data::ConstSimpleDataView<float> reconstructed,
    size_t primary,
    size_t residual
) {
    CATCH_REQUIRE(original.size() == reconstructed.size());
    CATCH_REQUIRE(original.dimensions() == reconstructed.dimensions());

    // Find the medioid of the dataset.
    auto mock_threadpool = svs::threads::SequentialThreadPool();
    auto center = svs::utils::compute_medioid(original, mock_threadpool);

    for (size_t i = 0, imax = original.size(); i < imax; ++i) {
        auto o = original.get_datum(i);

        // Find the minimum and maximum values after removing the center.
        auto max = std::numeric_limits<double>::min();
        auto min = std::numeric_limits<double>::max();
        for (size_t j = 0; j < o.size(); ++j) {
            double c = o[j] - center[j];
            max = std::max(max, c);
            min = std::min(min, c);
        }

        auto error =
            (max - min) / ((std::pow(2.0, primary) - 1) * std::pow(2.0, residual)) +
            LVQ_MARGIN;
        auto r = reconstructed.get_datum(i);
        for (size_t j = 0; j < o.size(); ++j) {
            CATCH_REQUIRE(std::abs(o[j] - r[j]) <= error);
        }
    }
}

} // namespace svs_test
