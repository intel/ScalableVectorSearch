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
#include "svs/core/data.h"

namespace svs_test {

/// @brief Check the quality of the LVQ reconstruction.
void check_lvq_reconstruction(
    svs::data::ConstSimpleDataView<float> original,
    svs::data::ConstSimpleDataView<float> reconstructed,
    size_t primary,
    size_t residual
);

} // namespace svs_test
