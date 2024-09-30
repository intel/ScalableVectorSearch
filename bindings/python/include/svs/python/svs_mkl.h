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
#include <optional>

namespace svs::python {

/// Return `true` if svs was build to link with Intel(R) MKL. Otherwise, return `false`.
bool have_mkl();

///
/// @brief Return the number of threads used by Intel(R) MKL.
///
/// If ``have_mkl()`` returns false, return an empty optional.
std::optional<size_t> mkl_num_threads();

} // namespace svs::python
