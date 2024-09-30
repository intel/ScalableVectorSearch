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
#pragma once

// Header under test
#include "svs/quantization/lvq/compressed.h"

// tests
#include "tests/utils/generators.h"

namespace test_q {

///
/// Create a random number generator for numbers expressible using a small-word compression
/// with the given sign and number of bits.
///
template <typename Sign, size_t Bits> auto create_generator() {
    using Encoding = svs::quantization::lvq::Encoding<Sign, Bits>;
    using value_type = typename Encoding::value_type;
    value_type lower = Encoding::min();
    value_type upper = Encoding::max();
    return svs_test::make_generator<value_type>(lower, upper);
}

} // namespace test_q
