/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
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
