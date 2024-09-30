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
#include "tests/utils/require_error.h"

// stl
#include <type_traits>

namespace {

template <typename T>
    requires std::is_arithmetic_v<T>
struct Add {
    static constexpr bool value = true;
};

} // namespace

CATCH_TEST_CASE("SFINAE Checker") {
    CATCH_STATIC_REQUIRE(Add<int>::value);
    SVS_REQUIRE_COMPILES(int, Add<TestType>::value);
    SVS_REQUIRE_DOES_NOT_COMPILE(char*, Add<TestType>::value);
}
