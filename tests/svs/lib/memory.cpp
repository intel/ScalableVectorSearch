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
// svs
#include "svs/lib/memory.h"
#include "svs/lib/misc.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <memory>

CATCH_TEST_CASE("Allocator", "[lib][memory]") {
    auto alloc = svs::lib::Allocator<float>();
    using A = decltype(alloc);
    float* p = std::allocator_traits<A>::allocate(alloc, 10);
    std::allocator_traits<A>::deallocate(alloc, p, 10);

    CATCH_STATIC_REQUIRE(std::is_same_v<typename A::value_type, float>);
    auto other = svs::lib::rebind_allocator<int64_t>(alloc);
    using O = decltype(other);
    CATCH_STATIC_REQUIRE(std::is_same_v<typename O::value_type, int64_t>);
}
