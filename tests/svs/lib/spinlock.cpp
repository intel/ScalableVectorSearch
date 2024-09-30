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
// stdlib
#include <mutex>

// svs
#include "svs/lib/spinlock.h"

// catch2
#include "catch2/catch_test_macros.hpp"

CATCH_TEST_CASE("Spin Lock", "[core][utils][spinlock]") {
    svs::SpinLock spinlock{};
    CATCH_REQUIRE(spinlock.islocked() == false);
    {
        std::lock_guard guard{spinlock};
        CATCH_REQUIRE(spinlock.islocked() == true);
        for (size_t i = 0; i < 10; ++i) {
            CATCH_REQUIRE(spinlock.try_lock() == false);
        }
    }
    CATCH_REQUIRE(spinlock.islocked() == false);
}
