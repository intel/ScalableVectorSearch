/*
 * Copyright 2023 Intel Corporation
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
