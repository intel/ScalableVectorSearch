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
