/**
 *    Copyright (C) 2023, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
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
