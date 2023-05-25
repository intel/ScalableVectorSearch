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

// svs
#include "svs/lib/memory.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stdlib
#include <vector>

CATCH_TEST_CASE("Allocator Traits", "[allocators]") {
    namespace lib = svs::lib;
    namespace memory = lib::memory;
    using vector_traits = memory::PointerTraits<std::vector<int>>;
    CATCH_REQUIRE(memory::may_trivially_construct<lib::VectorAllocator>);
    CATCH_REQUIRE(memory::may_trivially_construct<vector_traits::allocator>);
}
