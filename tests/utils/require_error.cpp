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
