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
