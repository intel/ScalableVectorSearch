/*
 * Copyright 2025 Intel Corporation
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

// header under test
#include "svs/third-party/eve.h"

// catch2
#include "catch2/catch_test_macros.hpp"

CATCH_TEST_CASE("evelib", "[eve]") {
    eve::wide<float, eve::fixed<4>> x = {1.0f, 2.0f, 3.0f, 4.0f};

    eve::wide<float, eve::fixed<4>> expected = {2.0f, 4.0f, 6.0f, 8.0f};
    CATCH_REQUIRE(eve::all(2 * x == expected));
}
