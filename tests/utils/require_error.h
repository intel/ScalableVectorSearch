/*
 * Copyright 2024 Intel Corporation
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

#pragma once

#include "catch2/catch_test_macros.hpp"

// Check that the provided expression does not compile.
// See: https://github.com/catchorg/Catch2/issues/2610
#define SVS_REQUIRE_DOES_NOT_COMPILE(Type, ...)            \
    {                                                      \
        constexpr auto Result = [&]<typename TestType>() { \
            return requires { __VA_ARGS__; };              \
        }.template operator()<Type>();                     \
        CATCH_STATIC_REQUIRE_FALSE(Result);                \
    }

#define SVS_REQUIRE_COMPILES(Type, ...)                    \
    {                                                      \
        constexpr auto Result = [&]<typename TestType>() { \
            return requires { __VA_ARGS__; };              \
        }.template operator()<Type>();                     \
        CATCH_STATIC_REQUIRE(Result);                      \
    }
