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
