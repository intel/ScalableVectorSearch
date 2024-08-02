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
