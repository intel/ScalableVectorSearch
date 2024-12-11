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

#pragma once

#include <sstream>
#include <string>
#include <tuple>

#include "catch2/benchmark/catch_benchmark_all.hpp"

#include "tests/utils/utils.h"

/////
///// Naming Benchmarks for better discovery.
/////

namespace svs_test {
template <typename T, typename... Args>
void type_names(std::ostringstream& stream, bool first = true) {
    if (!first) {
        stream << "_";
    }
    stream << type_name<T>();
    if constexpr (sizeof...(Args) > 0) {
        type_names<Args...>(stream, false);
    }
}

template <typename T> struct TypeNameGenerator {};
template <typename... Args> struct TypeNameGenerator<std::tuple<Args...>> {
    static void generate(std::ostringstream& stream) {
        if constexpr (sizeof...(Args) > 0) {
            stream << "_<";
            type_names<Args...>(stream);
            stream << ">";
        }
    };
};

template <typename T, typename... Args>
void value_names(std::ostringstream& stream, T x, Args... args) {
    stream << "_" << x;
    if constexpr (sizeof...(Args) > 0) {
        value_names(stream, args...);
    }
}

// Create a named tests with the given test parameters.
template <typename T, typename... Args>
std::string benchmark_name(const std::string& prefix, Args... args) {
    // Initialize the stream stream and seek to the end to append more characters.
    std::ostringstream stream{prefix, std::ios_base::ate};
    TypeNameGenerator<T>::generate(stream);
    if constexpr (sizeof...(Args) > 0) {
        value_names(stream, args...);
    }
    return stream.str();
}
} // namespace svs_test

// Named Benchmarks
#define REMOVE_PARENS(...) __VA_ARGS__
#define BENCHMARK_NAME_TEMPLATE(name, types, ...) \
    svs_test::benchmark_name<std::tuple<REMOVE_PARENS types>>(name, ##__VA_ARGS__)

// Wrappers for benchmark naming
#define NAMED_TEMPLATE_BENCHMARK(name, types, ...) \
    CATCH_BENCHMARK((BENCHMARK_NAME_TEMPLATE(name, types, ##__VA_ARGS__)))
