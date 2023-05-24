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
