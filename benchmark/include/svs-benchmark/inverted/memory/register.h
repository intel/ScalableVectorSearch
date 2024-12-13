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

// svsbenchmark
#include "svs-benchmark/benchmark.h"

// stl
#include <memory>
#include <string_view>

namespace svsbenchmark::inverted::memory {

// Main entry-point for memory-based executables.
void register_executables(ExecutableDispatcher& dispatcher);

///// Executables

/// Perform static index construction and query the constructed index.
std::unique_ptr<Benchmark> static_build();

inline constexpr std::string_view static_build_name() {
    return "inverted_static_memory_build";
}

/// Perform static searching
std::unique_ptr<Benchmark> static_search();

inline constexpr std::string_view static_search_name() {
    return "inverted_static_memory_search";
}

std::unique_ptr<Benchmark> test_generator();

} // namespace svsbenchmark::inverted::memory
