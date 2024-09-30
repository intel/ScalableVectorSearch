/*
 * Copyright (C) 2024 Intel Corporation
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
