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
