#pragma once

// svs-benchmark
#include "svs-benchmark/inverted/memory/build.h"
#include "svs-benchmark/inverted/memory/search.h"
#include "svs-benchmark/inverted/memory/test.h"

namespace svsbenchmark::inverted::memory {
// build
void register_lvq_memory_build(svsbenchmark::inverted::memory::MemoryBuildDispatcher&);
// search
void register_lvq_memory_search(svsbenchmark::inverted::memory::MemorySearchDispatcher&);
// test
std::vector<inverted::memory::TestFunction> register_lvq_test_routines();
} // namespace svsbenchmark::inverted::memory
