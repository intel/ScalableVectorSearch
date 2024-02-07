#pragma once

// svs-benchmark
#include "svs-benchmark/inverted/memory/build.h"
#include "svs-benchmark/inverted/memory/search.h"
#include "svs-benchmark/inverted/memory/test.h"

namespace svsbenchmark::inverted::memory {
// build
void register_uncompressed_memory_build(memory::MemoryBuildDispatcher&);
// search
void register_uncompressed_memory_search(memory::MemorySearchDispatcher&);
// test
std::vector<inverted::memory::TestFunction> register_uncompressed_test_routines();
} // namespace svsbenchmark::inverted::memory
