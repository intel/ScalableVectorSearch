#pragma once

// svs-benchmark
#include "svs-benchmark/vamana/build.h"
#include "svs-benchmark/vamana/search.h"
#include "svs-benchmark/vamana/test.h"

// stl
#include <vector>

namespace svsbenchmark::vamana {

///// target-registration
// search
void register_uncompressed_static_search(vamana::StaticSearchDispatcher&);

// build
void register_uncompressed_static_build(vamana::StaticBuildDispatcher&);
void register_uncompressed_dynamic_build(vamana::DynamicBuildDispatcher&);

// test
std::vector<TestFunction> register_uncompressed_test_routines();

} // namespace svsbenchmark::vamana
