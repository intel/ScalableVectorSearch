#pragma once

// svs-benchmark
#include "svs-benchmark/benchmark.h"
#include "svs-benchmark/vamana/build.h"
#include "svs-benchmark/vamana/test.h"

// stl
#include <vector>

namespace svsbenchmark::vamana {

///// target-registration
// search
void register_lvq_static_search(vamana::StaticSearchDispatcher&);

// build
void register_lvq_static_build(vamana::StaticBuildDispatcher&);
void register_lvq_dynamic_build(vamana::DynamicBuildDispatcher&);

// test
std::vector<TestFunction> register_lvq_test_routines();

} // namespace svsbenchmark::vamana
