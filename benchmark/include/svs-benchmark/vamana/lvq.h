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
