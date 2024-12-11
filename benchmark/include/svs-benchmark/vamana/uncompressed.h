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
