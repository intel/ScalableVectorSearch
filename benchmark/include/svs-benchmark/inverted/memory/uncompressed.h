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
