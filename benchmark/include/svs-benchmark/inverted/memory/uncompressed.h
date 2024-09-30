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
