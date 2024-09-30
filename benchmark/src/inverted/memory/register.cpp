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

// svs-benchmark
#include "svs-benchmark/inverted/memory/register.h"
#include "svs-benchmark/inverted/memory/build.h"

namespace svsbenchmark::inverted::memory {

void register_executables(ExecutableDispatcher& dispatcher) {
    dispatcher.register_executable(svsbenchmark::inverted::memory::static_build());
    dispatcher.register_executable(svsbenchmark::inverted::memory::static_search());
    dispatcher.register_executable(svsbenchmark::inverted::memory::test_generator());
}

} // namespace svsbenchmark::inverted::memory
