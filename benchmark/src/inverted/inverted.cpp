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
#include "svs-benchmark/inverted/inverted.h"
#include "svs-benchmark/benchmark.h"

#include "svs-benchmark/inverted/memory/register.h"

namespace svsbenchmark::inverted {

// Register executables for the Inverted index.
void register_executables(ExecutableDispatcher& dispatcher) {
    svsbenchmark::inverted::memory::register_executables(dispatcher);
}

} // namespace svsbenchmark::inverted
