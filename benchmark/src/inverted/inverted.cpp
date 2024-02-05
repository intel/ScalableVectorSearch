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
