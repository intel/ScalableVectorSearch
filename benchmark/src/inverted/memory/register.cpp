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
