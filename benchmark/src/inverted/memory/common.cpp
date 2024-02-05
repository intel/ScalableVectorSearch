// Corresponding header.
#include "svs-benchmark/inverted/memory/common.h"

// svs
#include "svs/lib/exception.h"

// stl
#include <string_view>

namespace svsbenchmark::inverted::memory {

ClusterStrategy parse_strategy(std::string_view str) {
    for (auto strategy : cluster_strategies) {
        if (str == name(strategy)) {
            return strategy;
        }
    }
    throw ANNEXCEPTION("Unrecognized strategy {}", str);
}

} // namespace svsbenchmark::inverted::memory
