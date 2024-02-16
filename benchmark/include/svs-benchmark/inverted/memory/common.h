#pragma once

// svs-benchmark
#include "svs-benchmark/benchmark.h"

// svs
#include "svs/index/inverted/memory_based.h"
#include "svs/lib/saveload.h"

// stl
#include <string>
#include <string_view>

namespace svsbenchmark::inverted::memory {

// The backend strategy to use for the in-memory version.
enum class ClusterStrategy { Sparse, Dense };
constexpr inline std::array<ClusterStrategy, 2> cluster_strategies = {
    ClusterStrategy::Sparse,
    ClusterStrategy::Dense,
};

constexpr std::string_view name(ClusterStrategy s) {
    switch (s) {
        using enum ClusterStrategy;
        case Sparse: {
            return "sparse";
        }
        case Dense: {
            return "dense";
        }
    }
    throw ANNEXCEPTION("Unhandled cluster strategy!");
}

ClusterStrategy parse_strategy(std::string_view str);

} // namespace svsbenchmark::inverted::memory

namespace svs::lib {
template <> struct Saver<svsbenchmark::inverted::memory::ClusterStrategy> {
    static SaveNode save(svsbenchmark::inverted::memory::ClusterStrategy s) {
        return name(s);
    }
};

template <> struct Loader<svsbenchmark::inverted::memory::ClusterStrategy> {
    using toml_type = toml::value<std::string>;
    svsbenchmark::inverted::memory::ClusterStrategy load(const toml_type& v) const {
        return svsbenchmark::inverted::memory::parse_strategy(v.get());
    }
};
} // namespace svs::lib

template <>
struct svs::lib::DispatchConverter<
    svsbenchmark::inverted::memory::ClusterStrategy,
    svs::index::inverted::SparseStrategy> {
    using T = svsbenchmark::inverted::memory::ClusterStrategy;

    static constexpr bool match(T strategy) { return strategy == T::Sparse; }
    static constexpr svs::index::inverted::SparseStrategy convert(T SVS_UNUSED(strategy)) {
        return {};
    }

    static std::string description() {
        constexpr std::string_view v = name(T::Sparse);
        return std::string(v);
    }
};

template <>
struct svs::lib::DispatchConverter<
    svsbenchmark::inverted::memory::ClusterStrategy,
    svs::index::inverted::DenseStrategy> {
    using T = svsbenchmark::inverted::memory::ClusterStrategy;

    static constexpr bool match(T strategy) { return strategy == T::Dense; }
    static constexpr svs::index::inverted::DenseStrategy convert(T SVS_UNUSED(strategy)) {
        return {};
    }

    static std::string description() {
        constexpr std::string_view v = name(T::Dense);
        return std::string(v);
    }
};
