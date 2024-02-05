#pragma once

#include "svs-benchmark/datasets.h"

#include "svs/lib/dispatcher.h"

namespace svsbenchmark {

/// A fully typed - uncompressed dataset used as the target for dispatch conversion.
template <typename T> struct TypedUncompressed {};

} // namespace svsbenchmark

namespace svs::lib {

template <typename T>
struct DispatchConverter<svsbenchmark::Uncompressed, svsbenchmark::TypedUncompressed<T>> {
    static bool match(const svsbenchmark::Uncompressed& x) {
        return x.data_type_ == svs::datatype_v<T>;
    }

    static svsbenchmark::TypedUncompressed<T>
    convert([[maybe_unused]] svsbenchmark::Uncompressed x) {
        assert(match(x));
        return svsbenchmark::TypedUncompressed<T>();
    }

    static std::string description() {
        return fmt::format("uncompressed ({})", svs::datatype_v<T>);
    }
};

template <typename T>
struct DispatchConverter<svsbenchmark::Dataset, svsbenchmark::TypedUncompressed<T>> {
    using From = svsbenchmark::Uncompressed;
    using To = svsbenchmark::TypedUncompressed<T>;
    static int64_t match(const svsbenchmark::Dataset& x) {
        if (auto* ptr = std::get_if<From>(&x.kinds_)) {
            return svs::lib::dispatch_match<From, To>(*ptr);
        }
        return svs::lib::invalid_match;
    }
    static To convert(svsbenchmark::Dataset x) {
        return svs::lib::dispatch_convert<From, To>(std::get<From>(std::move(x.kinds_)));
    }

    static std::string description() { return svs::lib::dispatch_description<From, To>(); }
};
} // namespace svs::lib
