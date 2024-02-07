#pragma once

#include "svs-benchmark/datasets.h"

#include "svs/lib/dispatcher.h"
#include "svs/quantization/lvq/lvq.h"

namespace svsbenchmark {

/// A fully typed - uncompressed dataset used as the target for dispatch conversion.
template <
    size_t Primary,
    size_t Residual,
    svs::quantization::lvq::LVQPackingStrategy Strategy>
struct TypedLVQ {};

namespace detail {
template <typename T> struct LVQStrategyMap;

template <> struct LVQStrategyMap<svs::quantization::lvq::Sequential> {
    static constexpr LVQPackingStrategy kind = LVQPackingStrategy::Sequential;
};
template <> struct LVQStrategyMap<svs::quantization::lvq::Turbo<16, 8>> {
    static constexpr LVQPackingStrategy kind = LVQPackingStrategy::Turbo16x8;
};
template <> struct LVQStrategyMap<svs::quantization::lvq::Turbo<16, 4>> {
    static constexpr LVQPackingStrategy kind = LVQPackingStrategy::Turbo16x4;
};

} // namespace detail

template <typename T>
inline constexpr svsbenchmark::LVQPackingStrategy lvq_packing_strategy_v =
    detail::LVQStrategyMap<T>::kind;

} // namespace svsbenchmark

namespace svs::lib {

template <
    size_t Primary,
    size_t Residual,
    svs::quantization::lvq::LVQPackingStrategy Strategy>
struct DispatchConverter<
    svsbenchmark::LVQ,
    svsbenchmark::TypedLVQ<Primary, Residual, Strategy>> {
    using To = svsbenchmark::TypedLVQ<Primary, Residual, Strategy>;

    static bool match(const svsbenchmark::LVQ& x) {
        // Match Kinds
        if (x.primary_ != Primary || x.residual_ != Residual) {
            return false;
        }

        // Match Strategies
        return x.strategy_ == svsbenchmark::detail::LVQStrategyMap<Strategy>::kind;
    }

    static To convert([[maybe_unused]] svsbenchmark::LVQ x) {
        assert(match(x));
        return To{};
    }

    static std::string description() {
        return fmt::format(
            "lvq {}x{} - {}",
            Primary,
            Residual,
            name(svsbenchmark::detail::LVQStrategyMap<Strategy>::kind)
        );
    }
};

template <
    size_t Primary,
    size_t Residual,
    svs::quantization::lvq::LVQPackingStrategy Strategy>
struct DispatchConverter<
    svsbenchmark::Dataset,
    svsbenchmark::TypedLVQ<Primary, Residual, Strategy>> {
    using From = svsbenchmark::LVQ;
    using To = svsbenchmark::TypedLVQ<Primary, Residual, Strategy>;

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
