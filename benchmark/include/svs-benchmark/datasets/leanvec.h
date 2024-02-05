#pragma once

#include "svs-benchmark/datasets.h"

#include "svs/leanvec/leanvec.h"
#include "svs/lib/dispatcher.h"

namespace svsbenchmark {

namespace detail {

template <typename T> struct LeanVecKindMap;

template <> struct LeanVecKindMap<svs::Float16> {
    static constexpr LeanVecKind kind = LeanVecKind::float16;
};
template <> struct LeanVecKindMap<float> {
    static constexpr LeanVecKind kind = LeanVecKind::float32;
};
template <> struct LeanVecKindMap<svs::leanvec::UsingLVQ<8>> {
    static constexpr LeanVecKind kind = LeanVecKind::lvq8;
};
template <> struct LeanVecKindMap<svs::leanvec::UsingLVQ<4>> {
    static constexpr LeanVecKind kind = LeanVecKind::lvq4;
};

} // namespace detail

template <typename T>
inline constexpr svsbenchmark::LeanVecKind leanvec_kind_v = detail::LeanVecKindMap<T>::kind;

/// A fully typed - uncompressed dataset used as the target for dispatch conversion.
template <typename Primary, typename Secondary, size_t LeanVecDims> struct TypedLeanVec {
  public:
    std::optional<svs::leanvec::LeanVecMatrices<LeanVecDims>> transformation_;

  public:
    TypedLeanVec(
        const std::optional<std::filesystem::path>& data_matrix = {},
        const std::optional<std::filesystem::path>& query_matrix = {}
    )
        : transformation_{std::nullopt} {
        if (data_matrix.has_value() != query_matrix.has_value()) {
            throw ANNEXCEPTION("Either provide both the matrices or provide none of them!");
        }
        if (data_matrix) {
            using matrix_type =
                typename svs::leanvec::LeanVecMatrices<LeanVecDims>::leanvec_matrix_type;
            transformation_.emplace(
                matrix_type::load(data_matrix.value()),
                matrix_type::load(query_matrix.value())
            );
        }
    }
};

} // namespace svsbenchmark

namespace svs::lib {

template <typename Primary, typename Secondary, size_t LeanVecDims>
struct DispatchConverter<
    svsbenchmark::LeanVec,
    svsbenchmark::TypedLeanVec<Primary, Secondary, LeanVecDims>> {
    static bool match(const svsbenchmark::LeanVec& x) {
        if (x.primary_ != svsbenchmark::detail::LeanVecKindMap<Primary>::kind) {
            return false;
        }
        if (x.secondary_ != svsbenchmark::detail::LeanVecKindMap<Secondary>::kind) {
            return false;
        }
        return x.leanvec_dims_ == LeanVecDims;
    }

    static svsbenchmark::TypedLeanVec<Primary, Secondary, LeanVecDims>
    convert(svsbenchmark::LeanVec x) {
        return svsbenchmark::TypedLeanVec<Primary, Secondary, LeanVecDims>{
            x.data_matrix_, x.query_matrix_};
    }

    static std::string description() {
        return fmt::format(
            "leanvec ({}, {}) - {} (static)",
            name(svsbenchmark::detail::LeanVecKindMap<Primary>::kind),
            name(svsbenchmark::detail::LeanVecKindMap<Secondary>::kind),
            LeanVecDims
        );
    }
};

template <typename Primary, typename Secondary, size_t LeanVecDims>
struct DispatchConverter<
    svsbenchmark::Dataset,
    svsbenchmark::TypedLeanVec<Primary, Secondary, LeanVecDims>> {
    using From = svsbenchmark::LeanVec;
    using To = svsbenchmark::TypedLeanVec<Primary, Secondary, LeanVecDims>;
    static int64_t match(const svsbenchmark::Dataset& x) {
        if (const auto* ptr = std::get_if<From>(&x.kinds_)) {
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
