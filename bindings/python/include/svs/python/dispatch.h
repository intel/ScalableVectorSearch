#pragma once

// svs python bindings
#include "svs/python/core.h"

// svs
#include "svs/core/data.h"
#include "svs/leanvec/leanvec.h"
#include "svs/lib/dispatcher.h"
#include "svs/lib/saveload.h"
#include "svs/quantization/lvq/lvq.h"

// Dispatch rule for serialized objects to a VectorDataLoader.
template <typename T, size_t N>
struct svs::lib::DispatchConverter<
    svs::lib::SerializedObject,
    svs::VectorDataLoader<T, N, svs::python::RebindAllocator<T>>> {
    using To = svs::VectorDataLoader<T, N, svs::python::RebindAllocator<T>>;

    static int64_t match(const svs::lib::SerializedObject& object) {
        auto ex = svs::lib::try_load<svs::data::Matcher>(object);
        if (!ex) {
            // Could not load for some reason.
            // Invalid match.
            return lib::invalid_match;
        }

        // Check suitability.
        const auto& matcher = ex.value();
        auto code = svs::data::detail::check_match<T, N>(matcher.eltype, matcher.dims);
        return code;
    }

    static To convert(const svs::lib::SerializedObject& object) {
        return To{object.context().get_directory()};
    }
};

template <
    size_t Primary,
    size_t Residual,
    size_t Extent,
    svs::quantization::lvq::LVQPackingStrategy Strategy>
struct svs::lib::DispatchConverter<
    svs::lib::SerializedObject,
    svs::quantization::lvq::LVQLoader<
        Primary,
        Residual,
        Extent,
        Strategy,
        svs::python::RebindAllocator<std::byte>>> {
    using To = svs::quantization::lvq::LVQLoader<
        Primary,
        Residual,
        Extent,
        Strategy,
        svs::python::RebindAllocator<std::byte>>;

    using LVQStrategyDispatch = svs::quantization::lvq::LVQStrategyDispatch;

    static int64_t match(const svs::lib::SerializedObject& object) {
        // TODO: Use a LoadTable directly instead of forcing reparsing every time.
        auto ex = svs::lib::try_load<svs::quantization::lvq::Matcher>(object);
        if (!ex) {
            return svs::lib::invalid_match;
        }

        return svs::quantization::lvq::overload_score<Primary, Residual, Extent, Strategy>(
            ex.value(), LVQStrategyDispatch::Auto
        );
    }

    static To convert(const svs::lib::SerializedObject& object) {
        return To{
            svs::quantization::lvq::Reload{std::move(object.context().get_directory())},
            0,
            svs::python::RebindAllocator<std::byte>()};
    }
};

template <typename PrimaryKind, typename SecondaryKind, size_t LeanVecDims, size_t Extent>
struct svs::lib::DispatchConverter<
    svs::lib::SerializedObject,
    svs::leanvec::LeanVecLoader<
        PrimaryKind,
        SecondaryKind,
        LeanVecDims,
        Extent,
        svs::python::RebindAllocator<std::byte>>> {
    using To = leanvec::LeanVecLoader<
        PrimaryKind,
        SecondaryKind,
        LeanVecDims,
        Extent,
        svs::python::RebindAllocator<std::byte>>;

    static int64_t match(const svs::lib::SerializedObject& object) {
        // TODO: Use a LoadTable directly instead of forcing reparsing every time.
        auto ex = svs::lib::try_load<svs::leanvec::Matcher>(object);
        if (!ex) {
            return svs::lib::invalid_match;
        }

        return svs::leanvec::
            overload_score<PrimaryKind, SecondaryKind, LeanVecDims, Extent>(ex.value());
    }

    static To convert(const svs::lib::SerializedObject& object) {
        return To{
            leanvec::Reload{object.context().get_directory()},
            LeanVecDims, // TODO: This is a hack for now. Since we're reloading, it doesn't
                         // matter.
            std::nullopt,
            0,
            svs::python::RebindAllocator<std::byte>()};
    }
};
