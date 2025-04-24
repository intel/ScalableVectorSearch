/*
 * Copyright 2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifndef USE_PROPRIETARY

#include "svs/quantization/lvq/lvq_fallback.h"

#else // USE_PROPRIETARY

#include "../../../../../include/svs/quantization/lvq/lvq.h"

#endif // USE_PROPRIETARY

namespace svs {
namespace quantization {
namespace lvq {

/////
///// Load Helpers
/////

// Types to use for lazy compression.
inline constexpr lib::Types<float, Float16> CompressionTs{};

// How are we expecting to obtain the data.
struct OnlineCompression {
  public:
    explicit OnlineCompression(const std::filesystem::path& path, DataType type)
        : path{path}
        , type{type} {
        if (!lib::in(type, CompressionTs)) {
            throw ANNEXCEPTION("Invalid type!");
        }
    }

    ///// Members
    std::filesystem::path path;
    DataType type;
};

///
/// @brief Dispatch type indicating that a compressed dataset should be reloaded
/// directly.
///
/// LVQ based loaders can either perform dataset compression online, or reload a
/// previously saved dataset.
///
/// Using this type in LVQ loader constructors indicates that reloading is
/// desired.
///
struct Reload {
  public:
    ///
    /// @brief Construct a new Reloader.
    ///
    /// @param directory The directory where a LVQ compressed dataset was
    /// previously saved.
    ///
    explicit Reload(const std::filesystem::path& directory)
        : directory{directory} {}

    ///// Members
    std::filesystem::path directory;
};

// The various ways we can instantiate LVQ-based datasets..
using SourceTypes = std::variant<OnlineCompression, Reload>;

// Forward Declaration.
template <
    size_t Primary,
    size_t Residual,
    size_t Extent,
    LVQPackingStrategy Strategy,
    typename Alloc>
struct LVQLoader;

struct Matcher {
    // Load a matcher for either one or two level datasets.
    static bool check_load_compatibility(std::string_view schema, lib::Version version) {
        if (schema == one_level_serialization_schema && version == one_level_save_version) {
            return true;
        }
        if (schema == two_level_serialization_schema && version == two_level_save_version) {
            return true;
        }
        if (schema == fallback_serialization_schema && version == fallback_save_version) {
            return true;
        }
        return false;
    }

    static Matcher load(const lib::ContextFreeLoadTable& table) {
        auto schema = table.schema();
        auto primary_summary = lib::load_at<lvq::DatasetSummary>(table, "primary");
        if (schema == one_level_serialization_schema) {
            return Matcher{
                .primary = primary_summary.bits,
                .residual = 0,
                .dims = primary_summary.dims};
        }
        if (schema == two_level_serialization_schema) {
            auto residual_summary = lib::load_at<lvq::DatasetSummary>(table, "residual");
            return Matcher{
                .primary = primary_summary.bits,
                .residual = residual_summary.bits,
                .dims = primary_summary.dims};
        }
        if (schema == fallback_serialization_schema) {
            return Matcher{
                .primary = primary_summary.bits,
                .residual = 0,
                .dims = primary_summary.dims};
        }
        throw ANNEXCEPTION(
            "Unreachable reached with schema and version ({}, {})!",
            table.schema(),
            table.version()
        );
    }

    static lib::TryLoadResult<Matcher> try_load(const lib::ContextFreeLoadTable& table) {
        // The saving and loading framework will check schema compatibility before
        // calling try-load.
        //
        // In that case, the logic behind `try_load` and `load` are the same.
        // Note that `load` will throw if sub-keys do not match, but that is okay
        // because mismatching sub-keys means we have an invalid schema.
        return load(table);
    }

    constexpr bool friend operator==(const Matcher&, const Matcher&) = default;

    ///// Members
    size_t primary;
    size_t residual;
    size_t dims;
};

template <LVQPackingStrategy Strategy>
int64_t overload_match_strategy(LVQStrategyDispatch strategy) {
    constexpr bool is_sequential = std::is_same_v<Strategy, lvq::Sequential>;
    constexpr bool is_turbo = lvq::TurboLike<Strategy>;

    switch (strategy) {
        // If sequential is requested - we can only match sequential.
        case LVQStrategyDispatch::Sequential: {
            return is_sequential ? lib::perfect_match : lib::invalid_match;
        }
        // If turbo is requested - we can only match turbo.
        case LVQStrategyDispatch::Turbo: {
            return is_turbo ? lib::perfect_match : lib::invalid_match;
        }
        case LVQStrategyDispatch::Auto: {
            // Preference:
            // (1) Turbo
            // (2) Sequential
            return is_turbo ? 0 : 1;
        }
    }
    throw ANNEXCEPTION("Unreachable!");
}

// Compatibility ranking for LVQ
template <size_t Primary, size_t Residual, size_t Extent, LVQPackingStrategy Strategy>
int64_t overload_score(size_t p, size_t r, size_t e, LVQStrategyDispatch strategy) {
    // Reject easy matches.
    if (lvq::check_primary_residual<Primary, Residual>(p, r)) {
        return lib::invalid_match;
    }

    // Check static dimensionality.
    auto extent_match =
        lib::dispatch_match<lib::ExtentArg, lib::ExtentTag<Extent>>(lib::ExtentArg{e});

    // If the extent match fails - abort immediately.
    if (extent_match < 0) {
        return lib::invalid_match;
    }

    // We know dimensionality matches, now we have to try to match strategy.
    auto strategy_match = overload_match_strategy<Strategy>(strategy);
    if (lvq::check_strategy_match(strategy_match)) {
        return lib::invalid_match;
    }

    // Prioritize matching dimensionality over better strategies.
    // Dispatch matching prefers lower return values over larger return values.
    //
    // By multiplying the `extent_match`, we enter a regime where better extent
    // matches always have precedence over strategy matches.
    constexpr size_t extent_multiplier = 1000;
    return strategy_match + extent_multiplier * extent_match;
}

template <size_t Primary, size_t Residual, size_t Extent, LVQPackingStrategy Strategy>
int64_t overload_score(Matcher matcher, LVQStrategyDispatch strategy) {
    return overload_score<Primary, Residual, Extent, Strategy>(
        matcher.primary, matcher.residual, matcher.dims, strategy
    );
}

template <typename Alloc = lib::Allocator<std::byte>> struct ProtoLVQLoader {
  public:
    // Constructors
    ProtoLVQLoader() = default;

    // TODO: Propagate allocator request.
    explicit ProtoLVQLoader(
        const UnspecializedVectorDataLoader<Alloc>& datafile,
        size_t primary,
        size_t residual,
        size_t alignment = 0,
        LVQStrategyDispatch strategy = LVQStrategyDispatch::Auto
    )
        : source_{std::in_place_type_t<OnlineCompression>(), datafile.path_, datafile.type_}
        , primary_{primary}
        , residual_{residual}
        , dims_{datafile.dims_}
        , alignment_{alignment}
        , strategy_{strategy}
        , allocator_{datafile.allocator_} {}

    explicit ProtoLVQLoader(
        Reload reloader,
        size_t alignment,
        LVQStrategyDispatch strategy = LVQStrategyDispatch::Auto,
        const Alloc& allocator = {}
    )
        : source_{std::move(reloader)}
        , primary_{0}
        , residual_{0}
        , dims_{0}
        , alignment_{alignment}
        , strategy_{strategy}
        , allocator_{allocator} {
        const auto& directory = std::get<Reload>(source_).directory;
        auto result = lib::try_load_from_disk<Matcher>(directory);
        if (!result) {
            throw ANNEXCEPTION(
                "Cannot determine primary, residual, and dimensions "
                "from data source {}. "
                "Code {}!",
                directory,
                static_cast<int64_t>(result.error())
            );
        }
        const auto& match = result.value();
        primary_ = match.primary;
        residual_ = match.residual;
        dims_ = match.dims;
    }

    template <
        size_t Primary,
        size_t Residual,
        size_t Extent,
        LVQPackingStrategy Strategy,
        typename F = std::identity>
    LVQLoader<
        Primary,
        Residual,
        Extent,
        Strategy,
        std::decay_t<std::invoke_result_t<F, const Alloc&>>>
    refine(lib::Val<Extent>, F&& f = std::identity()) const {
        using ARet = std::decay_t<std::invoke_result_t<F, const Alloc&>>;
        // Make sure the pre-set values are correct.
        if constexpr (Extent != Dynamic) {
            if (Extent != dims_) {
                throw ANNEXCEPTION("Invalid specialization!");
            }
        }
        if (lvq::check_primary_residual<Primary, Residual>(primary_, residual_)) {
            throw ANNEXCEPTION("Encoding bits mismatched!");
        }
        if (!detail::is_compatible<Strategy>(strategy_)) {
            throw ANNEXCEPTION("Trying to dispatch to an inappropriate strategy!");
        }

        return LVQLoader<Primary, Residual, Extent, Strategy, ARet>(
            source_, alignment_, f(allocator_)
        );
    }

  public:
    SourceTypes source_;
    size_t primary_;
    size_t residual_;
    size_t dims_;
    size_t alignment_;
    LVQStrategyDispatch strategy_;
    Alloc allocator_;
};

template <
    size_t Primary,
    size_t Residual,
    size_t Extent,
    LVQPackingStrategy Strategy,
    typename Alloc>
struct LVQLoader {
  public:
    using loaded_type = LVQDataset<Primary, Residual, Extent, Strategy, Alloc>;

    explicit LVQLoader(SourceTypes source, size_t alignment, const Alloc& allocator)
        : source_{std::move(source)}
        , alignment_{alignment}
        , allocator_{allocator} {}

    loaded_type load() const {
        auto pool = threads::SequentialThreadPool();
        return load(pool);
    }

    template <typename F>
    LVQLoader<
        Primary,
        Residual,
        Extent,
        Strategy,
        std::decay_t<std::invoke_result_t<F, const Alloc&>>>
    rebind_alloc(const F& f) {
        return LVQLoader<
            Primary,
            Residual,
            Extent,
            Strategy,
            std::decay_t<std::invoke_result_t<F, const Alloc&>>>{
            source_, alignment_, f(allocator_)};
    }

    template <threads::ThreadPool Pool> loaded_type load(Pool& threadpool) const {
        return std::visit<loaded_type>(
            [&](auto source) {
                using T = std::decay_t<decltype(source)>;
                if constexpr (std::is_same_v<T, Reload>) {
                    return lib::load_from_disk<loaded_type>(
                        source.directory, alignment_, allocator_
                    );
                } else {
                    return lib::match(
                        CompressionTs,
                        source.type,
                        [&]<typename T>(lib::Type<T> SVS_UNUSED(type)) {
                            return loaded_type::compress(
                                data::SimpleData<T>::load(source.path),
                                threadpool,
                                alignment_,
                                allocator_
                            );
                        }
                    );
                }
            },
            source_
        );
    }

  private:
    SourceTypes source_;
    size_t alignment_;
    Alloc allocator_;
};

} // namespace lvq
} // namespace quantization

// Define dispatch conversion from ProtoLVQLoader to LVQLoader.
template <
    size_t Primary,
    size_t Residual,
    size_t Extent,
    quantization::lvq::LVQPackingStrategy Strategy,
    typename Alloc>
struct lib::DispatchConverter<
    quantization::lvq::ProtoLVQLoader<Alloc>,
    quantization::lvq::LVQLoader<Primary, Residual, Extent, Strategy, Alloc>> {
    static int64_t match(const quantization::lvq::ProtoLVQLoader<Alloc>& loader) {
        return quantization::lvq::overload_score<Primary, Residual, Extent, Strategy>(
            loader.primary_, loader.residual_, loader.dims_, loader.strategy_
        );
    }

    static quantization::lvq::LVQLoader<Primary, Residual, Extent, Strategy, Alloc>
    convert(const quantization::lvq::ProtoLVQLoader<Alloc>& loader) {
        return loader.template refine<Primary, Residual, Extent, Strategy>(lib::Val<Extent>(
        ));
    }

    static std::string description() {
        auto dims = []() {
            if constexpr (Extent == Dynamic) {
                return "any";
            } else {
                return Extent;
            }
        }();

        return fmt::format(
            "LVQLoader {}x{} ({}) with {} dimensions",
            Primary,
            Residual,
            Strategy::name(),
            dims
        );
    }
};

} // namespace svs
