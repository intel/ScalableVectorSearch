/**
 *    Copyright (C) 2023 Intel Corporation
 *
 *    This software and the related documents are Intel copyrighted materials,
 *    and your use of them is governed by the express license under which they
 *    were provided to you ("License"). Unless the License provides otherwise,
 *    you may not use, modify, copy, publish, distribute, disclose or transmit
 *    this software or the related documents without Intel's prior written
 *    permission.
 *
 *    This software and the related documents are provided as is, with no
 *    express or implied warranties, other than those that are expressly stated
 *    in the License.
 */

#pragma once

#include "svs/core/data/simple.h"
#include "svs/lib/threads.h"
#include "svs/lib/saveload/save.h"
#include "svs/fallback/fallback_mode.h"

namespace fallback = svs::fallback;

namespace svs {
namespace quantization {
namespace lvq {

// TODO: should these be fully defined?
struct Sequential {
    static constexpr std::string_view name() { return "sequential"; }
};
template <size_t Lanes, size_t ElementsPerLane> struct Turbo {
    static constexpr std::string name() {
        return fmt::format("turbo<{}x{}>", Lanes, ElementsPerLane);
    }
};

namespace detail {

// Trait to identify and dispatch based on the Turbo class itself.
template <typename T> inline constexpr bool is_turbo_like_v = false;
template <typename T> inline constexpr bool is_lvq_packing_strategy_v = false;

template <size_t Lanes, size_t ElementsPerLane>
inline constexpr bool is_turbo_like_v<lvq::Turbo<Lanes, ElementsPerLane>> = true;

template <> inline constexpr bool is_lvq_packing_strategy_v<lvq::Sequential> = true;
template <size_t Lanes, size_t ElementsPerLane>

inline constexpr bool is_lvq_packing_strategy_v<lvq::Turbo<Lanes, ElementsPerLane>> = true;

template <typename A> inline constexpr bool is_blocked = false;
template <typename A> inline constexpr bool is_blocked<data::Blocked<A>> = true;

template <typename T, typename A, bool = is_blocked<A>> struct select_rebind_allocator {
    using type = lib::rebind_allocator_t<T, A>;
};
template <typename T, typename A> struct select_rebind_allocator<T, A, true> {
    using base_allocator = typename A::allocator_type;
    using rebind_base_allocator = lib::rebind_allocator_t<T, base_allocator>;
    using type = data::Blocked<rebind_base_allocator>;
};
template <typename T, typename A>
using select_rebind_allocator_t = typename select_rebind_allocator<T, A>::type;

} // namespace detail

template <typename T>
concept LVQPackingStrategy = detail::is_lvq_packing_strategy_v<T>;

enum class LVQStrategyDispatch { Auto, Sequential, Turbo };

// LVQDataset
template <
    size_t Primary,
    size_t Residual = 0,
    size_t Extent = Dynamic,
    LVQPackingStrategy Strategy = Sequential,
    typename Alloc = lib::Allocator<std::byte>>
class LVQDataset {
  public:
    using allocator_type = detail::select_rebind_allocator_t<float, Alloc>;
  private:
    data::SimpleData<float, Extent, allocator_type> primary_;
  public:
    static constexpr bool is_resizeable = detail::is_blocked<Alloc>;
    using const_value_type = typename data::SimpleData<float, Extent, allocator_type>::const_value_type;
    using element_type = float;
    using value_type = const_value_type;
    using primary_type = data::SimpleData<float, Extent, allocator_type>;
    void resize(size_t new_size)
        requires is_resizeable
    {
        primary_.resize(new_size);
    }
    template <std::integral I, threads::ThreadPool Pool>
        requires is_resizeable
    void
    compact(std::span<const I> new_to_old, Pool& threadpool, size_t batchsize = 1'000'000) {
        primary_.compact(new_to_old, threadpool, batchsize);
    }

    template <data::ImmutableMemoryDataset Dataset>
        LVQDataset(Dataset primary): primary_{primary} {
        if (fallback::get_mode() == fallback::FallbackMode::Error) {
            throw fallback::UnsupportedHardwareError();
        } else if (fallback::get_mode() == fallback::FallbackMode::Warning) {
            fmt::print(fallback::fallback_warning);
        }
    }

    size_t size() const { return primary_.size(); }
    size_t dimensions() const { return primary_.dimensions(); }
    const_value_type get_datum(size_t i) const { return primary_.get_datum(i); }
    void prefetch(size_t i) const { primary_.prefetch(i); }

    template <typename QueryType, size_t N>
    void set_datum(size_t i, std::span<QueryType, N> datum, size_t SVS_UNUSED(centroid_selector) = 0) {
        primary_.set_datum(i, datum);
    }

    template <data::ImmutableMemoryDataset Dataset>
    static LVQDataset compress(const Dataset& data, const Alloc& allocator = {}) {
        return compress(data, 1, 0, allocator);
    }

    template <data::ImmutableMemoryDataset Dataset>
    static LVQDataset compress(
        const Dataset& data,
        size_t num_threads,
        size_t alignment,
        const Alloc& allocator = {}
    ) {
        auto pool = threads::NativeThreadPool{num_threads};
        return compress(data, pool, alignment, allocator);
    }

    template <data::ImmutableMemoryDataset Dataset, threads::ThreadPool Pool>
    static LVQDataset compress(
        const Dataset& data,
        Pool& SVS_UNUSED(threadpool),
        size_t SVS_UNUSED(alignment),
        const Alloc& allocator = {}
    ) {
        primary_type primary = primary_type{data.size(), data.dimensions(), allocator_type{allocator}};
        svs::data::copy(data, primary);
        return LVQDataset{primary};
    }

    
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);
    static constexpr std::string_view serialization_schema = "lvq_fallback";
    lib::SaveTable save(const lib::SaveContext& ctx) const {
        return lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(primary, ctx)}
        );
    }

    static LVQDataset load(
        const lib::LoadTable& table,
        size_t SVS_UNUSED(alignment) = 0,
        const Alloc& allocator = {}
    ) {
        return LVQDataset{SVS_LOAD_MEMBER_AT_(table, primary, allocator)};
    }
};

struct Reload {
  public:
    explicit Reload(const std::filesystem::path& directory)
        : directory{directory} {}

    std::filesystem::path directory;
};

template <
    size_t Primary,
    size_t Residual,
    size_t Extent,
    LVQPackingStrategy Strategy,
    typename Alloc>
struct LVQLoader;

inline constexpr std::string_view one_level_serialization_schema = "one_level_lvq_dataset";
inline constexpr lib::Version one_level_save_version = lib::Version(0, 0, 2);
inline constexpr std::string_view two_level_serialization_schema = "two_level_lvq_dataset";
inline constexpr lib::Version two_level_save_version = lib::Version(0, 0, 3);
inline constexpr lib::Types<float, Float16> CompressionTs{};
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
using SourceTypes = std::variant<OnlineCompression, Reload>;

enum class DatasetSchema { Compressed, ScaledBiased };
struct Signed {
    static constexpr std::string_view name = "signed";
};
inline constexpr std::string_view get_schema(DatasetSchema kind) {
    switch (kind) {
        using enum DatasetSchema;
        case Compressed: {
            return "lvq_compressed_dataset";
        }
        case ScaledBiased: {
            return "lvq_with_scaling_constants";
        }
    }
    throw ANNEXCEPTION("Invalid schema!");
}
inline constexpr lib::Version get_current_version(DatasetSchema kind) {
    switch (kind) {
        using enum DatasetSchema;
        case Compressed: {
            return lib::Version(0, 0, 0);
        }
        case ScaledBiased: {
            return lib::Version(0, 0, 3);
        }
    }
    throw ANNEXCEPTION("Invalid schema!");
}
struct DatasetSummary {
    static bool check_load_compatibility(std::string_view schema, lib::Version version) {
        using enum DatasetSchema;
        if (schema == get_schema(Compressed) &&
            version == get_current_version(Compressed)) {
            return true;
        }
        if (schema == get_schema(ScaledBiased) &&
            version == get_current_version(ScaledBiased)) {
            return true;
        }
        return false;
    }

    static DatasetSummary load(const lib::ContextFreeLoadTable& table) {
        using enum DatasetSchema;
        auto schema = table.schema();
        if (schema == get_schema(Compressed)) {
            return DatasetSummary{
                .kind = Compressed,
                .is_signed =
                    (lib::load_at<std::string>(table, "sign") == lvq::Signed::name),
                .dims = lib::load_at<size_t>(table, "ndims"),
                .bits = lib::load_at<size_t>(table, "bits")};
        }
        if (schema == get_schema(ScaledBiased)) {
            return DatasetSummary{
                .kind = ScaledBiased,
                .is_signed = false, // ScaledBiased always uses unsigned codes.
                .dims = lib::load_at<size_t>(table, "logical_dimensions"),
                .bits = lib::load_at<size_t>(table, "bits")};
        }
        throw ANNEXCEPTION("Invalid table schema {}!", schema);
    }

    ///// Members
    // The kind of the leaf dataset.
    DatasetSchema kind;
    // Whether each LVQ element is signed.
    bool is_signed;
    // The logical number of dimensions in the dataset.
    size_t dims;
    // The number of bits used for compression.
    size_t bits;
};

template <typename T>
concept TurboLike = detail::is_turbo_like_v<T>;
namespace detail {
template <LVQPackingStrategy Strategy>
constexpr bool is_compatible(LVQStrategyDispatch strategy) {
    switch (strategy) {
        case LVQStrategyDispatch::Auto: {
            return true;
        }
        case LVQStrategyDispatch::Sequential: {
            return std::is_same_v<Strategy, Sequential>;
        }
        case LVQStrategyDispatch::Turbo: {
            return TurboLike<Strategy>;
        }
    }
    throw ANNEXCEPTION("Could not match strategy!");
}
}
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

struct Matcher {
    // Load a matcher for either one or two level datasets.
    static bool check_load_compatibility(std::string_view schema, lib::Version version) {
        if (schema == one_level_serialization_schema && version == one_level_save_version) {
            return true;
        }
        if (schema == two_level_serialization_schema && version == two_level_save_version) {
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

// Compatibility ranking for LVQ
template <size_t Primary, size_t Residual, size_t Extent, LVQPackingStrategy Strategy>
int64_t overload_score(size_t p, size_t r, size_t e, LVQStrategyDispatch strategy) {
    // Reject easy matches.
    if (p != Primary || r != Residual) {
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
    if (strategy_match < 0) {
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
        if (Primary != primary_ || Residual != residual_) {
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
}
