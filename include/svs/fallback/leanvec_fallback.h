/*
 * Copyright 2023 Intel Corporation
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

#include "svs/fallback/lvq_fallback.h"
#include "svs/fallback/fallback_mode.h"

namespace fallback = svs::fallback;

namespace svs {
namespace leanvec {

template <size_t Bits> struct UsingLVQ {};

template <size_t Extent> struct LeanVecMatrices {
  // temporary (?) additions
  public:
    using leanvec_matrix_type = data::SimpleData<float, Extent>;

    LeanVecMatrices() = default;
    LeanVecMatrices(leanvec_matrix_type data_matrix, leanvec_matrix_type query_matrix)
        : data_matrix_{std::move(data_matrix)}
        , query_matrix_{std::move(query_matrix)} {
        // Check that the size and dimensionality of both the matrices should be same
        if (data_matrix_.size() != query_matrix_.size()) {
            throw ANNEXCEPTION("Mismatched data and query matrix sizes!");
        }
        if (data_matrix_.dimensions() != query_matrix_.dimensions()) {
            throw ANNEXCEPTION("Mismatched data and query matrix dimensions!");
        }
    }
  private:
    leanvec_matrix_type data_matrix_;
    leanvec_matrix_type query_matrix_;
};

enum class LeanVecKind { float32, float16, lvq8, lvq4 };

// is this necessary or duplicate of LVQ?
namespace detail {
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
}

template <
    typename T1,
    typename T2,
    size_t LeanVecDims,
    size_t Extent,
    typename Alloc = lib::Allocator<std::byte>>
class LeanDataset {
  public:
    using allocator_type = detail::select_rebind_allocator_t<float, Alloc>;
  private:
    data::SimpleData<float, Extent, allocator_type> primary_;
  public:
    static constexpr bool is_resizeable = detail::is_blocked<Alloc>;
    using leanvec_matrices_type = LeanVecMatrices<LeanVecDims>;
    using const_value_type = typename data::SimpleData<float, Extent, allocator_type>::const_value_type;
    using element_type = float;
    using value_type = const_value_type;
    using primary_type = data::SimpleData<float, Extent, allocator_type>;

    LeanDataset(primary_type primary): primary_{std::move(primary)} {
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
    template <typename U, size_t N> void set_datum(size_t i, std::span<U, N> datum) {
        primary_.set_datum(i, datum);
    }

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
    static LeanDataset reduce(
        const Dataset& data,
        size_t num_threads = 1,
        size_t alignment = 0,
        lib::MaybeStatic<LeanVecDims> leanvec_dims = {},
        const Alloc& allocator = {}
    ) {
        return reduce(data, std::nullopt, num_threads, alignment, leanvec_dims, allocator);
    }

    template <data::ImmutableMemoryDataset Dataset>
    static LeanDataset reduce(
        const Dataset& data,
        std::optional<leanvec_matrices_type> matrices,
        size_t num_threads = 1,
        size_t alignment = 0,
        lib::MaybeStatic<LeanVecDims> leanvec_dims = {},
        const Alloc& allocator = {}
    ) {
        auto pool = threads::NativeThreadPool{num_threads};
        return reduce(data, std::move(matrices), pool, alignment, leanvec_dims, allocator);
    }

    template <data::ImmutableMemoryDataset Dataset, threads::ThreadPool Pool>
    static LeanDataset reduce(
        const Dataset& data,
        std::optional<leanvec_matrices_type> SVS_UNUSED(matrices),
        Pool& SVS_UNUSED(threadpool),
        size_t SVS_UNUSED(alignment) = 0,
        lib::MaybeStatic<LeanVecDims> SVS_UNUSED(leanvec_dims) = {},
        const Alloc& allocator = {}
    ) {
        primary_type primary = primary_type{data.size(), data.dimensions(), allocator_type{allocator}};
        svs::data::copy(data, primary);
        return LeanDataset{primary};
    }

    static constexpr lib::Version save_version = lib::Version(0, 0, 0);
    static constexpr std::string_view serialization_schema = "leanvec_fallback";
    lib::SaveTable save(const lib::SaveContext& ctx) const {
        return lib::SaveTable(
            serialization_schema,
            save_version,
            {SVS_LIST_SAVE_(primary, ctx)}
        );
    }

    static LeanDataset load(
        const lib::LoadTable& table,
        size_t SVS_UNUSED(alignment) = 0,
        const Alloc& allocator = {}
    ) {
        return LeanDataset{SVS_LOAD_MEMBER_AT_(table, primary, allocator)};
    }
};

struct Reload {
  public:
    explicit Reload(const std::filesystem::path& directory)
        : directory{directory} {}

  public:
    std::filesystem::path directory;   
};
inline constexpr lib::Types<float, Float16> LeanVecSourceTypes{};
struct OnlineLeanVec {
  public:
    explicit OnlineLeanVec(const std::filesystem::path& path, DataType type)
        : path{path}
        , type{type} {
        if (!lib::in(type, LeanVecSourceTypes)) {
            throw ANNEXCEPTION("Invalid type!");
        }
    }
  public:
    std::filesystem::path path;
    DataType type;
};
using SourceTypes = std::variant<OnlineLeanVec, Reload>;
inline constexpr std::string_view lean_dataset_schema = "leanvec_dataset";
inline constexpr lib::Version lean_dataset_save_version = lib::Version(0, 0, 0);

namespace detail {

template <typename T> struct LeanVecPicker;

template <> struct LeanVecPicker<float> {
    static constexpr LeanVecKind value = LeanVecKind::float32;
};
template <> struct LeanVecPicker<svs::Float16> {
    static constexpr LeanVecKind value = LeanVecKind::float16;
};
template <> struct LeanVecPicker<UsingLVQ<8>> {
    static constexpr LeanVecKind value = LeanVecKind::lvq8;
};
template <> struct LeanVecPicker<UsingLVQ<4>> {
    static constexpr LeanVecKind value = LeanVecKind::lvq4;
};

} // namespace detail

template <typename T>
inline constexpr LeanVecKind leanvec_kind_v = detail::LeanVecPicker<T>::value;

// LeanDataset Matcher
struct Matcher {
  private:
    struct DatasetLayout {
        size_t dims;
        LeanVecKind kind;
    };

    static lib::TryLoadResult<DatasetLayout>
    detect_data(const lib::ContextFreeNodeView<toml::node>& node) {
        // Is it an uncompressed dataset?
        auto maybe_uncompressed = lib::try_load<svs::data::Matcher>(node);
        auto failure = lib::Unexpected{lib::TryLoadFailureReason::Other};

        // On success - determine if this one of the recognized types.
        if (maybe_uncompressed) {
            const auto& matcher = maybe_uncompressed.value();
            size_t dims = matcher.dims;
            switch (matcher.eltype) {
                case DataType::float16: {
                    return DatasetLayout{dims, LeanVecKind::float16};
                }
                case DataType::float32: {
                    return DatasetLayout{dims, LeanVecKind::float32};
                }
                default: {
                    return failure;
                }
            }
        }

        // Failed to match the uncompressed layout. Try LVQ.
        auto maybe_lvq = lib::try_load<svs::quantization::lvq::Matcher>(node);
        if (maybe_lvq) {
            const auto& matcher = maybe_lvq.value();
            size_t dims = matcher.dims;
            size_t primary = matcher.primary;
            switch (primary) {
                case 4: {
                    return DatasetLayout{dims, LeanVecKind::lvq4};
                }
                case 8: {
                    return DatasetLayout{dims, LeanVecKind::lvq8};
                }
                default: {
                    return failure;
                }
            }
        }
        return lib::Unexpected(lib::TryLoadFailureReason::InvalidSchema);
    }

  public:
    ///// Loading.
    static bool check_load_compatibility(std::string_view schema, lib::Version version) {
        return schema == lean_dataset_schema && version == lean_dataset_save_version;
    }

    static lib::TryLoadResult<Matcher> try_load(const lib::ContextFreeLoadTable& table) {
        // For each of the primary and secondary, use the combinations of expected
        // expected types until we have a successful match.
        auto primary_expected = detect_data(table.at("primary"));
        if (!primary_expected) {
            return lib::Unexpected(primary_expected.error());
        }

        auto secondary_expected = detect_data(table.at("secondary"));
        if (!secondary_expected) {
            return lib::Unexpected(secondary_expected.error());
        }

        const auto& primary = primary_expected.value();
        const auto& secondary = secondary_expected.value();

        return Matcher{
            .leanvec_dims = primary.dims,
            .total_dims = secondary.dims,
            .primary_kind = primary.kind,
            .secondary_kind = secondary.kind};
    }

    static Matcher load(const lib::ContextFreeLoadTable& table) {
        // For each of the primary and secondary, use the combinations of expected
        // expected types until we have a successful match.
        auto primary_expected = detect_data(table.at("primary"));
        if (!primary_expected) {
            throw ANNEXCEPTION("Could not match the primary dataset!");
        }

        auto secondary_expected = detect_data(table.at("secondary"));
        if (!secondary_expected) {
            throw ANNEXCEPTION("Could not match the secondary dataset!");
        }

        const auto& primary = primary_expected.value();
        const auto& secondary = secondary_expected.value();

        return Matcher{
            .leanvec_dims = primary.dims,
            .total_dims = secondary.dims,
            .primary_kind = primary.kind,
            .secondary_kind = secondary.kind};
    }

    constexpr bool friend operator==(const Matcher&, const Matcher&) = default;

    ///// Members
    size_t leanvec_dims;
    size_t total_dims;
    LeanVecKind primary_kind;
    LeanVecKind secondary_kind;
};

// Overload Matching Rules
template <typename T1, typename T2, size_t LeanVecDims, size_t Extent>
int64_t overload_score(
    LeanVecKind primary, size_t primary_dims, LeanVecKind secondary, size_t secondary_dims
) {
    // Check primary kind
    if (primary != leanvec::leanvec_kind_v<T1>) {
        return lib::invalid_match;
    }

    // Check secondary kind
    if (secondary != leanvec::leanvec_kind_v<T2>) {
        return lib::invalid_match;
    }

    // Check extent-tags.
    auto extent_match = lib::dispatch_match<lib::ExtentArg, lib::ExtentTag<Extent>>(
        lib::ExtentArg{secondary_dims}
    );

    // If extents don't match, then we abort immediately.
    if (extent_match < 0) {
        return lib::invalid_match;
    }

    // Check leanvec_dims-tags.
    auto leanvec_dims_match =
        lib::dispatch_match<lib::ExtentArg, lib::ExtentTag<LeanVecDims>>(lib::ExtentArg{
            primary_dims});

    // If leanvec_dims don't match, then we abort immediately.
    if (leanvec_dims_match < 0) {
        return lib::invalid_match;
    }

    return extent_match + leanvec_dims_match;
}

template <typename T1, typename T2, size_t LeanVecDims, size_t Extent>
int64_t overload_score(const Matcher& matcher) {
    return overload_score<T1, T2, LeanVecDims, Extent>(
        matcher.primary_kind,
        matcher.leanvec_dims,
        matcher.secondary_kind,
        matcher.total_dims
    );
}

template <typename T1, typename T2, size_t LeanVecDims, size_t Extent, typename Alloc>
struct LeanVecLoader;

template <typename Alloc = lib::Allocator<std::byte>> struct ProtoLeanVecLoader {
  public:
    ProtoLeanVecLoader() = default;
    explicit ProtoLeanVecLoader(
        const UnspecializedVectorDataLoader<Alloc>& datafile,
        size_t leanvec_dims,
        LeanVecKind primary_kind,
        LeanVecKind secondary_kind,
        std::optional<LeanVecMatrices<Dynamic>> matrices,
        size_t alignment = 0
    )
        : source_{std::in_place_type<OnlineLeanVec>, datafile.path_, datafile.type_}
        , leanvec_dims_{leanvec_dims}
        , dims_{datafile.dims_}
        , primary_kind_{primary_kind}
        , secondary_kind_{secondary_kind}
        , matrices_{std::move(matrices)}
        , alignment_{alignment}
        , allocator_{datafile.allocator_} {}

    explicit ProtoLeanVecLoader(
        Reload reloader,
        // size_t leanvec_dims,
        // size_t dims,
        // LeanVecKind primary_kind,
        // LeanVecKind secondary_kind,
        size_t alignment = 0,
        const Alloc& allocator = {}
    )
        : source_{std::move(reloader)}
        , matrices_{std::nullopt}
        , alignment_{alignment}
        , allocator_{allocator} {
        // Produce a hard error if we cannot load and match the dataset.
        auto matcher = lib::load_from_disk<Matcher>(std::get<Reload>(source_).directory);
        primary_kind_ = matcher.primary_kind;
        secondary_kind_ = matcher.secondary_kind;
        leanvec_dims_ = matcher.leanvec_dims;
        dims_ = matcher.total_dims;
    }

    template <
        typename T1,
        typename T2,
        size_t LeanVecDims,
        size_t Extent,
        typename F = std::identity>
    LeanVecLoader<
        T1,
        T2,
        LeanVecDims,
        Extent,
        std::decay_t<std::invoke_result_t<F, const Alloc&>>>
    refine(lib::Val<Extent>, F&& f = std::identity()) const {
        using ARet = std::decay_t<std::invoke_result_t<F, const Alloc&>>;
        // Make sure the pre-set values are correct.
        if constexpr (Extent != Dynamic) {
            if (Extent != dims_) {
                throw ANNEXCEPTION("Invalid Extent specialization!");
            }
        }

        if constexpr (LeanVecDims != Dynamic) {
            if (LeanVecDims != leanvec_dims_) {
                throw ANNEXCEPTION("Invalid LeanVecDims specialization!");
            }
        }

        if (leanvec_kind_v<T1> != primary_kind_) {
            throw ANNEXCEPTION("Invalid Primary kind specialization!");
        }

        if (leanvec_kind_v<T2> != secondary_kind_) {
            throw ANNEXCEPTION("Invalid Secondary kind specialization!");
        }

        // Convert dynamic Extent matrices to static LeanVecDims
        auto matrices = std::optional<LeanVecMatrices<LeanVecDims>>(matrices_);

        return LeanVecLoader<T1, T2, LeanVecDims, Extent, ARet>(
            source_, leanvec_dims_, std::move(matrices), alignment_, f(allocator_)
        );
    }

  public:
    SourceTypes source_;
    size_t leanvec_dims_;
    size_t dims_;
    LeanVecKind primary_kind_;
    LeanVecKind secondary_kind_;
    std::optional<LeanVecMatrices<Dynamic>> matrices_;
    size_t alignment_;
    Alloc allocator_;
};

template <typename T1, typename T2, size_t LeanVecDims, size_t Extent, typename Alloc>
struct LeanVecLoader {
  public:
    using loaded_type = LeanDataset<T1, T2, LeanVecDims, Extent, Alloc>;

    explicit LeanVecLoader(
        SourceTypes source,
        size_t leanvec_dims,
        std::optional<LeanVecMatrices<LeanVecDims>> matrices,
        size_t alignment,
        const Alloc& allocator
    )
        : source_{std::move(source)}
        , leanvec_dims_{leanvec_dims}
        , matrices_{std::move(matrices)}
        , alignment_{alignment}
        , allocator_{allocator} {}

    loaded_type load() const {
        auto pool = threads::SequentialThreadPool();
        return load(pool);
    }

    template <typename F>
    LeanVecLoader<
        T1,
        T2,
        LeanVecDims,
        Extent,
        std::decay_t<std::invoke_result_t<F, const Alloc&>>>
    rebind_alloc(const F& f) {
        return LeanVecLoader<
            T1,
            T2,
            LeanVecDims,
            Extent,
            std::decay_t<std::invoke_result_t<F, const Alloc&>>>{
            source_, leanvec_dims_, matrices_, alignment_, f(allocator_)};
    }

    template <threads::ThreadPool Pool> loaded_type load(Pool& threadpool) const {
        return std::visit<loaded_type>(
            [&](auto source) {
                using U = std::decay_t<decltype(source)>;
                if constexpr (std::is_same_v<U, Reload>) {
                    return lib::load_from_disk<loaded_type>(
                        source.directory, alignment_, allocator_
                    );
                } else {
                    return lib::match(
                        LeanVecSourceTypes,
                        source.type,
                        [&]<typename V>(lib::Type<V> SVS_UNUSED(type)) {
                            using rebind_type = detail::select_rebind_allocator_t<V, Alloc>;
                            return loaded_type::reduce(
                                data::SimpleData<V, Extent, rebind_type>::load(source.path),
                                matrices_,
                                threadpool,
                                alignment_,
                                leanvec_dims_,
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
    lib::MaybeStatic<LeanVecDims> leanvec_dims_;
    std::optional<LeanVecMatrices<LeanVecDims>> matrices_;
    size_t alignment_;
    Alloc allocator_;
};

} // namespace leanvec

// Define dispatch conversion from ProtoLeanVecLoader to LeanVecLoader.
template <
    typename Primary,
    typename Secondary,
    size_t LeanVecDims,
    size_t Extent,
    typename Alloc>
struct lib::DispatchConverter<
    leanvec::ProtoLeanVecLoader<Alloc>,
    leanvec::LeanVecLoader<Primary, Secondary, LeanVecDims, Extent, Alloc>> {
    static int64_t match(const leanvec::ProtoLeanVecLoader<Alloc>& loader) {
        return overload_score<Primary, Secondary, LeanVecDims, Extent>(
            loader.primary_kind_, loader.leanvec_dims_, loader.secondary_kind_, loader.dims_
        );
        // if (loader.primary_kind_ != leanvec::leanvec_kind_v<Primary>) {
        //     return lib::invalid_match;
        // }

        // // Check secondary kind
        // if (loader.secondary_kind_ != leanvec::leanvec_kind_v<Secondary>) {
        //     return lib::invalid_match;
        // }

        // // Check extent-tags.
        // auto extent_match = lib::dispatch_match<lib::ExtentArg,
        // lib::ExtentTag<Extent>>(
        //     lib::ExtentArg{loader.dims_}
        // );

        // // If extents don't match, then we abort immediately.
        // if (extent_match < 0) {
        //     return lib::invalid_match;
        // }

        // // Check leanvec_dims-tags.
        // auto leanvec_dims_match =
        //     lib::dispatch_match<lib::ExtentArg,
        //     lib::ExtentTag<LeanVecDims>>(lib::ExtentArg{ loader.leanvec_dims_});
        // // If leanvec_dims don't match, then we abort immediately.
        // if (leanvec_dims_match < 0) {
        //     return lib::invalid_match;
        // }

        // return extent_match + leanvec_dims_match;
    }

    static leanvec::LeanVecLoader<Primary, Secondary, LeanVecDims, Extent, Alloc>
    convert(const leanvec::ProtoLeanVecLoader<Alloc>& loader) {
        return loader.template refine<Primary, Secondary, LeanVecDims, Extent>(
            lib::Val<Extent>()
        );
    }

    static std::string description() {
        auto dims = []() {
            if constexpr (Extent == Dynamic) {
                return "any";
            } else {
                return Extent;
            }
        }();

        auto leanvec_dims = []() {
            if constexpr (LeanVecDims == Dynamic) {
                return "any";
            } else {
                return LeanVecDims;
            }
        }();

        return fmt::format("LeanVecLoader dims-{}x{}", dims, leanvec_dims);
    }
};

} // namespace svs
