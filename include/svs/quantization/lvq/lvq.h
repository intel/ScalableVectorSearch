/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

#pragma once

// Quantization
#include "svs/quantization/lvq/codec.h"
#include "svs/quantization/lvq/compressed.h"
#include "svs/quantization/lvq/datasets.h"
#include "svs/quantization/lvq/ops.h"
#include "svs/quantization/lvq/vectors.h"

// svs
#include "svs/core/data.h"
#include "svs/core/kmeans.h"
#include "svs/lib/meta.h"
#include "svs/lib/saveload.h"
#include "svs/lib/traits.h"

// stl
#include <string>
#include <string_view>
#include <variant>

namespace svs {
namespace quantization {
namespace lvq {

// Loader traits
struct CompressorTag : public lib::AbstractLoaderTag {};

class GlobalMinMax {
  private:
    float min_ = std::numeric_limits<float>::max();
    float max_ = std::numeric_limits<float>::lowest();

  public:
    // Constructors
    GlobalMinMax() = default;
    explicit GlobalMinMax(float min, float max)
        : min_{min}
        , max_{max} {}

    float min() const { return min_; }
    float max() const { return max_; }

    /// Compute the two-constant scale for the given minimum and maximum.
    float scale(size_t nbits) const {
        return (max() - min()) / (std::pow(2.0f, lib::narrow_cast<float>(nbits)) - 1);
    }

    // Update
    void update(float v) {
        min_ = std::min(v, min_);
        max_ = std::max(v, max_);
    }

    void update(GlobalMinMax other) {
        min_ = std::min(min(), other.min());
        max_ = std::max(max(), other.max());
    }
};

///
/// Compute the global extrema after applying the operation `map` each element of the
/// given dataset.
///
template <data::ImmutableMemoryDataset Data, typename Map, threads::ThreadPool Pool>
GlobalMinMax mapped_extrema(const Data& data, const Map& map, Pool& threadpool) {
    auto extrema_tls = threads::SequentialTLS<GlobalMinMax>(threadpool.size());
    threads::run(
        threadpool,
        threads::DynamicPartition{data.size(), 100'000},
        [&](const auto& is, uint64_t tid) {
            auto map_local = map;
            auto& extrema = extrema_tls[tid];
            for (auto i : is) {
                auto mapped = map_local(data.get_datum(i));
                for (auto j : mapped) {
                    extrema.update(lib::narrow<float>(j));
                }
            }
        }
    );

    auto final_extrema = GlobalMinMax();
    extrema_tls.visit([&final_extrema](const auto& other) { final_extrema.update(other); });
    return final_extrema;
}

///
/// Compress a dataset.
///
template <
    data::MemoryDataset Compressed,
    data::ImmutableMemoryDataset Original,
    typename Map,
    threads::ThreadPool Pool>
void generic_compress(
    Compressed& compressed, const Original& original, Map&& map, Pool& threadpool
) {
    if (compressed.size() != original.size()) {
        throw ANNEXCEPTION("Compressed and original dataset have mismatched sizes!");
    }
    threads::run(
        threadpool,
        threads::DynamicPartition(original.size(), 100'000),
        [&](const auto& is, uint64_t /*tid*/) {
            // Construct a thread-local copy of the original map.
            auto map_local = map;
            for (auto i : is) {
                compressed.set_datum(i, map_local(original.get_datum(i)));
            }
        }
    );
}

template <
    data::MemoryDataset Residual,
    data::ImmutableMemoryDataset Primary,
    data::ImmutableMemoryDataset Original,
    typename Map1,
    typename Map2,
    threads::ThreadPool Pool>
void generic_compress_residual(
    Residual& residual,
    const Primary& primary,
    const Original& original,
    Map1&& map_outer,
    Map2&& map_inner,
    Pool& threadpool
) {
    if (primary.size() != original.size()) {
        throw ANNEXCEPTION("Primary and original dataset have mismatched sizes!");
    }
    if (primary.size() != residual.size()) {
        throw ANNEXCEPTION("Primary and residual dataset have mismatched sizes!");
    }
    threads::run(
        threadpool,
        threads::DynamicPartition(original.size(), 100'000),
        [&](const auto& is, uint64_t /*tid*/) {
            // Construct a thread-local copy of the original map.
            auto map_outer_local = map_outer;
            auto map_inner_local = map_inner;
            for (auto i : is) {
                const auto& compressed = map_outer_local(
                    primary.get_datum(i), map_inner_local(original.get_datum(i))
                );
                residual.set_datum(i, compressed);
            }
        }
    );
}

// Partial template specializations to get the access mode value types set-up correctly.
// Can't define these inside the LVQDataset classes themselves because partial alias
// template specialization is not allowed in classes.
namespace detail {

// Default version - don't implement to hopefully get better error messages if a particular
// combination is not implemented.
template <data::AccessMode Mode, size_t Primary, size_t Residual, size_t Extent>
struct ValueType;

template <size_t Primary, size_t Residual, size_t Extent>
struct ValueType<data::FullAccess, Primary, Residual, Extent> {
    using type = ScaledBiasedWithResidual<Primary, Residual, Extent>;
};

template <size_t Primary, size_t Residual, size_t Extent>
struct ValueType<data::FastAccess, Primary, Residual, Extent> {
    using type = ScaledBiasedVector<Primary, Extent>;
};

template <data::AccessMode Mode, size_t Primary, size_t Residual, size_t Extent>
using const_value_type = typename ValueType<Mode, Primary, Residual, Extent>::type;

} // namespace detail

// Multi-level Dataset
template <size_t Primary, size_t Residual, size_t Extent, typename Storage = FlatStorage>
class LVQDataset {
    // Require that both the primary and residual datasets have the same backend storage.
    // This ensures that both are resizeable if being used in a dynamic context.
    static_assert(Storage::is_lvq_storage_tag);

    // Class invariants:
    //
    // * primary_.size() == residual_.size();
    // * primary_.dimensions() == residual_.dimensiosn();

  public:
    static constexpr bool is_resizeable = Storage::is_resizeable;
    using primary_type = ScaledBiasedDataset<Primary, Extent, Storage>;
    using residual_type = CompressedDataset<Signed, Residual, Extent, Storage>;

    // Members
  private:
    primary_type primary_;
    residual_type residual_;

    // Methods
  public:
    // Define the base template that is never meant to actually be instantiated.
    // Instead, we wish to use the full specializations instead.
    using const_value_type = ScaledBiasedWithResidual<Primary, Residual, Extent>;
    using value_type = const_value_type;

    template <data::AccessMode Mode>
    using mode_const_value_type = detail::const_value_type<Mode, Primary, Residual, Extent>;
    template <data::AccessMode Mode> using mode_value_type = mode_const_value_type<Mode>;

    ///// Constructors

    // Construct from constituent parts.
    LVQDataset(primary_type primary, residual_type residual)
        : primary_{std::move(primary)}
        , residual_{std::move(residual)} {
        auto primary_size = primary_.size();
        auto residual_size = residual_.size();
        if (primary_size != residual_size) {
            auto msg = fmt::format(
                "Primary size is {} while residual size is {}!", primary_size, residual_size
            );
            throw ANNEXCEPTION(msg);
        }

        auto primary_dims = primary_.dimensions();
        auto residual_dims = residual_.dimensions();
        if (primary_dims != residual_dims) {
            auto msg = fmt::format(
                "Primary dimensions is {} while residual dimensions is {}!",
                primary_dims,
                residual_dims
            );
            throw ANNEXCEPTION(msg);
        }
    }

    // Full dataset API.
    size_t size() const { return primary_.size(); }
    size_t dimensions() const { return primary_.dimensions(); }

    /// @brief Access just the first level of the two level dataset.
    ScaledBiasedVector<Primary, Extent>
    get_datum(size_t i, data::FastAccess SVS_UNUSED(mode)) const {
        return primary_.get_datum(i);
    }

    ///
    /// @brief Access both levels of the two-level dataset.
    ///
    /// Return a type that lazily combines the primary and residual.
    ///
    ScaledBiasedWithResidual<Primary, Residual, Extent>
    get_datum(size_t i, data::FullAccess SVS_UNUSED(mode)) const {
        return combine(primary_.get_datum(i), residual_.get_datum(i));
    }

    const_value_type get_datum(size_t i) const { return get_datum(i, data::full_access); }

    /// @brief Prefetch data in the first-level dataset.
    void prefetch(size_t i, data::FastAccess SVS_UNUSED(mode)) const {
        primary_.prefetch(i);
    }

    /// @brief Prefetch data in the first and second level datasets.
    void prefetch(size_t i, data::FullAccess SVS_UNUSED(mode)) const {
        prefetch(i, data::fast_access);
        residual_.prefetch(i);
    }

    void prefetch(size_t i) const { prefetch(i, data::full_access); }

    ///// Resizing
    void resize(size_t new_size)
        requires is_resizeable
    {
        // TODO: Should we roll-back in case of failure?
        primary_.resize(new_size);
        residual_.resize(new_size);
    }

    ///// Compaction
    template <typename I, typename Alloc, threads::ThreadPool Pool>
        requires is_resizeable
    void compact(
        const std::vector<I, Alloc>& new_to_old,
        Pool& threadpool,
        size_t batchsize = 1'000'000
    ) {
        primary_.compact(new_to_old, threadpool, batchsize);
        residual_.compact(new_to_old, threadpool, batchsize);
    }

    ///// Insertion
    template <typename QueryType, size_t N>
    void set_datum(size_t i, std::span<QueryType, N> datum) {
        auto dims = dimensions();
        assert(datum.size() == dims);

        // First, find the nearest centroid.
        auto selector = find_nearest(datum, *(primary_.view_centroids())).id();

        // Now that we have the nearest neighbor, perform primary compression, followed
        // by residual compression.
        auto buffer = std::vector<double>(dims);
        const auto& centroid = primary_.get_centroid(selector);
        for (size_t i = 0; i < dims; ++i) {
            buffer[i] = datum[i] - centroid[i];
        }

        // Compress and save primary.
        auto compressor = MinRange<Primary, Extent>(lib::MaybeStatic<Extent>(dims));
        primary_.set_datum(i, compressor(buffer, lib::narrow_cast<selector_t>(selector)));

        // Compress and save residual.
        auto residual_compressor = ResidualEncoder<Residual>();
        residual_.set_datum(i, residual_compressor(primary_.get_datum(i), buffer));
    }

    ///// Distance Adaptors
    EuclideanBiased adapt_distance(const distance::DistanceL2& SVS_UNUSED(dist)) const {
        return EuclideanBiased(primary_.view_centroids());
    }

    InnerProductBiased adapt_distance(const distance::DistanceIP& SVS_UNUSED(dist)) const {
        return InnerProductBiased(primary_.view_centroids());
    }

    Decompressor decompressor() const { return Decompressor{primary_.view_centroids()}; }

    DecompressionAdaptor<EuclideanBiased>
    self_distance(const distance::DistanceL2& SVS_UNUSED(dist)) const {
        return DecompressionAdaptor<EuclideanBiased>{
            std::in_place, primary_.view_centroids()};
    }

    DecompressionAdaptor<InnerProductBiased>
    self_distance(const distance::DistanceIP& SVS_UNUSED(dist)) const {
        return DecompressionAdaptor<InnerProductBiased>{
            std::in_place, primary_.view_centroids()};
    }

    ///// Saving
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);
    lib::SaveType save(const lib::SaveContext& ctx) const {
        auto table = toml::table(
            {{"primary", lib::recursive_save(primary_, ctx)},
             {"residual", lib::recursive_save(residual_, ctx)}}
        );
        return lib::SaveType(std::move(table), save_version);
    }
};

// Specialize one-level LVQ
template <size_t Primary, size_t Extent, typename Storage>
class LVQDataset<Primary, 0, Extent, Storage> {
    static_assert(Storage::is_lvq_storage_tag);

    // Type aliases
  public:
    static constexpr bool is_resizeable = Storage::is_resizeable;
    using primary_type = ScaledBiasedDataset<Primary, Extent, Storage>;

    // Members
  private:
    primary_type primary_;

    // Methods
  public:
    using value_type = typename primary_type::value_type;
    using const_value_type = typename primary_type::const_value_type;

    template <data::AccessMode Mode> using mode_const_value_type = const_value_type;
    template <data::AccessMode Mode> using mode_value_type = value_type;

    ///// Constructors
    LVQDataset(primary_type primary)
        : primary_{std::move(primary)} {}

    // Dataset API
    size_t size() const { return primary_.size(); }
    size_t dimensions() const { return primary_.dimensions(); }

    ///
    /// @brief Return the stored data at position `i`.
    ///
    /// @param i The index to access.
    /// @param mode The accessing mode.
    ///
    /// This class does not have different behavior under different access modes.
    /// It exposes the access mode API for compatibility purposes.
    ///
    template <data::AccessMode Mode = data::DefaultAccess>
    const_value_type get_datum(size_t i, Mode SVS_UNUSED(mode) = {}) const {
        return primary_.get_datum(i);
    }

    template <data::AccessMode Mode = data::DefaultAccess>
    void prefetch(size_t i, Mode SVS_UNUSED(mode) = {}) const {
        primary_.prefetch(i);
    }

    ///// Resizing
    void resize(size_t new_size)
        requires is_resizeable
    {
        primary_.resize(new_size);
    }

    ///// Compaction
    template <typename I, typename Alloc, threads::ThreadPool Pool>
        requires is_resizeable
    void compact(
        const std::vector<I, Alloc>& new_to_old,
        Pool& threadpool,
        size_t batchsize = 1'000'000
    ) {
        primary_.compact(new_to_old, threadpool, batchsize);
    }

    ///// Insertion
    template <typename QueryType, size_t N>
    void set_datum(size_t i, std::span<QueryType, N> datum) {
        auto dims = dimensions();
        if constexpr (checkbounds_v) {
            if (datum.size() != dims) {
                throw ANNEXCEPTION("Dimensions mismatch!");
            }
        }

        // First, map the data to its nearest centroid.
        auto selector = find_nearest(datum, *primary_.view_centroids()).id();

        // Now that we have a nearest neighbor, compress and perform the assignment.
        // We first subtract out the centroid from the data, then use the two-level
        // compression codec to finish up.
        auto buffer = std::vector<double>(dims);
        const auto& centroid = primary_.get_centroid(selector);
        for (size_t i = 0; i < dims; ++i) {
            buffer[i] = datum[i] - centroid[i];
        }

        auto compressor = MinRange<Primary, Extent>(lib::MaybeStatic<Extent>(dims));
        primary_.set_datum(i, compressor(buffer, lib::narrow_cast<selector_t>(selector)));
    }

    ///// Distance Adaptors
    EuclideanBiased adapt_distance(const distance::DistanceL2& SVS_UNUSED(dist)) const {
        return EuclideanBiased(primary_.view_centroids());
    }

    InnerProductBiased adapt_distance(const distance::DistanceIP& SVS_UNUSED(dist)) const {
        return InnerProductBiased(primary_.view_centroids());
    }

    Decompressor decompressor() const { return Decompressor{primary_.view_centroids()}; }

    DecompressionAdaptor<EuclideanBiased>
    self_distance(const distance::DistanceL2& SVS_UNUSED(dist)) const {
        return DecompressionAdaptor<EuclideanBiased>{
            std::in_place, primary_.view_centroids()};
    }

    DecompressionAdaptor<InnerProductBiased>
    self_distance(const distance::DistanceIP& SVS_UNUSED(dist)) const {
        return DecompressionAdaptor<InnerProductBiased>{
            std::in_place, primary_.view_centroids()};
    }

    ///// Saving
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);
    lib::SaveType save(const lib::SaveContext& ctx) const {
        auto table = toml::table({{"primary", lib::recursive_save(primary_, ctx)}});
        return lib::SaveType(std::move(table), save_version);
    }
};

/////
///// Mid Level Implementations
/////

// Valid source types for compression.
using source_element_types_t = meta::Types<float, svs::Float16>;
inline constexpr source_element_types_t SOURCE_ELEMENT_TYPES{};

// How are we expecting to obtain the data.
struct OnlineCompression {
  public:
    explicit OnlineCompression(const std::filesystem::path& path, DataType type)
        : path{path}
        , type{type} {
        if (!meta::in(type, SOURCE_ELEMENT_TYPES)) {
            throw ANNEXCEPTION("Invalid type!");
        }
    }

    // Members
  public:
    std::filesystem::path path;
    DataType type;
};

///
/// @brief Dispatch type indicating that a compressed dataset should be reloaded directly.
///
/// LVQ based loaders can either perform dataset compression online, or reload a previously
/// saved dataset.
///
/// Using this type in LVQ loader constructors indicates that reloading is desired.
///
struct Reload {
  public:
    ///
    /// @brief Construct a new Reloader.
    ///
    /// @param directory The directory where a LVQ compressed dataset was previously saved.
    ///
    explicit Reload(const std::filesystem::path& directory)
        : directory{directory} {}

    // Members
  public:
    std::filesystem::path directory;
};

// The various ways we can instantiate LVQ-based datasets.
using SourceTypes = std::variant<OnlineCompression, Reload>;

// Forward Declarations
template <size_t Primary, size_t Extent> class OneLevelWithBias;

template <size_t Primary> struct UnspecializedOneLevelWithBias {
    // Constructors
    UnspecializedOneLevelWithBias() = default;

    // TODO: Propagate allocator request.
    template <typename Allocator>
    UnspecializedOneLevelWithBias(
        const UnspecializedVectorDataLoader<Allocator>& datafile, size_t padding = 0
    )
        : source_{std::in_place_type_t<OnlineCompression>(), datafile.path_, datafile.type_}
        , dims_{datafile.dims_}
        , padding_{padding} {}

    UnspecializedOneLevelWithBias(Reload reloader, size_t dims, size_t padding = 0)
        : source_{std::move(reloader)}
        , dims_{dims}
        , padding_{padding} {}

    ///
    /// @brief Construct a fully-typed compressor from the generic compressor.
    ///
    template <size_t Extent>
    OneLevelWithBias<Primary, Extent> refine(meta::Val<Extent> /*unused*/) const {
        return OneLevelWithBias<Primary, Extent>{*this};
    }

    ///// Members
    SourceTypes source_;
    size_t dims_;
    size_t padding_;
};

///
/// @brief One-level locally-adaptive vector quantization loader
///
/// @tparam Primary The number of bits to use for each component.
/// @tparam Extent The compile-time logical number of dimensions for the dataset to
///     be loaded.
///
/// This class is responsible for loading and compressing a dataset using one-level
/// locally-adaptive vector quantization.
///
/// This class can be constructed in multiple ways which affects what happens when the
/// ``load`` method is called.
///
/// * If constructed from a ``svs::VectorDataLoader``, this class will lazily compress
///   the data pointed to by the loader.
/// * If constructed from an ``svs::quantization::lvq::Reload`` will reload a previously
///   compressed dataset. See ``svs::quantization::lvq::DistanceMainResidual`` for some
///   idea of how to save a compressed dataset.
///
template <size_t Primary, size_t Extent = svs::Dynamic> class OneLevelWithBias {
  public:
    // Traits
    using loader_tag = CompressorTag;

    // Sources from which compressed files can be obtained.
    using op_type = VectorBias;

    using default_builder_type = data::PolymorphicBuilder<HugepageAllocator>;

    // Type of the primary encoded dataset.
    template <typename Builder>
    using primary_type = ScaledBiasedDataset<Primary, Extent, get_storage_tag<Builder>>;

    template <typename Builder>
    using return_type = LVQDataset<Primary, 0, Extent, get_storage_tag<Builder>>;

  private:
    SourceTypes source_;
    size_t padding_;

  public:
    ///
    /// @brief Construct a LVQ compressor that will lazily compress the provided dataset.
    ///
    /// @param source The uncompressed vector data to compress with LVQ.
    /// @param padding The desired alignment for the start of each compressed vector.
    ///
    template <typename T>
    OneLevelWithBias(const VectorDataLoader<T, Extent>& source, size_t padding = 0)
        : source_{std::in_place_type_t<OnlineCompression>(), source.get_path(), datatype_v<T>}
        , padding_{padding} {
        static_assert(meta::in<T>(SOURCE_ELEMENT_TYPES));
    }

    ///
    /// @brief Reload a previously saved LVQ dataset.
    ///
    /// @param reload Reload with the directory containing the compressed dataset.
    /// @param padding The desired alignment for the start of each compressed vector.
    ///
    OneLevelWithBias(Reload reload, size_t padding = 0)
        : source_{reload}
        , padding_{padding} {}

    OneLevelWithBias(const UnspecializedOneLevelWithBias<Primary>& unspecialized)
        : source_{unspecialized.source_}
        , padding_{unspecialized.padding_} {
        // Perform a dimensionality check.
        if constexpr (Extent != Dynamic) {
            auto source_dims = unspecialized.dims_;
            if (source_dims != Extent) {
                throw ANNEXCEPTION("Dims mismatch!");
            }
        }
    }

    ///
    /// @brief Load a compressed dataset using the given distance function.
    ///
    /// @param distance The distance functor to use for the compressed vector data elements.
    /// @param num_threads The number of threads to use for compression.
    ///     Only used if this class was constructed from a ``svs::VectorDataLoader``.
    ///
    template <typename Builder = default_builder_type>
    return_type<Builder>
    load(const Builder& builder = {}, [[maybe_unused]] size_t num_threads = 1) const {
        return std::visit<return_type<Builder>>(
            [&](auto source) {
                using T = std::decay_t<decltype(source)>;
                if constexpr (std::is_same_v<T, OnlineCompression>) {
                    return compress_dispatch(
                        source.path, source.type, builder, num_threads
                    );
                } else if constexpr (std::is_same_v<T, Reload>) {
                    return reload(source.directory, builder);
                }
            },
            source_
        );
    }

    template <typename Builder = default_builder_type>
    return_type<Builder> compress_dispatch(
        const std::filesystem::path& path,
        DataType source_eltype,
        const Builder& builder = {},
        size_t num_threads = 1
    ) const {
        return match(
            SOURCE_ELEMENT_TYPES,
            source_eltype,
            [&]<typename T>(meta::Type<T> /*unused*/) {
                return compress_file<T>(path, builder, num_threads);
            }
        );
    }

    template <typename SourceType, typename Builder = default_builder_type>
    return_type<Builder> compress_file(
        const std::filesystem::path& path,
        const Builder& builder = {},
        size_t num_threads = 1
    ) const {
        auto data = VectorDataLoader<SourceType>(path).load();
        return compress(data, builder, num_threads);
    }

    template <data::ImmutableMemoryDataset Data, typename Builder = default_builder_type>
    return_type<Builder>
    compress(const Data& data, const Builder& builder = {}, size_t num_threads = 1) const {
        threads::NativeThreadPool threadpool{num_threads};

        if constexpr (Extent != Dynamic) {
            size_t data_dims = data.dimensions();
            if (data_dims != Extent) {
                throw ANNEXCEPTION(
                    "File data has dimensions ",
                    data_dims,
                    " while compression engine is expecting ",
                    Extent,
                    " dimensions!"
                );
            }
        }

        // Primary Compression.
        VectorBias op{};
        // Derive dataset per-vector means and construct a vector-wise operator `map`
        // That can be applied to each element in the dataset to remove this mean.
        auto [map, centroid] = op(data, threadpool);
        // Allocate the compressed dataset.
        auto dims = lib::MaybeStatic<Extent>(data.dimensions());
        primary_type<Builder> primary{data.size(), dims, padding_, builder};

        // Need to do a little dance to get the means into a form that can be cleanly
        // assigned to the dataset.
        auto means_f32 = std::vector<float>(centroid.begin(), centroid.end());
        auto means = data::SimpleData<float>(1, means_f32.size());
        means.set_datum(0, means_f32);
        primary.set_centroids(means);

        // Compress the dataset by:
        // 1. Lazily removing the per-vector bias using `map`.
        // 2. Using the `MinRange` compression codec to compress the result of `map`.
        generic_compress(
            primary,
            data,
            lib::Compose(MinRange<Primary, Extent>(dims), std::move(map)),
            threadpool
        );

        return return_type<Builder>{std::move(primary)};
    }

    template <typename Builder = default_builder_type>
    return_type<Builder>
    reload(const std::filesystem::path& dir, const Builder& builder = {}) const {
        auto loader = lib::LoadOverride{[&](const toml::table& table,
                                            const lib::LoadContext& ctx,
                                            const lib::Version& SVS_UNUSED(version)) {
            // auto this_name = get(table, "name").value();
            // if (this_name != LVQSaveParameters::name) {
            //     throw ANNException("Name mismatch!");
            // }

            return return_type<Builder>{
                lib::recursive_load<primary_type<Builder>>(
                    subtable(table, "primary"), ctx, builder
                ),
            };
        }};
        return lib::load(loader, dir);
    }
};

// Forward Declaration.
template <size_t Primary, size_t Residual, size_t Extent> class TwoLevelWithBias;

template <size_t Primary, size_t Residual> struct UnspecializedTwoLevelWithBias {
    UnspecializedTwoLevelWithBias() = default;

    template <typename Allocator>
    UnspecializedTwoLevelWithBias(
        const UnspecializedVectorDataLoader<Allocator>& datafile, size_t padding = 0
    )
        : source_{std::in_place_type_t<OnlineCompression>(), datafile.path_, datafile.type_}
        , dims_{datafile.dims_}
        , padding_{padding} {}

    UnspecializedTwoLevelWithBias(Reload reloader, size_t dims, size_t padding = 0)
        : source_{std::move(reloader)}
        , dims_{dims}
        , padding_{padding} {}

    ///
    /// @brief Construct a fully-typed compressor from the generic compressor.
    ///
    template <size_t Extent>
    TwoLevelWithBias<Primary, Residual, Extent> refine(meta::Val<Extent> /*unused*/) const {
        return TwoLevelWithBias<Primary, Residual, Extent>{*this};
    }

    ///// Members
    SourceTypes source_;
    size_t dims_;
    size_t padding_;
};

///
/// @brief Two-level locally-adaptive vector quantization loader
///
/// @tparam Primary The number of bits to use for each component for the primary dataset.
/// @tparam Residual The number of bits to use for each component in the residual dataset.
/// @tparam Extent The compile-time logical number of dimensions for the dataset to
///     be loaded.
///
/// This class is responsible for loading and compressing a dataset using one-level
/// locally-adaptive vector quantization.
///
/// This class can be constructed in multiple ways which affects what happens when the
/// ``load`` method is called.
///
/// * If constructed from a ``svs::VectorDataLoader``, this class will lazily compress
///   the data pointed to by the loader.
/// * If constructed from an ``svs::quantization::lvq::Reload`` will reload a previously
///   compressed dataset. See ``svs::quantization::lvq::DistanceMainResidual`` for some
///   idea of how to save a compressed dataset.
///
template <size_t Primary, size_t Residual, size_t Extent = Dynamic> class TwoLevelWithBias {
  public:
    // Traits
    using loader_tag = CompressorTag;

    // Sources from which compressed files can be obtained.
    using op_type = VectorBias;

    using default_builder_type = data::PolymorphicBuilder<HugepageAllocator>;

    ///
    /// @brief Type of the primary encoded dataset.
    ///
    /// @tparam Extent The static length of the encoded data.
    ///
    template <typename Builder>
    using primary_type = ScaledBiasedDataset<Primary, Extent, get_storage_tag<Builder>>;

    ///
    /// @brief Type of the residual dataset.
    ///
    /// @tparam Extent The static length of the encoded data.
    ///
    template <typename Builder>
    using residual_type =
        CompressedDataset<Signed, Residual, Extent, get_storage_tag<Builder>>;

    ///
    /// @brief The composite return type following application of this dataset loader.
    ///
    /// @tparam Distance The distance type to use for the original dataset. Implementations
    ///     of compression loaders may modify the distance type to help undo some
    ///     steps performed during the compression process.
    ///
    /// @tparam Extent The number of dimensions in this dataset.
    ///
    template <typename Builder>
    using return_type = LVQDataset<Primary, Residual, Extent, get_storage_tag<Builder>>;

  private:
    SourceTypes source_;
    size_t padding_;

  public:
    ///
    /// @brief Construct a new loader from the given source.
    ///
    /// @param source The source of the dataset.
    /// @param padding Extra padding per compressed vector in bytes. Setting this to a
    ///     multiple of either a half or a full cache line can substantially improve
    ///     performance at the cost of lower compression.
    ///
    template <typename T>
    TwoLevelWithBias(const VectorDataLoader<T, Extent>& source, size_t padding = 0)
        : source_{std::in_place_type_t<OnlineCompression>(), source.get_path(), datatype_v<T>}
        , padding_{padding} {
        static_assert(meta::in<T>(SOURCE_ELEMENT_TYPES));
    }

    ///
    /// @brief Reload a previously saved LVQ dataset.
    ///
    /// @param reload Reloader with the directory containing the compressed dataset.
    /// @param padding The desired alignment for the start of each compressed vector.
    ///
    TwoLevelWithBias(Reload reload, size_t padding = 0)
        : source_{reload}
        , padding_{padding} {}

    TwoLevelWithBias(const UnspecializedTwoLevelWithBias<Primary, Residual>& unspecialized)
        : source_{unspecialized.source_}
        , padding_{unspecialized.padding_} {
        // Perform a dimensionality check.
        if constexpr (Extent != Dynamic) {
            auto source_dims = unspecialized.dims_;
            if (source_dims != Extent) {
                throw ANNEXCEPTION("Dims mismatch!");
            }
        }
    }

    ///
    /// @brief Load a compressed dataset using the given distance function.
    ///
    /// @param distance The distance functor to use for the compressed vector data elements.
    /// @param num_threads The number of threads to use for compression.
    ///     Only used if this class was constructed from a ``svs::VectorDataLoader``.
    ///
    template <typename Builder = default_builder_type>
    return_type<Builder>
    load(const Builder& builder = {}, [[maybe_unused]] size_t num_threads = 1) const {
        return std::visit<return_type<Builder>>(
            [&](auto source) {
                using T = std::decay_t<decltype(source)>;
                if constexpr (std::is_same_v<T, OnlineCompression>) {
                    return compress_dispatch(
                        source.path, source.type, builder, num_threads
                    );
                } else if constexpr (std::is_same_v<T, Reload>) {
                    return reload(source.directory, builder);
                }
            },
            source_
        );
    }

    ///
    /// @brief Load a compressed dataset.
    ///
    template <typename Builder = default_builder_type>
    return_type<Builder> compress_dispatch(
        const std::filesystem::path& path,
        DataType source_eltype,
        const Builder& builder = {},
        size_t num_threads = 1
    ) const {
        return match(
            SOURCE_ELEMENT_TYPES,
            source_eltype,
            [&]<typename T>(meta::Type<T> /*unused*/) {
                return compress_file<T>(path, builder, num_threads);
            }
        );
    }

    template <typename SourceType, typename Builder = default_builder_type>
    return_type<Builder> compress_file(
        const std::filesystem::path& path,
        const Builder& builder = {},
        size_t num_threads = 1
    ) const {
        auto data = VectorDataLoader<SourceType>(path).load();
        return compress(data, builder, num_threads);
    }

    template <data::ImmutableMemoryDataset Data, typename Builder = default_builder_type>
    return_type<Builder>
    compress(const Data& data, const Builder& builder = {}, size_t num_threads = 1) const {
        auto static_ndims = lib::MaybeStatic<Extent>(data.dimensions());
        threads::NativeThreadPool threadpool{num_threads};

        if constexpr (Extent != Dynamic) {
            size_t data_dims = data.dimensions();
            if (data_dims != Extent) {
                throw ANNEXCEPTION(
                    "File data has dimensions ",
                    data_dims,
                    " while compression engine is expecting ",
                    Extent,
                    " dimensions!"
                );
            }
        }

        // Primary Compression.
        VectorBias op{};
        auto&& [map, centroid] = op(data, threadpool);
        auto primary = primary_type<Builder>{data.size(), static_ndims, padding_, builder};

        auto means_f32 = std::vector<float>(centroid.begin(), centroid.end());
        auto means = data::SimpleData<float>(1, means_f32.size());
        means.set_datum(0, means_f32);
        primary.set_centroids(means);

        generic_compress(
            primary,
            data,
            lib::Compose(MinRange<Primary, Extent>(static_ndims), map),
            threadpool
        );

        // Residual Compression.
        auto residual = residual_type<Builder>{data.size(), static_ndims, builder};
        generic_compress_residual(
            residual, primary, data, ResidualEncoder<Residual>(), map, threadpool
        );
        return return_type<Builder>{std::move(primary), std::move(residual)};
    }

    template <typename Builder = default_builder_type>
    return_type<Builder>
    reload(const std::filesystem::path& dir, const Builder& builder = {}) const {
        auto loader = lib::LoadOverride{[&](const toml::table& table,
                                            const lib::LoadContext& ctx,
                                            const lib::Version& SVS_UNUSED(version)) {
            // auto this_name = get(table, "name").value();
            // if (this_name != LVQSaveParameters::name) {
            //     throw ANNException("Name mismatch!");
            // }

            return return_type<Builder>{
                lib::recursive_load<primary_type<Builder>>(
                    subtable(table, "primary"), ctx, builder
                ),
                lib::recursive_load<residual_type<Builder>>(
                    subtable(table, "residual"), ctx, builder
                )};
        }};
        return lib::load(loader, dir);
    }
};
} // namespace lvq
} // namespace quantization
} // namespace svs
