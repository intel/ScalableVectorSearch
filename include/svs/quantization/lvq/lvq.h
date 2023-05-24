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

/// @brief Sentinel to represent that the particular encoding has no residual dataset.
struct NoResidual {
    static constexpr std::string_view name = "lvq no residual";
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);

    lib::SaveType save(const lib::SaveContext& SVS_UNUSED(ctx)) const {
        return lib::SaveType(toml::table({{"name", name}}), save_version);
    }

    static NoResidual load(
        const toml::table& table,
        const lib::LoadContext& SVS_UNUSED(ctx),
        const lib::Version& version
    ) {
        if (version != save_version) {
            throw ANNEXCEPTION("Loading version mismatch!");
        }
        auto this_name = get(table, "name").value();
        if (this_name != name) {
            throw ANNEXCEPTION("Name mismatch!");
        }
        return NoResidual();
    }
};

///// Dataset Saving

struct LVQSaveParameters {
    // Save Parameters.
    static constexpr std::string_view name = "lvq dataset";
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);
};

template <typename Distance, typename Main, typename Residual>
struct DistanceMainResidualRef {
    // Constructor
    DistanceMainResidualRef(
        const Distance& distance, const Main& main, const Residual& residual
    )
        : distance{distance}
        , main{main}
        , residual{residual} {}

    // Saving.
    lib::SaveType save(const lib::SaveContext& ctx) const {
        auto table = toml::table(
            {{"name", LVQSaveParameters::name},
             {"distance", lib::recursive_save(distance, ctx)},
             {"primary", lib::recursive_save(main, ctx)},
             {"residual", lib::recursive_save(residual, ctx)}}
        );
        return lib::SaveType(std::move(table), LVQSaveParameters::save_version);
    }

    // Members as const-reference
    // Allows for lazy reconstruction.
    const Distance& distance;
    const Main& main;
    const Residual& residual;
};

///
/// @brief Return type for vector compression loading algorithms.
///
/// @tparam Distance The type of the modified distance functor to compensate for LVQ
///     compression. The exact value is an implementation detail.
/// @tparam Main The type of the primary compressed dataset. The exact value is an
///     implementation detail.
/// @tparam Residual The type of the residual dataset. The exact value is an
///     implementation detail.
///
/// Generally, this class is used as an intermediate class for use after dataset
/// compression/loading prior to assignment in an index implementation.
///
/// Users should not interact with this class directly outside of calling the ``save``
/// method to save a compressed dataset to disk.
///
template <typename Distance, typename Main, typename Residual> struct DistanceMainResidual {
    // Saving - delegate to the the reference version.
    lib::SaveType save(const lib::SaveContext& ctx) const {
        return DistanceMainResidualRef{distance, main, residual}.save(ctx);
    }

    ///
    /// @brief Save the compressed dataset to the directory.
    ///
    /// The corresponding directory can be used to construct a ``Reload`` class to enable
    /// reloading of the compressed dataset.
    ///
    void save(const std::filesystem::path directory) const { lib::save(*this, directory); }

    DistanceMainResidual(Distance distance, Main main, Residual residual)
        : distance{std::move(distance)}
        , main{std::move(main)}
        , residual{std::move(residual)} {}

    // Distance function to use for uncompressed vectors on the left and compressed vectors
    // on the right.
    Distance distance;
    // The primary compressed dataset.
    Main main;
    // The residual dataset. If the selected compression technique does not require a
    // residual, than this will be an instance of NoResidual.
    Residual residual;
};

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
void compress(
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
void compress_residual(
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

    // Type of the primary encoded dataset.
    using primary_type = ScaledBiasedDataset<Primary, Extent>;

    // No residual datasets with one-level compression.
    using residual_type = NoResidual;

    // Modified distance type for use with this dataset compression.
    template <typename Distance>
    using distance_type = typename op_type::distance_type<Distance>;

    ///
    /// @brief The composite return type following application of this dataset loader.
    ///
    /// @tparam Distance The distance type to use for the original dataset. Implementations
    ///     of compression loaders may modify the distance type to help undo some
    ///     steps performed during the compression process.
    ///
    template <typename Distance>
    using return_type =
        DistanceMainResidual<distance_type<Distance>, primary_type, residual_type>;

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
    template <typename Distance>
    return_type<Distance>
    load(const Distance& distance, [[maybe_unused]] size_t num_threads = 1) const {
        return std::visit<return_type<Distance>>(
            [&](auto source) {
                using T = std::decay_t<decltype(source)>;
                if constexpr (std::is_same_v<T, OnlineCompression>) {
                    return compress_dispatch(
                        distance, source.path, source.type, num_threads
                    );
                } else if (std::is_same_v<T, Reload>) {
                    return reload(distance, source.directory);
                }
            },
            source_
        );
    }

    template <typename Distance>
    return_type<Distance> compress_dispatch(
        const Distance& distance,
        const std::filesystem::path& path,
        DataType source_eltype,
        size_t num_threads = 1
    ) const {
        return match(
            SOURCE_ELEMENT_TYPES,
            source_eltype,
            [&]<typename T>(meta::Type<T> /*unused*/) {
                return compress_file<T>(distance, path, num_threads);
            }
        );
    }

    template <typename SourceType, typename Distance>
    return_type<Distance> compress_file(
        const Distance& distance, const std::filesystem::path& path, size_t num_threads = 1
    ) const {
        auto data = VectorDataLoader<SourceType>(path).load();
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
        auto&& [distance_modified, map, misc] = op(distance, data, threadpool);
        // Allocate the compressed dataset.
        auto dims = lib::MaybeStatic<Extent>(data.dimensions());
        primary_type primary{data.size(), dims, padding_};
        // Compress the dataset by:
        // 1. Lazily removing the per-vector bias using `map`.
        // 2. Using the `MinRange` compression codec to compress the result of `map`.
        compress(
            primary,
            data,
            lib::Compose(MinRange<Primary, Extent>(dims), std::move(map)),
            threadpool
        );
        return return_type<Distance>{
            std::move(distance_modified), std::move(primary), NoResidual()};
    }

    template <typename Distance>
    return_type<Distance>
    reload(const Distance& SVS_UNUSED(distance), const std::filesystem::path& dir) const {
        auto loader = lib::LoadOverride{[&](const toml::table& table,
                                            const lib::LoadContext& ctx,
                                            const lib::Version& SVS_UNUSED(version)) {
            auto this_name = get(table, "name").value();
            if (this_name != LVQSaveParameters::name) {
                throw ANNException("Name mismatch!");
            }

            return return_type<Distance>(
                lib::recursive_load<distance_type<Distance>>(
                    subtable(table, "distance"), ctx
                ),
                lib::recursive_load<primary_type>(subtable(table, "primary"), ctx),
                lib::recursive_load<residual_type>(subtable(table, "residual"), ctx)
            );
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
template <size_t Primary, size_t Residual, size_t Extent> class TwoLevelWithBias {
  public:
    // Traits
    using loader_tag = CompressorTag;

    // Sources from which compressed files can be obtained.
    using op_type = VectorBias;

    ///
    /// @brief Type of the primary encoded dataset.
    ///
    /// @tparam Extent The static length of the encoded data.
    ///
    using primary_type = ScaledBiasedDataset<Primary, Extent>;

    ///
    /// @brief Type of the residual dataset.
    ///
    /// @tparam Extent The static length of the encoded data.
    ///
    using residual_type = CompressedDataset<Signed, Residual, Extent>;

    /// @brief Modified distance type for use with this dataset compression.
    template <typename Distance>
    using distance_type = typename op_type::distance_type<Distance>;

    ///
    /// @brief The composite return type following application of this dataset loader.
    ///
    /// @tparam Distance The distance type to use for the original dataset. Implementations
    ///     of compression loaders may modify the distance type to help undo some
    ///     steps performed during the compression process.
    ///
    /// @tparam Extent The number of dimensions in this dataset.
    ///
    template <typename Distance>
    using return_type =
        DistanceMainResidual<distance_type<Distance>, primary_type, residual_type>;

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
    template <typename Distance>
    return_type<Distance>
    load(const Distance& distance, [[maybe_unused]] size_t num_threads = 1) const {
        return std::visit<return_type<Distance>>(
            [&](auto source) {
                using T = std::decay_t<decltype(source)>;
                if constexpr (std::is_same_v<T, OnlineCompression>) {
                    return compress_dispatch(
                        distance, source.path, source.type, num_threads
                    );
                } else if (std::is_same_v<T, Reload>) {
                    return reload(distance, source.directory);
                }
            },
            source_
        );
    }

    ///
    /// @brief Load a compressed dataset.
    ///
    template <typename Distance>
    return_type<Distance> compress_dispatch(
        const Distance& distance,
        const std::filesystem::path& path,
        DataType source_eltype,
        size_t num_threads = 1
    ) const {
        return match(
            SOURCE_ELEMENT_TYPES,
            source_eltype,
            [&]<typename T>(meta::Type<T> /*unused*/) {
                return compress_file<T>(distance, path, num_threads);
            }
        );
    }

    // Memory map the given file and compress the dataset from that file.
    template <typename ElementType, typename Distance>
    return_type<Distance> compress_file(
        const Distance& distance, const std::filesystem::path& path, size_t num_threads = 1
    ) const {
        auto data = VectorDataLoader<ElementType>(path).load();
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
        auto&& [distance_modified, map, misc] = op(distance, data, threadpool);
        auto primary = primary_type{data.size(), static_ndims, padding_};
        compress(
            primary,
            data,
            lib::Compose(MinRange<Primary, Extent>(static_ndims), map),
            threadpool
        );

        // Residual Compression.
        auto residual =
            CompressedDataset<Signed, Residual, Extent>{data.size(), static_ndims};
        compress_residual(
            residual, primary, data, ResidualEncoder<Residual>(), map, threadpool
        );
        return return_type<Distance>{
            std::move(distance_modified), std::move(primary), std::move(residual)};
    }

    template <typename Distance>
    return_type<Distance>
    reload(const Distance& SVS_UNUSED(distance), const std::filesystem::path& dir) const {
        auto loader = lib::LoadOverride{[&](const toml::table& table,
                                            const lib::LoadContext& ctx,
                                            const lib::Version& SVS_UNUSED(version)) {
            auto this_name = get(table, "name").value();
            if (this_name != LVQSaveParameters::name) {
                throw ANNException("Name mismatch!");
            }

            return return_type<Distance>(
                lib::recursive_load<distance_type<Distance>>(
                    subtable(table, "distance"), ctx
                ),
                lib::recursive_load<primary_type>(subtable(table, "primary"), ctx),
                lib::recursive_load<residual_type>(subtable(table, "residual"), ctx)
            );
        }};
        return lib::load(loader, dir);
    }
};

/////
///// One Level Global Quantization.
/////

// Forward Declaration
template <size_t Primary, size_t Extent> class GlobalOneLevelWithBias;

template <size_t Primary> struct UnspecializedGlobalOneLevelWithBias {
    // Constructors
    UnspecializedGlobalOneLevelWithBias() = default;

    // TODO: Propagate allocator request.
    template <typename Allocator>
    UnspecializedGlobalOneLevelWithBias(
        const UnspecializedVectorDataLoader<Allocator>& datafile, size_t padding = 0
    )
        : source_{std::in_place_type_t<OnlineCompression>(), datafile.path_, datafile.type_}
        , dims_{datafile.dims_}
        , padding_{padding} {}

    UnspecializedGlobalOneLevelWithBias(Reload reloader, size_t dims, size_t padding = 0)
        : source_{std::move(reloader)}
        , dims_{dims}
        , padding_{padding} {}

    ///
    /// @brief Construct a fully-typed compressor from the generic compressor.
    ///
    template <size_t Extent>
    GlobalOneLevelWithBias<Primary, Extent> refine(meta::Val<Extent> /*unused*/) const {
        return GlobalOneLevelWithBias<Primary, Extent>{*this};
    }

    ///// Members
    SourceTypes source_;
    size_t dims_;
    size_t padding_;
};

// One-Level Compressor subtracting out the dataset bias with global constants
template <size_t Primary, size_t Extent> class GlobalOneLevelWithBias {
  public:
    // Traits
    using loader_tag = CompressorTag;

    // Sources
    using op_type = VectorBias;

    /// Type of the primary encoded dataset.
    using primary_type = GlobalScaledBiasedDataset<Primary, Extent>;

    /// The residual to use with this dataset (none).
    using residual_type = NoResidual;

    /// Modified distance type for use with this dataset compression.
    template <typename Distance>
    using distance_type = typename op_type::distance_type<Distance>;

    /// The composite return type following application of this dataset loader.
    template <typename Distance>
    using return_type =
        DistanceMainResidual<distance_type<Distance>, primary_type, residual_type>;

  private:
    SourceTypes source_;
    size_t padding_;

  public:
    // Constructors
    template <typename T>
    GlobalOneLevelWithBias(const VectorDataLoader<T, Extent>& source, size_t padding = 0)
        : source_{std::in_place_type_t<OnlineCompression>(), source.get_path(), datatype_v<T>}
        , padding_{padding} {
        static_assert(meta::in<T>(SOURCE_ELEMENT_TYPES));
    }

    /// @brief Reload a previously saved LVQ dataset.
    GlobalOneLevelWithBias(Reload reload, size_t padding = 0)
        : source_{reload}
        , padding_{padding} {}

    GlobalOneLevelWithBias(const UnspecializedGlobalOneLevelWithBias<Primary>& unspecialized
    )
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

    template <typename Distance>
    return_type<Distance>
    load(const Distance& distance, [[maybe_unused]] size_t num_threads = 1) const {
        return std::visit<return_type<Distance>>(
            [&](auto source) {
                using T = std::decay_t<decltype(source)>;
                if constexpr (std::is_same_v<T, OnlineCompression>) {
                    return compress_dispatch(
                        distance, source.path, source.type, num_threads
                    );
                } else if (std::is_same_v<T, Reload>) {
                    return reload(distance, source.directory);
                }
            },
            source_
        );
    }

    template <typename Distance>
    return_type<Distance> compress_dispatch(
        const Distance& distance,
        const std::filesystem::path& path,
        DataType source_eltype,
        size_t num_threads = 1
    ) const {
        return match(
            SOURCE_ELEMENT_TYPES,
            source_eltype,
            [&]<typename T>(meta::Type<T> /*unused*/) {
                return compress_file<T>(distance, path, num_threads);
            }
        );
    }

    ///
    /// Memory map the given file and compress the dataset from that file.
    ///
    template <typename ElementType, typename Distance>
    return_type<Distance> compress_file(
        const Distance& distance, const std::string& path, size_t num_threads = 1
    ) const {
        auto data = VectorDataLoader<ElementType>(path).load();
        auto static_ndims = lib::MaybeStatic<Extent>(data.dimensions());
        threads::NativeThreadPool threadpool{num_threads};

        if constexpr (Extent != Dynamic) {
            if (data.dimensions() != Extent) {
                throw ANNEXCEPTION(
                    "File data has dimensions ",
                    data.dimensions(),
                    " while compression engine is expecting ",
                    Extent,
                    " dimensions!"
                );
            }
        }

        // Primary Compression.
        VectorBias op{};
        auto&& [distance_modified, map, misc] = op(distance, data, threadpool);
        auto extrema = mapped_extrema(data, map, threadpool);
        auto primary = primary_type{
            data.size(), static_ndims, extrema.scale(Primary), extrema.min(), padding_};

        compress(
            primary,
            data,
            lib::Compose(
                MinRange<Primary, Extent>(extrema.min(), extrema.max(), static_ndims), map
            ),
            threadpool
        );
        return return_type<Distance>{
            std::move(distance_modified), std::move(primary), NoResidual()};
    }

    template <typename Distance>
    return_type<Distance>
    reload(const Distance& SVS_UNUSED(distance), const std::filesystem::path& dir) const {
        auto loader = lib::LoadOverride{[&](const toml::table& table,
                                            const lib::LoadContext& ctx,
                                            const lib::Version& SVS_UNUSED(version)) {
            auto this_name = get(table, "name").value();
            if (this_name != LVQSaveParameters::name) {
                throw ANNException("Name mismatch!");
            }

            return return_type<Distance>(
                lib::recursive_load<distance_type<Distance>>(
                    subtable(table, "distance"), ctx
                ),
                lib::recursive_load<primary_type>(subtable(table, "primary"), ctx),
                lib::recursive_load<residual_type>(subtable(table, "residual"), ctx)
            );
        }};
        return lib::load(loader, dir);
    }
};

/////
///// Global Two Level
/////

// Forward Declaration
template <size_t Primary, size_t Residual, size_t Extent> class GlobalTwoLevelWithBias;

template <size_t Primary, size_t Residual> struct UnspecializedGlobalTwoLevelWithBias {
    // Constructors
    UnspecializedGlobalTwoLevelWithBias() = default;

    // TODO: Propagate allocator request.
    template <typename Allocator>
    UnspecializedGlobalTwoLevelWithBias(
        const UnspecializedVectorDataLoader<Allocator>& datafile, size_t padding = 0
    )
        : source_{std::in_place_type_t<OnlineCompression>(), datafile.path_, datafile.type_}
        , dims_{datafile.dims_}
        , padding_{padding} {}

    UnspecializedGlobalTwoLevelWithBias(Reload reloader, size_t dims, size_t padding = 0)
        : source_{std::move(reloader)}
        , dims_{dims}
        , padding_{padding} {}

    ///
    /// @brief Construct a fully-typed compressor from the generic compressor.
    ///
    template <size_t Extent>
    GlobalTwoLevelWithBias<Primary, Residual, Extent>
    refine(meta::Val<Extent> /*unused*/) const {
        return GlobalTwoLevelWithBias<Primary, Residual, Extent>{*this};
    }

    ///// Members
    SourceTypes source_;
    size_t dims_;
    size_t padding_;
};

template <size_t Primary, size_t Residual, size_t Extent> class GlobalTwoLevelWithBias {
  public:
    // Traits
    using loader_tag = CompressorTag;

    // Sources
    using op_type = VectorBias;

    using primary_type = GlobalScaledBiasedDataset<Primary, Extent>;
    using residual_type = CompressedDataset<Signed, Residual, Extent>;

    template <typename Distance>
    using distance_type = typename op_type::distance_type<Distance>;

    template <typename Distance>
    using return_type =
        DistanceMainResidual<distance_type<Distance>, primary_type, residual_type>;

  private:
    SourceTypes source_;
    size_t padding_;

  public:
    // Constructors
    template <typename T>
    GlobalTwoLevelWithBias(const VectorDataLoader<T, Extent>& source, size_t padding = 0)
        : source_{std::in_place_type_t<OnlineCompression>(), source.get_path(), datatype_v<T>}
        , padding_{padding} {
        static_assert(meta::in<T>(SOURCE_ELEMENT_TYPES));
    }

    /// @brief Reload a previously saved LVQ dataset.
    GlobalTwoLevelWithBias(Reload reload, size_t padding = 0)
        : source_{reload}
        , padding_{padding} {}

    GlobalTwoLevelWithBias(
        const UnspecializedGlobalTwoLevelWithBias<Primary, Residual>& unspecialized
    )
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

    template <typename Distance>
    return_type<Distance>
    load(const Distance& distance, [[maybe_unused]] size_t num_threads = 1) const {
        return std::visit<return_type<Distance>>(
            [&](auto source) {
                using T = std::decay_t<decltype(source)>;
                if constexpr (std::is_same_v<T, OnlineCompression>) {
                    return compress_dispatch(
                        distance, source.path, source.type, num_threads
                    );
                } else if (std::is_same_v<T, Reload>) {
                    return reload(distance, source.directory);
                }
            },
            source_
        );
    }

    template <typename Distance>
    return_type<Distance> compress_dispatch(
        const Distance& distance,
        const std::filesystem::path& path,
        DataType source_eltype,
        size_t num_threads = 1
    ) const {
        return match(
            SOURCE_ELEMENT_TYPES,
            source_eltype,
            [&]<typename T>(meta::Type<T> /*unused*/) {
                return compress_file<T>(distance, path, num_threads);
            }
        );
    }

    template <typename ElementType, typename Distance>
    return_type<Distance> compress_file(
        const Distance& distance, const std::string& path, size_t num_threads = 1
    ) const {
        auto data = VectorDataLoader<ElementType>(path).load();
        auto static_ndims = lib::MaybeStatic<Extent>(data.dimensions());
        threads::NativeThreadPool threadpool{num_threads};

        if constexpr (Extent != Dynamic) {
            if (data.dimensions() != Extent) {
                throw ANNEXCEPTION(
                    "File data has dimensions ",
                    data.dimensions(),
                    " while compression engine is expecting ",
                    Extent,
                    " dimensions!"
                );
            }
        }

        // Primary Compression.
        VectorBias op{};
        auto&& [distance_modified, map, misc] = op(distance, data, threadpool);
        auto extrema = mapped_extrema(data, map, threadpool);
        auto primary = primary_type{
            data.size(), static_ndims, extrema.scale(Primary), extrema.min(), padding_};

        compress(
            primary,
            data,
            lib::Compose(
                MinRange<Primary, Extent>(extrema.min(), extrema.max(), static_ndims), map
            ),
            threadpool
        );

        // Residual Compression.
        auto residual =
            CompressedDataset<Signed, Residual, Extent>{data.size(), static_ndims};
        compress_residual(
            residual, primary, data, ResidualEncoder<Residual>(), map, threadpool
        );
        return return_type<Distance>{
            std::move(distance_modified), std::move(primary), std::move(residual)};
    }

    template <typename Distance>
    return_type<Distance>
    reload(const Distance& SVS_UNUSED(distance), const std::filesystem::path& dir) const {
        auto loader = lib::LoadOverride{[&](const toml::table& table,
                                            const lib::LoadContext& ctx,
                                            const lib::Version& SVS_UNUSED(version)) {
            auto this_name = get(table, "name").value();
            if (this_name != LVQSaveParameters::name) {
                throw ANNException("Name mismatch!");
            }

            return return_type<Distance>(
                lib::recursive_load<distance_type<Distance>>(
                    subtable(table, "distance"), ctx
                ),
                lib::recursive_load<primary_type>(subtable(table, "primary"), ctx),
                lib::recursive_load<residual_type>(subtable(table, "residual"), ctx)
            );
        }};
        return lib::load(loader, dir);
    }
};

} // namespace lvq
} // namespace quantization
} // namespace svs
