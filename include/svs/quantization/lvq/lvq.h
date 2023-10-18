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
#include "svs/lib/misc.h"
#include "svs/lib/saveload.h"

// stl
#include <string>
#include <string_view>
#include <variant>

namespace svs {
namespace quantization {
namespace lvq {

namespace detail {
// This type alias is shared between both one-level and two-level LVQ datasets.
using centroid_type = data::SimpleData<float, Dynamic>;
} // namespace detail

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

// Multi-level Dataset
template <
    size_t Primary,
    size_t Residual = 0,
    size_t Extent = Dynamic,
    typename Alloc = lib::Allocator<std::byte>>
class LVQDataset {
    // Class invariants:
    //
    // * primary_.size() == residual_.size();
    // * primary_.dimensions() == residual_.dimensiosn();

  public:
    constexpr static size_t primary_bits = Primary;
    constexpr static size_t residual_bits = Residual;
    static constexpr bool is_resizeable = detail::is_blocked<Alloc>;
    using primary_type = ScaledBiasedDataset<Primary, Extent, Alloc>;
    using residual_type = CompressedDataset<Signed, Residual, Extent, Alloc>;
    using centroid_type = detail::centroid_type;
    using allocator_type = Alloc;

    // Members
  private:
    primary_type primary_;
    residual_type residual_;
    std::shared_ptr<centroid_type> centroids_;

    // Methods
  public:
    using const_primary_value_type = ScaledBiasedVector<Primary, Extent>;

    // Define the base template that is never meant to actually be instantiated.
    // Instead, we wish to use the full specializations instead.
    using const_value_type = ScaledBiasedWithResidual<Primary, Residual, Extent>;
    using value_type = const_value_type;

    ///// Constructors

    // Construct from constituent parts.
    LVQDataset(primary_type primary, residual_type residual, const centroid_type& centroids)
        : primary_{std::move(primary)}
        , residual_{std::move(residual)}
        , centroids_{
              std::make_shared<centroid_type>(centroids.size(), centroids.dimensions())} {
        auto primary_size = primary_.size();
        auto residual_size = residual_.size();
        if (primary_size != residual_size) {
            throw ANNEXCEPTION(
                "Primary size is {} while residual size is {}!", primary_size, residual_size
            );
        }
        data::copy(centroids, *centroids_);

        auto primary_dims = primary_.dimensions();
        auto residual_dims = residual_.dimensions();
        auto centroid_dims = centroids.dimensions();
        if (primary_dims != residual_dims) {
            throw ANNEXCEPTION(
                "Primary dimensions is {} while residual dimensions is {}!",
                primary_dims,
                residual_dims
            );
        }

        if (primary_dims != centroid_dims) {
            throw ANNEXCEPTION(
                "Primary dimension is {} while centroids is {}", primary_dims, centroid_dims
            );
        }
    }

    // Full dataset API.
    size_t size() const { return primary_.size(); }
    size_t dimensions() const { return primary_.dimensions(); }

    ///
    /// @brief Access both levels of the two-level dataset.
    ///
    /// Return a type that lazily combines the primary and residual.
    ///
    const_value_type get_datum(size_t i) const {
        return combine(primary_.get_datum(i), residual_.get_datum(i));
    }

    /// @brief Prefetch data in the first and second level datasets.
    void prefetch(size_t i) const {
        primary_.prefetch(i);
        residual_.prefetch(i);
    }

    /// @brief Access only the first level of the dataset.
    const_primary_value_type get_primary(size_t i) const { return primary_.get_datum(i); }

    /// @brief Prefetch only the primary dataset.
    void prefetch_primary(size_t i) const { primary_.prefetch(i); }

    ///// Resizing
    void resize(size_t new_size)
        requires is_resizeable
    {
        // TODO: Should we roll-back in case of failure?
        primary_.resize(new_size);
        residual_.resize(new_size);
    }

    ///// Compaction
    template <typename I, typename A, threads::ThreadPool Pool>
        requires is_resizeable
    void compact(
        const std::vector<I, A>& new_to_old, Pool& threadpool, size_t batchsize = 1'000'000
    ) {
        primary_.compact(new_to_old, threadpool, batchsize);
        residual_.compact(new_to_old, threadpool, batchsize);
    }

    // Return a shared copy of the LVQ centroids.
    std::shared_ptr<const centroid_type> view_centroids() const { return centroids_; }
    std::span<const float> get_centroid(size_t i) const { return centroids_->get_datum(i); }

    ///// Insertion
    template <typename QueryType, size_t N>
    void set_datum(size_t i, std::span<QueryType, N> datum) {
        auto dims = dimensions();
        assert(datum.size() == dims);

        // First, find the nearest centroid.
        auto selector = find_nearest(datum, *centroids_).id();

        // Now that we have the nearest neighbor, perform primary compression, followed
        // by residual compression.
        auto buffer = std::vector<double>(dims);
        const auto& centroid = get_centroid(selector);
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

    ///// Decompressor
    Decompressor decompressor() const { return Decompressor{centroids_}; }

    ///// Static Constructors.
    template <data::ImmutableMemoryDataset Dataset>
    static LVQDataset compress(const Dataset& data, const allocator_type& allocator = {}) {
        return compress(data, 1, 0, allocator);
    }

    template <data::ImmutableMemoryDataset Dataset>
    static LVQDataset compress(
        const Dataset& data,
        size_t num_threads,
        size_t alignment,
        const allocator_type& allocator = {}
    ) {
        auto pool = threads::NativeThreadPool{num_threads};
        return compress(data, pool, alignment, allocator);
    }

    template <data::ImmutableMemoryDataset Dataset, threads::ThreadPool Pool>
    static LVQDataset compress(
        const Dataset& data,
        Pool& threadpool,
        size_t alignment,
        const allocator_type& allocator = {}
    ) {
        size_t data_dims = data.dimensions();
        auto static_ndims = lib::MaybeStatic<Extent>(data_dims);
        if (Extent != Dynamic) {
            size_t data_dims = data.dimensions();
            if (data_dims != Extent) {
                throw ANNEXCEPTION("Dimension mismatch!");
            }
        }

        // Primary Compression.
        VectorBias op{};
        auto&& [map, centroid] = op(data, threadpool);
        auto primary = primary_type{data.size(), static_ndims, alignment, allocator};

        // Need to do a little dance to get the means into a form that can be cleanly
        // assigned to the dataset.
        auto centroids = centroid_type(1, centroid.size());
        centroids.set_datum(0, centroid);
        generic_compress(
            primary,
            data,
            lib::Compose(MinRange<Primary, Extent>(static_ndims), map),
            threadpool
        );

        // Residual Compression.
        auto residual = residual_type{data.size(), static_ndims, allocator};
        generic_compress_residual(
            residual, primary, data, ResidualEncoder<Residual>(), map, threadpool
        );
        return LVQDataset{std::move(primary), std::move(residual), centroids};
    }

    ///// Saving

    // Version History
    // v0.0.1 - BREAKING
    //   - Moved LVQ centroid storage location into the LVQ dataset instead of with
    //     the primary dataset.
    static constexpr lib::Version save_version = lib::Version(0, 0, 1);
    lib::SaveTable save(const lib::SaveContext& ctx) const {
        return lib::SaveTable(
            save_version,
            {SVS_LIST_SAVE_(primary, ctx),
             SVS_LIST_SAVE_(residual, ctx),
             {"centroids", lib::save(*centroids_, ctx)}}
        );
    }

    static LVQDataset load(
        const toml::table& table,
        const lib::LoadContext& ctx,
        const lib::Version& version,
        const allocator_type& allocator = {}
    ) {
        if (version != save_version) {
            throw ANNEXCEPTION(
                "LVQ Dataset was saved using version {}, which is incompatible with the "
                "current version ({}).",
                version,
                save_version
            );
        }
        return LVQDataset{
            SVS_LOAD_MEMBER_AT_(table, primary, ctx, allocator),
            SVS_LOAD_MEMBER_AT_(table, residual, ctx, allocator),
            lib::load_at<centroid_type>(table, "centroids", ctx)};
    }
};

// Specialize one-level LVQ
template <size_t Primary, size_t Extent, typename Alloc>
class LVQDataset<Primary, 0, Extent, Alloc> {
  public:
    constexpr static size_t primary_bits = Primary;
    constexpr static size_t residual_bits = 0;

    static constexpr bool is_resizeable = detail::is_blocked<Alloc>;
    using allocator_type = Alloc;
    using primary_type = ScaledBiasedDataset<Primary, Extent, allocator_type>;
    using centroid_type = detail::centroid_type;

    // Members
  private:
    primary_type primary_;
    std::shared_ptr<centroid_type> centroids_;

    // Methods
  public:
    using value_type = typename primary_type::value_type;
    using const_value_type = typename primary_type::const_value_type;

    ///// Constructors
    LVQDataset(primary_type primary, const centroid_type& centroids)
        : primary_{std::move(primary)}
        , centroids_{
              std::make_shared<centroid_type>(centroids.size(), centroids.dimensions())} {
        data::copy(centroids, *centroids_);
    }

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
    const_value_type get_datum(size_t i) const { return primary_.get_datum(i); }

    void prefetch(size_t i) const { primary_.prefetch(i); }

    ///// Resizing
    void resize(size_t new_size)
        requires is_resizeable
    {
        primary_.resize(new_size);
    }

    ///// Compaction
    template <typename I, typename A, threads::ThreadPool Pool>
        requires is_resizeable
    void compact(
        const std::vector<I, A>& new_to_old, Pool& threadpool, size_t batchsize = 1'000'000
    ) {
        primary_.compact(new_to_old, threadpool, batchsize);
    }

    std::shared_ptr<const centroid_type> view_centroids() const { return centroids_; }
    std::span<const float> get_centroid(size_t i) const { return centroids_->get_datum(i); }

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
        auto selector = find_nearest(datum, *centroids_).id();

        // Now that we have a nearest neighbor, compress and perform the assignment.
        // We first subtract out the centroid from the data, then use the two-level
        // compression codec to finish up.
        auto buffer = std::vector<double>(dims);
        const auto& centroid = get_centroid(selector);
        for (size_t i = 0; i < dims; ++i) {
            buffer[i] = datum[i] - centroid[i];
        }

        auto compressor = MinRange<Primary, Extent>(lib::MaybeStatic<Extent>(dims));
        primary_.set_datum(i, compressor(buffer, lib::narrow_cast<selector_t>(selector)));
    }

    ///// Decompressor
    Decompressor decompressor() const { return Decompressor{centroids_}; }

    ///// Static Constructors
    template <data::ImmutableMemoryDataset Dataset>
    static LVQDataset compress(const Dataset& data, const allocator_type& allocator = {}) {
        return compress(data, 1, 0, allocator);
    }

    template <data::ImmutableMemoryDataset Dataset>
    static LVQDataset compress(
        const Dataset& data,
        size_t num_threads,
        size_t alignment,
        const allocator_type& allocator = {}
    ) {
        auto pool = threads::NativeThreadPool{num_threads};
        return compress(data, pool, alignment, allocator);
    }

    template <data::ImmutableMemoryDataset Dataset, threads::ThreadPool Pool>
    static LVQDataset compress(
        const Dataset& data,
        Pool& threadpool,
        size_t alignment,
        const allocator_type& allocator = {}
    ) {
        if (Extent != Dynamic) {
            size_t data_dims = data.dimensions();
            if (data_dims != Extent) {
                throw ANNEXCEPTION("Dimension mismatch!");
            }
        }

        // Primary Compression.
        VectorBias op{};
        // Derive dataset per-vector means and construct a vector-wise operator `map`
        // That can be applied to each element in the dataset to remove this mean.
        auto [map, centroid] = op(data, threadpool);
        // Allocate the compressed dataset.
        auto dims = lib::MaybeStatic<Extent>(data.dimensions());
        auto primary = primary_type{data.size(), dims, alignment, allocator};

        // Need to do a little dance to get the means into a form that can be cleanly
        // assigned to the dataset.
        auto centroids = centroid_type{1, centroid.size()};
        centroids.set_datum(0, centroid);

        // Compress the dataset by:
        // 1. Lazily removing the per-vector bias using `map`.
        // 2. Using the `MinRange` compression codec to compress the result of `map`.
        generic_compress(
            primary,
            data,
            lib::Compose(MinRange<Primary, Extent>(dims), std::move(map)),
            threadpool
        );
        return LVQDataset{std::move(primary), centroids};
    }

    ///// Saving

    // Version History
    // v0.0.1 - BREAKING
    //   - Moved LVQ centroid storage location into the LVQ dataset instead of with
    //     the primary dataset.
    static constexpr lib::Version save_version = lib::Version(0, 0, 1);
    lib::SaveTable save(const lib::SaveContext& ctx) const {
        return lib::SaveTable(
            save_version,
            {SVS_LIST_SAVE_(primary, ctx), {"centroids", lib::save(*centroids_, ctx)}}
        );
    }

    static LVQDataset load(
        const toml::table& table,
        const lib::LoadContext& ctx,
        const lib::Version& version,
        const allocator_type& allocator = {}
    ) {
        if (version != save_version) {
            throw ANNEXCEPTION(
                "LVQ Dataset was saved using version {}, which is incompatible with the "
                "current version ({}).",
                version,
                save_version
            );
        }

        return LVQDataset{
            SVS_LOAD_MEMBER_AT_(table, primary, ctx, allocator),
            lib::load_at<centroid_type>(table, "centroids", ctx)};
    }
};

/////
///// LVQDataset Concept
/////

template <typename T> inline constexpr bool is_lvq_dataset = false;
template <size_t Primary, size_t Residual, size_t Extent, typename Storage>
inline constexpr bool is_lvq_dataset<LVQDataset<Primary, Residual, Extent, Storage>> = true;

template <typename T>
concept IsLVQDataset = is_lvq_dataset<T>;

template <typename T>
concept IsTwoLevelDataset = is_lvq_dataset<T> && (T::residual_bits != 0);

// Accessor for obtaining the primary level of a two-level dataset.
struct PrimaryAccessor {
    template <IsTwoLevelDataset Data>
    using const_value_type = typename Data::const_primary_value_type;

    template <IsTwoLevelDataset Data>
    const_value_type<Data> operator()(const Data& data, size_t i) const {
        return data.get_primary(i);
    }

    template <IsTwoLevelDataset Data> void prefetch(const Data& data, size_t i) const {
        return data.prefetch_primary(i);
    }
};

/////
///// Distance Adaptation.
/////

///
/// @brief Adapt the distance functor for use with the LVQ dataset.
///
/// @param dataset The dataset from which LVQ vectors will be obtained.
/// @param distance The distance functor to modify.
///
/// The returned distance functor will be appropiate for use with compressed dataset vector
/// data on the left and LVQ vectors originating from the dataset on the right.
///
template <IsLVQDataset Data, typename Distance>
biased_distance_t<Distance>
adapt(const Data& dataset, const Distance& SVS_UNUSED(distance)) {
    return biased_distance_t<Distance>(dataset.view_centroids());
}

///
/// @brief Adapt the distance functor for self-distance use over the LVQ dataset.
///
/// @param dataset The dataset from which LVQ vectors will be obtained.
/// @param distance The distance functor to modify.
///
/// The returned distance functor can be used to compute distances between two elements of
/// the LVQ dataset.
///
template <IsLVQDataset Data, typename Distance>
DecompressionAdaptor<biased_distance_t<Distance>>
adapt_for_self(const Data& dataset, const Distance& SVS_UNUSED(distance)) {
    return DecompressionAdaptor<biased_distance_t<Distance>>(
        std::in_place, dataset.view_centroids()
    );
}

/////
///// Load Helpers
/////

// Types to use for lazy compression.
inline constexpr meta::Types<float, Float16> CompressionTs{};

// How are we expecting to obtain the data.
struct OnlineCompression {
  public:
    explicit OnlineCompression(const std::filesystem::path& path, DataType type)
        : path{path}
        , type{type} {
        if (!meta::in(type, CompressionTs)) {
            throw ANNEXCEPTION("Invalid type!");
        }
    }

    // Members
  public:
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
/// Using this type in LVQ loader constructors indicates that reloading is desired.
///
struct Reload {
  public:
    ///
    /// @brief Construct a new Reloader.
    ///
    /// @param directory The directory where a LVQ compressed dataset was previously
    /// saved.
    ///
    explicit Reload(const std::filesystem::path& directory)
        : directory{directory} {}

    // Members
  public:
    std::filesystem::path directory;
};

// The various ways we can instantiate LVQ-based datasets..
using SourceTypes = std::variant<OnlineCompression, Reload>;

// Forward Declaration.
template <size_t Primary, size_t Residual, size_t Extent, typename Alloc> struct LVQLoader;

template <size_t Primary, size_t Residual = 0, typename Alloc = lib::Allocator<std::byte>>
struct ProtoLVQLoader {
  public:
    // Constructors
    ProtoLVQLoader() = default;

    // TODO: Propagate allocator request.
    explicit ProtoLVQLoader(
        const UnspecializedVectorDataLoader<Alloc>& datafile, size_t alignment = 0
    )
        : source_{std::in_place_type_t<OnlineCompression>(), datafile.path_, datafile.type_}
        , dims_{datafile.dims_}
        , alignment_{alignment}
        , allocator_{datafile.allocator_} {}

    explicit ProtoLVQLoader(
        Reload reloader, size_t dims, size_t alignment, const Alloc& allocator
    )
        : source_{std::move(reloader)}
        , dims_{dims}
        , alignment_{alignment}
        , allocator_{allocator} {}

    explicit ProtoLVQLoader(Reload reloader, size_t dims, size_t alignment)
        : ProtoLVQLoader(std::move(reloader), dims, alignment, Alloc()) {}

    template <size_t Extent, typename F = std::identity>
    LVQLoader<
        Primary,
        Residual,
        Extent,
        std::decay_t<std::invoke_result_t<F, const Alloc&>>>
    refine(meta::Val<Extent>, F&& f = std::identity()) const {
        using ARet = std::decay_t<std::invoke_result_t<F, const Alloc&>>;
        // Make sure the pre-set values are correct.
        if constexpr (Extent != Dynamic) {
            if (Extent != dims_) {
                throw ANNEXCEPTION("Invalid specialization!");
            }
        }
        return LVQLoader<Primary, Residual, Extent, ARet>(
            source_, alignment_, f(allocator_)
        );
    }

  public:
    SourceTypes source_;
    size_t dims_;
    size_t alignment_;
    Alloc allocator_;
};

template <size_t Primary, size_t Residual, size_t Extent, typename Alloc> struct LVQLoader {
  public:
    using loaded_type = LVQDataset<Primary, Residual, Extent, Alloc>;

    explicit LVQLoader(SourceTypes source, size_t alignment, const Alloc& allocator)
        : source_{std::move(source)}
        , alignment_{alignment}
        , allocator_{allocator} {}

    loaded_type load() const {
        auto pool = threads::SequentialThreadPool();
        return load(pool);
    }

    template <threads::ThreadPool Pool> loaded_type load(Pool& threadpool) const {
        return std::visit<loaded_type>(
            [&](auto source) {
                using T = std::decay_t<decltype(source)>;
                if constexpr (std::is_same_v<T, Reload>) {
                    // TODO: Enable different loaded alignment.
                    return lib::load_from_disk<loaded_type>(source.directory, allocator_);
                } else {
                    return meta::match(
                        CompressionTs,
                        source.type,
                        [&]<typename T>(meta::Type<T> SVS_UNUSED(type)) {
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
} // namespace svs
