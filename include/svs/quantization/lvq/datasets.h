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

#include "svs/quantization/lvq/compressed.h"
#include "svs/quantization/lvq/vectors.h"

#include "svs/lib/exception.h"
#include "svs/lib/saveload.h"

namespace svs::quantization::lvq {

//
// Summary of Dataset Types defined in this file.
//
// -- CompressedDataset<Sign, Bits, Extent>
// A compressed dataset storing only quantized data with no scaling or bias.
//
// -- ScaledBiasedDataset<Bits, Extent>
// Each vector is encoded using two constants, a bias (the smallest value any dimension
// can take) and a scaling parameter.
//

// Backing storage for LVQ datasets.
struct FlatStorage {
    // Require this static member to provide some static asserts in the dataset
    // implementations.
    //
    // Basically, leave some bread-crumbs for when we inevitably return to it.
    static constexpr bool is_lvq_storage_tag = true;
    static constexpr bool is_resizeable = false;

    template <size_t Extent = Dynamic>
    using type = data::SimplePolymorphicData<std::byte, Extent>;

    using builder = data::PolymorphicBuilder<HugepageAllocator>;

    template <size_t Extent = Dynamic>
    using loader_type = VectorDataLoader<std::byte, Extent, builder>;
};

struct BlockedStorage {
    static constexpr bool is_lvq_storage_tag = true;
    static constexpr bool is_resizeable = true;

    template <size_t Extent = Dynamic> using type = data::BlockedData<std::byte, Extent>;

    using builder = data::BlockedBuilder;

    template <size_t Extent = Dynamic>
    using loader_type = VectorDataLoader<std::byte, Extent, builder>;
};

namespace detail {

template <typename T> struct StorageTag;
template <typename Allocator> struct StorageTag<data::PolymorphicBuilder<Allocator>> {
    using type = FlatStorage;
};

template <> struct StorageTag<data::BlockedBuilder> {
    using type = BlockedStorage;
};
} // namespace detail

template <typename T> using get_storage_tag = typename detail::StorageTag<T>::type;

/////
///// Layout Helpers
/////

struct __attribute__((packed)) ScalarBundle {
    Float16 scale;
    Float16 bias;
    selector_t selector;
};
// Ensure the compiler respects our request for packing.
static_assert(sizeof(ScalarBundle) == 2 * sizeof(Float16) + sizeof(selector_t));

///
/// Layout for `ScaledBiasedVector` where the scaling constants are stored inline after the
/// `CompressedVector` storing the data.
///
template <size_t Bits, size_t Extent> class ScaledBiasedVectorLayout {
  public:
    using cv_type = CompressedVector<Unsigned, Bits, Extent>;

    using const_value_type = ScaledBiasedVector<Bits, Extent>;
    using scalar_type = typename const_value_type::scalar_type;

    explicit ScaledBiasedVectorLayout(lib::MaybeStatic<Extent> dims)
        : dims_{dims} {}

    constexpr size_t total_bytes() const {
        return cv_type::compute_bytes(dims_) + 2 * sizeof(scalar_type) + sizeof(selector_t);
    }

    lib::MaybeStatic<Extent> static_size() const { return dims_; }
    size_t size() const { return static_size(); }

    template <typename T, size_t N>
        requires(std::is_same_v<std::remove_cv_t<T>, std::byte>)
    CompressedVectorBase<Unsigned, Bits, Extent, std::is_const_v<T>>
    vector(std::span<T, N> raw_data) const {
        return CompressedVectorBase<Unsigned, Bits, Extent, std::is_const_v<T>>(
            AllowShrinkingTag(), dims_, raw_data
        );
    }

    ///
    /// get.
    ///
    template <size_t N> const_value_type get(std::span<const std::byte, N> raw_data) const {
        assert(raw_data.size() >= total_bytes());
        auto cv = vector(raw_data);
        auto bundle = ScalarBundle{};
        const auto* start = raw_data.data() + cv.size_bytes();
        memcpy(&bundle, start, sizeof(ScalarBundle));
        return const_value_type{bundle.scale, bundle.bias, bundle.selector, cv};
    }

    ///
    /// set.
    ///
    template <size_t N, std::integral I>
    void
    set(std::span<std::byte, N> raw_data,
        float scale,
        float bias,
        selector_t selector,
        const std::vector<I>& src) const {
        assert(raw_data.size() >= total_bytes());
        auto cv = vector(raw_data);
        cv.copy_from(src);
        auto tof16 = [](float x) { return lib::narrow_cast<scalar_type>(x); };
        auto bundle = ScalarBundle{tof16(scale), tof16(bias), selector};
        auto* start = raw_data.data() + cv.size_bytes();
        memcpy(start, &bundle, sizeof(ScalarBundle));
    }

    template <size_t N>
    void set(std::span<std::byte, N> raw_data, const const_value_type& src) const {
        assert(raw_data.size() >= total_bytes());
        auto cv = vector(raw_data);
        cv.copy_from(src.data);
        auto* start = raw_data.data() + cv.size_bytes();
        auto bundle = ScalarBundle{src.scale, src.bias, src.get_selector()};
        memcpy(start, &bundle, sizeof(ScalarBundle));
    }

  private:
    [[no_unique_address]] lib::MaybeStatic<Extent> dims_;
};

/////
///// Compressed Datasets
/////

namespace detail {
template <typename T, typename U> void assert_equal(T&& x, U&& y) {
    if (x != y) {
        throw ANNEXCEPTION("Validation mismatch. Got ", x, ". Expected ", y, '!');
    }
}
} // namespace detail

///
/// Compressed Dataset
///
template <typename Sign, size_t Bits, size_t Extent, typename Storage = FlatStorage>
class CompressedDataset {
    static_assert(Storage::is_lvq_storage_tag);

  public:
    static constexpr bool is_resizeable = Storage::is_resizeable;

    ///
    /// The number of bits used for this encoding.
    ///
    static constexpr size_t encoding_bits = Bits;

    /// Dataset type aliases
    using value_type = MutableCompressedVector<Sign, Bits, Extent>;
    using const_value_type = CompressedVector<Sign, Bits, Extent>;

    ///
    /// The compile-time dimensionality of the raw byte storage backing the compressed
    /// data.
    ///
    static constexpr size_t data_extent = compute_storage_extent(Bits, Extent);
    using dataset_type = typename Storage::template type<data_extent>;
    using dataset_loader = typename Storage::template loader_type<data_extent>;
    using default_builder_type = typename Storage::builder;

    static size_t total_bytes(lib::MaybeStatic<Extent> dims) {
        return const_value_type::compute_bytes(dims);
    }

    ///
    /// Allocate an empty dataset.
    ///
    template <typename Builder = default_builder_type>
    CompressedDataset(
        size_t size, lib::MaybeStatic<Extent> dims = {}, const Builder& builder = Builder{}
    )
        : dims_{dims}
        , data_{data::build<std::byte, data_extent>(builder, size, total_bytes(dims))} {}

    CompressedDataset(dataset_type data, lib::MaybeStatic<Extent> dims = {})
        : dims_{dims}
        , data_{std::move(data)} {
        if (data_.dimensions() < total_bytes(dims)) {
            throw ANNEXCEPTION("Insert helpful error message here!");
        }
    }

    ///// Dataset Inteface

    size_t size() const { return data_.size(); }
    lib::MaybeStatic<Extent> static_dims() const { return dims_; }
    size_t dimensions() const { return static_dims(); }
    void prefetch(size_t i) const { data_.prefetch(i); }

    value_type get_datum(size_t i) {
        return value_type(AllowShrinkingTag(), static_dims(), data_.get_datum(i));
    }

    const_value_type get_datum(size_t i) const {
        return const_value_type(AllowShrinkingTag(), static_dims(), data_.get_datum(i));
    }

    template <std::integral I> void set_datum(size_t i, const std::vector<I>& data) {
        get_datum(i).copy_from(data);
    }

    void set_datum(size_t i, const const_value_type& data) { get_datum(i).copy_from(data); }

    ///// Resizing
    void resize(size_t new_size)
        requires is_resizeable
    {
        data_.resize(new_size);
    }

    ///// Compaction
    // Use perfect forwarding to the compacting algorithm of the backing buffer.
    template <typename... Args>
        requires is_resizeable
    void compact(Args&&... args) {
        data_.compact(std::forward<Args>(args)...);
    }

    /////
    ///// Saving and Loading.
    /////

    static constexpr std::string_view kind = "compressed dataset";
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);
    lib::SaveType save(const lib::SaveContext& ctx) const {
        return lib::SaveType(
            toml::table({
                {"inner", lib::recursive_save(data_, ctx)},
                {"kind", kind},
                {"sign", Sign::name},
                {"bits", prepare(Bits)},
                {"ndims", prepare(dimensions())},
                {"data_dims", prepare(data_.dimensions())},
                {"num_points", prepare(size())},
            }),
            save_version
        );
    }

    static CompressedDataset load(
        const toml::table& table,
        const lib::LoadContext& ctx,
        const lib::Version& version,
        const default_builder_type& builder = {}
    ) {
        if (version != save_version) {
            throw ANNEXCEPTION("Unhandled version!");
        }

        // Parse and validate.
        detail::assert_equal(get(table, "kind").value(), kind);
        detail::assert_equal(get(table, "sign").value(), Sign::name);
        detail::assert_equal(get<size_t>(table, "bits"), Bits);
        auto ndims = get<size_t>(table, "ndims");
        if constexpr (Extent != Dynamic) {
            detail::assert_equal(ndims, Extent);
        }
        // Load the sub-table.
        return CompressedDataset(
            lib::recursive_load(
                dataset_loader(lib::InferPath(), builder), subtable(table, "inner"), ctx
            ),
            lib::MaybeStatic<Extent>(ndims)
        );
    }

  private:
    [[no_unique_address]] lib::MaybeStatic<Extent> dims_;
    dataset_type data_;
};

///
/// ScaledBiasedDataset
///

template <size_t Bits, size_t Extent, typename Storage = FlatStorage>
class ScaledBiasedDataset {
    // Sanity check on the storage type parameter.
    static_assert(Storage::is_lvq_storage_tag);

  public:
    static constexpr bool is_resizeable = Storage::is_resizeable;
    using helper_type = ScaledBiasedVectorLayout<Bits, Extent>;

    using compressed_vector_type = CompressedVector<Unsigned, Bits, Extent>;
    using Encoded_value_type = typename compressed_vector_type::value_type;
    static constexpr bool supports_saving = false;
    // Pad data extent to be a multiple of half the underlying cache line size for better
    // bandwidth characteristics.
    static constexpr size_t encoding_bits = Bits;
    using dataset_type = typename Storage::template type<Dynamic>;
    using centroid_type = data::SimpleData<float, Dynamic>;
    using dataset_loader = typename Storage::template loader_type<Dynamic>;
    using default_builder_type = typename Storage::builder;

    static constexpr size_t compressed_vector_extent =
        compressed_vector_type::storage_extent;

    static constexpr size_t
    compute_data_dimensions(const helper_type& layout, size_t alignment = 0) {
        // Alignment = 0 implies no additional alignment.
        // Use the minimum possible space.
        size_t unaligned_size = layout.total_bytes();
        if (alignment == 0) {
            return unaligned_size;
        } else {
            return lib::round_up_to_multiple_of(unaligned_size, alignment);
        }
    }

    ///
    /// Allocate an empty dataset.
    ///
    /// Flat storage can accept an allocator.
    ///
    template <typename Builder = default_builder_type>
    ScaledBiasedDataset(
        size_t size,
        lib::MaybeStatic<Extent> dims = {},
        size_t alignment = 0,
        const Builder& builder = Builder{}
    )
        : layout_helper_{dims}
        , alignment_{alignment}
        , centroids_{std::make_shared<centroid_type>(1, dims)}
        , data_{data::build<std::byte, Dynamic>(
              builder, size, compute_data_dimensions(layout_helper_, alignment)
          )} {}

    ScaledBiasedDataset(
        dataset_type data,
        std::shared_ptr<centroid_type> centroids,
        size_t alignment,
        lib::MaybeStatic<Extent> dims = {}
    )
        : layout_helper_{dims}
        , alignment_{alignment}
        , centroids_{std::move(centroids)}
        , data_{std::move(data)} {
        auto data_dims = data_.dimensions();
        if (data_dims < compute_data_dimensions(layout_helper_)) {
            throw ANNEXCEPTION("Insert helpful error message here!");
        }
        // If the data dimensions doesn't match the alignment, then we were constructed
        // incorrectly.
        if (data_dims % alignment_) {
            throw ANNEXCEPTION("Misaligned data");
        }
    }

    size_t get_alignment() const { return alignment_; }

    ///// Set the centroids.
    template <size_t OtherExtent>
    void set_centroids(const data::SimpleData<float, OtherExtent>& centroids) {
        // Perform a size check on the other centroids.
        auto centroid_dims = centroids.dimensions();
        auto dims = dimensions();
        if (centroid_dims != dims) {
            auto message = fmt::format(
                "Trying to assign centroids with {} dimensions to a dataset with {}",
                centroid_dims,
                dims
            );
            throw ANNEXCEPTION(message);
        }

        // Create an appropriately sized destination and copy.
        centroids_ = std::make_shared<centroid_type>(centroids.size(), centroid_dims);
        data::copy(centroids, *centroids_);
    }

    ///// Dataset Inteface
    // N.B.: ScaledBiasedVector is immutable.
    using value_type = ScaledBiasedVector<Bits, Extent>;
    using const_value_type = ScaledBiasedVector<Bits, Extent>;
    using scalar_type = typename value_type::scalar_type;

    size_t size() const { return data_.size(); }
    lib::MaybeStatic<Extent> static_dims() const { return layout_helper_.static_size(); }
    size_t dimensions() const { return static_dims(); }
    void prefetch(size_t i) const { data_.prefetch(i); }

    const_value_type get_datum(size_t i) const {
        return layout_helper_.get(data_.get_datum(i));
    }

    template <std::integral I>
    void set_datum(
        size_t i, float scale, float bias, selector_t selector, const std::vector<I>& data
    ) {
        layout_helper_.set(data_.get_datum(i), scale, bias, selector, data);
    }

    void set_datum(size_t i, const value_type& data) {
        layout_helper_.set(data_.get_datum(i), data);
    }

    ///// Get the underlying centroids.
    std::shared_ptr<const centroid_type> view_centroids() const { return centroids_; }
    std::span<const float> get_centroid(size_t i) const { return centroids_->get_datum(i); }

    ///// Resizing
    void resize(size_t new_size)
        requires is_resizeable
    {
        data_.resize(new_size);
    }

    ///// Compaction
    // Use perfect forwarding to the compacting algorithm of the backing buffer.
    template <typename... Args>
        requires is_resizeable
    void compact(Args&&... args) {
        data_.compact(std::forward<Args>(args)...);
    }

    /////
    ///// Saving and Loading.
    /////

    static constexpr std::string_view kind = "scaled biased compressed dataset";
    static constexpr lib::Version save_version = lib::Version(0, 0, 1);

    lib::SaveType save(const lib::SaveContext& ctx) const {
        // TODO: Enable support for saving and reloading padded datasets.
        size_t padding = data_.dimensions() - layout_helper_.total_bytes();
        return lib::SaveType(
            toml::table(
                {{"inner", lib::recursive_save(data_, ctx)},
                 {"centroids", lib::recursive_save(*centroids_, ctx)},
                 {"num_centroids", prepare(centroids_->size())},
                 {"kind", kind},
                 {"bits", prepare(Bits)},
                 {"ndims", prepare(dimensions())},
                 {"data_dims", prepare(data_.dimensions())},
                 {"padding", prepare(padding)},
                 {"num_points", prepare(size())}}
            ),
            save_version
        );
    }

    static ScaledBiasedDataset load(
        const toml::table& table,
        const lib::LoadContext& ctx,
        const lib::Version& version,
        const default_builder_type& builder = default_builder_type()
    ) {
        if (version != save_version) {
            throw ANNEXCEPTION("Unhandled version!");
        }

        // Parse and validate.
        detail::assert_equal(get(table, "kind").value(), kind);
        detail::assert_equal(get<size_t>(table, "bits"), Bits);
        auto ndims = get<size_t>(table, "ndims");
        if constexpr (Extent != Dynamic) {
            detail::assert_equal(ndims, Extent);
        }

        // TODO: Enable direct loading to SimpleData
        auto centroids_poly = lib::recursive_load(
            VectorDataLoader<float>(), subtable(table, "centroids"), ctx
        );

        auto centroids = std::make_shared<centroid_type>(
            centroids_poly.size(), centroids_poly.dimensions()
        );
        data::copy(centroids_poly, *centroids);

        // Load the sub-table.
        return ScaledBiasedDataset(
            lib::recursive_load(
                dataset_loader(lib::InferPath(), builder), subtable(table, "inner"), ctx
            ),
            std::move(centroids),
            get<size_t>(table, "data_dims"),
            lib::MaybeStatic<Extent>(ndims)
        );
    }

  private:
    [[no_unique_address]] helper_type layout_helper_;
    size_t alignment_;
    std::shared_ptr<centroid_type> centroids_;
    dataset_type data_;
};
} // namespace svs::quantization::lvq
