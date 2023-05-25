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
// -- GlobalScaledBiasedDataset<Bits, Extent>
// A dataset similar to `ScaledBiasedDataset<Bits, Extent>` except that the constants
// for each vector are global for the entire dataset rather than per-vector.
//

/////
///// Layout Helpers
/////

// To avoid repeating a massive amount of work for the various dataset types we need to
// support, the methods of interacting with compressed data in the form of a span of
// `std::byte` are factored out for better reuse.

///
/// Layout helper for inline `CompressedVector`s.
///
template <typename Sign, size_t Bits, size_t Extent> class CompressedVectorLayout {
  public:
    using value_type = MutableCompressedVector<Sign, Bits, Extent>;
    using const_value_type = CompressedVector<Sign, Bits, Extent>;
    using encoding_type = typename value_type::value_type;

    constexpr explicit CompressedVectorLayout(lib::MaybeStatic<Extent> size)
        : size_{size} {}

    ///
    /// The expected number of bytes occupied by the compressed vector.
    ///
    static constexpr size_t compressed_extent = compute_storage_extent(Bits, Extent);
    constexpr size_t total_bytes() const { return compute_storage(Bits, size()); }

    /// Return the potentially static dimensionality of the underlying data.
    constexpr lib::MaybeStatic<Extent> static_size() const { return size_; }
    /// Return the dimensionality of the underlying data.
    constexpr size_t size() const { return static_size(); }

    // Get Datum - mutable
    template <size_t N> value_type get(std::span<std::byte, N> raw_data) const {
        assert(raw_data.size() >= total_bytes());
        return value_type{size_, raw_data.template subspan<0, compressed_extent>()};
    }

    // Get Datum - const
    template <size_t N> const_value_type get(std::span<const std::byte, N> raw_data) const {
        assert(raw_data.size() >= total_bytes());
        return const_value_type{size_, raw_data.template subspan<0, compressed_extent>()};
    }

    // Set Datum
    template <size_t N, std::integral I>
    void set(std::span<std::byte, N> raw_dst, const std::vector<I>& src) const {
        value_type dst = get(raw_dst);
        for (size_t i = 0, imax = dst.size(); i < imax; ++i) {
            dst.set(src.at(i), i);
        }
    }

    template <size_t N>
    void set(std::span<std::byte, N> raw_dst, const const_value_type& src) const {
        assert(raw_dst.size_bytes() >= src.size_bytes());
        memcpy(raw_dst.data(), src.data(), src.size_bytes());
    }

  private:
    [[no_unique_address]] lib::MaybeStatic<Extent> size_;
};

///
/// Layout for `ScaledBiasedVector` where the scaling constants are stored inline after the
/// `CompressedVector` storing the data.
///
template <size_t Bits, size_t Extent> class ScaledBiasedVectorLayout {
  public:
    using compressed_helper = CompressedVectorLayout<Unsigned, Bits, Extent>;
    using const_value_type = ScaledBiasedVector<Bits, Extent>;
    using scalar_type = typename const_value_type::scalar_type;

    explicit ScaledBiasedVectorLayout(lib::MaybeStatic<Extent> size)
        : helper_{size} {}

    static constexpr size_t compressed_extent = compute_storage_extent(Bits, Extent);
    constexpr size_t total_bytes() const {
        return helper_.total_bytes() + 2 * sizeof(scalar_type);
    }

    lib::MaybeStatic<Extent> static_size() const { return helper_.static_size(); }
    size_t size() const { return static_size(); }

    // const
    template <size_t N> const_value_type get(std::span<const std::byte, N> raw_data) const {
        assert(raw_data.size() >= total_bytes());
        auto compressed_data = helper_.get(raw_data);
        // Copy out the scalar value.
        scalar_type scale;
        scalar_type bias;
        const auto* start = raw_data.data() + helper_.total_bytes();
        memcpy(&scale, start, sizeof(scalar_type));
        memcpy(&bias, start + sizeof(scalar_type), sizeof(scalar_type));
        return const_value_type{scale, bias, compressed_data};
    }

    ///
    /// set.
    ///
    template <size_t N, std::integral I>
    void
    set(std::span<std::byte, N> raw_data, float scale, float bias, const std::vector<I>& src
    ) const {
        // Store the compressed vector at the beginning of the raw data.
        helper_.set(raw_data, src);
        // Append the scalar to the end of the compressed data.
        auto scale_converted = lib::narrow_cast<scalar_type>(scale);
        auto bias_converted = lib::narrow_cast<scalar_type>(bias);
        auto* start = raw_data.data() + helper_.total_bytes();
        memcpy(start, &scale_converted, sizeof(scalar_type));
        memcpy(start + sizeof(scalar_type), &bias_converted, sizeof(scalar_type));
    }

    template <size_t N>
    void set(std::span<std::byte, N> raw_data, const const_value_type& src) const {
        helper_.set(raw_data, src.data);
        auto* start = raw_data.data() + helper_.total_bytes();
        memcpy(start, &(src.scale), sizeof(scalar_type));
        memcpy(start + sizeof(scalar_type), &(src.bias), sizeof(scalar_type));
    }

  private:
    [[no_unique_address]] compressed_helper helper_;
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
template <typename Sign, size_t Bits, size_t Extent> class CompressedDataset {
  public:
    using helper_type = CompressedVectorLayout<Sign, Bits, Extent>;
    using encoded_value_type = typename helper_type::value_type;

    ///
    /// The number of bits used for this encoding.
    ///
    static constexpr size_t encoding_bits = Bits;

    ///
    /// The compile-time dimensionality of the raw byte storage backing the compressed
    /// data.
    ///
    static constexpr size_t data_extent = helper_type::compressed_extent;
    using dataset_type = data::SimplePolymorphicData<std::byte, data_extent>;
    using dataset_loader = VectorDataLoader<std::byte, data_extent>;

    ///
    /// Allocate an empty dataset.
    ///
    template <typename Allocator = HugepageAllocator>
    CompressedDataset(
        size_t size,
        lib::MaybeStatic<Extent> dims = {},
        Allocator allocator = HugepageAllocator{}
    )
        : layout_helper_{dims}
        , data_{allocator, size, layout_helper_.total_bytes()} {}

    CompressedDataset(dataset_type data, lib::MaybeStatic<Extent> dims = {})
        : layout_helper_{dims}
        , data_{std::move(data)} {
        if (data_.dimensions() < layout_helper_.total_bytes()) {
            throw ANNEXCEPTION("Insert helpful error message here!");
        }
    }

    ///// Dataset Inteface
    using value_type = CompressedVector<Sign, Bits, Extent>;
    using const_value_type = CompressedVector<Sign, Bits, Extent>;

    size_t size() const { return data_.size(); }
    lib::MaybeStatic<Extent> static_dims() const { return layout_helper_.static_size(); }
    size_t dimensions() const { return static_dims(); }
    void prefetch(size_t i) const { data_.prefetch(i); }

    const_value_type get_datum(size_t i) const {
        return layout_helper_.get(data_.get_datum(i));
    }

    template <std::integral I> void set_datum(size_t i, const std::vector<I>& data) {
        layout_helper_.set(data_.get_datum(i), data);
    }

    void set_datum(size_t i, const const_value_type& data) {
        layout_helper_.set(data_.get_datum(i), data);
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
        const toml::table& table, const lib::LoadContext& ctx, const lib::Version& version
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
            lib::recursive_load(dataset_loader(), subtable(table, "inner"), ctx),
            lib::MaybeStatic<Extent>(ndims)
        );
    }

  private:
    [[no_unique_address]] helper_type layout_helper_;
    dataset_type data_;
};

///
/// ScaledBiasedDataset
///

template <size_t Bits, size_t Extent> class ScaledBiasedDataset {
  public:
    using helper_type = ScaledBiasedVectorLayout<Bits, Extent>;

    using compressed_vector_type = CompressedVector<Unsigned, Bits, Extent>;
    using encoded_value_type = typename compressed_vector_type::value_type;
    static constexpr bool supports_saving = false;
    // Pad data extent to be a multiple of half the underlying cache line size for better
    // bandwidth characteristics.
    static constexpr size_t encoding_bits = Bits;
    using dataset_type = data::SimplePolymorphicData<std::byte, Dynamic>;
    using dataset_loader = VectorDataLoader<std::byte, Dynamic>;
    static constexpr size_t compressed_vector_extent =
        compressed_vector_type::storage_extent;

    static constexpr size_t
    compute_data_dimensions(const helper_type& layout, size_t padding = 0) {
        // Padding = 0 implies no additional padding.
        // Use the minimum possible space.
        size_t unpadded_size = layout.total_bytes();
        if (padding == 0) {
            return unpadded_size;
        } else {
            return lib::round_up_to_multiple_of(unpadded_size, padding);
        }
    }

    ///
    /// Allocate an empty dataset.
    ///
    template <typename Allocator = HugepageAllocator>
    ScaledBiasedDataset(
        size_t size,
        lib::MaybeStatic<Extent> dims = {},
        size_t padding = 0,
        Allocator allocator = HugepageAllocator{}
    )
        : layout_helper_{dims}
        , data_{allocator, size, compute_data_dimensions(layout_helper_, padding)} {}

    ScaledBiasedDataset(dataset_type data, lib::MaybeStatic<Extent> dims = {})
        : layout_helper_{dims}
        , data_{std::move(data)} {
        if (data_.dimensions() < compute_data_dimensions(layout_helper_)) {
            throw ANNEXCEPTION("Insert helpful error message here!");
        }
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
    void set_datum(size_t i, float scale, float bias, const std::vector<I>& data) {
        layout_helper_.set(data_.get_datum(i), scale, bias, data);
    }

    void set_datum(size_t i, const value_type& data) {
        layout_helper_.set(data_.get_datum(i), data);
    }

    /////
    ///// Saving and Loading.
    /////

    static constexpr std::string_view kind = "scaled biased compressed dataset";
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);

    lib::SaveType save(const lib::SaveContext& ctx) const {
        // TODO: Enable support for saving and reloading padded datasets.
        size_t padding = data_.dimensions() - layout_helper_.total_bytes();
        if (padding != 0) {
            throw ANNEXCEPTION("Cannot yet save padded datasets!");
        }
        return lib::SaveType(
            toml::table(
                {{"inner", lib::recursive_save(data_, ctx)},
                 {"kind", kind},
                 {"bits", prepare(Bits)},
                 {"ndims", prepare(dimensions())},
                 {"data_dims", prepare(data_.dimensions())},
                 {"num_points", prepare(size())}}
            ),
            save_version
        );
    }

    static ScaledBiasedDataset load(
        const toml::table& table, const lib::LoadContext& ctx, const lib::Version& version
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
        // Load the sub-table.
        return ScaledBiasedDataset(
            lib::recursive_load(dataset_loader(), subtable(table, "inner"), ctx),
            lib::MaybeStatic<Extent>(ndims)
        );
    }

  private:
    [[no_unique_address]] helper_type layout_helper_;
    dataset_type data_;
};

/////
///// Global Quantization
/////

template <size_t Bits, size_t Extent> class GlobalScaledBiasedDataset {
  public:
    using helper_type = CompressedVectorLayout<Unsigned, Bits, Extent>;
    static constexpr bool supports_saving = false;
    // Pad data extent to be a multiple of half the underlying cache line size for better
    // bandwidth characteristics.
    static constexpr size_t encoding_bits = Bits;
    using dataset_type = data::SimplePolymorphicData<std::byte>;
    using dataset_loader = VectorDataLoader<std::byte>;

    static constexpr size_t
    compute_data_dimensions(const helper_type& helper, size_t padding = 0) {
        // Padding = 0 implies no additional padding.
        // Use the minimum possible space.
        size_t unpadded_size = helper.total_bytes();
        if (padding == 0) {
            return unpadded_size;
        } else {
            return lib::round_up_to_multiple_of(unpadded_size, padding);
        }
    }

    ///
    /// Allocate an empty dataset.
    ///
    template <typename Allocator = HugepageAllocator>
    GlobalScaledBiasedDataset(
        size_t size,
        lib::MaybeStatic<Extent> dims,
        float scale,
        float bias,
        size_t padding = 0,
        Allocator allocator = HugepageAllocator{}
    )
        : layout_helper_{dims}
        , scale_{scale}
        , bias_{bias}
        , data_{allocator, size, compute_data_dimensions(layout_helper_, padding)} {}

    GlobalScaledBiasedDataset(
        lib::MaybeStatic<Extent> dims, float scale, float bias, dataset_type data
    )
        : layout_helper_{dims}
        , scale_{scale}
        , bias_{bias}
        , data_{std::move(data)} {
        if (data_.dimensions() < compute_data_dimensions(layout_helper_)) {
            throw ANNEXCEPTION("Unhelpful error message!");
        }
    }

    float get_scale() const { return scale_; }
    float get_bias() const { return bias_; }

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
        return const_value_type{scale_, bias_, layout_helper_.get(data_.get_datum(i))};
    }

    template <std::integral I> void set_datum(size_t i, const std::vector<I>& data) {
        layout_helper_.set(data_.get_datum(i), data);
    }

    void set_datum(size_t i, const value_type& data) {
        // Ensure that the scale and bias of the incoming data match with the pre-computed
        // value in this dataset.
        //
        // Perform the comparison in the `Float16` domain because the conversion a to
        // `Float16` scalar is lossy so `data.scale` may not exactly convert back to the
        // original `float` value stored in `scale_`.
        auto check_constants = [](std::string_view kind, float got, float expected) {
            if (got != expected) {
                auto message = fmt::format(
                    "{} mismatch. Data has {} but pre-configured value is {}!",
                    kind,
                    got,
                    expected
                );
                throw ANNEXCEPTION(message);
            }
        };

        check_constants("Scale", data.scale, static_cast<Float16>(scale_));
        check_constants("Bias", data.bias, static_cast<Float16>(bias_));
        layout_helper_.set(data_.get_datum(i), data.data);
    }

    ///// Saving and Loading.
    static constexpr std::string_view kind = "global scaled biased dataset";
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);

    lib::SaveType save(const lib::SaveContext& ctx) const {
        size_t padding = data_.dimensions() - layout_helper_.total_bytes();
        if (padding != 0) {
            throw ANNEXCEPTION("Cannot yet save padded datasets!");
        }

        return lib::SaveType(
            toml::table(
                {{"inner", lib::recursive_save(data_, ctx)},
                 {"kind", kind},
                 {"bits", prepare(Bits)},
                 {"ndims", prepare(dimensions())},
                 {"data_dims", prepare(data_.dimensions())},
                 {"num_points", prepare(size())},
                 {"scale", prepare(scale_)},
                 {"bias", prepare(bias_)}}
            ),
            save_version
        );
    }

    static GlobalScaledBiasedDataset load(
        const toml::table& table, const lib::LoadContext& ctx, const lib::Version& version
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

        float scale = get<float>(table, "scale");
        float bias = get<float>(table, "bias");

        // Load the sub-table.
        return GlobalScaledBiasedDataset(
            lib::MaybeStatic<Extent>(ndims),
            scale,
            bias,
            lib::recursive_load(dataset_loader(), subtable(table, "inner"), ctx)
        );
    }

  private:
    [[no_unique_address]] helper_type layout_helper_;
    float scale_;
    float bias_;
    dataset_type data_;
};

} // namespace svs::quantization::lvq
