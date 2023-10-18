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
// -- CompressedDataset<Sign, Bits, Extent, Allocator>
// A compressed dataset storing only quantized data with no scaling or bias.
//
// -- ScaledBiasedDataset<Bits, Extent, Allocator>
// Each vector is encoded using two constants, a bias (the smallest value any dimension
// can take) and a scaling parameter.
//

namespace detail {
template <typename A> inline constexpr bool is_blocked = false;
template <typename A> inline constexpr bool is_blocked<data::Blocked<A>> = true;
} // namespace detail

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
    CompressedVectorBase<Unsigned, Bits, Extent, std::is_const_v<T>> vector(
        std::span<T, N> raw_data
    ) const {
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
        throw ANNEXCEPTION("Validation mismatch. Got {}. Expected {}!", x, y);
    }
}
} // namespace detail

///
/// Compressed Dataset
///
template <
    typename Sign,
    size_t Bits,
    size_t Extent,
    typename Alloc = lib::Allocator<std::byte>>
class CompressedDataset {
  public:
    static constexpr bool is_resizeable = detail::is_blocked<Alloc>;
    using allocator_type = Alloc;

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
    using dataset_type = data::SimpleData<std::byte, Dynamic, allocator_type>;

    static size_t total_bytes(lib::MaybeStatic<Extent> dims) {
        return const_value_type::compute_bytes(dims);
    }

    ///
    /// Allocate an empty dataset.
    ///
    CompressedDataset(
        size_t size, lib::MaybeStatic<Extent> dims, const allocator_type& allocator
    )
        : dims_{dims}
        , data_{size, total_bytes(dims), allocator} {}

    CompressedDataset(size_t size, lib::MaybeStatic<Extent> dims = {})
        : CompressedDataset{size, dims, allocator_type{}} {}

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
    const allocator_type& get_allocator() const { return data_.get_allocator(); }

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
    lib::SaveTable save(const lib::SaveContext& ctx) const {
        return lib::SaveTable(
            save_version,
            {
                {"inner", lib::save(data_, ctx)},
                SVS_LIST_SAVE(kind),
                {"sign", lib::save(Sign::name)},
                {"bits", lib::save(Bits)},
                {"ndims", lib::save(dimensions())},
                {"data_dims", lib::save(data_.dimensions())},
                {"num_points", lib::save(size())},
            }
        );
    }

    static CompressedDataset load(
        const toml::table& table,
        const lib::LoadContext& ctx,
        const lib::Version& version,
        const allocator_type& allocator = {}
    ) {
        if (version != save_version) {
            throw ANNEXCEPTION("Unhandled version!");
        }

        // Parse and validate.
        detail::assert_equal(lib::load_at<std::string>(table, "kind"), kind);
        detail::assert_equal(lib::load_at<std::string>(table, "sign"), Sign::name);
        detail::assert_equal(lib::load_at<size_t>(table, "bits"), Bits);
        auto ndims = lib::load_at<size_t>(table, "ndims");
        if constexpr (Extent != Dynamic) {
            detail::assert_equal(ndims, Extent);
        }
        // Load the sub-table.
        return CompressedDataset(
            lib::load_at<dataset_type>(table, "inner", ctx, allocator),
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

template <size_t Bits, size_t Extent, typename Alloc = lib::Allocator<std::byte>>
class ScaledBiasedDataset {
  public:
    static constexpr bool is_resizeable = detail::is_blocked<Alloc>;
    using helper_type = ScaledBiasedVectorLayout<Bits, Extent>;
    using allocator_type = Alloc;

    using compressed_vector_type = CompressedVector<Unsigned, Bits, Extent>;
    using Encoded_value_type = typename compressed_vector_type::value_type;
    // Pad data extent to be a multiple of half the underlying cache line size for better
    // bandwidth characteristics.
    static constexpr size_t encoding_bits = Bits;
    using dataset_type = data::SimpleData<std::byte, Dynamic, allocator_type>;

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
    ScaledBiasedDataset(
        size_t size,
        lib::MaybeStatic<Extent> dims,
        size_t alignment,
        const allocator_type& allocator
    )
        : layout_helper_{dims}
        , alignment_{alignment}
        , data_{size, compute_data_dimensions(layout_helper_, alignment), allocator} {}

    ScaledBiasedDataset(
        size_t size, lib::MaybeStatic<Extent> dims = {}, size_t alignment = 0
    )
        : ScaledBiasedDataset(size, dims, alignment, allocator_type{}) {}

    ScaledBiasedDataset(
        dataset_type data, size_t alignment, lib::MaybeStatic<Extent> dims = {}
    )
        : layout_helper_{dims}
        , alignment_{alignment}
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
    const allocator_type& get_allocator() const { return data_.get_allocator(); }

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

    // Version History
    // v0.0.1 - Unknown Change.
    // v0.0.2 - BREAKING
    //   - Removed centroids from being stored with the ScaledBiasedCompressedDataset.
    //     Centroids are now stored in the higher level LVQ dataset.
    static constexpr lib::Version save_version = lib::Version(0, 0, 2);

    lib::SaveTable save(const lib::SaveContext& ctx) const {
        // TODO: Enable support for saving and reloading padded datasets.
        size_t padding = data_.dimensions() - layout_helper_.total_bytes();
        return lib::SaveTable(
            save_version,
            {{"inner", lib::save(data_, ctx)},
             {"kind", lib::save(kind)},
             {"bits", lib::save(Bits)},
             {"ndims", lib::save(dimensions())},
             {"data_dims", lib::save(data_.dimensions())},
             {"padding", lib::save(padding)},
             {"num_points", lib::save(size())}}
        );
    }

    static ScaledBiasedDataset load(
        const toml::table& table,
        const lib::LoadContext& ctx,
        const lib::Version& version,
        const allocator_type& allocator = {}
    ) {
        if (version != save_version) {
            throw ANNEXCEPTION("Unhandled version!");
        }

        // Parse and validate.
        detail::assert_equal(lib::load_at<std::string>(table, "kind"), kind);
        detail::assert_equal(lib::load_at<size_t>(table, "bits"), Bits);
        auto ndims = lib::load_at<size_t>(table, "ndims");
        if constexpr (Extent != Dynamic) {
            detail::assert_equal(ndims, Extent);
        }

        // Load the sub-table.
        return ScaledBiasedDataset(
            lib::load_at<dataset_type>(table, "inner", ctx, allocator),
            lib::load_at<size_t>(table, "data_dims"),
            lib::MaybeStatic<Extent>(ndims)
        );
    }

  private:
    [[no_unique_address]] helper_type layout_helper_;
    size_t alignment_;
    dataset_type data_;
};
} // namespace svs::quantization::lvq
