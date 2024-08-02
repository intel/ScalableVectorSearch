/**
 *    Copyright (C) 2023, Intel Corporation
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
// -- ScaledBiasedDataset<Bits, Extent, Strategy, Allocator>
// Each vector is encoded using two constants, a bias (the smallest value any dimension
// can take) and a scaling parameter.
//

// Forward Declaration
template <size_t Bits, size_t Extent, LVQPackingStrategy Strategy, typename Allocator>
class ScaledBiasedDataset;

namespace detail {
// Trait to determine if an allocator is blocked or not.
// Used to SFINAE away resizing methods if the allocator is not blocked.
template <typename A> inline constexpr bool is_blocked = false;
template <typename A> inline constexpr bool is_blocked<data::Blocked<A>> = true;
} // namespace detail

/////
///// Layout Helpers
/////

// LVQ constants.
// Define as `packed` because the start byte is not necessarily aligned to the 2-byte
// boundary usually required by float16.
struct __attribute__((packed)) ScalarBundle {
    scaling_t scale;
    scaling_t bias;
    selector_t selector;
};
// Ensure the compiler respects our request for packing.
static_assert(sizeof(ScalarBundle) == 2 * sizeof(scaling_t) + sizeof(selector_t));

///
/// Layout for `ScaledBiasedVector` where the scaling constants are stored inline after the
/// `CompressedVector` storing the data.
///
template <size_t Bits, size_t Extent, typename Strategy = Sequential>
class ScaledBiasedVectorLayout {
  public:
    using cv_type = CompressedVector<Unsigned, Bits, Extent, Strategy>;

    using const_value_type = ScaledBiasedVector<Bits, Extent, Strategy>;
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
    CompressedVectorBase<Unsigned, Bits, Extent, std::is_const_v<T>, Strategy> vector(
        std::span<T, N> raw_data
    ) const {
        return CompressedVectorBase<Unsigned, Bits, Extent, std::is_const_v<T>, Strategy>(
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
        auto bundle = ScalarBundle{
            lvq::through_scaling_type(scale), lvq::through_scaling_type(bias), selector};
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

namespace detail {
// Accessor for putting LVQ compressed data into a canonical ordering.
// The canonical ordering consists of bit-packed, sequentially ordered data followed by
// the constants in Float16 form.
class Canonicalizer {
  public:
    Canonicalizer() = default;

    template <size_t Bits, size_t Extent>
    using canonical_layout_type = ScaledBiasedVectorLayout<Bits, Extent, Sequential>;

    /// @brief Convert the given LVQ compressed vector to a canonical dense representation.
    template <size_t Bits, size_t Extent, typename Strategy>
    std::span<const std::byte>
    to_canonical(const ScaledBiasedVector<Bits, Extent, Strategy>& v) {
        auto dims = lib::MaybeStatic<Extent>(v.size());

        // Determine the memory needed for canonical storage.
        // Make sure the underlying buffer is the correct size.
        auto canonical_layout = canonical_layout_type<Bits, Extent>(dims);
        canonical_form_.resize(canonical_layout.total_bytes());

        // Fast-path: If sequential is already used, we can use memcopy rather than
        // element-wise assignment
        if constexpr (std::is_same_v<Strategy, Sequential>) {
            canonical_layout.set(lib::as_span(canonical_form_), v);
        } else {
            // Fall-back: Use element-wise assignment into the canonical form.
            auto mut = compressed_buffer_
                           .template view<lvq::Unsigned, Bits, Extent, lvq::Sequential>(dims

                           );

            // Delegate the job of bit unpacking to the strategy permutation.
            for (size_t i = 0; i < dims; ++i) {
                mut.set(v.data.get(i), i);
            }

            // Construct an intermediate ScaledBiasedVector to and use the layout to store
            // the final result.
            auto canonical = ScaledBiasedVector<Bits, Extent, lvq::Sequential>(
                v.scale, v.bias, v.selector, mut
            );

            canonical_layout.set(lib::as_span(canonical_form_), canonical);
        }
        return lib::as_const_span(canonical_form_);
    }

    ///
    /// @brief Convert the canonical representation into an intermediate representation.
    ///
    /// @param dispatch_tag Dispatch type containing the full type of the scaled vector
    ///     being created.
    /// @param raw_data The raw bytes containing the canonical layout LVQ data.
    /// @param logical_dimensions The logical number of dimensions of the serialized LVQ
    ///     vector.
    ///
    /// Notes: The returned `ScaledBiasedVector` may contain pointers into either the
    /// `raw_data` argument or local data structures owned by the `Canonicalizer`,
    /// depending on which method is most efficient.
    ///
    /// The resulting `ScaledBiasedVector` must be completely used before the storage
    /// behind `raw_data` is changed and before further decoding methods are called on
    /// the Canonicalizer.
    ///
    template <size_t Bits, size_t Extent, typename Strategy>
    ScaledBiasedVector<Bits, Extent, Strategy> from_canonical(
        lib::Type<ScaledBiasedVector<Bits, Extent, Strategy>> SVS_UNUSED(dispatch_tag),
        std::span<const std::byte> raw_data,
        lib::MaybeStatic<Extent> logical_dimensions
    ) {
        auto canonical_layout = canonical_layout_type<Bits, Extent>(logical_dimensions);
        assert(raw_data.size() == canonical_layout.total_bytes());
        auto canonical = canonical_layout.get(raw_data);

        // If we are loading into a sequential layout, then we can return the interpreted
        // raw data directly.
        //
        // Otherwise, we need to construct an intermediate layout with the same strategy
        // as requested.
        if constexpr (std::is_same_v<Strategy, Sequential>) {
            return canonical;
        } else {
            // Resize storage for the permuted compressed-vector.
            auto mut =
                compressed_buffer_.template view<lvq::Unsigned, Bits, Extent, Strategy>(
                    logical_dimensions
                );

            for (size_t i = 0; i < logical_dimensions; ++i) {
                mut.set(canonical.data.get(i), i);
            }

            return ScaledBiasedVector<Bits, Extent, Strategy>{
                canonical.scale, canonical.bias, canonical.selector, mut};
        }
    }

  private:
    // If the original dataset uses a Turbo-layout, we need an intermediate vector to
    // store the sequential encodings.
    CVStorage compressed_buffer_{};
    // Allocated storage for the canonical layout.
    std::vector<std::byte> canonical_form_{};
};

class CanonicalAccessor {
  public:
    CanonicalAccessor() = default;

    // Read access
    template <size_t Bits, size_t Extent, typename Strategy, typename Allocator>
    std::span<const std::byte>
    get(const ScaledBiasedDataset<Bits, Extent, Strategy, Allocator>& dataset, size_t i) {
        return canonicalizer_.to_canonical(dataset.get_datum(i));
    }

    // IO Compatibility
    // The raw number of bytes that will be written for this representation.
    template <size_t Bits, size_t Extent, typename Strategy, typename Allocator>
    size_t serialized_dimensions(
        const ScaledBiasedDataset<Bits, Extent, Strategy, Allocator>& dataset
    ) const {
        return Canonicalizer::canonical_layout_type<Bits, Extent>(dataset.static_dims())
            .total_bytes();
    }

    // Write access
    template <size_t Bits, size_t Extent, typename Strategy, typename Allocator>
    void
    set(ScaledBiasedDataset<Bits, Extent, Strategy, Allocator>& dataset,
        size_t i,
        std::span<const std::byte> raw_data) {
        using T = typename ScaledBiasedDataset<Bits, Extent, Strategy, Allocator>::
            const_value_type;

        dataset.set_datum(
            i,
            canonicalizer_.from_canonical(lib::Type<T>(), raw_data, dataset.static_dims())
        );
    }

    // IO compatibility
    // The raw serialization format is using bytes.
    template <
        size_t Bits,
        size_t Extent,
        LVQPackingStrategy Strategy,
        typename Allocator,
        typename File>
    typename File::template reader_type<std::byte> reader(
        const ScaledBiasedDataset<Bits, Extent, Strategy, Allocator>& SVS_UNUSED(dataset),
        const File& file
    ) const {
        return file.reader(lib::Type<std::byte>());
    }

  private:
    Canonicalizer canonicalizer_{};
};

} // namespace detail

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
/// Support for deduction.
///
enum class DatasetSchema { Compressed, ScaledBiased };
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
    using value_type = MutableCompressedVector<Sign, Bits, Extent, Sequential>;
    using const_value_type = CompressedVector<Sign, Bits, Extent, Sequential>;

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
    static constexpr std::string_view serialization_schema =
        get_schema(DatasetSchema::Compressed);
    static constexpr lib::Version save_version =
        get_current_version(DatasetSchema::Compressed);

    lib::SaveTable save(const lib::SaveContext& ctx) const {
        return lib::SaveTable(
            serialization_schema,
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

    static CompressedDataset
    load(const lib::LoadTable& table, const allocator_type& allocator = {}) {
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
            lib::load_at<dataset_type>(table, "inner", allocator),
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

template <
    size_t Bits,
    size_t Extent,
    LVQPackingStrategy Strategy,
    typename Alloc = lib::Allocator<std::byte>>
class ScaledBiasedDataset {
  public:
    static constexpr bool is_resizeable = detail::is_blocked<Alloc>;
    using strategy = Strategy;
    using helper_type = ScaledBiasedVectorLayout<Bits, Extent, Strategy>;
    using allocator_type = Alloc;

    using compressed_vector_type = CompressedVector<Unsigned, Bits, Extent, Strategy>;
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
    using value_type = ScaledBiasedVector<Bits, Extent, Strategy>;
    using const_value_type = ScaledBiasedVector<Bits, Extent, Strategy>;
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
    // v0.0.3 - BREAKING
    //   - Canonicalize the layout of serialized LVQ to be sequential with no padding.
    //     This allows different packing strategies and paddings to be used upon reload.
    static constexpr lib::Version save_version =
        get_current_version(DatasetSchema::ScaledBiased);
    static constexpr std::string_view serialization_schema =
        get_schema(DatasetSchema::ScaledBiased);

    lib::SaveTable save(const lib::SaveContext& ctx) const {
        // Prepare for binary-file serialization.
        // Use the serialization utilities in core but provide an accessor that lazily
        // reorganizes the LVQ packing to the canonical form (if needed).
        auto uuid = lib::UUID();
        auto filename = ctx.generate_name("lvq_data");
        {
            auto canonical_accessor = detail::CanonicalAccessor{};
            io::save(*this, canonical_accessor, io::NativeFile(filename), uuid);
        }

        return lib::SaveTable(
            serialization_schema,
            save_version,
            {{"kind", lib::save(serialization_schema)},
             {"binary_file", lib::save(filename.filename())},
             {"file_uuid", uuid.str()},
             {"num_vectors", lib::save(size())},
             {"logical_dimensions", lib::save(dimensions())},
             {"bits", lib::save(Bits)}}
        );
    }

    static ScaledBiasedDataset load(
        const lib::LoadTable& table,
        size_t alignment = 0,
        const allocator_type& allocator = {}
    ) {
        // Parse and validate.
        detail::assert_equal(
            lib::load_at<std::string>(table, "kind"), serialization_schema
        );
        detail::assert_equal(lib::load_at<size_t>(table, "bits"), Bits);
        auto ndims = lib::load_at<size_t>(table, "logical_dimensions");
        if constexpr (Extent != Dynamic) {
            detail::assert_equal(ndims, Extent);
        }

        // Load the binary data.
        auto uuid = lib::load_at<lib::UUID>(table, "file_uuid");
        auto binary_file = io::find_uuid(table.context().get_directory(), uuid);
        if (!binary_file.has_value()) {
            throw ANNEXCEPTION("Could not open file with uuid {}!", uuid.str());
        }

        // Setup and execute the binary loading.
        auto expected_size = lib::load_at<size_t>(table, "num_vectors");
        auto lazy_constructor =
            lib::Lazy([&](size_t size, size_t SVS_UNUSED(raw_dimensions)) {
                if (size != expected_size) {
                    throw ANNEXCEPTION(
                        "Expected {} vectors in loaded file. Instead, got {}!",
                        expected_size,
                        size
                    );
                }

                // Ignore the number of dimensions returned from the file deduction.
                // The number of provided dimensions corresponds to the number of bytes in
                // the canonical layout.
                return ScaledBiasedDataset(
                    size, lib::MaybeStatic<Extent>(ndims), alignment, allocator
                );
            });
        auto write_accessor = detail::CanonicalAccessor();
        return io::load_dataset(binary_file.value(), write_accessor, lazy_constructor);
    }

  private:
    [[no_unique_address]] helper_type layout_helper_;
    size_t alignment_;
    dataset_type data_;
};
} // namespace svs::quantization::lvq
