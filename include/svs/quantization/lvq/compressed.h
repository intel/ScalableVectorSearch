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

// svs
#include "svs/concepts/distance.h"
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/lib/meta.h"
#include "svs/lib/misc.h"
#include "svs/lib/preprocessor.h"
#include "svs/lib/type_traits.h"
#include "svs/third-party/eve.h"

// stl
#include <bit>
#include <concepts>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace svs::quantization::lvq {

///
/// The encoding to use for centroid selection.
///
using selector_t = uint8_t;

///
/// @brief Compute the number of bytes to store a compressed vector.
///
/// @param nbits The number of bits used to encode each vector component.
/// @param length The number of vector components.
///
/// Given a compressed vector using `nbits` per element and a length `length` compute
/// the number of bytes required to store the vector.
///
constexpr size_t compute_storage(size_t nbits, size_t length) {
    return lib::div_round_up(nbits * length, 8);
}

///
/// @brief Compute the static extent of spans for compressed vectors.
///
/// @param nbits The number of bits used to encode each vector component.
/// @param extent The compile time number of dimensions, of ``Dynamic`` if unknown.
///
/// Compile-time computation of storage extent.
/// If ``extent == Dynamic``, then propagate ``Dynamic``. Otherwise, compute the static
/// storage requirement using ``compute_storage``.
///
/// @sa ``compute_storage``
///
constexpr size_t compute_storage_extent(size_t nbits, size_t extent) {
    return extent == Dynamic ? Dynamic : compute_storage(nbits, extent);
}

namespace detail {
///
/// Return a bit-mask of type `T` with ones from `lo` to `hi` (inclusive).
///
template <std::integral T> constexpr T bitmask(T lo, T hi) {
    T one{1};
    return static_cast<T>(one << (hi + one)) - static_cast<T>(one << lo);
}

///
/// Helper struct giving upper and lower byte and bit bounds for accessing a densely packed
/// N-bit packing where `N < 8`.
///
/// The following assumptions are made:
/// * The `N`-bit packing is dense and begins at a byte boundary. For example, if `N = 7`,
///   then the packing looks like this.
///
///        Byte 0             Byte 1             Byte 2             Byte 3
///   +-------------+    +-------------+    +-------------+    +-------------+   ...
///   V             V    V             V    V             V    V             V
///   0 1 2 3 4 5 6 7 -- 0 1 2 3 4 5 6 7 -- 0 1 2 3 4 5 6 7 -- 0 1 2 3 4 5 6 7   ...
///   |           | |              | |              | |              |
///   +-----------+ +--------------+ +--------------+ +--------------+           ...
///      Value 0         Value 1          Value 2          Value 3
///
///
/// This struct struct contains contains 4 values: `byte_start`, `byte_stop`, `bit_start`,
/// and `bit_stop`.
///
/// The intervals defined by these values are inclusive.
/// If the packed value lives entirely within a byte (i.e., `byte_start == byte_stop`), then
/// the mask defined by `bitmask(bit_start, bit_stop)` is appropriate for either an 8-bit
/// or a 16-bit load starting at `byte_start`. Of these, an 8-bit load should be preferred
/// because we don't ensure padding for the very last value (i.e., a 16-bit load would read
/// out of bounds).
///
/// If the packed value straddles two consecutive bytes, than `byte_start + 1 == byte_stop`.
/// In this case, the mask `bitmask(bit_start, bit_stop)` is suitable to mask a 16-bit
/// load beginning at `byte_start`.
///
/// For example, the struct returned for `value 1` in the above example above would be
/// ```
/// IndexRange(byte_start = 1, byte_stop = 2, bit_start = 6, bit_stop = 12)
/// ```
///
struct IndexRange {
    // Constructors
    IndexRange(size_t byte_start, size_t byte_stop, uint8_t bit_start, uint8_t bit_stop)
        : byte_start{byte_start}
        , byte_stop{byte_stop}
        , bit_start{bit_start}
        , bit_stop{bit_stop} {}

    template <size_t Bits>
    IndexRange(meta::Val<Bits> /*unused*/, size_t i)
        : byte_start((Bits * i) / 8)
        , byte_stop((Bits * (i + 1) - 1) / 8)
        , bit_start(Bits * i - 8 * byte_start)
        , bit_stop(Bits * (i + 1) - 8 * byte_start - 1) {}

    size_t byte_start;
    size_t byte_stop;
    uint8_t bit_start;
    uint8_t bit_stop;
};

///
/// Compare the contents of two `IndexRange` instances.
/// Useful for testing.
///
inline bool operator==(const IndexRange& x, const IndexRange& y) {
    return (x.byte_start == y.byte_start) && (x.byte_stop == y.byte_stop) &&
           (x.bit_start == y.bit_start) && (x.bit_stop == y.bit_stop);
}

///
/// Print operator for easier debugging.
///
inline std::ostream& operator<<(std::ostream& stream, const IndexRange& r) {
    return stream << "IndexRange(" << r.byte_start << ", " << r.byte_stop << ", "
                  << uint16_t(r.bit_start) << ", " << uint16_t(r.bit_stop) << ")";
}
} // namespace detail

///
/// Place-holder to indicate that a given direct compression stores its values as
/// signed integers (taking positive and negative values in accordance with a two-s
/// complement encoding).
///
struct Signed {
    static constexpr std::string_view name = "signed";
};

///
/// Place-holder to indicate that a given direct compression stores its values as
/// unsigned integers.
///
struct Unsigned {
    static constexpr std::string_view name = "unsigned";
};

/// Place holder for specializations.
template <typename T, size_t Bits> struct Encoding;

///
/// A signed encoding using `Bits` bits per component.
///
template <size_t Bits> struct Encoding<Signed, Bits> {
    Encoding() = default;

    ///
    /// Return the number of bytes required to store `length` densely packed `Bits`-sized
    /// elements.
    ///
    static constexpr size_t bytes(size_t length) { return compute_storage(Bits, length); }

    // Type Aliases
    using value_type = int8_t;
    static constexpr size_t bits = Bits;

    static constexpr value_type max() { return (1 << (Bits - 1)) - 1; }
    static constexpr value_type min() { return -(1 << (Bits - 1)); }
    static constexpr size_t absmax() { return -(static_cast<int64_t>(min())); }

    // Internally, we convert signed values to unsigned values by adding in a bias to
    // turn values of type `min()` to zero.
    //
    // This avoids complications related to restoring the sign bit when unpacking values.
    static value_type decode(uint8_t raw) {
        static_assert(Bits <= 8);
        if constexpr (Bits == 8) {
            return std::bit_cast<int8_t>(raw);
        } else {
            // Since we're using less than 8 bits to encode the value, the maximum of the
            // encoded small integer can fit inside a signed 8-bit number.
            //
            // Therefore, this bitcast is lossless.
            //
            // After converting to a signed 8-bit integer, we need to apply the 2's
            // complement shift to restore signedness.
            return std::bit_cast<int8_t>(raw) + min();
        }
    }

    static uint8_t encode(value_type value) {
        static_assert(Bits <= 8);
        if constexpr (Bits == 8) {
            return std::bit_cast<uint8_t>(value);
        } else {
            return lib::narrow<uint8_t>(value - min());
        }
    }

    template <std::signed_integral I> static bool check_bounds(I value) {
        return (min() <= value) && (value <= max());
    }
};

template <size_t Bits> struct Encoding<Unsigned, Bits> {
    Encoding() = default;

    ///
    /// Return the number of bytes required to store `length` densly packed `Bits`-sized
    /// elements.
    ///
    static constexpr size_t bytes(size_t length) { return compute_storage(Bits, length); }

    // Type Aliases
    using value_type = uint8_t;
    static constexpr size_t bits = Bits;

    // Helper functions.
    static constexpr value_type max() { return (1 << Bits) - 1; }
    static constexpr value_type min() { return 0; }
    static constexpr size_t absmax() { return max(); }

    // No adjustment required for unsigned types since we mask out the upper order
    // bits anyways.
    static value_type decode(uint8_t raw) { return raw; }
    static uint8_t encode(uint8_t raw) { return raw; }

    template <std::unsigned_integral I> static bool check_bounds(I value) {
        return value <= max();
    }
};

///
/// Allow span resizing when constructing a CompressedVector.
///
struct AllowShrinkingTag {};

///
/// Compressed Vector
///
/// Expresses a view over a span of bytes containing the encoded data.
///
/// @tparam Sign One of the types `Signed` or `Unsigned` indicating the signedness
///     of the encoded values.
/// @tparam Bits The number of bits used to encode each component.
/// @tparam Extent The compile-time dimensionality of the vector.
/// @tparam IsConst Boolean flag indicating whether the view is constant or not.
///
template <typename Sign, size_t Bits, size_t Extent, bool IsConst>
class CompressedVectorBase {
  public:
    ///
    /// Static member describing if this is const.
    ///
    static constexpr bool is_const = IsConst;

    ///
    /// The encoding to use for this combination of sign and number of bits.
    ///
    using encoding_type = Encoding<Sign, Bits>;

    ///
    /// The smallest native integer type capable of storing the uncompressed values
    /// in the vector.
    ///
    using value_type = typename encoding_type::value_type;

    ///
    /// The number of bits used to encode each value in the compressed vector.
    ///
    static constexpr size_t bits = Bits;

    ///
    /// Return the compile-time length of the vector, or ``Dynamic`` if unknown.
    ///
    static constexpr size_t extent = Extent;

    ///
    /// The compile-time number of bytes used for the underlying bytes the compressed
    /// vector interprets.
    ///
    static constexpr size_t storage_extent = compute_storage_extent(bits, Extent);

    ///
    /// The type of the storage span backing this vector.
    ///
    using const_span_type = std::span<const std::byte, storage_extent>;
    using mutable_span_type = std::span<std::byte, storage_extent>;
    using span_type = std::conditional_t<IsConst, const_span_type, mutable_span_type>;

    static constexpr size_t compute_bytes()
        requires(Extent != Dynamic)
    {
        constexpr size_t bytes = compute_storage(Bits, Extent);
        return bytes;
    }

    static constexpr size_t compute_bytes(lib::MaybeStatic<Extent> sz) {
        return compute_storage(Bits, sz);
    }

    // Disable default construction.
    CompressedVectorBase() = delete;

    ///
    /// Construct a `CompressedVector` over the contents of `data`.
    /// This is a view and will not take ownership nor extent the lifetime of `data`.
    ///
    /// Only valid if ``Extent != Dynamic``.
    ///
    explicit CompressedVectorBase(span_type data)
        requires(Extent != Dynamic)
        : size_{}
        , data_{data} {}

    ///
    /// Construct a ``CompressedVector`` with the given size over the contents of `data`.
    /// This is a view and will not take ownership nor extent the lifetime of `data`.
    ///
    explicit CompressedVectorBase(lib::MaybeStatic<Extent> size, span_type data)
        : size_{size}
        , data_{data} {
        // Debug size check if running in dynamic mode.
        if constexpr (Extent == Dynamic) {
            if (data.size() != compute_bytes(size)) {
                throw ANNEXCEPTION("Incorrect size!");
            }
        }
    }

    ///
    /// Construct a CompressedVector from potentially oversized span.
    ///
    /// @param tag Indicate that it is okay to use a subset of the provided span.
    /// @param size The requested number of elements in the compressed view.
    /// @param source The source span to construct a view over.
    ///
    /// Construct a CompressedVector view over the given span.
    /// If necessary, the constructed view will be over only a subset of the provided span.
    /// If this is the case, the subset will begin at the start of ``source``.
    ///
    template <typename T, size_t N>
    explicit CompressedVectorBase(
        AllowShrinkingTag SVS_UNUSED(tag),
        lib::MaybeStatic<Extent> size,
        std::span<T, N> source
    )
        : size_{size}
        , data_{source.begin(), compute_bytes(size)} {
        // Refuse to compile if we can prove that the source span is too short.
        static_assert(storage_extent == Dynamic || N == Dynamic || N >= storage_extent);
        assert(source.size_bytes() >= compute_bytes(size));
    }

    ///
    /// Allow conversion to const from non-const.
    ///
    operator CompressedVectorBase<Sign, Bits, Extent, true>() const {
        return CompressedVectorBase<Sign, Bits, Extent, true>{size_, data_};
    }

    ///
    /// Convert to the const version of the vector.
    ///
    CompressedVectorBase<Sign, Bits, Extent, true> as_const() const { return *this; }

    ///
    /// Return the length of the compressed vector.
    ///
    constexpr size_t size() const { return size_.size(); }

    ///
    /// Return a constant pointer to the start of the underlying storage.
    ///
    const std::byte* data() const { return data_.data(); }

    ///
    /// Return a mutable pointer to the start of the underlying storage.
    ///
    std::byte* data()
        requires(!IsConst)
    {
        return data_.data();
    }

    ///
    /// Return the size in bytes of the underlying storage.
    ///
    constexpr size_t size_bytes() const {
        assert(data_.size_bytes() == compute_bytes(size_));
        return data_.size_bytes();
    }

    ///
    /// Return the uncompressed value at index `i`.
    /// **Preconditions:**
    ///
    /// * `0 <= i < size()`
    ///
    value_type get(size_t i) const {
        auto [byte_start, byte_stop, bit_start, bit_stop] =
            detail::IndexRange(meta::Val<bits>{}, i);
        if (byte_start == byte_stop) {
            auto mask8 = detail::bitmask<uint8_t>(bit_start, bit_stop);
            return decode((extract<uint8_t>(byte_start) & mask8) >> bit_start);
        } else {
            auto mask16 = detail::bitmask<uint16_t>(bit_start, bit_stop);
            return decode((extract<uint16_t>(byte_start) & mask16) >> bit_start);
        }
    }

    template <typename T>
    void set(T v, size_t i)
        requires(!IsConst)
    {
        set(lib::narrow<value_type>(v), i);
    }

    ///
    /// Assign the compressed value `v0` to index `i`.
    /// If `v0` is not exactly expressible using the current encoding, throws
    /// `ANNException.`
    ///
    /// **Preconditions:**
    ///
    /// * `0 <= i < size()`
    ///
    void set(value_type v0, size_t i)
        requires(!IsConst)
    {
        if (!(encoding_type::check_bounds(v0))) {
            throw ANNEXCEPTION(
                "Value of type {} cannot be expressed using {} bits!",
                static_cast<int>(v0),
                bits
            );
        }
        uint8_t v = encode(v0);

        auto [byte_start, byte_stop, bit_start, bit_stop] =
            detail::IndexRange(meta::Val<bits>{}, i);

        if (byte_start == byte_stop) {
            auto m8 = detail::bitmask<uint8_t>(bit_start, bit_stop);
            auto v8 = extract<uint8_t>(byte_start);
            uint8_t newvalue = (v8 & ~m8) | ((v << bit_start) & m8);
            insert<uint8_t>(newvalue, byte_start);
        } else {
            auto m16 = detail::bitmask<uint16_t>(bit_start, bit_stop);
            auto v16 = extract<uint16_t>(byte_start);
            uint16_t newvalue = (v16 & ~m16) | ((v << bit_start) & m16);
            insert<uint16_t>(newvalue, byte_start);
        }
    }

    ///
    /// @brief Copy to contents of another compressed vector view.
    ///
    /// Requires that the other CompressedVectorBase has the same run-time dimensions.
    ///
    template <size_t OtherExtent, bool OtherConst>
        requires(!IsConst)
    void copy_from(const CompressedVectorBase<Sign, Bits, OtherExtent, OtherConst>& other) {
        static_assert(Extent == Dynamic || OtherExtent == Dynamic || Extent == OtherExtent);
        assert(other.size() == size());
        memcpy(data(), other.data(), size_bytes());
    }

    ///
    /// @brief Assign the contents of ``other`` to the compressed vector.
    ///
    /// Requires that each element of ``other`` can be losslessly converted to an integer
    /// in the span ``[Encoding::min(), Encoding::max()]``.
    ///
    template <typename I, typename Alloc>
        requires(!IsConst)
    void copy_from(const std::vector<I, Alloc>& other) {
        assert(size() == other.size());
        for (size_t i = 0, imax = size(); i < imax; ++i) {
            set(other[i], i);
        }
    }

    ///
    /// Safely extract a value of type `T` beginning at byte `i`.
    /// Allow caller to specify the number of bytes to help with AVX decoding.
    ///
    template <typename T> T extract(size_t i) const {
        T v{};
        // N.B.: Memcpy should be optimized away into just an unaligned load.
        // We use `memcpy` instead of `reinterpret_cast` and load because `memcpy`
        // is a bit more "blessed" from a C++ stand point when it comes to winking
        // objects into and out of existence from a bunch of bytes.
        std::memcpy(&v, &data_[i], sizeof(T));
        return v;
    }

    template <typename T> T extract_subset(size_t i, size_t bytes) const {
        assert(bytes <= sizeof(T));
        T v{0};
        std::memcpy(&v, &data_[i], bytes);
        return v;
    }

    ///
    /// Insert a value of type `T` into the underlying bytes, starting at byte `i`.
    ///
    template <typename T>
    void insert(T v, size_t i)
        requires(!IsConst)
    {
        std::memcpy(&data_[i], &v, sizeof(T));
    }

    ///
    /// Perform any necessary steps to convert a raw extracted, unsigned, zero padded
    /// byte to the `value_type` of the encoder.
    ///
    static value_type decode(uint8_t value) { return encoding_type::decode(value); }

    ///
    /// Encode a `value_type` to an unsigned byte suitable for encoding in
    /// `bits` nubmer of bits.
    ///
    static uint8_t encode(value_type value) { return encoding_type::encode(value); }

  private:
    [[no_unique_address]] lib::MaybeStatic<Extent> size_;
    span_type data_;
};

template <typename Sign, size_t Bits, size_t Extent>
using CompressedVector = CompressedVectorBase<Sign, Bits, Extent, true>;

template <typename Sign, size_t Bits, size_t Extent>
using MutableCompressedVector = CompressedVectorBase<Sign, Bits, Extent, false>;

///
/// Base type for vector quantization codecs.
/// Provides storage for a `MutableCompressedVector`.
///
class CVStorage {
  public:
    CVStorage()
        : storage_{} {}

    template <typename Sign, size_t Bits, size_t Extent>
    MutableCompressedVector<Sign, Bits, Extent> view(lib::MaybeStatic<Extent> size = {}) {
        using Mut = MutableCompressedVector<Sign, Bits, Extent>;
        storage_.resize(Mut::compute_bytes(size));
        return Mut{size, typename Mut::span_type{storage_}};
    }

    template <typename Sign, size_t Bits>
    MutableCompressedVector<Sign, Bits, Dynamic> view(size_t size) {
        return view<Sign, Bits, Dynamic>(lib::MaybeStatic(size));
    }

  private:
    std::vector<std::byte> storage_;
};

///
/// Extract a value of type `T` from the raw bytes containing the compressed vector
/// beginning at byte `i`.
///
template <typename T, typename Sign, size_t Bits, size_t Extent>
T extract(const CompressedVector<Sign, Bits, Extent>& v, size_t i) {
    return v.template extract<T>(i);
}

template <typename T, typename Sign, size_t Bits, size_t Extent>
T extract_predicated(
    const CompressedVector<Sign, Bits, Extent>& v, size_t i, eve::ignore_none_ /*unused*/
) {
    return v.template extract<T>(i);
}

template <typename T, typename Sign, size_t Bits, size_t Extent>
T extract_predicated(
    const CompressedVector<Sign, Bits, Extent>& v, size_t i, eve::keep_first keep_first
) {
    size_t bytes = lib::div_round_up(Bits * keep_first.count(eve::as<size_t>()), 8);
    return v.template extract_subset<T>(i, bytes);
}

/////
///// AVX Accelerated unpacking of compressed data.
/////

namespace detail {
///
/// AVX register width compatibility traits.
///
template <size_t Bits, size_t VecWidth> inline constexpr bool is_compatible = false;

// 8-wide SIMD
template <> inline constexpr bool is_compatible<3, 8> = true;
template <> inline constexpr bool is_compatible<4, 8> = true;
template <> inline constexpr bool is_compatible<5, 8> = true;
template <> inline constexpr bool is_compatible<6, 8> = true;
template <> inline constexpr bool is_compatible<7, 8> = true;
template <> inline constexpr bool is_compatible<8, 8> = true;

// 16-wide SIMD
template <> inline constexpr bool is_compatible<4, 16> = true;
template <> inline constexpr bool is_compatible<8, 16> = true;

///
/// The preferred SIMD width to use with the number of bits used to encode compressed
/// values.
///
template <size_t Bits> inline constexpr size_t preferred_simd_width = 0;
template <> inline constexpr size_t preferred_simd_width<3> = 8;
template <> inline constexpr size_t preferred_simd_width<4> = 16;
template <> inline constexpr size_t preferred_simd_width<5> = 8;
template <> inline constexpr size_t preferred_simd_width<6> = 8;
template <> inline constexpr size_t preferred_simd_width<7> = 8;
template <> inline constexpr size_t preferred_simd_width<8> = 16;

///
/// By default, the encoded values need to unpack into 64-bit integers. However,
/// encodings whose width is more "natural" (e.g., 8-bits, or 4-bits), we can use
/// 32-bit integers to take advantage of more values per AVX register.
///
template <size_t VecWidth, size_t Bits> struct IntegerMapping {
    using type = int64_t;
};
template <> struct IntegerMapping<16, 8> {
    using type = int32_t;
};
template <> struct IntegerMapping<8, 8> {
    using type = int32_t;
};
template <> struct IntegerMapping<16, 4> {
    using type = int32_t;
};
template <> struct IntegerMapping<8, 4> {
    using type = int32_t;
};
template <> struct IntegerMapping<8, 3> {
    using type = int32_t;
};

///
/// The primitive integer type that will hold the unpacked values from the compressed
/// vector.
///
template <size_t VecWidth, size_t Bits>
using integer_t = typename IntegerMapping<VecWidth, Bits>::type;

///
/// SIMD register type to hold `VecWidth` unpacked values.
///
template <size_t VecWidth, size_t Bits>
using integer_wide_t = wide_<integer_t<VecWidth, Bits>, VecWidth>;

template <typename T, size_t Bits> wide_<T, 8> shifts_x8() {
    return wide_<T, 8>{0, Bits, 2 * Bits, 3 * Bits, 4 * Bits, 5 * Bits, 6 * Bits, 7 * Bits};
}

template <size_t VecWidth, typename Sign, size_t Bits> struct UnpackerBase {
    static const size_t simd_width = VecWidth;
    using int_type = integer_t<VecWidth, Bits>;
    using int_wide_type = integer_wide_t<VecWidth, Bits>;
    using accum_type = wide_<float, VecWidth>;
    using raw_type = typename Encoding<Sign, Bits>::value_type;

    static const size_t bits = Bits;

    inline static const wide_<int_type, 8> shifts_x8 = detail::shifts_x8<int_type, Bits>();

    static const int_type mask = bitmask<int_type>(0, Bits - 1);
    static const int_type bias = Encoding<Sign, Bits>::min();
};

/////
///// Select widest integer.
/////
template <std::signed_integral Left, std::signed_integral Right>
using biggest_int_t = std::conditional_t<sizeof(Left) >= sizeof(Right), Left, Right>;

template <std::signed_integral Left, std::signed_integral Right, size_t N>
using common_wide_t = wide_<biggest_int_t<Left, Right>, N>;
} // namespace detail

// Pick the SIMD width for a pair of compressed vectors.
template <size_t ABits> constexpr size_t pick_simd_width() {
    return detail::preferred_simd_width<ABits>;
}

template <size_t ABits, size_t BBits> constexpr size_t pick_simd_width() {
    constexpr size_t a_width = detail::preferred_simd_width<ABits>;
    constexpr size_t b_width = detail::preferred_simd_width<BBits>;
    constexpr size_t width = std::min(a_width, b_width);
    // Check  SIMD width compatibility.
    static_assert(detail::is_compatible<ABits, width> && detail::is_compatible<BBits, width>);
    return width;
}

template <typename A> constexpr size_t pick_simd_width() {
    return pick_simd_width<A::bits>();
}

template <typename A, typename B> constexpr size_t pick_simd_width() {
    return pick_simd_width<A::bits, B::bits>();
}

template <size_t VecWidth, typename Sign, size_t Bits, size_t Extent>
struct Unpacker : public detail::UnpackerBase<VecWidth, Sign, Bits> {
    // Make sure this is a considered combination.
    static_assert(detail::is_compatible<Bits, VecWidth>);

    Unpacker(meta::Val<VecWidth> /*unused*/, CompressedVector<Sign, Bits, Extent> data)
        : data_{data} {}

    Unpacker(CompressedVector<Sign, Bits, Extent> data)
        : data_{data} {}

    using parent_type = detail::UnpackerBase<VecWidth, Sign, Bits>;
    using int_wide_type = typename parent_type::int_wide_type;

    // Rebind the unpacker to a new SIMD width.
    template <size_t NewVecWidth> Unpacker<NewVecWidth, Sign, Bits, Extent> rebind() const {
        return Unpacker<NewVecWidth, Sign, Bits, Extent>{data_};
    }

    template <typename P = eve::ignore_none_>
    int_wide_type get(size_t i, P pred = eve::ignore_none_()) const {
        return unpack(meta::Val<VecWidth>(), data_, i, pred);
    }

    constexpr size_t size() const { return data_.size(); }

    // member
    CompressedVector<Sign, Bits, Extent> data_;
};

template <
    size_t VecWidth,
    typename Sign,
    size_t Bits,
    size_t Extent,
    typename T = eve::ignore_none_>
auto unpack(
    meta::Val<VecWidth> /*unused*/,
    const CompressedVector<Sign, Bits, Extent>& x,
    size_t i,
    T predicate = eve::ignore_none_()
) -> typename Unpacker<VecWidth, Sign, Bits, Extent>::int_wide_type {
    using Unpacker = Unpacker<VecWidth, Sign, Bits, Extent>;

    using integer_t = typename Unpacker::int_type;
    using integer_wide_t = typename Unpacker::int_wide_type;

    // Load elements from the compressed dataset.
    integer_wide_t broadcast{
        extract_predicated<integer_t>(x, Unpacker::bits * i, predicate)};
    return eve::add[predicate.else_(0)](
        (broadcast >> Unpacker::shifts_x8) & Unpacker::mask, Unpacker::bias
    );
}

///
/// Specialize for 8-bit packing.
///
/// With 8-bits, there's no need for masking, shifting, or bias re-adjustment. Instead,
/// we can perform a direct predicated load and avoid many unnecessary instructions.
///
template <size_t VecWidth, typename Sign, size_t Extent, typename T = eve::ignore_none_>
auto unpack(
    meta::Val<VecWidth> /*unused*/,
    const CompressedVector<Sign, 8, Extent>& x,
    size_t i,
    T predicate = eve::ignore_none_()
) -> typename Unpacker<VecWidth, Sign, 8, Extent>::int_wide_type {
    using Unpacker = Unpacker<VecWidth, Sign, 8, Extent>;
    using integer_t = typename Unpacker::int_type;
    using raw_t = typename Unpacker::raw_type;

    auto packed = eve::load[predicate.else_(0)](
        reinterpret_cast<const raw_t*>(x.data() + VecWidth * i),
        eve::as<wide_<raw_t, VecWidth>>()
    );

    return eve::convert(packed, eve::as<integer_t>());
}

///
/// Specialize 4-bit unpacking.
///
/// With 4-bit unpacking, a full 8 encoded values live in each 32-bit word.
/// As such, we can work directly with 32-bit integers and make better use of our SIMD
/// lanes.
///
template <typename Sign, size_t Extent>
auto unpack(
    meta::Val<16> /*unused*/,
    const CompressedVector<Sign, 4, Extent>& x,
    size_t i,
    eve::ignore_none_ predicate = eve::ignore_none_()
) -> typename Unpacker<16, Sign, 4, Extent>::int_wide_type {
    using Unpacker = Unpacker<16, Sign, 4, Extent>;
    using integer_t = typename Unpacker::int_type;
    using integer_wide_t = typename Unpacker::int_wide_type;
    using half_integer_wide_t = wide_<integer_t, 8>;

    auto i_long = lib::narrow_cast<int>(i);
    integer_wide_t shifts{Unpacker::shifts_x8, Unpacker::shifts_x8};
    half_integer_wide_t lo{extract<integer_t>(x, 4 * 2 * i_long)};
    half_integer_wide_t hi{extract<integer_t>(x, 4 * (2 * i_long + 1))};
    integer_wide_t combined{lo, hi};
    return eve::add[predicate.else_(0)](
        (combined >> shifts) & Unpacker::mask, Unpacker::bias
    );
}

template <typename Sign, size_t Extent>
auto unpack(
    meta::Val<16> /*unused*/,
    const CompressedVector<Sign, 4, Extent>& x,
    size_t i,
    eve::keep_first predicate
) -> typename Unpacker<16, Sign, 4, Extent>::int_wide_type {
    using Unpacker = Unpacker<16, Sign, 4, Extent>;
    using integer_t = typename Unpacker::int_type;
    using integer_wide_t = typename Unpacker::int_wide_type;

    using half_integer_wide_t = wide_<integer_t, 8>;
    auto count = predicate.count(eve::as<int64_t>());
    int64_t count_lo = count >= 8 ? 8 : count;
    int64_t count_hi = count >= 8 ? count - 8 : 0;

    integer_wide_t shifts{Unpacker::shifts_x8, Unpacker::shifts_x8};
    auto lo = half_integer_wide_t{
        extract_predicated<integer_t>(x, 4 * (2 * i), eve::keep_first(count_lo))};
    auto hi = half_integer_wide_t{
        extract_predicated<integer_t>(x, 4 * (2 * i + 1), eve::keep_first(count_hi))};
    integer_wide_t combined{lo, hi};
    return eve::add[predicate.else_(0)](
        (combined >> shifts) & Unpacker::mask, Unpacker::bias
    );
}

///
/// Use SIMD acceleration to unpack the compressed vector to the destination.
///
template <typename Sign, size_t Bits, size_t Extent>
void unpack(
    std::vector<typename CompressedVector<Sign, Bits, Extent>::value_type>& v,
    CompressedVector<Sign, Bits, Extent> cv
) {
    v.resize(cv.size());
    unpack(std::span(v.data(), v.size()), cv);
}

template <typename Sign, size_t Bits, size_t Extent>
void unpack(
    std::span<typename CompressedVector<Sign, Bits, Extent>::value_type> v,
    CompressedVector<Sign, Bits, Extent> cv
) {
    // Instantiate the unpacker.
    unpack(v, Unpacker(meta::Val<pick_simd_width<Bits>()>(), cv));
}

// Generic entry-point for unpacking into a destination.
template <std::integral I, size_t VecWidth, typename Sign, size_t Bits, size_t Extent>
void unpack(std::span<I> v, Unpacker<VecWidth, Sign, Bits, Extent> unpacker) {
    assert(v.size() == unpacker.size());
    I* base = v.data();
    size_t iterations = unpacker.size() / VecWidth;
    size_t tail = unpacker.size() % VecWidth;

    // Main iterations
    for (size_t i = 0; i < iterations; ++i) {
        eve::store(eve::convert(unpacker.get(i), eve::as<I>()), base);
        base += VecWidth;
    }

    // Tail Iterations
    if (tail != 0) {
        auto predicate = eve::keep_first(lib::narrow<int64_t>(tail));
        eve::store[predicate](
            eve::convert(unpacker.get(iterations, predicate), eve::as<I>()), base
        );
    }
}

} // namespace svs::quantization::lvq
