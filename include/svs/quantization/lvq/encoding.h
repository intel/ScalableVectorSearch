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

// svs
#include "svs/lib/float16.h"
#include "svs/lib/meta.h"
#include "svs/lib/misc.h"
#include "svs/lib/preprocessor.h"

// stl
#include <bit>
#include <concepts>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace svs::quantization::lvq {

/// The storage format for LVQ constants.
using scaling_t = svs::Float16;

/// Route floating point numbers through the scaling type.
inline scaling_t through_scaling_type(svs::Float16 x) { return x; }
inline scaling_t through_scaling_type(float x) { return lib::narrow_cast<scaling_t>(x); }
inline scaling_t through_scaling_type(double x) { return lib::narrow_cast<scaling_t>(x); }

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
    IndexRange(lib::Val<Bits> /*unused*/, size_t i)
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

} // namespace svs::quantization::lvq
