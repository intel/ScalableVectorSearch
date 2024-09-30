/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
 */
#pragma once

// svs
#include "svs/lib/meta.h"
#include "svs/lib/misc.h"
#include "svs/lib/preprocessor.h"
#include "svs/lib/static.h"
#include "svs/third-party/eve.h"

// svs-lvq
#include "svs/quantization/lvq/config.h"
#include "svs/quantization/lvq/encoding.h"

// third-party
#include "svs/third-party/fmt.h"

// stl
#include <bit>
#include <concepts>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

namespace svs::quantization::lvq {

/////
///// Strategies for storing compressed data.
/////

// Strategies for storing packed data.
struct Sequential {
    static constexpr std::string_view name() { return "sequential"; }
    static constexpr size_t compute_bytes(size_t nbits, size_t length) {
        return lib::div_round_up(nbits * length, 8);
    }

    // No permutation required.
    static constexpr size_t logical_to_linear(size_t i) { return i; }
    static constexpr size_t linear_to_logical(size_t i) { return i; }
};

// Blockwise strategy.
template <size_t Lanes, size_t ElementsPerLane> struct Turbo {
    static constexpr std::string name() {
        return fmt::format("turbo<{}x{}>", Lanes, ElementsPerLane);
    }
    static constexpr size_t lanes = Lanes;
    static constexpr size_t elements_per_lane = ElementsPerLane;
    static constexpr size_t block_size = Lanes * ElementsPerLane;

    static constexpr size_t compute_bytes(size_t nbits, size_t length) {
        assert(nbits == 4 || nbits == 8);

        size_t block_size_bytes = nbits * block_size / 8;
        size_t num_blocks = lib::div_round_up(length, block_size);
        return block_size_bytes * num_blocks;
    }

    static constexpr size_t logical_to_linear(size_t i) {
        // `a`: Which block we are in.
        // `b`: Tne entry in the block.
        // `c`: The offset in the lane
        // `d`: Which lane.
        auto [a, b] = detail::divrem(i, block_size);
        auto [c, d] = detail::divrem(b, Lanes);
        return block_size * a + ElementsPerLane * d + c;
    }

    static constexpr size_t linear_to_logical(size_t i) {
        // `a`: Which block we are in.
        // `b`: The entry in the block.
        auto [a, b] = detail::divrem(i, block_size);
        auto [c, d] = detail::divrem(b, ElementsPerLane);
        return block_size * a + Lanes * d + c;
    }

    static constexpr size_t num_blocks(size_t count) {
        return lib::round_up_to_multiple_of(count, block_size);
    }
};

namespace detail {

// Customization point object for logic equality.
struct LogicallyEqualType {
    template <typename Left, typename Right>
    bool operator()(const Left& left, const Right& right) const {
        return left.logically_equivalent_to(right);
    }
};

// Trait to identify and dispatch based on the Turbo class itself.
template <typename T> inline constexpr bool is_turbo_like_v = false;
template <typename T> inline constexpr bool is_lvq_packing_strategy_v = false;

template <size_t Lanes, size_t ElementsPerLane>
inline constexpr bool is_turbo_like_v<lvq::Turbo<Lanes, ElementsPerLane>> = true;

template <> inline constexpr bool is_lvq_packing_strategy_v<lvq::Sequential> = true;
template <size_t Lanes, size_t ElementsPerLane>
inline constexpr bool is_lvq_packing_strategy_v<lvq::Turbo<Lanes, ElementsPerLane>> = true;

} // namespace detail

template <typename T>
concept LVQPackingStrategy = detail::is_lvq_packing_strategy_v<T>;

template <typename T>
concept TurboLike = detail::is_turbo_like_v<T>;

template <typename T>
concept UsesSequential = std::is_same_v<typename T::strategy, Sequential>;

template <typename T>
concept UsesTurbo = TurboLike<typename T::strategy>;

///
/// @brief Return whether two LVQ compressed entities are logically equal.
///
/// @return Boolean indicating logical equality.
///
/// Two compressed vectors are logically equal if:
///
/// 1. The are encoded using the same number of bits and the same signedness.
/// 2. They have the same runtime length.
/// 3. The encodings for each pairwise logical dimension are equal.
///
/// In particular, logical equality can hold for compressed vectors using different
/// packing strategies.
///
inline constexpr detail::LogicallyEqualType logically_equal{};

///
/// Allow span resizing when constructing a CompressedVector.
///
struct AllowShrinkingTag {};

using DefaultStrategy = Sequential;

template <
    typename Sign,
    size_t Bits,
    size_t Extent,
    bool IsConst,
    LVQPackingStrategy Strategy = DefaultStrategy>
class CompressedVectorBase {
  public:
    /// The packing strategy used by the vector.
    using strategy = Strategy;
    /// Static member describing if this is const.
    static constexpr bool is_const = IsConst;
    /// The encoding to use for this combination of sign and number of bits.
    using encoding_type = Encoding<Sign, Bits>;
    /// The smallest native integer type capable of storing the uncompressed values
    /// in the vector.
    using value_type = typename encoding_type::value_type;
    /// The number of bits used to encode each value in the compressed vector.
    static constexpr size_t bits = Bits;
    /// Return the compile-time length of the vector, or ``Dynamic`` if unknown.
    static constexpr size_t extent = Extent;
    /// The maximum encoding value.
    static constexpr value_type max() { return encoding_type::max(); }
    /// The minimum encoding value.
    static constexpr value_type min() { return encoding_type::min(); }

    /// The compile-time number of bytes used for the underlying bytes the compressed
    /// vector interprets.
    static constexpr size_t storage_extent =
        Extent == Dynamic ? Dynamic : Strategy::compute_bytes(Bits, Extent);

    /// The type of the storage span backing this vector.
    using const_span_type = std::span<const std::byte, storage_extent>;
    using mutable_span_type = std::span<std::byte, storage_extent>;
    using span_type = std::conditional_t<IsConst, const_span_type, mutable_span_type>;
    using pointer = std::conditional_t<IsConst, const std::byte*, std::byte*>;

    static constexpr size_t compute_bytes()
        requires(Extent != Dynamic)
    {
        return CompressedVectorBase::compute_bytes(lib::MaybeStatic<Extent>());
    }

    static constexpr size_t compute_bytes(lib::MaybeStatic<Extent> sz) {
        return Strategy::compute_bytes(Bits, sz);
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
        : data_{data.data()}
        , size_{} {}

    ///
    /// Construct a ``CompressedVector`` with the given size over the contents of `data`.
    /// This is a view and will not take ownership nor extent the lifetime of `data`.
    ///
    explicit CompressedVectorBase(lib::MaybeStatic<Extent> size, span_type data)
        : data_{data.data()}
        , size_{size} {
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
        : data_{source.data()}
        , size_{size} {
        // Refuse to compile if we can prove that the source span is too short.
        static_assert(storage_extent == Dynamic || N == Dynamic || N >= storage_extent);
        assert(source.size_bytes() >= compute_bytes(size));
    }

    ///
    /// Allow conversion to const from non-const.
    ///
    operator CompressedVectorBase<Sign, Bits, Extent, true, Strategy>() const {
        return CompressedVectorBase<Sign, Bits, Extent, true, Strategy>{data_, size_};
    }

    ///
    /// Convert to the const version of the vector.
    ///
    CompressedVectorBase<Sign, Bits, Extent, true, Strategy> as_const() const {
        return *this;
    }

    ///
    /// Return the length of the compressed vector.
    ///
    constexpr size_t size() const { return size_.size(); }

    ///
    /// Return a constant pointer to the start of the underlying storage.
    ///
    const std::byte* data() const { return data_; }

    ///
    /// Return a mutable pointer to the start of the underlying storage.
    ///
    std::byte* data()
        requires(!IsConst)
    {
        return data_;
    }

    ///
    /// Return the size in bytes of the underlying storage.
    ///
    constexpr size_t size_bytes() const { return compute_bytes(size_); }

    ///
    /// Return the uncompressed value at index `i`.
    /// **Preconditions:**
    ///
    /// * `0 <= i < size()`
    ///
    value_type get(size_t i) const {
        auto j = Strategy::logical_to_linear(i);
        auto [byte_start, byte_stop, bit_start, bit_stop] =
            detail::IndexRange(lib::Val<bits>{}, j);
        if (byte_start == byte_stop) {
            auto mask8 = lib::bitmask<uint8_t>(bit_start, bit_stop);
            return decode((extract<uint8_t>(byte_start) & mask8) >> bit_start);
        }

        auto mask16 = lib::bitmask<uint16_t>(bit_start, bit_stop);
        return decode((extract<uint16_t>(byte_start) & mask16) >> bit_start);
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

        size_t j = Strategy::logical_to_linear(i);
        auto [byte_start, byte_stop, bit_start, bit_stop] =
            detail::IndexRange(lib::Val<bits>{}, j);

        if (byte_start == byte_stop) {
            auto m8 = lib::bitmask<uint8_t>(bit_start, bit_stop);
            auto v8 = extract<uint8_t>(byte_start);
            uint8_t newvalue = (v8 & ~m8) | ((v << bit_start) & m8);
            insert<uint8_t>(newvalue, byte_start);
        } else {
            auto m16 = lib::bitmask<uint16_t>(bit_start, bit_stop);
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
    void copy_from(
        const CompressedVectorBase<Sign, Bits, OtherExtent, OtherConst, Strategy>& other
    ) {
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
    /// Allow caller to specify the number of bytes to help with Intel(R) AVX decoding.
    ///
    template <typename T> T extract(size_t i) const {
        T v{};
        // N.B.: Memcpy should be optimized away into just an unaligned load.
        // We use `memcpy` instead of `reinterpret_cast` and load because `memcpy`
        // is a bit more "blessed" from a C++ stand point when it comes to winking
        // objects into and out of existence from a bunch of bytes.
        std::memcpy(&v, data() + i, sizeof(T));
        return v;
    }

    /// Extract a value by copying the specified bytes beginning at offset ``i``.
    ///
    /// This function behaves as if constructing a value of type ``T`` with an all-zero
    /// bit representation and performing a ``std::memcpy`` of the specified bytes
    /// into the representation of ``T``.
    ///
    /// In other words, copy only the requested bytes and zero pad the rest of ``T``.
    ///
    /// This function is safe to use in contexts where generating a read of ``sizeof(T)``
    /// would result in an out-of-bounds access.
    ///
    /// Prerequisites:
    /// * ``T`` is constexpr default constructible AND constructible from ``int(0)`` with
    ///   bit representation of the returned object consisting of zeroed memory.
    /// * ``T`` is trivially copyable.
    /// * ``bytes <= sizeof(T)``
    /// * ``0 < bytes``: At least one byte must be read and that read must be inbounds.
    /// @tparam T The type to extract.
    /// @tparam Static Whether this is being called in a static dimensional context or not.
    ///     This is a hint and should not affect the read value.
    template <typename T, bool Static = (Extent != svs::Dynamic)>
    SVS_FORCE_INLINE T extract_subset(size_t i, uint64_t bytes) const {
        static_assert(T{} == T{0});
        static_assert(std::is_trivially_copyable_v<T>);

        // Check pre-requisites.
        assert(bytes <= sizeof(T));
        assert(0 < bytes);
        assert(i + bytes <= size_bytes());

        // Variable length memcopies are quite slow.
        // If it is available, use a masked load.
        // Fault suppression will ensure that we don't trap on an invalid read.
        //
        // When static dimensionality is used, the compiler's constant-propagated
        // heuristics seem to do slightly better than the predicated load.
        constexpr bool use_predicated_load =
            !Static && sizeof(T) <= 16 && arch::have_avx512_vl && arch::have_avx512_bw;

        // Choose the strategy for loading the requested number of bytes.
        if constexpr (use_predicated_load) {
            T v{};
            constexpr uint64_t one = 1;
            auto mask = static_cast<__mmask16>((one << bytes) - one);
            auto reg = _mm_maskz_loadu_epi8(mask, data() + i);
            std::memcpy(&v, &reg, sizeof(T));
            return v;
        } else {
            T v{0};
            std::memcpy(&v, data() + i, bytes);
            return v;
        }
    }

    ///
    /// Insert a value of type `T` into the underlying bytes, starting at byte `i`.
    ///
    template <typename T>
    void insert(T v, size_t i)
        requires(!IsConst)
    {
        assert(i + sizeof(T) <= size_bytes());
        std::memcpy(data() + i, &v, sizeof(T));
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

    ///
    /// Logic Equivalence.
    ///
    /// Two vectors are logically equivalent if they contain the same number of dimensions
    /// and the encoding for each dimension
    ///
    template <size_t E2, bool C2, LVQPackingStrategy S2>
    bool logically_equivalent_to(const CompressedVectorBase<Sign, Bits, E2, C2, S2>& other
    ) const {
        if (size() != other.size()) {
            return false;
        }

        // Fast-path for sequential strategies.
        constexpr bool use_memcmp =
            std::is_same_v<Strategy, Sequential> && std::is_same_v<S2, Sequential>;

        if constexpr (use_memcmp) {
            return std::memcmp(data(), other.data(), size_bytes()) == 0;
        } else {
            for (size_t i = 0, imax = size(); i < imax; ++i) {
                if (get(i) != other.get(i)) {
                    return false;
                }
            }
            return true;
        }
    }

    // Make the mutable version a friend to the constant version to enable direct
    // construction when implicitly converting to const.
    friend class CompressedVectorBase<Sign, Bits, Extent, false, Strategy>;

  private:
    // Private constructor from a raw-pointer.
    CompressedVectorBase(pointer data, lib::MaybeStatic<Extent> size)
        : data_{data}
        , size_{size} {}

    ///// Members
    pointer data_;
    [[no_unique_address]] lib::MaybeStatic<Extent> size_;
};

template <
    typename Sign,
    size_t Bits,
    size_t Extent,
    LVQPackingStrategy Strategy = DefaultStrategy>
using CompressedVector = CompressedVectorBase<Sign, Bits, Extent, true, Strategy>;

template <
    typename Sign,
    size_t Bits,
    size_t Extent,
    LVQPackingStrategy Strategy = DefaultStrategy>
using MutableCompressedVector = CompressedVectorBase<Sign, Bits, Extent, false, Strategy>;

///
/// Base type for vector quantization codecs.
/// Provides storage for a `MutableCompressedVector`.
///
class CVStorage {
  public:
    CVStorage()
        : storage_{} {}

    template <
        typename Sign,
        size_t Bits,
        size_t Extent,
        LVQPackingStrategy Strategy = DefaultStrategy>
    MutableCompressedVector<Sign, Bits, Extent, Strategy>
    view(lib::MaybeStatic<Extent> size = {}) {
        using Mut = MutableCompressedVector<Sign, Bits, Extent, Strategy>;
        storage_.resize(Mut::compute_bytes(size));
        return Mut{size, typename Mut::span_type{storage_}};
    }

    template <typename Sign, size_t Bits, LVQPackingStrategy Strategy = DefaultStrategy>
    MutableCompressedVector<Sign, Bits, Dynamic, Strategy> view(size_t size) {
        return view<Sign, Bits, Dynamic>(lib::MaybeStatic(size));
    }

  private:
    std::vector<std::byte> storage_;
};

/////
///// Intel(R) AVX Helpers for Sequential LVQ
/////

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

namespace detail {

template <typename T, size_t Bits> wide_<T, 8> shifts_x8() {
    return wide_<T, 8>{0, Bits, 2 * Bits, 3 * Bits, 4 * Bits, 5 * Bits, 6 * Bits, 7 * Bits};
}

// Sentinal type representing no pre-computed data for unpack helping.
struct Empty {
    // Init
    constexpr Empty operator()() const { return {}; }
    // Combine
    constexpr Empty operator()(Empty, Empty) const { return {}; }
    // Reduce
    constexpr void operator()(Empty) const {}
};

inline constexpr Empty empty{};

} // namespace detail

/////
///// Unpacking
/////

// Compilers can be a little flaky on whether the successfully move pre-computed values
// like shifts and masks into registers, or whether they reload these constant values on
// every iteration.
//
// This prefix function helps us hoist these computations out of the main loop and increases
// our odds of the compiler "doing the right thing".
template <typename Sign, size_t Bits, size_t Extent>
SVS_FORCE_INLINE auto
prepare_unpack(const CompressedVector<Sign, Bits, Extent, Sequential>& SVS_UNUSED(x)) {
    if constexpr (Bits == 8) {
        return detail::Empty{};
    } else if constexpr (Bits > 4) {
        return std::make_pair(
            detail::shifts_x8<int64_t, Bits>(),
            wide_<int32_t, 16>(lib::bitmask<int32_t>(0, Bits - 1))
        );
    } else {
        auto half = detail::shifts_x8<int32_t, Bits>();
        return std::make_pair(
            wide_<int32_t, 16>{half, half},
            wide_<int32_t, 16>(lib::bitmask<int32_t>(0, Bits - 1))
        );
    }
}

// 8-bit specialization.
template <typename Sign, size_t Extent, typename Predicate = eve::ignore_none_>
SVS_FORCE_INLINE wide_<int32_t, 16> unpack_8_as(
    CompressedVector<Sign, 8, Extent, Sequential> x,
    size_t i,
    eve::as<wide_<int32_t, 16>>,
    detail::Empty,
    Predicate predicate = {}
) {
    using raw_t = typename Encoding<Sign, 8>::value_type;
    auto packed = eve::load[predicate.else_(0)](
        reinterpret_cast<const raw_t*>(x.data() + 16 * i), eve::as<wide_<raw_t, 16>>()
    );
    return eve::convert(packed, eve::as<int32_t>());
}

template <typename Sign, size_t Bits, size_t Extent, typename Helper>
SVS_FORCE_INLINE wide_<int32_t, 16> unpack_as(
    CompressedVector<Sign, Bits, Extent, Sequential> x,
    size_t i,
    [[maybe_unused]] eve::as<wide_<int32_t, 16>> as,
    const Helper& helper,
    eve::ignore_none_ SVS_UNUSED(predicate) = {}
) {
    // For 8-bits, use a code-path that works regardless of predicate.
    if constexpr (Bits == 8) {
        return unpack_8_as(x, i, as, helper);
    } else if constexpr (Bits > 4) {
        // When using more than 4-bits, we have to unpack each group of 8 elements as a
        // 64-bit integer.
        //
        // Use broadcasting and shifting to extract each of the elements into 8 lanes.
        // Do this for the next group of 8-elements as well.
        //
        // After broad-casting and shifting, the conversion to 32-bit integers is guarenteed
        // lossless, so we can horizontally concatenate the registers to get a full 16
        // -element bundle.
        auto lo = wide_<int64_t, 8>{extract<int64_t>(x, Bits * 2 * i)} >> helper.first;
        auto hi =
            wide_<int64_t, 8>{extract<int64_t>(x, Bits * (2 * i + 1))} >> helper.first;

        auto combined = wide_<int32_t, 16>(
            eve::convert(lo, eve::as<int32_t>()), eve::convert(hi, eve::as<int32_t>())
        );

        return (combined & helper.second) + Encoding<Sign, Bits>::min();
    } else {
        // Once we're at 4-bits or lower, we can use the same strategy, except this time
        // The initial load can be done using 32-bit integers to avoid the intermediate
        // conversion from 64-bit to 32-bit.
        auto lo = wide_<int32_t, 8>{extract<int32_t>(x, Bits * 2 * i)};
        auto hi = wide_<int32_t, 8>{extract<int32_t>(x, Bits * (2 * i + 1))};

        auto combined = wide_<int32_t, 16>{lo, hi};
        return ((combined >> helper.first) & helper.second) + Encoding<Sign, Bits>::min();
    }
}

namespace detail {

// A very light-weight container for constructing a pair of `eve::wide`.
template <typename T, size_t N> struct WidePair {
    // Convenience constructor broadcasting the arguments across all lanes in the
    // corresponding `wide`.
    WidePair(T first, T second)
        : first_{first}
        , second_{second} {}

    wide_<T, N> first_;
    wide_<T, N> second_;
};

// Utility pair for splitting predicates.
struct PredicatePair {
    eve::keep_first first_;
    eve::keep_first second_;
};

// Split a predicate in half.
template <int64_t N> SVS_FORCE_INLINE PredicatePair split_predicate(eve::keep_first p) {
    auto v = p.count(eve::as<int64_t>());
    assert(v <= 2 * N);
    auto ishigh = v > N;
    return PredicatePair{
        eve::keep_first(ishigh ? N : v), eve::keep_first(ishigh ? (v - N) : 0)};
}

} // namespace detail

// Notes on predicated unpacking:
//
// Static Dimensionality:
//
//    When compile-time dimensionality is used, the compilers (both GCC and Clang) are
//    generally able to constant-propagate the value (number of SIMD lanes to keep) of the
//    predicate through all these methods.
//
//    In particular, the number of bytes given to `CompressedVector::extract_predicated`
//    is known at compile time and an appropriate sequence of loads is used instead of
//    a full call to `memcpy`.
//
//    However, GCC only seems able to materialized sub-object loads when the value being
//    extracted from `CompressedVector::extract_predicate` is a primitive integral type.
//
//    In other words, trying to load something like `std::array<int64_t, 2>` is not
//    optimized correctly. Instead, GCC writes sub-object loads onto the stack before
//    reading back into registers (i.e., SROA fails).
//
//    So, when static dimensionality is used, we generate two predicated loads to native
//    integer types.
//
// Dynamic Dimensionality:
//
//    When dynamic dimensionality is used, the compiler for obvious reasons is unable to
//    constant propagate the value of the predicate.
//
//    In order to prevent actual calls to `memcpy` from appearing in the generated code
//    (this slows down end-to-end performance by A LOT) we try to hit the masked SIMD load
//    path in `CompressedVector::extract_predicated`.
//
//    To do this efficiently, we load the bit data as an array so that we only need to
//    invoke the masked load a single time.
template <typename Sign, size_t Bits, size_t Extent, typename Helper>
wide_<int32_t, 16> unpack_as(
    CompressedVector<Sign, Bits, Extent, Sequential> x,
    size_t i,
    [[maybe_unused]] eve::as<wide_<int32_t, 16>> as,
    Helper helper,
    eve::keep_first predicate
) {
    if constexpr (Bits == 8) {
        return unpack_8_as(x, i, as, helper, predicate);
    } else {
        // Utility lambda for picking the best strategy for unpacking.
        // Upon invocation, will yield a `detail::WidePair` consisting of two half-width
        // `wides` loading the `predicate` lanes in total.
        //
        // All masked out lanes will have zero as their value.
        auto unpack = [&]<std::integral T>() -> detail::WidePair<T, 8> {
            // With 4-bits, the alignment works out that we can load both the low and high
            // values in a single shot.
            //
            // However, doing this is only really beneficial when using dynamic
            // dimensionality because the compiler's handling of the masked load in the
            // static case (i.e., the contents of `predicate` are const-propagated) seem
            // to be slightly better.
            if constexpr (Extent == Dynamic && Bits == 4) {
                auto r = extract_predicated<std::array<T, 2>>(x, Bits * 2 * i, predicate);
                return {r[0], r[1]};
            } else {
                // Read the low and high values.
                // Only generate a read to the high register if at least one of the lanes
                // is valid (otherwise, we might call `memcpy` on an invalid pointer.
                auto [lomask, himask] = detail::split_predicate<8>(predicate);
                bool generate_upper_read = himask.count(eve::as<int64_t>()) != 0;
                return {
                    extract_predicated<T>(x, Bits * 2 * i, lomask),
                    generate_upper_read
                        ? extract_predicated<T>(x, Bits * (2 * i + 1), himask)
                        : T{0}};
            }
        };

        if constexpr (Bits > 4) {
            // Obtain low and high components.
            // Shift and mask before merging.
            auto [lo, hi] = unpack.template operator()<int64_t>();
            auto combined = wide_<int32_t, 16>(
                eve::convert(lo >> helper.first, eve::as<int32_t>()),
                eve::convert(hi >> helper.first, eve::as<int32_t>())
            );
            return eve::add[predicate.else_(0)](
                combined & helper.second, Encoding<Sign, Bits>::min()
            );
        } else {
            // Obtain low and high components.
            // Combine before shifting and masking.
            auto [lo, hi] = unpack.template operator()<int32_t>();
            auto combined = wide_<int32_t, 16>{lo, hi};
            return eve::add[predicate.else_(0)](
                ((combined >> helper.first) & helper.second), Encoding<Sign, Bits>::min()
            );
        }
    }
}

// Temporary representation for two compressed vectors to unpack together.
template <size_t Primary, size_t Residual, size_t Extent, LVQPackingStrategy Strategy>
struct Combined {
  public:
    // Return the logical number of dimensions in this vector.
    size_t size() const {
        assert(primary_.size() == residual_.size());
        return primary_.size();
    }

  public:
    CompressedVector<Unsigned, Primary, Extent, Strategy> primary_;
    // N.B.: Purposely leave the residual using the Sequential strategy because for 8-bit
    // residuals, it doesn't seem to make much difference.
    CompressedVector<Unsigned, Residual, Extent, Sequential> residual_;
};

// Prepare for unpacking a combined vector by unpacking the primary and residual components.
template <size_t Primary, size_t Residual, size_t Extent, LVQPackingStrategy Strategy>
auto prepare_unpack(const Combined<Primary, Residual, Extent, Strategy>& x) {
    return std::make_pair(prepare_unpack(x.primary_), prepare_unpack(x.residual_));
}

// Unpack a combined vector.
//
// The combined vector is assumed to model cascaded scalar quantization.
// Let:
// * p: The integer code for the primary encoding.
// * r: The integer encoding for the residual.
// * B: The number of bits used to encode the residual.
// * d: The scaling coefficient for the primary encoding.
//
// In this case, the formula for reconstruction looks like the following (excluding the
// application of offsets, which is performed elsewhere):
//
// d * p + (d / (2^B - 1)) * r = (d / (2 ^ B - 1)) * ((2^B - 1) * p + r)
//                             = (d / (2 ^ B - 1)) * (2^B * p - p + r)
//
// This function computes the (2^B * p - p + r) portion, which can be performed entirely
// as integer arithmetic.
template <
    size_t Primary,
    size_t Residual,
    size_t Extent,
    typename Helper,
    typename Predicate = eve::ignore_none_>
wide_<int32_t, 16> unpack_as(
    const Combined<Primary, Residual, Extent, Sequential> x,
    size_t i,
    eve::as<wide_<int32_t, 16>> as,
    Helper helper,
    Predicate predicate = {}
) {
    auto primary = unpack_as(x.primary_, i, as, helper.first, predicate);
    auto residual = unpack_as(x.residual_, i, as, helper.second, predicate);
    return (primary << Residual) - primary + residual;
}

/////
///// Turbo Implementation.
/////

// 8-bit Intel(R) AVX-512 turbo unpacking.
template <
    typename Sign,
    size_t Extent,
    typename Op,
    typename Init,
    typename Combine,
    typename Reduce>
SVS_FORCE_INLINE auto for_each_slice(
    CompressedVector<Sign, 8, Extent, Turbo<16, 4>> v,
    Op&& op,
    Init&& init,
    Combine&& combine,
    Reduce&& reduce
) {
    using CV = decltype(v);
    using turbo_type = typename CV::strategy;

    // Determine how many blocks we need to iterate over.
    constexpr size_t block_size = turbo_type::block_size;

    // Ensure that each block occupies a cache-line worth of space.
    static_assert(turbo_type::compute_bytes(CV::bits, block_size) == 64);

    // Precompute auxiliary values.
    constexpr size_t shift = 8;
    auto mask = wide_<int32_t, 16>(0xff);

    size_t sz = v.size();
    size_t num_blocks = sz / block_size;
    size_t remaining = sz - num_blocks * block_size;

    const auto* compressed_base = reinterpret_cast<const int32_t*>(v.data());

    auto a0 = init();
    auto a1 = init();
    auto a2 = init();
    auto a3 = init();

    size_t lane = 0;
    for (size_t block = 0; block < num_blocks; ++block) {
        // Load the entire block.
        const auto* ptr = compressed_base + turbo_type::lanes * block;
        auto packed_data = eve::load(ptr, eve::as<wide_<int32_t, 16>>());

        a0 = op(a0, lane, packed_data & mask, eve::ignore_none);
        ++lane;

        a1 = op(a1, lane, (packed_data >> 8) & mask, eve::ignore_none);
        ++lane;

        a2 = op(a2, lane, (packed_data >> 16) & mask, eve::ignore_none);
        ++lane;

        a3 = op(a3, lane, (packed_data >> 24) & mask, eve::ignore_none);
        ++lane;
    }

    a0 = combine(combine(a0, a1), combine(a2, a3));

    // Main loop has been completed.
    // When working with the remaining elements - we perform as many standard iterations as
    // possible until we need to mask the final operation.
    if (remaining != 0) {
        size_t full_lanes = remaining / turbo_type::lanes;
        const auto* ptr = compressed_base + turbo_type::lanes * num_blocks;
        auto packed_data = eve::load(ptr, eve::as<wide_<int32_t, 16>>());

        // Unroll the tail iterations.
        for (size_t i = 0; i < full_lanes; ++i) {
            a0 = op(a0, lane, packed_data & mask, eve::ignore_none);
            packed_data >>= shift;
            ++lane;
        };

        // Get the very last elements.
        size_t final_remaining = remaining - turbo_type::lanes * full_lanes;
        if (final_remaining != 0) {
            a0 = op(a0, lane, packed_data & mask, eve::keep_first(final_remaining));
        }
    }
    return reduce(a0);
}

// 4-bit Intel(R) AVX-512 turbo unpacking.
template <
    typename Sign,
    size_t Extent,
    typename Op,
    typename Init,
    typename Combine,
    typename Reduce>
SVS_FORCE_INLINE auto for_each_slice(
    CompressedVector<Sign, 4, Extent, Turbo<16, 8>> v,
    Op&& op,
    Init&& init,
    Combine&& combine,
    Reduce&& reduce
) {
    using CV = decltype(v);
    using turbo_type = typename CV::strategy;

    // Determine how many blocks we need to iterate over.
    constexpr size_t block_size = turbo_type::block_size;

    // Ensure that each block occupies a cache-line worth of space.
    static_assert(turbo_type::compute_bytes(CV::bits, block_size) == 64);

    // Precompute auxiliary values.
    constexpr size_t shift = 4;
    // auto shift = wide_<int32_t, 16>(4);
    auto mask = wide_<int32_t, 16>(0xf);

    size_t sz = v.size();
    size_t num_blocks = sz / block_size;
    size_t remaining = sz - num_blocks * block_size;

    const auto* compressed_base = reinterpret_cast<const int32_t*>(v.data());

    auto a0 = init();
    auto a1 = init();
    auto a2 = init();
    auto a3 = init();

    size_t lane = 0;
    for (size_t block = 0; block < num_blocks; ++block) {
        // Load the entire block.
        const auto* ptr = compressed_base + turbo_type::lanes * block;
        auto packed_data = eve::load(ptr, eve::as<wide_<int32_t, 16>>());

        // Manually unroll 8 iterations.
        a0 = op(a0, lane, packed_data & mask, eve::ignore_none);
        ++lane;

        a1 = op(a1, lane, (packed_data >> 4) & mask, eve::ignore_none);
        ++lane;

        a2 = op(a2, lane, (packed_data >> 8) & mask, eve::ignore_none);
        ++lane;

        a3 = op(a3, lane, (packed_data >> 12) & mask, eve::ignore_none);
        ++lane;

        a0 = op(a0, lane, (packed_data >> 16) & mask, eve::ignore_none);
        ++lane;

        a1 = op(a1, lane, (packed_data >> 20) & mask, eve::ignore_none);
        ++lane;

        a2 = op(a2, lane, (packed_data >> 24) & mask, eve::ignore_none);
        ++lane;

        a3 = op(a3, lane, (packed_data >> 28) & mask, eve::ignore_none);
        ++lane;
    }

    a0 = combine(combine(a0, a1), combine(a2, a3));

    // Main loop has been completed.
    // When working with the remaining elements - we perform as many standard iterations as
    // possible until we need to mask the final operation.
    if (remaining != 0) {
        size_t full_lanes = remaining / turbo_type::lanes;
        const auto* ptr = compressed_base + turbo_type::lanes * num_blocks;
        auto packed_data = eve::load(ptr, eve::as<wide_<int32_t, 16>>());

        // Unroll the tail iterations.
        for (size_t i = 0; i < full_lanes; ++i) {
            a0 = op(a0, lane, packed_data & mask, eve::ignore_none);
            packed_data >>= shift;
            ++lane;
        };

        // Get the very last elements.
        size_t final_remaining = remaining - turbo_type::lanes * full_lanes;
        if (final_remaining != 0) {
            a0 = op(a0, lane, packed_data & mask, eve::keep_first(final_remaining));
        }
    }
    return reduce(a0);
}

// Combined unpacking.
template <size_t Extent, typename Op, typename Init, typename Combine, typename Reduce>
SVS_FORCE_INLINE auto for_each_slice(
    Combined<4, 8, Extent, Turbo<16, 8>> c,
    Op&& op,
    Init&& init,
    Combine&& combine,
    Reduce&& reduce
) {
    auto p = c.primary_;
    auto r = c.residual_;
    auto helper = prepare_unpack(r);

    // Wrap `op` to mix-in the residual.
    return for_each_slice(
        p,
        [helper, r, &op](auto accum, size_t lane, wide_<int32_t, 16> primary, auto pred) {
            auto res = unpack_as(r, lane, eve::as<wide_<int32_t, 16>>(), helper, pred);
            return op(accum, lane, (primary << 8) - primary + res, pred);
        },
        SVS_FWD(init),
        SVS_FWD(combine),
        SVS_FWD(reduce)
    );
}

/////
///// SIMD accelerated bulk decompression.
/////

/// Use SIMD acceleration to unpack the compressed vector to the destination.
template <typename Sign, size_t Bits, size_t Extent, LVQPackingStrategy Strategy>
void unpack(
    std::vector<typename CompressedVector<Sign, Bits, Extent, Strategy>::value_type>& v,
    CompressedVector<Sign, Bits, Extent, Strategy> cv
) {
    v.resize(cv.size());
    unpack(std::span(v.data(), v.size()), cv);
}

/// Sequential Strategy
template <std::integral I, typename Sign, size_t Bits, size_t Extent>
void unpack(std::span<I> v, CompressedVector<Sign, Bits, Extent, Sequential> cv) {
    assert(v.size() == cv.size());
    const size_t simd_width = 16;
    using int_wide_t = wide_<int32_t, simd_width>;

    auto helper = prepare_unpack(cv);
    I* base = v.data();
    size_t iterations = cv.size() / simd_width;
    size_t tail = cv.size() % simd_width;

    // Main iterations
    for (size_t i = 0; i < iterations; ++i) {
        auto u = unpack_as(cv, i, eve::as<int_wide_t>(), helper);
        eve::store(eve::convert(u, eve::as<I>()), base);
        base += simd_width;
    }

    // Tail Iterations
    if (tail != 0) {
        auto predicate = eve::keep_first(lib::narrow<int64_t>(tail));
        auto u = unpack_as(cv, iterations, eve::as<int_wide_t>(), helper, predicate);
        eve::store[predicate](eve::convert(u, eve::as<I>()), base);
    }
}

/// Turbo Strategy.
namespace detail {
// CV can be either a turbo CompressedVector, or a turbo CombinedVector.
template <std::integral I, typename CV>
void unpack_turbo(std::span<I> dst, const CV& compressed_like) {
    assert(dst.size() == compressed_like.size());

    // Store the unpacked elements into the destination buffer.
    auto op = [dst](detail::Empty, size_t lane, wide_<int32_t, 16> unpacked, auto pred) {
        eve::store[pred](eve::convert(unpacked, eve::as<I>()), dst.data() + 16 * lane);
        return detail::Empty();
    };

    for_each_slice(
        compressed_like,
        op,                                                           // op
        []() { return detail::Empty(); },                             // init
        [](detail::Empty, detail::Empty) { return detail::Empty{}; }, // combine
        [](detail::Empty) {}                                          // reduce
    );
}
} // namespace detail

template <std::integral I, typename Sign, size_t Bits, size_t Extent, TurboLike T>
void unpack(std::span<I> dst, CompressedVector<Sign, Bits, Extent, T> v) {
    return detail::unpack_turbo(dst, v);
}

template <std::integral I, size_t Primary, size_t Residual, size_t Extent, TurboLike T>
void unpack(std::span<I> dst, const Combined<Primary, Residual, Extent, T>& v) {
    return detail::unpack_turbo(dst, v);
}

} // namespace svs::quantization::lvq
