/*
 * Copyright 2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// svs
#include "eve/algo.hpp"
#include "svs/core/data.h"
#include "svs/core/distance.h"
#include "svs/core/kmeans.h"
#include "svs/lib/dispatcher.h"
#include "svs/lib/meta.h"
#include "svs/lib/misc.h"
#include "svs/lib/saveload.h"

// stl
#include <memory>
#include <string>
#include <string_view>
#include <variant>

namespace svs {
namespace quantization {
namespace lvq {

/// The encoding to use for centroid selection.
using selector_t = uint8_t;
/// The storage format for LVQ constants.
using scaling_t = svs::Float16;

// Strategies for storing packed data.
struct Sequential {
    static constexpr std::string_view name();
    static constexpr size_t compute_bytes(size_t nbits, size_t length) {
        return lib::div_round_up(nbits * length, 8);
    }

    // No permutation required.
    static constexpr size_t logical_to_linear(size_t i);
    static constexpr size_t linear_to_logical(size_t i);
};

// Blockwise strategy.
template <size_t Lanes, size_t ElementsPerLane> struct Turbo {
    static constexpr std::string name();
    static constexpr size_t lanes = Lanes;
    static constexpr size_t elements_per_lane = ElementsPerLane;
    static constexpr size_t block_size = Lanes * ElementsPerLane;

    // Need to define here because it is called at compile time
    static constexpr size_t compute_bytes(size_t nbits, size_t length) {
        assert(nbits == 4 || nbits == 8);

        size_t block_size_bytes = nbits * block_size / 8;
        size_t num_blocks = lib::div_round_up(length, block_size);
        return block_size_bytes * num_blocks;
    }

    static constexpr size_t logical_to_linear(size_t i);
    static constexpr size_t linear_to_logical(size_t i);
    static constexpr size_t num_blocks(size_t count);
};

// Auxiliary struct to help with distance computations.
struct ScaleBias {
    float scale;
    float bias;
};

struct Unsigned;
struct Signed;
using DefaultStrategy = Sequential;

///
/// Allow span resizing when constructing a CompressedVector.
///
struct AllowShrinkingTag {};

/// Place holder for specializations.
template <typename T, size_t Bits> struct Encoding;

template <size_t Bits> struct Encoding<Signed, Bits> {
    Encoding() = default;

    ///
    /// Return the number of bytes required to store `length` densely packed
    /// `Bits`-sized elements.
    ///
    static constexpr size_t bytes(size_t length);

    // Type Aliases
    using value_type = int8_t;
    static constexpr size_t bits = Bits;

    static constexpr value_type max();
    static constexpr value_type min();
    static constexpr size_t absmax();

    // Internally, we convert signed values to unsigned values by adding in a bias
    // to turn values of type `min()` to zero.
    //
    // This avoids complications related to restoring the sign bit when unpacking
    // values.
    static value_type decode(uint8_t raw);

    static uint8_t encode(value_type value);

    template <std::signed_integral I> static bool check_bounds(I value);
};

template <size_t Bits> struct Encoding<Unsigned, Bits> {
    Encoding() = default;

    ///
    /// Return the number of bytes required to store `length` densly packed
    /// `Bits`-sized elements.
    ///
    static constexpr size_t bytes(size_t length);

    // Type Aliases
    using value_type = uint8_t;
    static constexpr size_t bits = Bits;

    // Helper functions.
    static constexpr value_type max();
    static constexpr value_type min();
    static constexpr size_t absmax();

    // No adjustment required for unsigned types since we mask out the upper order
    // bits anyways.
    static value_type decode(uint8_t raw);
    static uint8_t encode(uint8_t raw);

    template <std::unsigned_integral I> static bool check_bounds(I value);
};

namespace detail {

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

template <
    typename Sign,
    size_t Bits,
    size_t Extent,
    bool IsConst,
    LVQPackingStrategy Strategy>
class CompressedVectorBase {
  public:
    /// The packing strategy used by the vector.
    using strategy = Strategy;
    /// Static member describing if this is const.
    static constexpr bool is_const = IsConst;
    /// The encoding to use for this combination of sign and number of bits.
    using encoding_type = Encoding<Sign, Bits>;
    /// The smallest native integer type capable of storing the uncompressed
    /// values in the vector.
    using value_type = typename encoding_type::value_type;
    /// The number of bits used to encode each value in the compressed vector.
    static constexpr size_t bits = Bits;
    /// Return the compile-time length of the vector, or ``Dynamic`` if unknown.
    static constexpr size_t extent = Extent;
    /// The maximum encoding value.
    static constexpr value_type max();
    /// The minimum encoding value.
    static constexpr value_type min();

    /// The compile-time number of bytes used for the underlying bytes the
    /// compressed vector interprets.
    static constexpr size_t storage_extent =
        Extent == Dynamic ? Dynamic : Strategy::compute_bytes(Bits, Extent);

    /// The type of the storage span backing this vector.
    using const_span_type = std::span<const std::byte, storage_extent>;
    using mutable_span_type = std::span<std::byte, storage_extent>;
    using span_type = std::conditional_t<IsConst, const_span_type, mutable_span_type>;
    using pointer = std::conditional_t<IsConst, const std::byte*, std::byte*>;

    static constexpr size_t compute_bytes()
        requires(Extent != Dynamic);

    static constexpr size_t compute_bytes(lib::MaybeStatic<Extent> sz);

    // Disable default construction.
    CompressedVectorBase() = delete;

    ///
    /// Construct a `CompressedVector` over the contents of `data`.
    /// This is a view and will not take ownership nor extent the lifetime of
    /// `data`.
    ///
    /// Only valid if ``Extent != Dynamic``.
    ///
    explicit CompressedVectorBase(span_type data)
        requires(Extent != Dynamic);

    ///
    /// Construct a ``CompressedVector`` with the given size over the contents of
    /// `data`. This is a view and will not take ownership nor extent the lifetime
    /// of `data`.
    ///
    explicit CompressedVectorBase(lib::MaybeStatic<Extent> size, span_type data);

    ///
    /// Construct a CompressedVector from potentially oversized span.
    ///
    /// @param tag Indicate that it is okay to use a subset of the provided span.
    /// @param size The requested number of elements in the compressed view.
    /// @param source The source span to construct a view over.
    ///
    /// Construct a CompressedVector view over the given span.
    /// If necessary, the constructed view will be over only a subset of the
    /// provided span. If this is the case, the subset will begin at the start of
    /// ``source``.
    ///
    template <typename T, size_t N>
    explicit CompressedVectorBase(
        AllowShrinkingTag SVS_UNUSED(tag),
        lib::MaybeStatic<Extent> size,
        std::span<T, N> source
    );

    ///
    /// Allow conversion to const from non-const.
    ///
    operator CompressedVectorBase<Sign, Bits, Extent, true, Strategy>() const;

    ///
    /// Convert to the const version of the vector.
    ///
    CompressedVectorBase<Sign, Bits, Extent, true, Strategy> as_const() const;

    ///
    /// Return the length of the compressed vector.
    ///
    constexpr size_t size() const;

    ///
    /// Return a constant pointer to the start of the underlying storage.
    ///
    const std::byte* data() const;

    ///
    /// Return a mutable pointer to the start of the underlying storage.
    ///
    std::byte* data()
        requires(!IsConst);

    ///
    /// Return the size in bytes of the underlying storage.
    ///
    constexpr size_t size_bytes() const;

    ///
    /// Return the uncompressed value at index `i`.
    /// **Preconditions:**
    ///
    /// * `0 <= i < size()`
    ///
    value_type get(size_t i) const;

    template <typename T>
    void set(T v, size_t i)
        requires(!IsConst);

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
        requires(!IsConst);

    ///
    /// @brief Copy to contents of another compressed vector view.
    ///
    /// Requires that the other CompressedVectorBase has the same run-time
    /// dimensions.
    ///
    template <size_t OtherExtent, bool OtherConst>
        requires(!IsConst)
    void copy_from(
        const CompressedVectorBase<Sign, Bits, OtherExtent, OtherConst, Strategy>& other
    );

    ///
    /// @brief Assign the contents of ``other`` to the compressed vector.
    ///
    /// Requires that each element of ``other`` can be losslessly converted to an
    /// integer in the span ``[Encoding::min(), Encoding::max()]``.
    ///
    template <typename I, typename Alloc>
        requires(!IsConst)
    void copy_from(const std::vector<I, Alloc>& other);

    ///
    /// Safely extract a value of type `T` beginning at byte `i`.
    /// Allow caller to specify the number of bytes to help with Intel(R) AVX
    /// decoding.
    ///
    template <typename T> T extract(size_t i) const;

    /// Extract a value by copying the specified bytes beginning at offset ``i``.
    ///
    /// This function behaves as if constructing a value of type ``T`` with an
    /// all-zero bit representation and performing a ``std::memcpy`` of the
    /// specified bytes into the representation of ``T``.
    ///
    /// In other words, copy only the requested bytes and zero pad the rest of
    /// ``T``.
    ///
    /// This function is safe to use in contexts where generating a read of
    /// ``sizeof(T)`` would result in an out-of-bounds access.
    ///
    /// Prerequisites:
    /// * ``T`` is constexpr default constructible AND constructible from
    /// ``int(0)`` with
    ///   bit representation of the returned object consisting of zeroed memory.
    /// * ``T`` is trivially copyable.
    /// * ``bytes <= sizeof(T)``
    /// * ``0 < bytes``: At least one byte must be read and that read must be
    /// inbounds.
    /// @tparam T The type to extract.
    /// @tparam Static Whether this is being called in a static dimensional
    /// context or not.
    ///     This is a hint and should not affect the read value.
    template <typename T, bool Static = (Extent != svs::Dynamic)>
    SVS_FORCE_INLINE T extract_subset(size_t i, uint64_t bytes) const;

    ///
    /// Insert a value of type `T` into the underlying bytes, starting at byte
    /// `i`.
    ///
    template <typename T>
    void insert(T v, size_t i)
        requires(!IsConst);

    ///
    /// Perform any necessary steps to convert a raw extracted, unsigned, zero
    /// padded byte to the `value_type` of the encoder.
    ///
    static value_type decode(uint8_t value);

    ///
    /// Encode a `value_type` to an unsigned byte suitable for encoding in
    /// `bits` nubmer of bits.
    ///
    static uint8_t encode(value_type value);

    ///
    /// Logic Equivalence.
    ///
    /// Two vectors are logically equivalent if they contain the same number of
    /// dimensions and the encoding for each dimension
    ///
    template <size_t E2, bool C2, LVQPackingStrategy S2>
    bool logically_equivalent_to(const CompressedVectorBase<Sign, Bits, E2, C2, S2>& other
    ) const;

    // Make the mutable version a friend to the constant version to enable direct
    // construction when implicitly converting to const.
    friend class CompressedVectorBase<Sign, Bits, Extent, false, Strategy>;

  private:
    // Private constructor from a raw-pointer.
    CompressedVectorBase(pointer data, lib::MaybeStatic<Extent> size);

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

template <size_t Bits, size_t Extent, typename Strategy = DefaultStrategy>
struct ScaledBiasedVector {
  public:
    using strategy = Strategy;
    using scalar_type = lvq::scaling_t;
    using auxiliary_type = ScaleBias;
    using vector_type = CompressedVector<Unsigned, Bits, Extent, Strategy>;
    using mutable_vector_type = MutableCompressedVector<Unsigned, Bits, Extent, Strategy>;

    static constexpr size_t extent = Extent;

    // Construct from constant `CompressedVector`.
    template <Arithmetic T>
    ScaledBiasedVector(T scale, T bias, selector_t selector, vector_type data);

    // Construct from `MutableCompressedVector`.
    template <Arithmetic T>
    ScaledBiasedVector(T scale, T bias, selector_t selector, mutable_vector_type data);

    float get(size_t i) const;
    size_t size() const;
    selector_t get_selector() const;
    float get_scale() const;
    float get_bias() const;
    vector_type vector() const;

    const std::byte* pointer() const;
    constexpr size_t size_bytes() const;

    // Distance computation helpers.
    auxiliary_type prepare_aux() const;

    // Logical equivalence.
    template <size_t E2, LVQPackingStrategy OtherStrategy>
    bool logically_equivalent_to(const ScaledBiasedVector<Bits, E2, OtherStrategy>& other
    ) const;

    ///// Members
    // The vector-wise scaling constant.
    scalar_type scale;
    // The vector-wise offset.
    scalar_type bias;
    // Memory span for compressed data.
    vector_type data;
    // The index of the centroid this vector belongs to.
    selector_t selector;
};

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
    // N.B.: Purposely leave the residual using the Sequential strategy because
    // for 8-bit residuals, it doesn't seem to make much difference.
    CompressedVector<Unsigned, Residual, Extent, Sequential> residual_;
};

template <size_t Primary, size_t Residual, size_t N, typename Strategy = DefaultStrategy>
struct ScaledBiasedWithResidual {
  public:
    using strategy = Strategy;
    using auxiliary_type = ScaleBias;
    using strategy_type = Strategy;

    /// Return the decoded value at index `i` using both the primary and residual
    /// encodings.
    float get(size_t i) const;

    /// Return the number of elements in the vector.
    size_t size() const;

    /// Return the centroid selector.
    selector_t get_selector() const;

    float get_scale() const;
    float get_bias() const;

    /// Prepare for distance computations.
    auxiliary_type prepare_aux() const;

    Combined<Primary, Residual, N, Strategy> vector() const;

    template <size_t N2, LVQPackingStrategy OtherStrategy>
    bool logically_equivalent_to(
        const ScaledBiasedWithResidual<Primary, Residual, N2, OtherStrategy>& other
    ) const;

    ///// Members
    // For now - only the primary vector is allowed to have variable strategy.
    // The residual is always kept as sequential due to implementation challenges
    // and somewhat dubious ROI.
    ScaledBiasedVector<Primary, N, Strategy> primary_;
    CompressedVector<Unsigned, Residual, N, Sequential> residual_;
};

namespace detail {

// Extendable trait to overload common entry-points for LVQ vectors.
template <typename T> inline constexpr bool lvq_compressed_vector_v = false;

template <size_t Bits, size_t Extent, typename Strategy>
inline constexpr bool
    lvq_compressed_vector_v<lvq::ScaledBiasedVector<Bits, Extent, Strategy>> = true;

template <size_t Primary, size_t Residual, size_t N, typename Strategy>
inline constexpr bool
    lvq_compressed_vector_v<lvq::ScaledBiasedWithResidual<Primary, Residual, N, Strategy>> =
        true;
} // namespace detail

// Dispatch concept for LVQCompressedVectors
template <typename T>
concept LVQCompressedVector = detail::lvq_compressed_vector_v<T>;

class Decompressor {
  public:
    Decompressor() = delete;
    Decompressor(std::shared_ptr<const data::SimpleData<float>>&& centroids);

    template <LVQCompressedVector T> std::span<const float> operator()(const T& compressed);

  private:
    std::shared_ptr<const data::SimpleData<float>> centroids_;
    std::vector<float> buffer_ = {};
};

// declarations for BiasedDistance

///// Static dispatch traits.

class EuclideanBiased {
  public:
    using compare = std::less<>;
    // Biased versions are not implicitly broadcastable because they must maintain
    // per-query state.
    static constexpr bool implicit_broadcast = false;
    static constexpr bool must_fix_argument = true;

    // Constructors
    EuclideanBiased(const std::shared_ptr<const data::SimpleData<float>>& centroids);

    EuclideanBiased(std::shared_ptr<const data::SimpleData<float>>&& centroids);

    EuclideanBiased(const std::vector<float>& centroid);

    // Shallow Copy
    // Don't preserve the state of `processed_query_`.
    EuclideanBiased shallow_copy() const;

    ///
    /// Subtract each centroid from the query and store the result in
    /// `processed_query_`. This essentially moves the query by the same amount as
    /// the original data point, preserving L2 distance.
    ///
    template <typename T> void fix_argument(const std::span<T>& query) {
        // Check pre-conditions.
        assert(centroids_->dimensions() == query.size());

        // Convert the query to float
        auto query_fp32 = data::SimpleData<float>(1, query.size());
        query_fp32.set_datum(0, query);

        // Component-wise add the bias to the query and cache the result.
        auto jmax = query.size();
        for (size_t i = 0, imax = centroids_->size(); i < imax; ++i) {
            const auto& centroid = centroids_->get_datum(i);
            auto dst = processed_query_.get_datum(i);
            for (size_t j = 0; j < jmax; ++j) {
                dst[j] = query_fp32.get_datum(0)[j] - centroid[j];
            }
        }
    }

    // For testing purposes.
    template <typename T, std::integral I = size_t>
    float compute(const T& y, I selector = 0) const
        requires(lib::is_spanlike_v<T>);

    ///
    /// Compute the Euclidean difference between a quantized vector `y` and a
    /// cached shifted query.
    ///
    template <LVQCompressedVector T> float compute(const T& y) const;

    std::span<const float> view_query(size_t i) const;

    ///
    /// Return the global bias as a `std::span`.
    ///
    data::ConstSimpleDataView<float> view_bias() const;

    std::span<const float> get_centroid(size_t i) const;

  private:
    data::SimpleData<float> processed_query_;
    std::shared_ptr<const data::SimpleData<float>> centroids_;
};

class InnerProductBiased {
  public:
    using compare = std::greater<>;
    // Biased versions are not implicitly broadcastable because they must maintain
    // per-query state.
    static constexpr bool implicit_broadcast = false;
    static constexpr bool must_fix_argument = true;

    // Constructor
    InnerProductBiased(const std::shared_ptr<const data::SimpleData<float>>& centroids);

    InnerProductBiased(std::shared_ptr<const data::SimpleData<float>>&& centroids);

    InnerProductBiased(const std::vector<float>& centroid);

    // Shallow Copy
    InnerProductBiased shallow_copy() const;

    ///
    /// Precompute the inner product between the query and the global bias.
    /// This pre-computed value will be added to the result of standard distance
    /// computations using the distributive property where
    /// ```
    /// q . (x + b) == (q . x) + (q . b)
    /// ```
    ///
    template <typename T> void fix_argument(const std::span<T>& query) {
        // Check pre-conditions.
        assert(centroids_->dimensions() == query.size());
        assert(processed_query_.size() == centroids_->size());
        query_fp32_.set_datum(0, query);

        const auto query_fp32 = query_fp32_.get_datum(0);
        // Pre-compute the inner-product between the query and each centroid.
        distance::DistanceIP inner_distance{};
        for (size_t i = 0, imax = centroids_->size(); i < imax; ++i) {
            processed_query_[i] =
                distance::compute(inner_distance, query_fp32, centroids_->get_datum(i));
        }

        // This preprocessing needed for DistanceFastIP
        query_sum_ = eve::algo::reduce(query_fp32, 0.0f);
    }

    template <typename T, std::integral I = size_t>
    float compute(const T& y, I selector = 0) const
        requires(lib::is_spanlike_v<T>);

    template <LVQCompressedVector T> float compute(const T& y) const;

    std::span<const float> view_query() const;

    ///
    /// Return the global bias as a `std::span`.
    ///
    data::ConstSimpleDataView<float> view_bias() const;

    std::span<const float> get_centroid(size_t i) const;

  private:
    // Store fp32 version of the query.
    data::SimpleData<float> query_fp32_;
    // The results of computing the inner product between each centroid and the
    // query. Applied after the distance computation between the query and
    // compressed vector.
    std::vector<float> processed_query_;
    std::shared_ptr<const data::SimpleData<float>> centroids_;
    float query_sum_ = 0;
};

template <typename Distance> class DecompressionAdaptor {
  public:
    using distance_type = Distance;
    using compare = distance::compare_t<distance_type>;
    static constexpr bool implicit_broadcast = false;
    static constexpr bool must_fix_argument = true;

    DecompressionAdaptor(const distance_type& inner, size_t size_hint = 0);

    DecompressionAdaptor(distance_type&& inner, size_t size_hint = 0);

    ///
    /// @brief Construct the internal portion of DecompressionAdaptor directly.
    ///
    /// The goal of the decompression adaptor is to wrap around an inner distance
    /// functor and decompress the left-hand component when requested, forwarding
    /// the decompressed value to the inner functor upon future distance
    /// computations.
    ///
    /// The inner distance functor may have non-trivial state associated with it.
    /// This constructor allows to construction of that inner functor directly to
    /// avoid a copy or move constructor.
    ///
    template <typename... Args>
    DecompressionAdaptor(std::in_place_t SVS_UNUSED(tag), Args&&... args);

    DecompressionAdaptor shallow_copy() const;

    // Distance API.
    template <LVQCompressedVector Left> void fix_argument(Left left);

    template <LVQCompressedVector Right> float compute(const Right& right) const;

    std::span<const float> view() const;

  private:
    distance_type inner_;
    std::vector<float> decompressed_;
};

namespace detail {
// Trait to determine if an allocator is blocked or not.
// Used to SFINAE away resizing methods if the allocator is not blocked.
template <typename A> inline constexpr bool is_blocked = false;
template <typename A> inline constexpr bool is_blocked<data::Blocked<A>> = true;

} // namespace detail

template <typename A>
concept is_resizeable = detail::is_blocked<A>;

namespace detail {

// Map from baseline distance functors to the local versions.
template <typename T> struct BiasedDistance;

template <> struct BiasedDistance<distance::DistanceL2> {
    using type = EuclideanBiased;
};

template <> struct BiasedDistance<distance::DistanceIP> {
    using type = InnerProductBiased;
};

} // namespace detail

///
/// Compute the correct biased distance function to operate on compressed data
/// given the original distance function `T`.
///
template <typename T> using biased_distance_t = typename detail::BiasedDistance<T>::type;

namespace detail {
// This type alias is shared between both one-level and two-level LVQ datasets.
using centroid_type = data::SimpleData<float, Dynamic>;
} // namespace detail

// Schemas are independent of most type parameters.
// Hoist them as stand-alone variables to they are accessible to the auto load
// matchers as well.
inline constexpr std::string_view one_level_serialization_schema = "one_level_lvq_dataset";
inline constexpr lib::Version one_level_save_version = lib::Version(0, 0, 2);
inline constexpr std::string_view two_level_serialization_schema = "two_level_lvq_dataset";
inline constexpr lib::Version two_level_save_version = lib::Version(0, 0, 3);

template <size_t Bits, size_t Extent, typename Strategy> class ScaledBiasedVectorLayout {
  public:
    using cv_type = CompressedVector<Unsigned, Bits, Extent, Strategy>;

    using const_value_type = ScaledBiasedVector<Bits, Extent, Strategy>;
    using scalar_type = typename const_value_type::scalar_type;

    explicit ScaledBiasedVectorLayout(lib::MaybeStatic<Extent> dims);

    constexpr size_t total_bytes() const;

    lib::MaybeStatic<Extent> static_size() const;
    size_t size() const;

    template <typename T, size_t N>
        requires(std::is_same_v<std::remove_cv_t<T>, std::byte>)
    CompressedVectorBase<Unsigned, Bits, Extent, std::is_const_v<T>, Strategy> vector(
        std::span<T, N> raw_data
    ) const;

    ///
    /// get.
    ///
    template <size_t N> const_value_type get(std::span<const std::byte, N> raw_data) const;

    ///
    /// set.
    ///
    template <size_t N, std::integral I>
    void
    set(std::span<std::byte, N> raw_data,
        float scale,
        float bias,
        selector_t selector,
        const std::vector<I>& src) const;

    template <size_t N>
    void set(std::span<std::byte, N> raw_data, const const_value_type& src) const;

  private:
    [[no_unique_address]] lib::MaybeStatic<Extent> dims_;
};

enum class DatasetSchema { Compressed, ScaledBiased };
///
/// Support for deduction.
///
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
    static constexpr bool is_blocked = detail::is_blocked<Alloc>;
    using strategy = Strategy;
    using helper_type = ScaledBiasedVectorLayout<Bits, Extent, Strategy>;
    using allocator_type = Alloc;

    using compressed_vector_type = CompressedVector<Unsigned, Bits, Extent, Strategy>;
    using Encoded_value_type = typename compressed_vector_type::value_type;
    // Pad data extent to be a multiple of half the underlying cache line size for
    // better bandwidth characteristics.
    static constexpr size_t encoding_bits = Bits;
    using dataset_type = data::SimpleData<std::byte, Dynamic, allocator_type>;

    static constexpr size_t compressed_vector_extent =
        compressed_vector_type::storage_extent;

    static constexpr size_t
    compute_data_dimensions(const helper_type& layout, size_t alignment = 0);

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
    );

    ScaledBiasedDataset(
        size_t size, lib::MaybeStatic<Extent> dims = {}, size_t alignment = 0
    );

    ScaledBiasedDataset(
        dataset_type data, size_t alignment, lib::MaybeStatic<Extent> dims = {}
    );

    size_t get_alignment() const;

    const allocator_type& get_allocator() const;

    ///// Dataset Inteface
    // N.B.: ScaledBiasedVector is immutable.
    using value_type = ScaledBiasedVector<Bits, Extent, Strategy>;
    using const_value_type = ScaledBiasedVector<Bits, Extent, Strategy>;
    using scalar_type = typename value_type::scalar_type;

    size_t size() const;
    lib::MaybeStatic<Extent> static_dims() const;
    size_t dimensions() const;
    void prefetch(size_t i) const;

    const_value_type get_datum(size_t i) const;

    template <std::integral I>
    void set_datum(
        size_t i, float scale, float bias, selector_t selector, const std::vector<I>& data
    );

    void set_datum(size_t i, const value_type& data);

    ///// Resizing
    void resize(size_t new_size)
        requires is_resizeable<Alloc>;

    ///// Compaction
    // Use perfect forwarding to the compacting algorithm of the backing buffer.
    template <typename... Args>
    void compact(Args&&... args)
        requires is_resizeable<Alloc>;

    /////
    ///// Saving and Loading.
    /////

    static constexpr std::string_view kind = "scaled biased compressed dataset";

    // Version History
    // v0.0.1 - Unknown Change.
    // v0.0.2 - BREAKING
    //   - Removed centroids from being stored with the
    //   ScaledBiasedCompressedDataset.
    //     Centroids are now stored in the higher level LVQ dataset.
    // v0.0.3 - BREAKING
    //   - Canonicalize the layout of serialized LVQ to be sequential with no
    //   padding.
    //     This allows different packing strategies and paddings to be used upon
    //     reload.
    static constexpr lib::Version save_version =
        get_current_version(DatasetSchema::ScaledBiased);
    static constexpr std::string_view serialization_schema =
        get_schema(DatasetSchema::ScaledBiased);

    lib::SaveTable save(const lib::SaveContext& ctx) const;

    static ScaledBiasedDataset load(
        const lib::LoadTable& table,
        size_t alignment = 0,
        const allocator_type& allocator = {}
    );

  private:
    [[no_unique_address]] helper_type layout_helper_;
    size_t alignment_;
    dataset_type data_;
};

template <
    typename Sign,
    size_t Bits,
    size_t Extent,
    typename Alloc = lib::Allocator<std::byte>>
class CompressedDataset {
  public:
    static constexpr bool is_blocked = detail::is_blocked<Alloc>;
    using allocator_type = Alloc;

    ///
    /// The number of bits used for this encoding.
    ///
    static constexpr size_t encoding_bits = Bits;

    /// Dataset type aliases
    using value_type = MutableCompressedVector<Sign, Bits, Extent, Sequential>;
    using const_value_type = CompressedVector<Sign, Bits, Extent, Sequential>;

    ///
    /// The compile-time dimensionality of the raw byte storage backing the
    /// compressed data.
    ///
    using dataset_type = data::SimpleData<std::byte, Dynamic, allocator_type>;

    static size_t total_bytes(lib::MaybeStatic<Extent> dims);

    ///
    /// Allocate an empty dataset.
    ///
    CompressedDataset(
        size_t size, lib::MaybeStatic<Extent> dims, const allocator_type& allocator
    );

    CompressedDataset(size_t size, lib::MaybeStatic<Extent> dims = {});

    CompressedDataset(dataset_type data, lib::MaybeStatic<Extent> dims = {});

    ///// Dataset Inteface

    size_t size() const;
    lib::MaybeStatic<Extent> static_dims() const;
    size_t dimensions() const;
    void prefetch(size_t i) const;
    const allocator_type& get_allocator() const;

    value_type get_datum(size_t i);

    const_value_type get_datum(size_t i) const;

    template <std::integral I> void set_datum(size_t i, const std::vector<I>& data);

    void set_datum(size_t i, const const_value_type& data);

    ///// Resizing
    void resize(size_t new_size)
        requires is_resizeable<Alloc>;

    ///// Compaction
    // Use perfect forwarding to the compacting algorithm of the backing buffer.
    template <typename... Args>
    void compact(Args&&... args)
        requires is_resizeable<Alloc>;

    /////
    ///// Saving and Loading.
    /////

    static constexpr std::string_view kind = "compressed dataset";
    static constexpr std::string_view serialization_schema =
        get_schema(DatasetSchema::Compressed);
    static constexpr lib::Version save_version =
        get_current_version(DatasetSchema::Compressed);

    lib::SaveTable save(const lib::SaveContext& ctx) const;

    static CompressedDataset
    load(const lib::LoadTable& table, const allocator_type& allocator = {});

  private:
    [[no_unique_address]] lib::MaybeStatic<Extent> dims_;
    dataset_type data_;
};

template <
    size_t Primary,
    size_t Residual = 0,
    size_t Extent = Dynamic,
    LVQPackingStrategy Strategy = Sequential,
    typename Alloc = lib::Allocator<std::byte>>
class LVQDataset {
  public:
    constexpr static size_t primary_bits = Primary;
    constexpr static size_t residual_bits = Residual;
    constexpr static size_t extent = Extent;
    static constexpr bool is_blocked = detail::is_blocked<Alloc>;
    using strategy = Strategy;
    using primary_type = ScaledBiasedDataset<Primary, Extent, Strategy, Alloc>;
    using residual_type = CompressedDataset<Unsigned, Residual, Extent, Alloc>;
    using centroid_type = detail::centroid_type;
    using allocator_type = Alloc;

  private:
    // Class invariants:
    //
    // * primary_.size() == residual_.size();
    // * primary_.dimensions() == residual_.dimensiosn();
    primary_type primary_;
    residual_type residual_;
    std::shared_ptr<centroid_type> centroids_;

  public:
    using const_primary_value_type = ScaledBiasedVector<Primary, Extent, Strategy>;

    using const_value_type = ScaledBiasedWithResidual<Primary, Residual, Extent, Strategy>;
    using value_type = const_value_type;

    LVQDataset(size_t size, lib::MaybeStatic<Extent> dims, size_t alignment = 0);

    LVQDataset(
        primary_type primary, residual_type residual, const centroid_type& centroids
    );

    template <size_t OtherExtent>
    void
    reproducibility_set_centroids(data::ConstSimpleDataView<float, OtherExtent> centroids);

    /// @brief Return the alignment of the primary dataset.
    size_t primary_dataset_alignment() const;

    size_t size() const;
    size_t dimensions() const;

    ///
    /// @brief Access both levels of the two-level dataset.
    ///
    /// Return a type that lazily combines the primary and residual.
    ///
    const_value_type get_datum(size_t i) const;

    /// @brief Prefetch data in the first and second level datasets.
    void prefetch(size_t i) const;

    /// @brief Access only the first level of the dataset.
    const_primary_value_type get_primary(size_t i) const;

    /// @brief Prefetch only the primary dataset.
    void prefetch_primary(size_t i) const;

    void resize(size_t new_size)
        requires is_resizeable<Alloc>;

    template <std::integral I, threads::ThreadPool Pool>
    void
    compact(std::span<const I> new_to_old, Pool& threadpool, size_t batchsize = 1'000'000)
        requires is_resizeable<Alloc>;

    std::shared_ptr<const centroid_type> view_centroids() const;
    std::span<const float> get_centroid(size_t i) const;

    template <typename QueryType, size_t N>
    void set_datum(size_t i, std::span<QueryType, N> datum, size_t centroid_selector);

    template <typename QueryType, size_t N>
    void set_datum(size_t i, std::span<QueryType, N> datum);

    Decompressor decompressor() const;

    template <data::ImmutableMemoryDataset Dataset>
    static LVQDataset compress(const Dataset& data, const allocator_type& allocator = {});

    template <data::ImmutableMemoryDataset Dataset>
    static LVQDataset compress(
        const Dataset& data,
        size_t num_threads,
        size_t alignment,
        const allocator_type& allocator = {}
    );

    template <data::ImmutableMemoryDataset Dataset, threads::ThreadPool Pool>
    static LVQDataset compress(
        const Dataset& data,
        Pool& threadpool,
        size_t alignment,
        const allocator_type& allocator = {}
    );

    static constexpr lib::Version save_version = two_level_save_version;
    static constexpr std::string_view serialization_schema = two_level_serialization_schema;
    lib::SaveTable save(const lib::SaveContext& ctx) const;

    static LVQDataset load(
        const lib::LoadTable& table,
        size_t alignment = 0,
        const allocator_type& allocator = {}
    );
};

// Specialize one-level LVQ
template <size_t Primary, size_t Extent, LVQPackingStrategy Strategy, typename Alloc>
class LVQDataset<Primary, 0, Extent, Strategy, Alloc> {
  public:
    constexpr static size_t primary_bits = Primary;
    constexpr static size_t residual_bits = 0;
    constexpr static size_t extent = Extent;

    static constexpr bool is_blocked = detail::is_blocked<Alloc>;
    using strategy = Strategy;
    using allocator_type = Alloc;
    using primary_type = ScaledBiasedDataset<Primary, Extent, Strategy, allocator_type>;
    using centroid_type = detail::centroid_type;

  private:
    primary_type primary_;
    std::shared_ptr<centroid_type> centroids_;

    // Methods
  public:
    using value_type = ScaledBiasedVector<Primary, Extent, Strategy>;
    using const_value_type = ScaledBiasedVector<Primary, Extent, Strategy>;

    ///// Constructors
    LVQDataset(size_t size, lib::MaybeStatic<Extent> dims, size_t alignment = 0);

    LVQDataset(primary_type primary, const centroid_type& centroids);

    template <size_t OtherExtent>
    void
    reproducibility_set_centroids(data::ConstSimpleDataView<float, OtherExtent> centroids);

    /// @brief Return the alignment of the primary dataset.
    size_t primary_dataset_alignment() const;

    // Dataset API
    size_t size() const;
    size_t dimensions() const;

    const primary_type& get_primary_dataset() const;
    const allocator_type& get_allocator() const;

    ///
    /// @brief Return the stored data at position `i`.
    ///
    /// @param i The index to access.
    ///
    /// This class does not have different behavior under different access modes.
    /// It exposes the access mode API for compatibility purposes.
    ///
    const_value_type get_datum(size_t i) const;

    ///
    /// @brief Assign the stored data at position `i`.
    ///
    /// @param i The index to store data at.
    /// @param v The data to store.
    ///
    void set_datum(size_t i, const const_value_type& v);

    void prefetch(size_t i) const;

    ///// Resizing
    void resize(size_t new_size)
        requires is_resizeable<Alloc>;

    ///// Compaction
    template <std::integral I, threads::ThreadPool Pool>
    void
    compact(std::span<const I> new_to_old, Pool& threadpool, size_t batchsize = 1'000'000)
        requires is_resizeable<Alloc>;

    std::shared_ptr<const centroid_type> view_centroids() const;
    std::span<const float> get_centroid(size_t i) const;

    ///// Insertion
    // Set datum with a specified centroid.
    template <typename QueryType, size_t N>
    void set_datum(size_t i, std::span<QueryType, N> datum, size_t centroid_selector);

    template <typename QueryType, size_t N>
    void set_datum(size_t i, std::span<QueryType, N> datum);

    ///// Decompressor
    Decompressor decompressor() const;

    ///// Static Constructors
    template <data::ImmutableMemoryDataset Dataset>
    static LVQDataset compress(const Dataset& data, const allocator_type& allocator = {});

    template <data::ImmutableMemoryDataset Dataset>
    static LVQDataset compress(
        const Dataset& data,
        size_t num_threads,
        size_t alignment,
        const allocator_type& allocator = {}
    );

    template <data::ImmutableMemoryDataset Dataset, threads::ThreadPool Pool>
    static LVQDataset compress(
        const Dataset& data,
        Pool& threadpool,
        size_t alignment,
        const allocator_type& allocator = {}
    );

    static constexpr lib::Version save_version = one_level_save_version;
    static constexpr std::string_view serialization_schema = one_level_serialization_schema;
    lib::SaveTable save(const lib::SaveContext& ctx) const;

    static LVQDataset load(
        const lib::LoadTable& table,
        size_t alignment = 0,
        const allocator_type& allocator = {}
    );
};

/////
///// LVQDataset Concept
/////

template <typename T> inline constexpr bool is_lvq_dataset = false;
template <
    size_t Primary,
    size_t Residual,
    size_t Extent,
    LVQPackingStrategy Strategy,
    typename Allocator>
inline constexpr bool
    is_lvq_dataset<LVQDataset<Primary, Residual, Extent, Strategy, Allocator>> = true;

template <typename T>
concept IsLVQDataset = is_lvq_dataset<T>;

template <typename T>
concept IsTwoLevelDataset = is_lvq_dataset<T> && (T::residual_bits != 0);

class DecompressionAccessor {
  public:
    template <IsLVQDataset Data> DecompressionAccessor(const Data& dataset);

    // Access
    template <IsLVQDataset Data>
    std::span<const float> operator()(const Data& dataset, size_t i);

  private:
    lvq::Decompressor decompressor_;
};

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

} // namespace lvq
} // namespace quantization
} // namespace svs
