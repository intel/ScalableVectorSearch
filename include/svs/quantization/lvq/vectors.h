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
#include "svs/lib/saveload.h"
#include "svs/lib/type_traits.h"
#include "svs/quantization/lvq/compressed.h"
#include "svs/third-party/eve.h"

// stl
#include <bit>
#include <concepts>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace svs {
namespace quantization::lvq {

///
/// Trait for one-level compression to facilitate SIMD accelerated distance computations.
/// Expected Interface
/// ```
/// struct /*impl*/ {
///     // Unpacked auxiliary type used reconstruct the original data vector in either a
///     // scalar or vectorized fashion.
///     //
///     // Examples include `Scale` and `ScaleBias`.
///     using auxiliary_type = /* user defined */
///
///     // Compatible SIMD width to use when performing vectorized operations over the
///     // compressed data.
///     static constexpr size_t simd_width = /* user defined */
///
///     // The vectorized unpacker type for the preferred SIMD width.
///     using unpacker_type = /* user defined */
///
///     // Return the number of packed elements.
///     size_t size() const;
///
///     // Construct the auxiliary distance-computation data structure.
///     auxiliary_type prepare_aux() const;
///
///     // Construct an Unpacker over the compressed data.
///     // The optional simd width parameter can be used to override the default choice
///     // in instances where multiple such vectors are being combined.
///     template <size_t VecWidth = simd_width>
///     Unpacker<VecWidth, ...>
///     unpacker(meta::Val<VecWidth> = meta::Val<simd_width>()) const;
/// };
/// ```
///
template <typename T> inline constexpr bool is_onelevel_compression_v = false;

///
/// Trait for two-level compression.
///
template <typename T> inline constexpr bool is_twolevel_compression_v = false;

template <size_t N = 0> struct ScaleBias {
    float scale;
    float bias;
};

ScaleBias(float, float) -> ScaleBias<0>;

/////
///// Compressed Vector Implementations.
/////

///
/// A compressed vector two helper constants.
/// A bias and a scalar.
///
template <size_t Bits, size_t Extent> struct ScaledBiasedVector {
  public:
    using scalar_type = Float16;
    using auxiliary_type = ScaleBias<0>;
    using vector_type = CompressedVector<Unsigned, Bits, Extent>;
    using mutable_vector_type = MutableCompressedVector<Unsigned, Bits, Extent>;

    static constexpr size_t extent = Extent;
    static constexpr size_t simd_width = pick_simd_width<vector_type>();
    using unpacker_type = Unpacker<simd_width, Unsigned, Bits, Extent>;

    // Construct from constant `CompressedVector`.
    template <Arithmetic T>
    ScaledBiasedVector(T scale, T bias, vector_type data)
        : scale{scale}
        , bias{bias}
        , data{data} {}

    // Construct from `MutableCompressedVector`.
    template <Arithmetic T>
    ScaledBiasedVector(T scale, T bias, mutable_vector_type data)
        : scale{scale}
        , bias{bias}
        , data{data} {}

    float get(size_t i) const { return static_cast<float>(scale) * data.get(i) + bias; }
    size_t size() const { return data.size(); }

    const std::byte* pointer() const { return data.data(); }
    constexpr size_t size_bytes() const { return data.size_bytes(); }

    // Distance computation helprs.
    auxiliary_type prepare_aux() const {
        return ScaleBias{static_cast<float>(scale), static_cast<float>(bias)};
    }

    template <size_t VecWidth = simd_width>
    Unpacker<VecWidth, Unsigned, Bits, Extent>
    unpacker(meta::Val<VecWidth> v = meta::Val<simd_width>()) const {
        return Unpacker(v, data);
    }

    ///// Members
    scalar_type scale;
    scalar_type bias;
    vector_type data;
};

template <size_t Bits, size_t Extent>
float get_scale(const ScaledBiasedVector<Bits, Extent>& v) {
    return v.scale;
}

template <size_t Bits, size_t Extent>
inline constexpr bool is_onelevel_compression_v<ScaledBiasedVector<Bits, Extent>> = true;

template <size_t Primary, size_t Residual, size_t N> struct ScaledBiasedWithResidual {
    using auxiliary_type = ScaleBias<Residual>;
    static constexpr size_t simd_width = pick_simd_width<Primary, Residual>();
    using primary_unpacker_type = Unpacker<simd_width, Unsigned, Primary, N>;
    using residual_unpacker_type = Unpacker<simd_width, Signed, Residual, N>;

    ///
    /// Return the decoded value at index `i` using both the primary and residual encodings.
    ///
    float get(size_t i) const {
        float primary = primary_.get(i);
        float residual_step = get_scale(primary_) / std::pow(2, Residual);
        float residual = residual_.get(i) * residual_step;
        return primary + residual;
    }

    ///
    /// Return the number of elements in the vector.
    ///
    size_t size() const { return primary_.size(); }

    ///
    /// Prepare for distance computations.
    ///
    auxiliary_type prepare_aux() const {
        auto aux = primary_.prepare_aux();
        float rescale{aux.scale / static_cast<float>(std::pow(2, Residual))};
        return auxiliary_type{rescale, aux.bias};
    }

    primary_unpacker_type primary_unpacker() const {
        return primary_.unpacker(meta::Val<simd_width>());
    }

    residual_unpacker_type residual_unpacker() const {
        return residual_unpacker_type{meta::Val<simd_width>(), residual_};
    }

    ///// Members
    ScaledBiasedVector<Primary, N> primary_;
    CompressedVector<Signed, Residual, N> residual_;
};

template <size_t Primary, size_t Residual, size_t N>
inline constexpr bool
    is_twolevel_compression_v<ScaledBiasedWithResidual<Primary, Residual, N>> = true;

// clang-format off
template <size_t Primary, size_t Residual, size_t N>
ScaledBiasedWithResidual(
    const ScaledBiasedVector<Primary, N>&,
    const CompressedVector<Signed, Residual, N>&
) -> ScaledBiasedWithResidual<Primary, Residual, N>;
// clang-format on

// Combine primary and residuals.
template <size_t Primary, size_t Residual, size_t N>
ScaledBiasedWithResidual<Primary, Residual, N> combine(
    const ScaledBiasedVector<Primary, N>& primary,
    const CompressedVector<Signed, Residual, N>& residual
) {
    return ScaledBiasedWithResidual{primary, residual};
}

/////
///// Distances
/////

struct EuclideanReference {
    using compare = std::less<>;
    static constexpr bool implicit_broadcast = true;

    template <typename T> float compute(std::span<const float> x, const T& y) {
        float sum{0};
        for (size_t i = 0; i < y.size(); ++i) {
            auto z = x[i] - y.get(i);
            sum += z * z;
        }
        return sum;
    }
};

struct InnerProductReference {
    using compare = std::greater<>;
    static constexpr bool implicit_broadcast = true;

    template <typename T> float compute(std::span<const float> x, const T& y) {
        float sum{0};
        for (size_t i = 0; i < y.size(); ++i) {
            sum += x[i] * y.get(i);
        }
        return sum;
    }
};

/////
///// Distance Helpers
/////

// The role of a `DistanceHelper` is:
//
// 1. Perform any necessary pre-computation required for distance computation.
//    For example, this could include conversion of `Float16` scalars to `float`.
//
// 2. Determine the SIMD width to use for distance computations.
//    This will depend on the type of the Vector. For example, 8- or 4-bit vectors can use
//    a SIMD width of 16 efficiently while 5, 6, and 7-bit vectors need to use 8-wide SIMD
//    because we need full 64-bit integers in the decoding process.
//
//    Further compilcations arise when performing distance computations with residuals.
//    For example, a 5-bit primary with a 4-bit residual is limited to 8-wide SIMD by the
//    5-bit primary encoding, even through the 4-bit residual could feasibly use 16-wide
//    SIMD.
//
//    To that end, we require unpacking algorithms that work for wider vector widths to also
//    work efficiently for narrow SIMD widths (e.g., 4-bit is compatible with both 8 and
//    16-wide SIMD).
//
// 3. Perform an optionally predicated unpacking and application of a step in the distance
//    function.
//
// 4. Apply any post-op reductions to the SIMD accumulation register.
template <typename Distance, typename Vector> struct DistanceHelper;

template <typename T, int64_t N, size_t S, typename P>
wide_<float, N> decompress_step(wide_<T, N> x, ScaleBias<S> aux, size_t /*i*/, P pred) {
    return eve::add[pred.else_(0)](aux.scale * eve::convert(x, eve::as<float>()), aux.bias);
}

template <typename T, int64_t N, size_t S, typename P>
wide_<float, N> apply_step(
    distance::DistanceL2 /*unused*/,
    wide_<float, N> accum,
    wide_<float, N> x,
    wide_<T, N> y,
    ScaleBias<S> aux,
    size_t /*i*/,
    P pred
) {
    // Apply the scaling parameter and add in the bias.
    // If a predicate is supplied, we must maintain the masked lanes as zero, so use a
    // predicated addition.
    auto converted =
        eve::add[pred.else_(0)](aux.scale * eve::convert(y, eve::as<float>()), aux.bias);
    auto temp = x - converted;
    return accum + temp * temp;
}

template <typename T, int64_t N, typename P, size_t S>
wide_<float, N> apply_step(
    distance::DistanceIP /*unused*/,
    wide_<float, N> accum,
    wide_<float, N> x,
    wide_<T, N> y,
    ScaleBias<S> aux,
    size_t /*i*/,
    P /*unused*/
) {
    // In this case, we can leverage the fact that `x` will be set to zero in the masked
    // lanes, so we can unconditionally add in the bias.
    auto converted = (aux.scale * eve::convert(y, eve::as<float>())) + aux.bias;
    return accum + x * converted;
}

template <int64_t N, size_t S>
float finish_step(
    distance::DistanceL2 /*unused*/, wide_<float, N> accum, ScaleBias<S> /*aux*/
) {
    return eve::reduce(accum, eve::plus);
}

template <int64_t N, size_t S>
float finish_step(
    distance::DistanceIP /*unused*/, wide_<float, N> accum, ScaleBias<S> /*aux*/
) {
    // As part of the application step, we mix in the scaling parameter.
    // Therefore, there's nothing really to be done in this step.
    return eve::reduce(accum, eve::plus);
}

struct NoDistance {};
template <typename Distance, typename T>
    requires is_onelevel_compression_v<T>
struct DistanceHelper<Distance, T> {
    // Helpers
    using unpacker_type = typename T::unpacker_type;
    static const size_t simd_width = unpacker_type::simd_width;
    using accumulator_type = typename unpacker_type::accum_type;
    using integer_wide_type = typename unpacker_type::int_wide_type;
    using auxiliary_type = typename T::auxiliary_type;

    ///
    /// Construct a distance computer from a ScaledVector.
    ///
    DistanceHelper(Distance /*unused*/, const T& v)
        : unpacker_{v.unpacker()}
        , aux_{v.prepare_aux()} {}

    ///
    /// Create a zeroed accumulator.
    ///
    accumulator_type accumulator() { return accumulator_type(0); }

    ///
    /// Decompress one vector bundle.
    ///
    template <typename P = eve::ignore_none_>
    accumulator_type decompress(size_t i, P pred = eve::ignore_none_()) {
        assert(i <= lib::div_round_up(unpacker_.size(), simd_width));
        integer_wide_type unpacked = unpacker_.get(i, pred);
        return decompress_step(unpacked, aux_, i, pred);
    }

    ///
    /// Distance computation step.
    ///
    template <typename P = eve::ignore_none_>
    accumulator_type apply(
        accumulator_type current,
        accumulator_type left,
        size_t i,
        P pred = eve::ignore_none_()
    ) {
        integer_wide_type unpacked = unpacker_.get(i, pred);
        return apply_step(Distance(), current, left, unpacked, aux_, i, pred);
    }

    ///
    /// Perform the final reduction.
    ///
    float finish(accumulator_type sum) { return finish_step(Distance(), sum, aux_); }

    ///// Members
    unpacker_type unpacker_;
    auxiliary_type aux_;
};

// Two level helpers
template <typename T, typename U, int64_t N, size_t S>
wide_<detail::biggest_int_t<T, U>, N> combine(
    wide_<T, N> primary, wide_<U, N> residual, ScaleBias<S> /*unused*/
) {
    using Common = detail::biggest_int_t<T, U>;
    return eve::convert((primary << S), eve::as<Common>()) +
           eve::convert(residual, eve::as<Common>());
}

// Euclidean Distance - Two Level
template <typename Distance, typename T>
    requires is_twolevel_compression_v<T>
struct DistanceHelper<Distance, T> {
    using primary_unpacker_type = typename T::primary_unpacker_type;
    using residual_unpacker_type = typename T::residual_unpacker_type;
    using auxiliary_type = typename T::auxiliary_type;
    static const size_t simd_width = T::simd_width;

    using accumulator_type = typename primary_unpacker_type::accum_type;

    ///
    /// Construct a distance computer from a ScaledVector.
    ///
    DistanceHelper(Distance /*unused*/, const T& v)
        : primary_unpacker_{v.primary_unpacker()}
        , residual_unpacker_{v.residual_unpacker()}
        , aux_{v.prepare_aux()} {}

    ///
    /// Create a zeroed accumulator.
    ///
    accumulator_type accumulator() { return accumulator_type(0); }

    ///
    /// Distance computation step.
    ///
    template <typename P = eve::ignore_none_>
    accumulator_type apply(
        accumulator_type current,
        accumulator_type left,
        size_t i,
        P pred = eve::ignore_none_()
    ) {
        auto p = primary_unpacker_.get(i, pred);
        auto r = residual_unpacker_.get(i, pred);

        // Combine the primary and residual then apply the distance computation.
        return apply_step(Distance(), current, left, combine(p, r, aux_), aux_, i, pred);
    }

    ///
    /// Perform the final reduction.
    ///
    float finish(accumulator_type sum) { return finish_step(Distance(), sum, aux_); }

    ///// Members
    primary_unpacker_type primary_unpacker_;
    residual_unpacker_type residual_unpacker_;
    auxiliary_type aux_;
};

template <typename Distance, typename T>
DistanceHelper(Distance, const T&) -> DistanceHelper<Distance, T>;

/////
///// Optimized AVX implementations.
/////

template <typename T, size_t N>
    requires is_onelevel_compression_v<T>
void decompress(std::span<float, N> dst, const T& src) {
    assert(dst.size() == src.size());
    auto helper = DistanceHelper(NoDistance(), src);
    constexpr size_t simd_width = decltype(helper)::simd_width;

    size_t iterations = src.size() / simd_width;
    size_t remaining = src.size() % simd_width;

    // Main unrolled loop.
    float* base = dst.data();
    for (size_t i = 0; i < iterations; ++i) {
        eve::store(helper.decompress(i), base);
        base += simd_width;
    }

    // Handle tail elements.
    if (remaining != 0) {
        auto predicate = eve::keep_first(lib::narrow_cast<int64_t>(remaining));
        eve::store[predicate](helper.decompress(iterations, predicate), base);
    }
}

template <typename T>
    requires is_onelevel_compression_v<T>
void decompress(std::vector<float>& dst, const T& src) {
    dst.resize(src.size());
    decompress(lib::as_span(dst), src);
}

class Decompressor {
  public:
    Decompressor() = default;

    template <typename T>
        requires is_onelevel_compression_v<T>
    std::span<const float> operator()(const T& compressed) {
        decompress(buffer_, compressed);
        return lib::as_const_span(buffer_);
    }

  private:
    std::vector<float> buffer_ = {};
};

// Trait for dispatching on compressed vector types.
template <typename T>
inline constexpr bool is_compressed_vector =
    is_onelevel_compression_v<T> || is_twolevel_compression_v<T>;

// Trait describing distances that have optimized implementations.
template <typename T> inline constexpr bool fast_quantized = false;
template <> inline constexpr bool fast_quantized<distance::DistanceL2> = true;
template <> inline constexpr bool fast_quantized<distance::DistanceIP> = true;

template <typename Distance, typename T>
    requires fast_quantized<Distance> && is_compressed_vector<T>
float compute_quantized(Distance distance, std::span<const float> x, const T& y) {
    // Construct the distance helper.
    //
    // All necessary setup tasks (e.g., conversion of Float16 scalars to Float32) are
    // performed in the constructor.
    //
    // The `DistanceHelper` also defines the SIMD width to use, which is determined by
    // multiple factors within the quantized data structure.
    auto helper = DistanceHelper(distance, y);
    using accumulator_t = typename decltype(helper)::accumulator_type;
    constexpr size_t simd_width = decltype(helper)::simd_width;

    // Compute the number of main iterations and the remaining number of elements to be
    // handle in the trailing loop.
    size_t iterations = y.size() / simd_width;
    size_t tail_elements = y.size() % simd_width;

    // Main Computation Loop.
    //
    // N.B.: At the moment, our dimensionality is static (i.e., known at compile time).
    // As such, manually unrolling the loop is unnecessary as the compiler completely
    // unrolls it.
    //
    // In the future if we choose to support run-time dimensionality, we may need to
    // manually unroll the loop several times.
    accumulator_t accum = helper.accumulator();
    for (size_t i = 0; i < iterations; ++i) {
        accumulator_t left{&x[simd_width * i]};
        accum = helper.apply(accum, left, i);
    }

    // Handle tail elements.
    //
    // The responsiblity at this level is to perform a masked load of the query vector.
    // After that, it's up to the distance helper to correctly apply the predicate for
    // both loading the compressed data as well as applying the distance computation to the
    // partial accumulated values.
    if (tail_elements != 0) {
        size_t j = simd_width * iterations;
        auto predicate = eve::keep_first(lib::narrow_cast<int64_t>(tail_elements));
        accumulator_t left = eve::load[predicate.else_(0)](&x[j], eve::as<accumulator_t>());
        accum = helper.apply(accum, left, iterations, predicate);
    }
    return helper.finish(accum);
}
} // namespace quantization::lvq

//
// Overload `distance::compute` for the distances and compression techniques defined above.
//
namespace distance {
template <typename Distance, typename T>
    requires quantization::lvq::fast_quantized<Distance> &&
             quantization::lvq::is_compressed_vector<T>
float compute(Distance distance, std::span<const float> x, const T& y) {
    return quantization::lvq::compute_quantized(distance, x, y);
}
} // namespace distance

namespace quantization::lvq {

///
/// Distance computations supporting a global vector bias.
///
class EuclideanBiased {
  public:
    using compare = std::less<>;
    // Biased versions are not implicitly broadcastable because they must maintain per-query
    // state.
    static constexpr bool implicit_broadcast = false;

    // Constructors
    EuclideanBiased(const std::shared_ptr<std::vector<float>>& bias)
        : biased_query_(bias->size())
        , bias_{bias} {}

    EuclideanBiased(const std::vector<float>& bias)
        : EuclideanBiased{std::make_shared<std::vector<float>>(bias)} {}

    // Shallow Copy
    // Don't preserve the state of the `biased_query_` vector.
    EuclideanBiased shallow_copy() const { return EuclideanBiased{bias_}; }

    ///
    /// Apply the inverse bias to `query` and cache the result internally, using a
    /// one-argument `compute` method.
    ///
    void fix_argument(const std::span<const float>& query) {
        // Check pre-conditions.
        assert(bias_->size() == query.size());
        // Component-wise add the bias to the query and cache the result.
        std::transform(
            query.begin(), query.end(), bias_->cbegin(), biased_query_.begin(), std::minus()
        );
    }

    ///
    /// Compute the Euclidean difference between a quantized vector `y` and a cached
    /// shifted query.
    ///
    template <typename T> float compute(const T& y) const {
        // If the argument `y` is a `std::span`, it's not a compressed vector so fall-back
        // to doing normal distance computations.
        distance::DistanceL2 inner{};
        return distance::compute(inner, view_query(), y);
    }

    std::span<const float> view_query() const {
        return std::span<const float>{biased_query_.data(), biased_query_.size()};
    }

    ///
    /// Return the global bias as a `std::span`.
    ///
    std::span<const float> view_bias() const { return lib::as_const_span(*bias_); }

    ///// Saving and Loading.
    static constexpr std::string_view name = distance::DistanceL2::name;
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);
    lib::SaveType save(const lib::SaveContext& /*ctx*/) const {
        return lib::SaveType(
            toml::table({{"name", name}, {"bias", prepare(*bias_)}}), save_version
        );
    }

    static EuclideanBiased load(
        const toml::table& table,
        const lib::LoadContext& /*ctx*/,
        const lib::Version& version
    ) {
        if (version != save_version) {
            throw ANNEXCEPTION("Unhandled version!");
        }
        return EuclideanBiased(get_vector<float>(table, "bias"));
    }

  private:
    std::vector<float> biased_query_;
    std::shared_ptr<std::vector<float>> bias_;
};

inline bool operator==(const EuclideanBiased& x, const EuclideanBiased& y) {
    const auto& xbias = x.view_bias();
    const auto& ybias = y.view_bias();
    if (xbias.size() != ybias.size()) {
        return false;
    }
    return std::equal(xbias.begin(), xbias.end(), ybias.begin());
}

class InnerProductBiased {
  public:
    using compare = std::greater<>;
    // Biased versions are not implicitly broadcastable because they must maintain per-query
    // state.
    static constexpr bool implicit_broadcast = false;

    // Constructor
    InnerProductBiased(const std::shared_ptr<std::vector<float>>& bias)
        : bias_{bias} {}

    InnerProductBiased(const std::vector<float>& bias)
        : InnerProductBiased{std::make_shared<std::vector<float>>(bias)} {}

    // Shallow Copy
    InnerProductBiased shallow_copy() const { return InnerProductBiased{bias_}; }

    ///
    /// Return the global bias as a `std::span`.
    ///
    std::span<const float> view_bias() const { return lib::as_const_span(*bias_); }

    ///
    /// Precompute the inner product between the query and the global bias.
    /// This pre-computed value will be added to the result of standard distance
    /// computations using the distributive property where
    /// ```
    /// q . (x + b) == (q . x) + (q . b)
    /// ```
    ///
    void fix_argument(const std::span<const float>& query) {
        // Check pre-conditions.
        assert(bias_->size() == query.size());
        distance::DistanceIP inner_distance{};
        bias_product_ = distance::compute(inner_distance, query, view_bias());
    }

    template <typename T>
    float compute(const std::span<const float>& query, const T& y) const {
        // If the argument `y` is a `std::span`, it's not a compressed vector so fall-back
        // to doing normal distance computations.
        distance::DistanceIP inner{};
        return bias_product_ + distance::compute(inner, query, y);
    }

    ///// Saving and Loading
    static constexpr std::string_view name = distance::DistanceIP::name;
    static constexpr lib::Version save_version = lib::Version(0, 0, 0);
    lib::SaveType save(const lib::SaveContext& /*ctx*/) const {
        return lib::SaveType(
            toml::table({{"name", name}, {"bias", prepare(*bias_)}}), save_version
        );
    }

    static InnerProductBiased load(
        const toml::table& table,
        const lib::LoadContext& /*ctx*/,
        const lib::Version& version
    ) {
        if (version != save_version) {
            throw ANNEXCEPTION("Unhandled version!");
        }
        return InnerProductBiased(get_vector<float>(table, "bias"));
    }

  private:
    std::shared_ptr<std::vector<float>> bias_;
    float bias_product_ = 0;
};

inline bool operator==(const InnerProductBiased& x, const InnerProductBiased& y) {
    const auto& xbias = x.view_bias();
    const auto& ybias = y.view_bias();
    if (xbias.size() != ybias.size()) {
        return false;
    }
    return std::equal(xbias.begin(), xbias.end(), ybias.begin());
}

namespace detail {

template <typename T> struct BiasedDistance;
template <typename T> struct HasBiased : std::false_type {};

template <> struct BiasedDistance<distance::DistanceL2> {
    using type = EuclideanBiased;
};
template <> struct HasBiased<distance::DistanceL2> : std::true_type {};

template <> struct BiasedDistance<distance::DistanceIP> {
    using type = InnerProductBiased;
};
template <> struct HasBiased<distance::DistanceIP> : std::true_type {};

} // namespace detail

///
/// Determine if the distance type `T` has a pre-biased implementation.
///
template <typename T> inline constexpr bool has_biased_v = detail::HasBiased<T>::value;

///
/// Compute the correct biased distance function to operate on compressed data given the
/// original distance function `T`.
///
template <typename T> using biased_distance_t = typename detail::BiasedDistance<T>::type;

/////
///// Support for index building.
/////

// When performing index building, there are two situations to consider.
// (1) No medioid removal.
// (2) With medioid removal.
//
// In the first case, there is a relatively straight-forward way of dealing with distance
// computations between two compressed vectors.
//
// We introduce a `DecompressionAdaptor` which turns the LHS query from its compressed
// form to floats. At this point, we can simply opt-in to the the standard distance
// computation pipeline.
//
// The second case (when the entire dataset has had the per-component means removed)
// requires a bit more care. When we're using the Euclidean distance, we can skip any kind
// of medioid restoration.
//
// However, in the case of inner product, we need to first restore the removed medioid from
// the uncompressed LHS before proceeding with normal bias-based distance computations.

///
/// Adaptor to adjust a distance function with type `Distance` to enable index building
/// over a compressed dataset.
///
/// Essentially, allows for distance computations between two elements of a compressed
/// dataset.
///
template <typename Distance> class DecompressionAdaptor;

template <typename Distance>
    requires fast_quantized<Distance>
class DecompressionAdaptor<Distance> {
  public:
    using distance_type = Distance;
    // Use the same comparison as the wrapped distance function.
    using compare = distance::compare_t<Distance>;
    // We require per-query state, so cannot implicitly broadcast.
    static constexpr bool implicit_broadcast = false;

    // Constructor
    DecompressionAdaptor(const Distance& distance, size_t size_hint = 0)
        : distance_{distance}
        , decompressed_(size_hint) {}

    // Shallow copy
    DecompressionAdaptor shallow_copy() const {
        return DecompressionAdaptor(distance_, decompressed_.size());
    }

    ///
    /// Precompute the decompressed value for the compressed vector.
    ///
    template <typename Left>
        requires(is_onelevel_compression_v<Left>)
    void fix_argument(Left left) {
        decompress(decompressed_, left);
    }

    template <typename Right>
        requires(is_onelevel_compression_v<Right>)
    float compute(const Right& right) const {
        return distance::compute(distance_, view(), right);
    }

    std::span<const float> view() const {
        return std::span<const float>(decompressed_.data(), decompressed_.size());
    }

  private:
    distance_type distance_;
    std::vector<float> decompressed_;
};

///
/// Specialization of the DecompressionAdaptor for the `EuclideanBiased` distance
/// function.
///
/// Applies an optimization of assuming that the bias removal is uniform for the entire
/// dataset and thus skips reapplying any bias adjustment since it will have no effect
/// on the final result.
///
/// Implementation details: Essentially just a wrapper around
/// `DecompressionAdaptor<distance::DistanceL2>`.
///
template <>
class DecompressionAdaptor<EuclideanBiased>
    : public DecompressionAdaptor<distance::DistanceL2> {
  public:
    using parent_type = DecompressionAdaptor<distance::DistanceL2>;
    DecompressionAdaptor(const EuclideanBiased& /*distance*/, size_t size_hint = 0)
        : parent_type{distance::DistanceL2(), size_hint} {}
};

template <> class DecompressionAdaptor<InnerProductBiased> {
  public:
    using distance_type = InnerProductBiased;
    using compare = distance::compare_t<distance_type>;
    static constexpr bool implicit_broadcast = false;

    DecompressionAdaptor(const distance_type& inner, size_t size_hint = 0)
        : inner_{inner}
        , decompressed_(size_hint) {}

    DecompressionAdaptor shallow_copy() const {
        return DecompressionAdaptor(inner_, decompressed_.size());
    }

    // Distance API.
    template <typename Left>
        requires(is_onelevel_compression_v<Left>)
    void fix_argument(Left left) {
        decompress(decompressed_, left);

        const auto& bias = inner_.view_bias();
        assert(bias.size() == decompressed_.size());
        // Decompress the query and add in the bias to restore it to its original value.
        for (size_t i = 0, imax = left.size(); i < imax; ++i) {
            decompressed_[i] += bias[i];
        }
        // Pass the decompressed vector into the bias routine.
        inner_.fix_argument(view());
    }

    template <typename Right>
        requires(is_onelevel_compression_v<Right>)
    float compute(const Right& right) const {
        return inner_.compute(view(), right);
    }

    std::span<const float> view() const {
        return std::span<const float>(decompressed_.data(), decompressed_.size());
    }

  private:
    distance_type inner_;
    std::vector<float> decompressed_;
};

} // namespace quantization::lvq

// Wire up SelfDistance dispatch to support vector quantization.
template <typename Distance, typename VectorType>
    requires(quantization::lvq::is_onelevel_compression_v<VectorType>)
struct SelfDistance<Distance, VectorType> {
    using type = quantization::lvq::DecompressionAdaptor<Distance>;
    static constexpr type modify(const Distance& distance) { return type(distance); }
};
} // namespace svs
