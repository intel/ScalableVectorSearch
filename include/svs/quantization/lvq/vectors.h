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
#include "eve/algo.hpp"
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

// Implementation Notes
//
// SVS_GCC_NOINLINE: GCC (<=13) struggles with optimizing the body of distance computation
// functions after they have been inlined into non-trivial call-sites.
//
// In particular, it failed to pre-load LVQ shifts and masks into registers and instead
// reloads them on every iteration.
//
// Strategically applying `SVS_GCC_NOINLINE` keeps the complexity of the distance
// computations to a point where GCC can correctly optimize the implementation.

namespace svs {
namespace quantization::lvq {

///// Static dispatch traits.
namespace detail {
// Extendable trait to overload common entry-points for LVQ vectors.
template <typename T> inline constexpr bool lvq_compressed_vector_v = false;
} // namespace detail

// Dispatch concept for LVQCompressedVectors
template <typename T>
concept LVQCompressedVector = detail::lvq_compressed_vector_v<T>;

// Auxiliary struct to help with distance computations.
struct ScaleBias {
    float scale;
    float bias;
};

/////
///// Compressed Vector Implementations.
/////

///
/// A compressed vector two helper constants.
/// A bias and a scalar.
///
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
    ScaledBiasedVector(T scale, T bias, selector_t selector, vector_type data)
        : scale{scale}
        , bias{bias}
        , data{data}
        , selector{selector} {}

    // Construct from `MutableCompressedVector`.
    template <Arithmetic T>
    ScaledBiasedVector(T scale, T bias, selector_t selector, mutable_vector_type data)
        : scale{scale}
        , bias{bias}
        , data{data}
        , selector{selector} {}

    float get(size_t i) const { return static_cast<float>(scale) * data.get(i) + bias; }
    size_t size() const { return data.size(); }
    selector_t get_selector() const { return selector; }
    float get_scale() const { return scale; }
    float get_bias() const { return bias; }
    vector_type vector() const { return data; }

    const std::byte* pointer() const { return data.data(); }
    constexpr size_t size_bytes() const { return data.size_bytes(); }

    // Distance computation helpers.
    auxiliary_type prepare_aux() const {
        return ScaleBias{static_cast<float>(scale), static_cast<float>(bias)};
    }

    // Logical equivalence.
    template <size_t E2, LVQPackingStrategy OtherStrategy>
    bool logically_equivalent_to(const ScaledBiasedVector<Bits, E2, OtherStrategy>& other
    ) const {
        // Maybe be able to always return false if extent checks fail.
        if constexpr (Extent != Dynamic && E2 != Dynamic && Extent != E2) {
            return false;
        }
        // Compare scalar constants.
        if (scale != other.scale || bias != other.bias || selector != other.selector) {
            return false;
        }
        // So far so good - compare the underlying vectors for equivalence.
        return logically_equal(data, other.data);
    }

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

namespace detail {
template <size_t Bits, size_t Extent, typename Strategy>
inline constexpr bool
    lvq_compressed_vector_v<lvq::ScaledBiasedVector<Bits, Extent, Strategy>> = true;
}

template <size_t Primary, size_t Residual, size_t N, typename Strategy = DefaultStrategy>
struct ScaledBiasedWithResidual {
  public:
    using strategy = Strategy;
    using auxiliary_type = ScaleBias;
    using strategy_type = Strategy;

    /// Return the decoded value at index `i` using both the primary and residual encodings.
    float get(size_t i) const {
        float primary = primary_.get(i);
        float delta = primary_.get_scale();
        float residual_step = delta / (std::pow(2, Residual) - 1);
        float residual = residual_.get(i) * residual_step - delta / 2;
        return primary + residual;
    }

    /// Return the number of elements in the vector.
    size_t size() const { return primary_.size(); }

    /// Return the centroid selector.
    selector_t get_selector() const { return primary_.get_selector(); }

    float get_scale() const { return primary_.get_scale() / (std::pow(2, Residual) - 1); }
    float get_bias() const { return primary_.get_bias() - primary_.get_scale() / 2; }

    /// Prepare for distance computations.
    auxiliary_type prepare_aux() const {
        auto [scale, bias] = primary_.prepare_aux();
        float rescale{scale / (static_cast<float>(std::pow(2, Residual)) - 1)};
        return auxiliary_type{rescale, bias - (scale / 2)};
    }

    Combined<Primary, Residual, N, Strategy> vector() const {
        return Combined<Primary, Residual, N, Strategy>{primary_.vector(), residual_};
    }

    template <size_t N2, LVQPackingStrategy OtherStrategy>
    bool logically_equivalent_to(
        const ScaledBiasedWithResidual<Primary, Residual, N2, OtherStrategy>& other
    ) const {
        // Recurse over each member.
        if (!logically_equal(primary_, other.primary_)) {
            return false;
        }
        return logically_equal(residual_, other.residual_);
    }

    ///// Members
    // For now - only the primary vector is allowed to have variable strategy.
    // The residual is always kept as sequential due to implementation challenges and
    // somewhat dubious ROI.
    ScaledBiasedVector<Primary, N, Strategy> primary_;
    CompressedVector<Unsigned, Residual, N, Sequential> residual_;
};

namespace detail {
template <size_t Primary, size_t Residual, size_t N, typename Strategy>
inline constexpr bool
    lvq_compressed_vector_v<ScaledBiasedWithResidual<Primary, Residual, N, Strategy>> =
        true;
}

// Combine primary and residuals.
template <size_t Primary, size_t Residual, size_t N, typename Strategy>
ScaledBiasedWithResidual<Primary, Residual, N, Strategy> combine(
    const ScaledBiasedVector<Primary, N, Strategy>& primary,
    const CompressedVector<Unsigned, Residual, N, Sequential>& residual
) {
    return ScaledBiasedWithResidual<Primary, Residual, N, Strategy>{primary, residual};
}

/////
///// Distances
/////

// Reference Implementations: fallback to using scalar indexing for each component of
// the compressed vectors.
struct EuclideanReference {
    using compare = std::less<>;
    static constexpr bool implicit_broadcast = true;

    template <LVQCompressedVector T> float compute(std::span<const float> x, const T& y) {
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

    template <LVQCompressedVector T> float compute(std::span<const float> x, const T& y) {
        float sum{0};
        for (size_t i = 0; i < y.size(); ++i) {
            sum += x[i] * y.get(i);
        }
        return sum;
    }
};

// Optimized inner product for LVQ datasets
// <q, (scale * x + bias)> => scale * <q, x> + <q, bias>
// Since, scale and bias are per LVQ vector scalar constants
// <q, bias> = bias * sum(q). sum(q) is precomputed for a query. Therefore,
// first calculate only <q,x> and finally multiply add the constants
struct DistanceFastIP {
    float query_sum; // preprocessed query sum
};

// Decompressing *with* a component-wise add-in.
template <typename T, int64_t N, typename P>
wide_<float, N>
decompress_step(wide_<T, N> x, ScaleBias aux, size_t i, const float* centroid, P pred) {
    auto mask = pred.else_(0);

    // Load the corresponding centroid to be added component-wise to the reconstructed
    // vector fragment.
    auto centroid_chunk = eve::load[mask](&centroid[N * i], eve::as<wide_<float, N>>());
    return centroid_chunk +
           eve::add[mask](aux.scale * eve::convert(x, eve::as<float>()), aux.bias);
}

template <typename T, int64_t N, typename P>
wide_<float, N> apply_step(
    distance::DistanceL2 /*unused*/,
    wide_<float, N> accum,
    wide_<float, N> x,
    wide_<T, N> y,
    ScaleBias aux,
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

template <typename T, int64_t N, typename P>
wide_<float, N> apply_step(
    distance::DistanceIP /*unused*/,
    wide_<float, N> accum,
    wide_<float, N> x,
    wide_<T, N> y,
    ScaleBias aux,
    P /*unused*/
) {
    // In this case, we can leverage the fact that `x` will be set to zero in the masked
    // lanes, so we can unconditionally add in the bias.
    auto converted = (aux.scale * eve::convert(y, eve::as<float>())) + aux.bias;
    return accum + x * converted;
}

template <typename T, int64_t N, typename P>
wide_<float, N> apply_step(
    DistanceFastIP /*unused*/,
    wide_<float, N> accum,
    wide_<float, N> x,
    wide_<T, N> y,
    ScaleBias /*aux*/,
    P /*unused*/
) {
    // In the first step, just do <x,y>
    return accum + x * eve::convert(y, eve::as<float>());
}

template <int64_t N>
float finish_step(
    distance::DistanceL2 /*unused*/, wide_<float, N> accum, ScaleBias /*aux*/
) {
    return eve::reduce(accum, eve::plus);
}

template <int64_t N>
float finish_step(
    distance::DistanceIP /*unused*/, wide_<float, N> accum, ScaleBias /*aux*/
) {
    // As part of the application step, we mix in the scaling parameter.
    // Therefore, there's nothing really to be done in this step.
    return eve::reduce(accum, eve::plus);
}

template <int64_t N>
float finish_step(DistanceFastIP distance, wide_<float, N> accum, ScaleBias aux) {
    // Scale and add bias*query_sum only once in the final step
    return aux.scale * eve::reduce(accum, eve::plus) + aux.bias * distance.query_sum;
}

///// Intel(R) AVX implementations.
template <LVQCompressedVector T, size_t N>
    requires UsesSequential<T>
void decompress(std::span<float, N> dst, const T& src, const float* centroid) {
    assert(dst.size() == src.size());
    auto aux = src.prepare_aux();
    auto v = src.vector();
    auto helper = prepare_unpack(v);

    const size_t simd_width = 16;
    using int_wide_t = wide_<int32_t, simd_width>;

    size_t iterations = src.size() / simd_width;
    size_t remaining = src.size() % simd_width;

    // Main unrolled loop.
    float* base = dst.data();
    for (size_t i = 0; i < iterations; ++i) {
        auto unpacked = unpack_as(v, i, eve::as<int_wide_t>(), helper, eve::ignore_none);
        eve::store(decompress_step(unpacked, aux, i, centroid, eve::ignore_none), base);
        base += simd_width;
    }

    // Handle tail elements.
    if (remaining != 0) {
        auto predicate = eve::keep_first(lib::narrow_cast<int64_t>(remaining));
        auto unpacked = unpack_as(v, iterations, eve::as<int_wide_t>(), helper, predicate);
        eve::store[predicate](
            decompress_step(unpacked, aux, iterations, centroid, predicate), base
        );
    }
}

template <LVQCompressedVector T, size_t N>
    requires UsesTurbo<T>
void decompress(std::span<float, N> dst, const T& src, const float* centroid) {
    auto aux = src.prepare_aux();
    auto v = src.vector();
    const size_t simd_width = 16;
    auto* ptr = dst.data();

    auto op = [ptr, aux, centroid](
                  detail::Empty, size_t lane, wide_<int32_t, simd_width> unpacked, auto pred
              ) {
        eve::store[pred](
            decompress_step(unpacked, aux, lane, centroid, pred), ptr + simd_width * lane
        );
        return detail::empty;
    };

    constexpr auto empty_op = detail::empty;
    for_each_slice(v, op, empty_op, empty_op, empty_op);
}

template <LVQCompressedVector T>
void decompress(std::vector<float>& dst, const T& src, const float* centroid) {
    dst.resize(src.size());
    decompress(lib::as_span(dst), src, centroid);
}

// Compression aid for LVQ - provides RAII management for the decompressed data and
// maintains a reference to the centroid-group for the compressed vector.
class Decompressor {
  public:
    Decompressor() = delete;
    Decompressor(std::shared_ptr<const data::SimpleData<float>>&& centroids)
        : centroids_{std::move(centroids)}
        , buffer_(centroids_->dimensions()) {}

    template <LVQCompressedVector T>
    std::span<const float> operator()(const T& compressed) {
        decompress(
            buffer_, compressed, centroids_->get_datum(compressed.get_selector()).data()
        );
        return lib::as_const_span(buffer_);
    }

  private:
    std::shared_ptr<const data::SimpleData<float>> centroids_;
    std::vector<float> buffer_ = {};
};

// Trait describing distances that have optimized implementations.
template <typename T> inline constexpr bool fast_quantized = false;
template <> inline constexpr bool fast_quantized<distance::DistanceL2> = true;
template <> inline constexpr bool fast_quantized<distance::DistanceIP> = true;
template <> inline constexpr bool fast_quantized<DistanceFastIP> = true;

///// Sequantial LVQ
template <typename Distance, LVQCompressedVector T>
    requires UsesSequential<T>
SVS_GCC_NOINLINE float
compute_quantized(Distance distance, std::span<const float> x, const T& y) {
    auto aux = y.prepare_aux();
    auto v = y.vector();
    const auto helper = prepare_unpack(v);

    const size_t simd_width = 16;
    const size_t unroll = 4;

    size_t iterations = y.size() / simd_width;
    size_t unrolled_iterations = iterations / 4;
    size_t remaining = y.size() % simd_width;

    using accumulator_t = wide_<float, simd_width>;
    using int_wide_t = wide_<int32_t, simd_width>;

    // Lambda for general unpacking of the LVQ compressed vector.
    auto unpack = [&]<typename Pred = eve::ignore_none_>(size_t i, Pred pred = {}) {
        return unpack_as(v, i, eve::as<int_wide_t>(), helper, pred);
    };

    auto a0 = accumulator_t(0);
    if (unrolled_iterations > 0) {
        auto a1 = accumulator_t(0);
        auto a2 = accumulator_t(0);
        auto a3 = accumulator_t(0);
        for (size_t i = 0; i < unrolled_iterations; ++i) {
            size_t j = unroll * i;
            auto lhs0 = accumulator_t{&x[simd_width * j]};
            auto lhs1 = accumulator_t{&x[simd_width * (j + 1)]};
            auto lhs2 = accumulator_t{&x[simd_width * (j + 2)]};
            auto lhs3 = accumulator_t{&x[simd_width * (j + 3)]};

            auto unpacked0 = unpack(j);
            auto unpacked1 = unpack(j + 1);
            auto unpacked2 = unpack(j + 2);
            auto unpacked3 = unpack(j + 3);

            a0 = apply_step(distance, a0, lhs0, unpacked0, aux, eve::ignore_none);
            a1 = apply_step(distance, a1, lhs1, unpacked1, aux, eve::ignore_none);
            a2 = apply_step(distance, a2, lhs2, unpacked2, aux, eve::ignore_none);
            a3 = apply_step(distance, a3, lhs3, unpacked3, aux, eve::ignore_none);
        }

        // Reduce
        a0 = (a0 + a1) + (a2 + a3);
    }

    size_t end_of_unroll = unroll * unrolled_iterations;
    for (size_t i = end_of_unroll; i < iterations; ++i) {
        auto lhs = accumulator_t{&x[simd_width * i]};
        auto unpacked = unpack(i);
        a0 = apply_step(distance, a0, lhs, unpacked, aux, eve::ignore_none);
    }

    // Handle tail elements.
    //
    // The responsibility at this level is to perform a masked load of the query vector.
    // After that, it's up to the distance helper to correctly apply the predicate for
    // both loading the compressed data as well as applying the distance computation to the
    // partial accumulated values.
    if (remaining != 0) {
        size_t i = iterations;
        auto predicate = eve::keep_first(lib::narrow_cast<int64_t>(remaining));
        auto lhs =
            eve::load[predicate.else_(0)](&x[simd_width * i], eve::as<accumulator_t>());
        auto unpacked = unpack_as(v, i, eve::as<int_wide_t>(), helper, predicate);
        a0 = apply_step(distance, a0, lhs, unpacked, aux, predicate);
    }
    return finish_step(distance, a0, aux);
}

// Turbo-based distance computation.
template <typename Distance, LVQCompressedVector T>
    requires UsesTurbo<T>
SVS_GCC_NOINLINE float
compute_quantized(Distance distance, std::span<const float> x, const T& y) {
    // static_assert(y uses turbo strategy)
    auto aux = y.prepare_aux();
    auto v = y.vector();

    const size_t simd_width = 16;
    const auto* ptr = x.data();
    auto op = [distance, ptr, aux](
                  wide_<float, simd_width> accum,
                  size_t lane,
                  wide_<int32_t, simd_width> unpacked,
                  auto pred
              ) {
        auto left =
            eve::load[pred.else_(0)](ptr + simd_width * lane, eve::as<wide_<float, 16>>());
        return apply_step(distance, accum, left, unpacked, aux, pred);
    };

    return for_each_slice(
        v,
        op,
        []() { return wide_<float, simd_width>(0); },
        eve::plus,
        [distance, aux](wide_<float, simd_width> accum) {
            return finish_step(distance, accum, aux);
        }
    );
}

} // namespace quantization::lvq

/// Overload `distance::compute` for the distances and compression techniques defined above.
namespace distance {
template <typename Distance, quantization::lvq::LVQCompressedVector T>
    requires quantization::lvq::fast_quantized<Distance>
float compute(Distance distance, std::span<const float> x, const T& y) {
    return quantization::lvq::compute_quantized(distance, x, y);
}
} // namespace distance

namespace quantization::lvq {

/// Distance computations supporting a global vector bias.
class EuclideanBiased {
  public:
    using compare = std::less<>;
    // Biased versions are not implicitly broadcastable because they must maintain per-query
    // state.
    static constexpr bool implicit_broadcast = false;
    static constexpr bool must_fix_argument = true;

    // Constructors
    EuclideanBiased(const std::shared_ptr<const data::SimpleData<float>>& centroids)
        : processed_query_(centroids->size(), centroids->dimensions())
        , centroids_{centroids} {}

    EuclideanBiased(std::shared_ptr<const data::SimpleData<float>>&& centroids)
        : processed_query_(centroids->size(), centroids->dimensions())
        , centroids_{std::move(centroids)} {}

    EuclideanBiased(const std::vector<float>& centroid)
        : processed_query_(1, centroid.size())
        , centroids_{} {
        // Construct the shared pointer by first creating a non-const version, then
        // using the copy-constructor.
        auto centroids = std::make_shared<data::SimpleData<float>>(1, centroid.size());
        centroids->set_datum(0, centroid);
        centroids_ = centroids;
    }

    // Shallow Copy
    // Don't preserve the state of `processed_query_`.
    EuclideanBiased shallow_copy() const { return EuclideanBiased{centroids_}; }

    ///
    /// Subtract each centroid from the query and store the result in `processed_query_`.
    /// This essentially moves the query by the same amount as the original data point,
    /// preserving L2 distance.
    ///
    void fix_argument(const std::span<const float>& query) {
        // Check pre-conditions.
        assert(centroids_->dimensions() == query.size());
        // Component-wise add the bias to the query and cache the result.
        auto jmax = query.size();
        for (size_t i = 0, imax = centroids_->size(); i < imax; ++i) {
            const auto& centroid = centroids_->get_datum(i);
            auto dst = processed_query_.get_datum(i);
            for (size_t j = 0; j < jmax; ++j) {
                dst[j] = query[j] - centroid[j];
            }
        }
    }

    // For testing purposes.
    template <typename T, std::integral I = size_t>
    float compute(const T& y, I selector = 0) const
        requires(lib::is_spanlike_v<T>)
    {
        auto inner = distance::DistanceL2{};
        return distance::compute(inner, view_query(selector), y);
    }

    ///
    /// Compute the Euclidean difference between a quantized vector `y` and a cached
    /// shifted query.
    ///
    template <LVQCompressedVector T> float compute(const T& y) const {
        // If the argument `y` is a `std::span`, it's not a compressed vector so fall-back
        // to doing normal distance computations.
        distance::DistanceL2 inner{};
        return distance::compute(inner, view_query(y.get_selector()), y);
    }

    std::span<const float> view_query(size_t i) const {
        return processed_query_.get_datum(i);
    }

    ///
    /// Return the global bias as a `std::span`.
    ///
    data::ConstSimpleDataView<float> view_bias() const { return centroids_->cview(); }

    std::span<const float> get_centroid(size_t i) const { return centroids_->get_datum(i); }

  private:
    data::SimpleData<float> processed_query_;
    std::shared_ptr<const data::SimpleData<float>> centroids_;
};

inline bool operator==(const EuclideanBiased& x, const EuclideanBiased& y) {
    return x.view_bias() == y.view_bias();
}

class InnerProductBiased {
  public:
    using compare = std::greater<>;
    // Biased versions are not implicitly broadcastable because they must maintain per-query
    // state.
    static constexpr bool implicit_broadcast = false;
    static constexpr bool must_fix_argument = true;

    // Constructor
    InnerProductBiased(const std::shared_ptr<const data::SimpleData<float>>& centroids)
        : processed_query_(centroids->size())
        , centroids_{centroids} {}

    InnerProductBiased(std::shared_ptr<const data::SimpleData<float>>&& centroids)
        : processed_query_(centroids->size())
        , centroids_{std::move(centroids)} {}

    InnerProductBiased(const std::vector<float>& centroid)
        : processed_query_(1)
        , centroids_{} {
        // Construct the shared pointer by first creating a non-const version, then
        // using the copy-constructor.
        auto centroids = std::make_shared<data::SimpleData<float>>(1, centroid.size());
        centroids->set_datum(0, centroid);
        centroids_ = centroids;
    }

    // Shallow Copy
    InnerProductBiased shallow_copy() const { return InnerProductBiased{centroids_}; }

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
        assert(centroids_->dimensions() == query.size());
        assert(processed_query_.size() == centroids_->size());

        // Pre-compute the inner-product between the query and each centroid.
        distance::DistanceIP inner_distance{};
        for (size_t i = 0, imax = centroids_->size(); i < imax; ++i) {
            processed_query_[i] =
                distance::compute(inner_distance, query, centroids_->get_datum(i));
        }

        // This preprocessing needed for DistanceFastIP
        query_sum_ = eve::algo::reduce(query, 0.0f);
    }

    template <size_t N, typename T, std::integral I = size_t>
    float compute(const std::span<const float, N>& query, const T& y, I selector = 0) const
        requires(lib::is_spanlike_v<T>)
    {
        // If the argument `y` is a `std::span`, it's not a compressed vector so fall-back
        // to doing normal distance computations.
        auto inner = distance::DistanceIP{};
        return distance::compute(inner, query, y) + processed_query_[selector];
    }

    template <LVQCompressedVector T>
    float compute(const std::span<const float>& query, const T& y) const {
        // Defaults to optimized inner product calculation
        DistanceFastIP inner{query_sum_};
        return distance::compute(inner, query, y) + processed_query_[y.get_selector()];
    }

    ///
    /// Return the global bias as a `std::span`.
    ///
    data::ConstSimpleDataView<float> view_bias() const { return centroids_->cview(); }

    std::span<const float> get_centroid(size_t i) const { return centroids_->get_datum(i); }

  private:
    // The results of computing the inner product between each centroid and the query.
    // Applied after the distance computation between the query and compressed vector.
    std::vector<float> processed_query_;
    std::shared_ptr<const data::SimpleData<float>> centroids_;
    float query_sum_ = 0;
};

inline bool operator==(const InnerProductBiased& x, const InnerProductBiased& y) {
    return x.view_bias() == y.view_bias();
}

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
/// Compute the correct biased distance function to operate on compressed data given the
/// original distance function `T`.
///
template <typename T> using biased_distance_t = typename detail::BiasedDistance<T>::type;

/////
///// Support for index building.
/////

///
/// Adaptor to adjust a distance function with type `Distance` to enable index building
/// over a compressed dataset.
///
/// Essentially, allows for distance computations between two elements of a compressed
/// dataset.
///
template <typename Distance> class DecompressionAdaptor {
  public:
    using distance_type = Distance;
    using compare = distance::compare_t<distance_type>;
    static constexpr bool implicit_broadcast = false;
    static constexpr bool must_fix_argument = true;

    DecompressionAdaptor(const distance_type& inner, size_t size_hint = 0)
        : inner_{inner}
        , decompressed_(size_hint) {}

    DecompressionAdaptor(distance_type&& inner, size_t size_hint = 0)
        : inner_{std::move(inner)}
        , decompressed_(size_hint) {}

    ///
    /// @brief Construct the internal portion of DecompressionAdaptor directly.
    ///
    /// The goal of the decompression adaptor is to wrap around an inner distance functor
    /// and decompress the left-hand component when requested, forwarding the decompressed
    /// value to the inner functor upon future distance computations.
    ///
    /// The inner distance functor may have non-trivial state associated with it.
    /// This constructor allows to construction of that inner functor directly to avoid
    /// a copy or move constructor.
    ///
    template <typename... Args>
    DecompressionAdaptor(std::in_place_t SVS_UNUSED(tag), Args&&... args)
        : inner_{std::forward<Args>(args)...}
        , decompressed_() {}

    DecompressionAdaptor shallow_copy() const {
        return DecompressionAdaptor(inner_, decompressed_.size());
    }

    // Distance API.
    template <LVQCompressedVector Left> void fix_argument(Left left) {
        decompress(decompressed_, left, inner_.get_centroid(left.get_selector()).data());
        inner_.fix_argument(view());
    }

    template <LVQCompressedVector Right> float compute(const Right& right) const {
        return distance::compute(inner_, view(), right);
    }

    std::span<const float> view() const { return decompressed_; }

  private:
    distance_type inner_;
    std::vector<float> decompressed_;
};
} // namespace quantization::lvq
} // namespace svs
