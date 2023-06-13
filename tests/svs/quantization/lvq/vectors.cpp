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

// Header under test
#include "svs/quantization/lvq/vectors.h"
#include "svs/quantization/lvq/codec.h"

// tests
#include "tests/svs/quantization/lvq/common.h"
#include "tests/utils/generators.h"

// svs
#include "svs/lib/range.h"
#include "svs/lib/timing.h"

// Catch
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

namespace lvq = svs::quantization::lvq;

namespace {

namespace detail {
template <typename T> struct ReferenceDistance;
template <> struct ReferenceDistance<svs::distance::DistanceL2> {
    using type = lvq::EuclideanReference;
};
template <> struct ReferenceDistance<svs::distance::DistanceIP> {
    using type = lvq::InnerProductReference;
};
} // namespace detail

///
/// Map standard distance types to their reference implementations for quantized vectors.
///
template <typename T>
using reference_distance_t = typename detail::ReferenceDistance<T>::type;

// Extract the contents of a compressed vector variant into a vector of floats.
template <typename T> std::vector<float> slurp(const T& x) {
    const size_t sz = x.size();
    auto v = std::vector<float>(sz);
    for (size_t i = 0; i < sz; ++i) {
        v[i] = x.get(i);
    }
    return v;
}

///
/// Often, scale parameters are stored as `Float16` values inside the compressed structures.
/// By passing a floating point value through a `Float16`, we can ensure that the floating
/// point values we use are losslessly convertible to `Float16`.
///
/// This simplifies checking the operations are preserved without needing to resort to
/// approximate computations.
///
float through_float16(float x) { return svs::Float16(x); }

// Lazy way of getting the generator type.
template <typename Sign, size_t Bits>
using compressed_generator_t = decltype(test_q::create_generator<Sign, Bits>());

namespace test_fixtures {
template <size_t Bits, size_t Extent> struct ScaledBiased {
  private:
    [[no_unique_address]] svs::lib::MaybeStatic<Extent> size_;
    compressed_generator_t<lvq::Unsigned, Bits> generator_;
    Catch::Generators::GeneratorWrapper<float> float_;
    lvq::CVStorage compressed_{};
    std::vector<float> reference_{};

  public:
    using vector_type = lvq::ScaledBiasedVector<Bits, Extent>;
    static constexpr float float_min = -3;
    static constexpr float float_max = 3;

    ScaledBiased(svs::lib::MaybeStatic<Extent> size = {})
        : size_{size}
        , generator_{test_q::create_generator<lvq::Unsigned, Bits>()}
        , float_{svs_test::make_generator<float>(float_min, float_max)} {}

    svs::lib::MaybeStatic<Extent> static_size() const { return size_; }
    size_t size() const { return static_size(); }

    std::pair<vector_type, std::span<float>> generate() {
        // Generate reference encodings into the generator.
        reference_.resize(size());
        svs_test::populate(reference_, generator_);

        // Pipe the scaling parameter through Float16 precision to ensure that reconstructed
        // values are the same.
        float scale = through_float16(svs_test::generate(float_));
        float bias = through_float16(svs_test::generate(float_));
        auto cv = compressed_.template view<lvq::Unsigned, Bits, Extent>(static_size());
        CATCH_REQUIRE((cv.size() == reference_.size()));
        CATCH_REQUIRE((cv.size() == size()));
        for (size_t i = 0, imax = cv.size(); i < imax; ++i) {
            cv.set(reference_[i], i);
            reference_[i] = (scale * reference_[i]) + bias;
        }
        return std::make_pair(
            vector_type{scale, bias, 0, cv.as_const()}, svs::lib::as_span(reference_)
        );
    }

    // N.B.: The copy constructor for the Catch random number generator is deleted.
    // SO, we need to explicitly describe how to copy this.
    ScaledBiased copy() const { return ScaledBiased(static_size()); }
};

template <size_t Primary, size_t Residual, size_t Extent> struct ScaledBiasedWithResidual {
  private:
    [[no_unique_address]] svs::lib::MaybeStatic<Extent> size_;
    compressed_generator_t<lvq::Unsigned, Primary> primary_generator_;
    compressed_generator_t<lvq::Signed, Residual> residual_generator_;
    Catch::Generators::GeneratorWrapper<float> float_;
    lvq::CVStorage primary_{};
    lvq::CVStorage residual_{};
    std::vector<float> reference_{};

  public:
    using vector_type = lvq::ScaledBiasedWithResidual<Primary, Residual, Extent>;
    static constexpr float float_min = -3;
    static constexpr float float_max = 3;

    ScaledBiasedWithResidual(svs::lib::MaybeStatic<Extent> size = {})
        : size_{size}
        , primary_generator_{test_q::create_generator<lvq::Unsigned, Primary>()}
        , residual_generator_{test_q::create_generator<lvq::Signed, Residual>()}
        , float_{svs_test::make_generator<float>(float_min, float_max)} {}

    svs::lib::MaybeStatic<Extent> static_size() const { return size_; }
    size_t size() const { return static_size(); }

    std::pair<vector_type, std::span<float>> generate() {
        // Generate reference encodings into the generator.
        reference_.resize(size());
        auto temp = std::vector<float>(reference_.size());
        svs_test::populate(reference_, primary_generator_);
        svs_test::populate(temp, residual_generator_);

        // Pipe the scaling parameter through Float16 precision to ensure that reconstructed
        // values are the same.
        float scale = through_float16(svs_test::generate(float_));
        float bias = through_float16(svs_test::generate(float_));

        auto primary =
            primary_.template view<lvq::Unsigned, Primary, Extent>(static_size());
        auto residual =
            residual_.template view<lvq::Signed, Residual, Extent>(static_size());
        CATCH_REQUIRE((primary.size() == reference_.size()));
        CATCH_REQUIRE((primary.size() == size()));
        CATCH_REQUIRE((residual.size() == reference_.size()));
        CATCH_REQUIRE((residual.size() == size()));

        if (Extent != svs::Dynamic) {
            CATCH_REQUIRE((primary.size() == Extent));
            CATCH_REQUIRE((residual.size() == Extent));
        }

        float multiplier = std::pow(2, Residual);
        for (size_t i = 0, imax = reference_.size(); i < imax; ++i) {
            auto x = reference_[i];
            auto y = temp[i];

            primary.set(reference_[i], i);
            residual.set(temp[i], i);

            reference_[i] = scale * (x + (y / multiplier)) + bias;
        }

        auto v = vector_type{
            lvq::ScaledBiasedVector(scale, bias, 0, primary.as_const()),
            residual.as_const()};
        return std::make_pair(v, svs::lib::as_span(reference_));
    }

    // N.B.: The copy constructor for the Catch random number generator is deleted.
    // SO, we need to explicitly describe how to copy this.
    ScaledBiasedWithResidual copy() const {
        return ScaledBiasedWithResidual(static_size());
    }
};
} // namespace test_fixtures

/////
///// Distance Testing Routines.
/////

const size_t NUM_TESTS = 100;
template <typename TestGenerator, typename Distance>
void test_distance(TestGenerator& rhs, Distance distance, size_t num_tests = NUM_TESTS) {
    // Generator for random numbers for the LHS.
    auto generator = svs_test::make_generator<float>(-2, 2);
    auto lhs = std::vector<float>(rhs.size());

    auto compressed_distance_ref = reference_distance_t<Distance>();
    for (size_t i = 0; i < num_tests; ++i) {
        const auto& [rhs_compressed, rhs_ref] = rhs.generate();

        // Test that the generated values are the same.
        CATCH_REQUIRE((rhs_compressed.size() == rhs_ref.size()));
        for (size_t j = 0, jmax = rhs_compressed.size(); j < jmax; ++j) {
            // Compare with a tiny epsilon because the multilevel compression techniques
            // are subject to a small amount of floating point error.
            CATCH_REQUIRE(
                (rhs_compressed.get(j) == Catch::Approx(rhs_ref[j]).epsilon(0.00001))
            );
        }

        // Test distances
        svs_test::populate(lhs, generator);
        auto lhs_span = svs::lib::as_const_span(lhs);
        float reference = svs::distance::compute(distance, lhs_span, rhs_ref);

        // Reference distance computation.
        float compressed_ref =
            svs::distance::compute(compressed_distance_ref, lhs_span, rhs_compressed);
        CATCH_REQUIRE(
            reference == Catch::Approx(compressed_ref).epsilon(0.0001).margin(0.001)
        );

        // Accelerated distance computation.
        float compressed_avx = svs::distance::compute(distance, lhs_span, rhs_compressed);
        CATCH_REQUIRE(
            reference == Catch::Approx(compressed_avx).epsilon(0.0001).margin(0.001)
        );
    }
}

template <typename TestGenerator, typename Distance>
void test_biased_distance(
    TestGenerator& rhs, Distance distance, size_t num_tests = NUM_TESTS
) {
    // Generator for random numbers for the LHS.
    auto generator = svs_test::make_generator<float>(-2, 2);
    auto lhs = std::vector<float>(rhs.size());

    // Fill out a random bias.
    auto bias = std::vector<float>(rhs.size());
    svs_test::populate(bias, svs_test::make_generator<float>(-100, 100));

    // Instantiate the distance struct that contains the
    auto distance_bias = lvq::biased_distance_t<Distance>(bias);
    for (size_t i = 0; i < num_tests; ++i) {
        const auto& [rhs_compressed, rhs_ref] = rhs.generate();

        // Add the bias component to the reference RHS argument.
        CATCH_REQUIRE((bias.size() == rhs_ref.size()));
        for (size_t j = 0, jmax = rhs_ref.size(); j < jmax; ++j) {
            rhs_ref[j] += bias[j];
        }

        // Test distances
        svs_test::populate(lhs, generator);
        auto lhs_span = svs::lib::as_const_span(lhs);
        float reference = svs::distance::compute(distance, lhs_span, rhs_ref);

        svs::distance::maybe_fix_argument(distance_bias, lhs_span);
        float dist = svs::distance::compute(distance_bias, lhs_span, rhs_compressed);
        CATCH_REQUIRE(reference == Catch::Approx(dist).epsilon(0.0001).margin(0.001));
    }
}

///
/// Test computation of distances between two vectors using the same compression scheme
/// using a global bias.
///
template <typename TestGenerator, typename Distance>
void test_biased_self_distance(
    TestGenerator& rhs, Distance distance, size_t num_tests = NUM_TESTS
) {
    // Copy the generator to make an independent version for the left hand side.
    auto lhs = rhs.copy();

    auto bias = std::vector<float>(rhs.size());
    svs_test::populate(bias, svs_test::make_generator<float>(-10, 10));

    // Construct the self distance function through the biased distance.
    auto distance_bias = lvq::biased_distance_t<Distance>(bias);
    auto distance_self = lvq::DecompressionAdaptor{distance_bias};

    for (size_t i = 0; i < num_tests; ++i) {
        const auto& [lhs_compressed, lhs_ref] = lhs.generate();
        const auto& [rhs_compressed, rhs_ref] = rhs.generate();

        // Add the bias into the reference vectors.
        CATCH_REQUIRE((lhs_ref.size() == bias.size()));
        CATCH_REQUIRE((rhs_ref.size() == bias.size()));
        for (size_t j = 0, jmax = bias.size(); j < jmax; ++j) {
            auto b = bias[j];
            lhs_ref[j] += b;
            rhs_ref[j] += b;
        }

        // Test distances
        float reference = svs::distance::compute(distance, lhs_ref, rhs_ref);
        svs::distance::maybe_fix_argument(distance_self, lhs_compressed);
        float dist = svs::distance::compute(distance_self, lhs_compressed, rhs_compressed);
        CATCH_REQUIRE(reference == Catch::Approx(dist).epsilon(0.01).margin(0.02));
    }
}

// reduce visual clutter
template <size_t N> using Val = svs::meta::Val<N>;

} // namespace

CATCH_TEST_CASE("Compressed Vector Variants", "[quantization][lvq]") {
    using DistanceL2 = svs::distance::DistanceL2;
    using DistanceIP = svs::distance::DistanceIP;

    const size_t TEST_DIM = 37;
    auto bits = std::make_tuple(Val<8>(), Val<7>(), Val<6>(), Val<5>(), Val<4>(), Val<3>());

    CATCH_SECTION("ScaledBiasedVector") {
        // Statically Sized
        svs::lib::foreach (bits, []<size_t N>(Val<N> /*unused*/) {
            auto generator = test_fixtures::ScaledBiased<N, TEST_DIM>();
            test_distance(generator, DistanceL2());
            test_distance(generator, DistanceIP());
            test_biased_distance(generator, DistanceL2());
            test_biased_distance(generator, DistanceIP());
            test_biased_self_distance(generator, DistanceL2());
            test_biased_self_distance(generator, DistanceIP());
        });

        // Dynamically Sized
        svs::lib::foreach (bits, []<size_t N>(Val<N> /*unused*/) {
            auto generator =
                test_fixtures::ScaledBiased<N, svs::Dynamic>(svs::lib::MaybeStatic(TEST_DIM)
                );
            test_distance(generator, DistanceL2());
            test_distance(generator, DistanceIP());
            test_biased_distance(generator, DistanceL2());
            test_biased_distance(generator, DistanceIP());
            test_biased_self_distance(generator, DistanceL2());
            test_biased_self_distance(generator, DistanceIP());
        });
    }

    CATCH_SECTION("ScaledBiasedWithResidual") {
        auto residuals = std::make_tuple(Val<4>(), Val<3>());
        auto timer = svs::lib::Timer();
        // Static Case
        auto static_case = timer.push_back("static residul computation");
        svs::lib::foreach (bits, [&residuals]<size_t N>(Val<N> /*unused*/) {
            svs::lib::foreach (residuals, []<size_t M>(Val<M> /*unused*/) {
                auto generator = test_fixtures::ScaledBiasedWithResidual<N, M, TEST_DIM>();
                test_distance(generator, DistanceL2());
                test_distance(generator, DistanceIP());
                test_biased_distance(generator, DistanceL2());
                test_biased_distance(generator, DistanceIP());
                test_biased_self_distance(generator, DistanceL2());
                test_biased_self_distance(generator, DistanceIP());
            });
        });
        static_case.finish();

        // Dynamic Case
        auto dynamic_case = timer.push_back("dynamic residual computation");
        svs::lib::foreach (bits, [&residuals]<size_t N>(Val<N> /*unused*/) {
            svs::lib::foreach (residuals, []<size_t M>(Val<M> /*unused*/) {
                auto generator =
                    test_fixtures::ScaledBiasedWithResidual<N, M, svs::Dynamic>(
                        svs::lib::MaybeStatic(TEST_DIM)
                    );
                test_distance(generator, DistanceL2());
                test_distance(generator, DistanceIP());
                test_biased_distance(generator, DistanceL2());
                test_biased_distance(generator, DistanceIP());
                test_biased_self_distance(generator, DistanceL2());
                test_biased_self_distance(generator, DistanceIP());
            });
        });
        dynamic_case.finish();
        timer.print();
    }
}
