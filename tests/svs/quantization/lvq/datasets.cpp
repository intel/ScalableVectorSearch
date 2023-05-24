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

// header under test
#include "svs/quantization/lvq/datasets.h"

// random number generation.
#include "tests/svs/quantization/lvq/common.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// tests
#include "tests/utils/generators.h"
#include "tests/utils/test_dataset.h"

// Temporary helpers
#include "svs/index/flat/flat.h"
#include "svs/quantization/lvq/codec.h"

namespace lvq = svs::quantization::lvq;
using DistanceL2 = svs::distance::DistanceL2;
using DistanceIP = svs::distance::DistanceIP;

namespace {

template <typename T, typename Allocator>
std::span<T> as_span(std::vector<T, Allocator>& v) {
    return std::span<T>(v.data(), v.size());
}

template <typename T, typename Allocator>
std::span<const T> as_const_span(const std::vector<T, Allocator>& v) {
    return std::span<const T>(v.data(), v.size());
}

const size_t NTESTS = 10;

template <typename T, typename Other>
void compare(const std::vector<T>& x, const Other& y) {
    CATCH_REQUIRE((x.size() == y.size()));
    for (size_t j = 0, jmax = x.size(); j < jmax; ++j) {
        CATCH_REQUIRE((x.at(j) == y.get(j)));
    }
}

// Convert a float value to Float16 and back.
float through_float16(float x) { return svs::Float16(x); }

/////
///// Layout Helpers
/////

///
/// Test CompressedVector (CV) layout.
///
template <typename Sign, size_t Bits, size_t Extent>
void test_cv_layout(svs::lib::MaybeStatic<Extent> dims = {}) {
    using Layout = lvq::CompressedVectorLayout<Sign, Bits, Extent>;
    using T = typename Layout::encoding_type;
    auto layout = Layout{dims};
    size_t N = layout.total_bytes();

    auto reference = std::vector<T>(dims);
    // Backing buffers for vector views.
    auto a = std::vector<std::byte>(N);
    auto b = std::vector<std::byte>(N);
    auto generator = test_q::create_generator<Sign, Bits>();

    for (size_t i = 0; i < NTESTS; ++i) {
        svs_test::populate(reference, generator);
        layout.set(as_span(a), reference);
        compare(reference, layout.get(as_const_span(a)));

        // Test assignment of the direct data type.
        layout.set(as_span(b), layout.get(as_const_span(a)));
        compare(reference, layout.get(as_const_span(b)));
    }
}

///
/// Test ScaledBiasedVector (SBV) layout.
///
template <size_t Bits, size_t Extent>
void test_sbv_layout(svs::lib::MaybeStatic<Extent> dims = {}) {
    auto layout = lvq::ScaledBiasedVectorLayout<Bits, Extent>{dims};
    size_t N = layout.total_bytes();

    auto reference = std::vector<uint8_t>(dims);
    // Backing buffers for vector views.
    auto a = std::vector<std::byte>(N);
    auto b = std::vector<std::byte>(N);

    auto generator = test_q::create_generator<lvq::Unsigned, Bits>();
    auto float_generator = svs_test::make_generator<float>(0, 100);

    for (size_t i = 0; i < NTESTS; ++i) {
        svs_test::populate(reference, generator);
        float scale = svs_test::generate(float_generator);
        float bias = svs_test::generate(float_generator);

        // Assignment from vector and direct scale/bias.
        layout.set(as_span(a), scale, bias, reference);
        auto x = layout.get(as_const_span(a));
        CATCH_REQUIRE((x.scale == through_float16(scale)));
        CATCH_REQUIRE((x.bias == through_float16(bias)));
        compare(reference, x.data);

        // Test assignment through same data type.
        layout.set(as_span(b), layout.get(as_const_span(a)));
        auto y = layout.get(as_const_span(b));
        CATCH_REQUIRE((y.scale == through_float16(scale)));
        CATCH_REQUIRE((y.bias == through_float16(bias)));
        compare(reference, y.data);
    }
}

/////
///// Dataset Helpers
/////

template <typename Reference, typename Dataset>
void test_comparison(const Reference& x, const Dataset& y) {
    CATCH_REQUIRE((x.size() == y.size()));
    for (size_t i = 0, imax = x.size(); i < imax; ++i) {
        CATCH_REQUIRE(x.compare(i, y.get_datum(i)));
    }
}

template <typename Dataset> Dataset make_copy(const Dataset& data) {
    auto other = Dataset(data.size(), data.static_dims());
    for (size_t i = 0, imax = data.size(); i < imax; ++i) {
        other.set_datum(i, data.get_datum(i));
    }
    return other;
}

template <size_t Bits, size_t Extent>
lvq::GlobalScaledBiasedDataset<Bits, Extent>
make_copy(const lvq::GlobalScaledBiasedDataset<Bits, Extent>& data) {
    auto other = lvq::GlobalScaledBiasedDataset<Bits, Extent>(
        data.size(), data.static_dims(), data.get_scale(), data.get_bias()
    );
    for (size_t i = 0, imax = data.size(); i < imax; ++i) {
        other.set_datum(i, data.get_datum(i));
    }
    return other;
}

///
/// Compressed Reference
///
class CompressedReference {
  private:
    std::vector<std::vector<int32_t>> reference_;

  public:
    CompressedReference() = default;

    ///
    /// Reallocate reference data to have `size` vectors each with `ndims` dimensions.
    ///
    void configure(size_t ndims, size_t size) {
        reference_.resize(size, std::vector<int32_t>(ndims));
    }

    size_t size() const { return reference_.size(); }

    template <typename Sign, size_t Bits, size_t Extent>
    void populate(size_t size, svs::lib::MaybeStatic<Extent> dims = {}) {
        configure(dims, size);
        using Dataset = lvq::CompressedDataset<Sign, Bits, Extent>;
        // Create a random number generator for the dynamic range under test.
        auto generator = test_q::create_generator<Sign, Bits>();
        // Allocate the dataset and randomly generate the reference data while assiging
        // reference data to the compressed dataset.
        auto dataset = Dataset(size, dims);
        CATCH_REQUIRE((dataset.size() == size));
        CATCH_REQUIRE((dataset.dimensions() == dims));
        for (size_t i = 0; i < size; ++i) {
            auto& v = reference_.at(i);
            svs_test::populate(v, generator);
            dataset.set_datum(i, v);
        }
        // Make sure the dataset faithfully compresses the result.
        test_comparison(*this, dataset);
        test_comparison(*this, make_copy(dataset));

        // Make sure saving and loading works correctly.
        svs_test::prepare_temp_directory();
        auto dir = svs_test::temp_directory();
        svs::lib::save(dataset, dir);
        auto other = svs::lib::load<Dataset>(dir);
        test_comparison(*this, other);
    }

    template <typename Sign, size_t Bits, size_t N>
    bool compare(size_t i, lvq::CompressedVector<Sign, Bits, N> v) const {
        const auto& u = reference_.at(i);
        CATCH_REQUIRE((v.size() == u.size()));
        for (size_t j = 0, jmax = v.size(); j < jmax; ++j) {
            if (u.at(j) != v.get(j)) {
                return false;
            }
        }
        return true;
    }
};

///
/// Scaled Biased Dataset Reference
///
class ScaledBiasedReference {
  private:
    std::vector<std::vector<int32_t>> reference_;
    std::vector<svs::Float16> scales_;
    std::vector<svs::Float16> biases_;

  public:
    ScaledBiasedReference() = default;

    ///
    /// Reallocate reference data to have `size` vectors each with `ndims` dimensions.
    ///
    void configure(size_t ndims, size_t size) {
        reference_.resize(size, std::vector<int32_t>(ndims));
        scales_.resize(size);
        biases_.resize(size);
    }

    size_t size() const { return reference_.size(); }

    template <size_t Bits, size_t Extent>
    void populate(size_t size, svs::lib::MaybeStatic<Extent> dims = {}) {
        configure(dims, size);
        using Dataset = lvq::ScaledBiasedDataset<Bits, Extent>;
        auto generator = test_q::create_generator<lvq::Unsigned, Bits>();
        auto float_generator = svs_test::make_generator<float>(0, 100);

        auto dataset = Dataset(size, dims);
        CATCH_REQUIRE((dataset.size() == size));
        CATCH_REQUIRE((dataset.dimensions() == dims));
        if (Extent != svs::Dynamic) {
            CATCH_REQUIRE((dataset.dimensions() == Extent));
        }
        for (size_t i = 0; i < size; ++i) {
            // Randomly assign the scale and bias.
            float scale = svs_test::generate(generator);
            float bias = svs_test::generate(generator);
            scales_.at(i) = svs::lib::narrow_cast<svs::Float16>(scale);
            biases_.at(i) = svs::lib::narrow_cast<svs::Float16>(bias);
            auto& v = reference_.at(i);
            svs_test::populate(v, generator);
            dataset.set_datum(i, scale, bias, v);
        }
        // Make sure the dataset faithfully compresses the result.
        test_comparison(*this, dataset);
        test_comparison(*this, make_copy(dataset));

        // Make sure saving and loading works correctly.
        svs_test::prepare_temp_directory();
        auto dir = svs_test::temp_directory();
        svs::lib::save(dataset, dir);
        auto other = svs::lib::load<Dataset>(dir);
        test_comparison(*this, other);
    }

    template <size_t Bits, size_t Extent>
    bool compare(size_t i, lvq::ScaledBiasedVector<Bits, Extent> v) const {
        // Compare scale and bias.
        CATCH_REQUIRE((v.scale == scales_.at(i)));
        CATCH_REQUIRE((v.bias == biases_.at(i)));
        // Compare compressed data.
        const auto& u = reference_.at(i);
        CATCH_REQUIRE((v.size() == u.size()));
        for (size_t j = 0, jmax = v.size(); j < jmax; ++j) {
            if (u.at(j) != v.data.get(j)) {
                return false;
            }
        }
        return true;
    }
};

class GlobalScaledBiasedReference {
  private:
    std::vector<std::vector<int32_t>> reference_{};
    float scale_ = 0;
    float bias_ = 0;

  public:
    GlobalScaledBiasedReference() = default;

    ///
    /// Reallocate reference data to have `size` vectors each with `ndims` dimensions.
    ///
    void configure(size_t ndims, size_t size) {
        reference_.resize(size, std::vector<int32_t>(ndims));
    }

    size_t size() const { return reference_.size(); }

    template <size_t Bits, size_t Extent>
    void populate(size_t size, svs::lib::MaybeStatic<Extent> dims = {}) {
        configure(dims, size);
        using Dataset = lvq::GlobalScaledBiasedDataset<Bits, Extent>;
        auto generator = test_q::create_generator<lvq::Unsigned, Bits>();

        // Set the global scale and bias.
        auto float_generator = svs_test::make_generator<float>(0, 100);
        scale_ = svs_test::generate(generator);
        bias_ = svs_test::generate(generator);

        auto dataset = Dataset(size, dims, scale_, bias_);
        CATCH_REQUIRE((dataset.size() == size));
        CATCH_REQUIRE((dataset.dimensions() == dims));
        if (Extent != svs::Dynamic) {
            CATCH_REQUIRE(dataset.dimensions() == Extent);
        }

        for (size_t i = 0; i < size; ++i) {
            auto& v = reference_.at(i);
            svs_test::populate(v, generator);
            dataset.set_datum(i, v);
        }
        // Make sure the dataset faithfully compresses the result.
        test_comparison(*this, dataset);
        test_comparison(*this, make_copy(dataset));

        // Make sure  loading and saving works correctly.
        svs_test::prepare_temp_directory();
        auto dir = svs_test::temp_directory();
        svs::lib::save(dataset, dir);
        auto other = svs::lib::load<Dataset>(dir);
        test_comparison(*this, other);
    }

    template <size_t Bits, size_t N>
    bool compare(size_t i, lvq::ScaledBiasedVector<Bits, N> v) const {
        // Compare scale and bias.
        CATCH_REQUIRE((v.scale == scale_));
        CATCH_REQUIRE((v.bias == bias_));
        // Compare compressed data.
        const auto& u = reference_.at(i);
        CATCH_REQUIRE((v.size() == u.size()));
        for (size_t j = 0, jmax = v.size(); j < jmax; ++j) {
            if (u.at(j) != v.data.get(j)) {
                return false;
            }
        }
        return true;
    }
};

// Alias to reduce visual clutter.
template <size_t N> using Val = svs::meta::Val<N>;

} // namespace

CATCH_TEST_CASE("Compressed Dataset", "[quantization][vector_quantization]") {
    // Use a weird size of the test dimensions to ensure that odd-sized edge cases
    // are handled appropriately.
    constexpr size_t TEST_DIM = 37;
    const size_t dataset_size = 100;
    auto bits = std::make_tuple(Val<8>(), Val<7>(), Val<6>(), Val<5>(), Val<4>(), Val<3>());
    CATCH_SECTION("Layout Helpers") {
        svs::lib::foreach (bits, []<size_t N>(Val<N> /*unused*/) {
            test_cv_layout<lvq::Signed, N, TEST_DIM>();
            test_cv_layout<lvq::Unsigned, N, TEST_DIM>();
            test_cv_layout<lvq::Signed, N, svs::Dynamic>(svs::lib::MaybeStatic(TEST_DIM));
            test_cv_layout<lvq::Unsigned, N, svs::Dynamic>(svs::lib::MaybeStatic(TEST_DIM));

            // test_sv_layout<N, TEST_DIM>();
            test_sbv_layout<N, TEST_DIM>();
            test_sbv_layout<N, svs::Dynamic>(svs::lib::MaybeStatic(TEST_DIM));
        });
    }

    CATCH_SECTION("Compressed Dataset") {
        auto tester = CompressedReference();
        svs::lib::foreach (bits, [&]<size_t N>(Val<N> /*unused*/) {
            tester.template populate<lvq::Signed, N, TEST_DIM>(dataset_size);
            tester.template populate<lvq::Unsigned, N, TEST_DIM>(dataset_size);

            tester.template populate<lvq::Signed, N, svs::Dynamic>(
                dataset_size, svs::lib::MaybeStatic(TEST_DIM)
            );
            tester.template populate<lvq::Unsigned, N, svs::Dynamic>(
                dataset_size, svs::lib::MaybeStatic(TEST_DIM)
            );
        });
    }

    CATCH_SECTION("Scaled Biased Dataset") {
        auto tester = ScaledBiasedReference();
        svs::lib::foreach (bits, [&]<size_t N>(Val<N> /*unused*/) {
            tester.template populate<N, TEST_DIM>(dataset_size);
            tester.template populate<N, svs::Dynamic>(
                dataset_size, svs::lib::MaybeStatic(TEST_DIM)
            );
        });
    }

    CATCH_SECTION("Global Scaled Biased Dataset") {
        auto tester = GlobalScaledBiasedReference();
        svs::lib::foreach (bits, [&]<size_t N>(Val<N> /*unused*/) {
            tester.template populate<N, TEST_DIM>(dataset_size);
            tester.template populate<N, svs::Dynamic>(
                dataset_size, svs::lib::MaybeStatic(TEST_DIM)
            );
        });
    }
}
