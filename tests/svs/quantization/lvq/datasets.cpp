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

const size_t NTESTS = 10;

template <size_t N = svs::Dynamic> using MaybeStatic = svs::lib::MaybeStatic<N>;

template <typename T, typename Other>
void compare(const std::vector<T>& x, const Other& y) {
    CATCH_REQUIRE((x.size() == y.size()));
    for (size_t j = 0, jmax = x.size(); j < jmax; ++j) {
        CATCH_REQUIRE((x.at(j) == y.get(j)));
    }
}

// Convert a float value to Float16 and back.
float through_float16(float x) { return svs::Float16(x); }

template <typename T>
std::vector<T> compact_vector(const std::vector<T>& original, std::span<const size_t> ids) {
    CATCH_REQUIRE(std::is_sorted(ids.begin(), ids.end()));
    auto result = std::vector<T>();
    for (const auto& id : ids) {
        result.push_back(original.at(id));
    }
    return result;
}

template <typename T> std::vector<T> get_last(const std::vector<T>& original, size_t n) {
    CATCH_REQUIRE(original.size() >= n);
    return std::vector<T>(original.end() - n, original.end());
}

/////
///// Layout Helpers
/////

///
/// Test ScaledBiasedVector (SBV) layout.
///
template <size_t Bits, size_t Extent, lvq::LVQPackingStrategy Strategy = lvq::Sequential>
void test_sbv_layout(MaybeStatic<Extent> dims = {}) {
    auto layout = lvq::ScaledBiasedVectorLayout<Bits, Extent, Strategy>{dims};
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
        layout.set(svs::lib::as_span(a), scale, bias, 10, reference);
        auto x = layout.get(svs::lib::as_const_span(a));
        CATCH_REQUIRE((x.scale == through_float16(scale)));
        CATCH_REQUIRE((x.bias == through_float16(bias)));
        CATCH_REQUIRE(x.get_selector() == 10);
        compare(reference, x.data);

        // Test assignment through same data type.
        layout.set(svs::lib::as_span(b), layout.get(svs::lib::as_const_span(a)));
        auto y = layout.get(svs::lib::as_const_span(b));
        CATCH_REQUIRE((y.scale == through_float16(scale)));
        CATCH_REQUIRE((y.bias == through_float16(bias)));
        CATCH_REQUIRE(y.get_selector() == 10);
        compare(reference, y.data);
    }
}

/////
///// Canonicalizer
/////

// Test canonicalization with both static and dynamic extents with the given source
template <size_t Bits, size_t Extent, lvq::LVQPackingStrategy Strategy>
void test_canonicalizer() {
    // Storage for the source buffer.
    auto source_buffer = std::vector<std::byte>();
    auto source_codes = std::vector<uint8_t>(Extent);
    auto to_canonical = lvq::detail::Canonicalizer();
    auto from_canonical = lvq::detail::Canonicalizer();

    auto rng = test_q::create_generator<lvq::Unsigned, Bits>();
    auto float_rng = svs_test::make_generator<float>(0, 100);

    // Inner lambda to test combinations of static and dynamic dimensions.
    auto do_test = [&]<size_t N1>(svs::lib::MaybeStatic<N1> dims) {
        // Create the source object.
        auto layout = lvq::ScaledBiasedVectorLayout<Bits, N1, Strategy>(dims);
        source_buffer.resize(layout.total_bytes());

        svs_test::populate(source_codes, rng);
        layout.set(
            svs::lib::as_span(source_buffer),
            svs_test::generate(float_rng),
            svs_test::generate(float_rng),
            0,
            source_codes
        );

        ///// Convert to the canonical form.
        auto source = layout.get(svs::lib::as_const_span(source_buffer));
        std::span<const std::byte> canonical = to_canonical.to_canonical(source);

        // Ensure that when we interpret the canonical layout as a ScaledBiasedVector, we
        // get something that is logically equivalent to the original vector.
        auto canonical_layout =
            lvq::ScaledBiasedVectorLayout<Bits, svs::Dynamic, lvq::Sequential>(
                svs::lib::MaybeStatic(dims.size())
            );
        CATCH_REQUIRE(canonical.size() == canonical_layout.total_bytes());
        auto canonical_vector = canonical_layout.get(canonical);
        CATCH_REQUIRE(canonical_vector.scale == source.scale);
        CATCH_REQUIRE(canonical_vector.bias == source.bias);
        CATCH_REQUIRE(canonical_vector.selector == source.selector);
        CATCH_REQUIRE(lvq::logically_equal(canonical_vector.data, source.data));

        ///// Convert from canonical form.
        auto reconstructed = from_canonical.from_canonical(
            svs::lib::Type<lvq::ScaledBiasedVector<Bits, N1, Strategy>>(), canonical, dims
        );

        // Reconstructed type should match exactly.
        CATCH_STATIC_REQUIRE(std::is_same_v<decltype(reconstructed), decltype(source)>);

        // Here - we require both
        CATCH_REQUIRE(reconstructed.scale == source.scale);
        CATCH_REQUIRE(reconstructed.bias == source.bias);
        CATCH_REQUIRE(reconstructed.selector == source.selector);
        CATCH_REQUIRE(lvq::logically_equal(reconstructed.data, source.data));
    };

    // Test static and dynamic combinations.
    do_test(svs::lib::MaybeStatic<Extent>());
    do_test(svs::lib::MaybeStatic<svs::Dynamic>(Extent));
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

template <typename Sign, size_t Bits, size_t Extent, typename Alloc>
lvq::CompressedDataset<Sign, Bits, Extent, Alloc>
make_copy(lvq::CompressedDataset<Sign, Bits, Extent, Alloc>& data) {
    auto other = lvq::CompressedDataset<Sign, Bits, Extent, Alloc>(
        data.size(), data.static_dims(), data.get_allocator()
    );
    for (size_t i = 0, imax = data.size(); i < imax; ++i) {
        other.set_datum(i, data.get_datum(i));
    }
    return other;
}

template <size_t Bits, size_t Extent, lvq::LVQPackingStrategy Strategy, typename Alloc>
lvq::ScaledBiasedDataset<Bits, Extent, Strategy, Alloc>
make_copy(lvq::ScaledBiasedDataset<Bits, Extent, Strategy, Alloc>& data) {
    auto other = lvq::ScaledBiasedDataset<Bits, Extent, Strategy, Alloc>(
        data.size(), data.static_dims(), data.get_alignment(), data.get_allocator()
    );
    for (size_t i = 0, imax = data.size(); i < imax; ++i) {
        other.set_datum(i, data.get_datum(i));
    }
    return other;
}

// Take both arguments by value so we can mutate them without affecting the caller.
template <typename Reference, typename Dataset> void test_dynamic(Reference x, Dataset y) {
    test_comparison(x, y);
    // First, decrease the size by 10;
    CATCH_REQUIRE(x.size() >= 100);
    auto back = x.copy_last(10);
    x.resize(x.size() - 10);
    y.resize(y.size() - 10);
    test_comparison(x, y);
    // Add the points back.
    auto newsize = y.size();
    x.put_back(back);
    y.resize(x.size());
    back.assign(y, newsize);
    test_comparison(x, y);

    // Test compactions.
    auto compact_ids = std::vector<size_t>{};
    for (size_t i = 0, imax = x.size(); i < imax; i += 2) {
        compact_ids.push_back(i);
    }
    auto newx = x.compact(svs::lib::as_const_span(compact_ids));
    auto other = make_copy(y);
    other.compact(svs::lib::as_const_span(compact_ids));
    other.resize(newx.size());
    test_comparison(newx, other);
}

///
/// Compressed Reference
///
class CompressedReference {
  private:
    std::vector<std::vector<int32_t>> reference_;

  public:
    CompressedReference() = default;
    CompressedReference(std::vector<std::vector<int32_t>> reference)
        : reference_{std::move(reference)} {}

    ///
    /// Reallocate reference data to have `size` vectors each with `ndims` dimensions.
    ///
    void configure(size_t ndims, size_t size) {
        reference_.resize(size, std::vector<int32_t>(ndims));
    }

    size_t size() const { return reference_.size(); }
    void resize(size_t new_size) { reference_.resize(new_size); }

    template <typename Sign, size_t Bits, size_t Extent, typename Alloc>
    void assign(lvq::CompressedDataset<Sign, Bits, Extent, Alloc>& data, size_t start) {
        // Make sure we're filling to the end.
        CATCH_REQUIRE(data.size() == start + size());
        for (size_t i = 0, imax = size(); i < imax; ++i) {
            data.set_datum(i + start, reference_.at(i));
        }
    }

    template <typename Sign, size_t Bits, size_t Extent, typename Alloc>
    void populate(size_t size, MaybeStatic<Extent> dims = {}, const Alloc& allocator = {}) {
        configure(dims, size);
        using Dataset = lvq::CompressedDataset<Sign, Bits, Extent, Alloc>;

        // Create a random number generator for the dynamic range under test.
        auto generator = test_q::create_generator<Sign, Bits>();
        // Allocate the dataset and randomly generate the reference data while assiging
        // reference data to the compressed dataset.
        auto dataset = Dataset(size, dims, allocator);
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
        svs::lib::save_to_disk(dataset, dir);
        auto other = svs::lib::load_from_disk<Dataset>(dir, dataset.get_allocator());
        test_comparison(*this, other);

        // Test DatasetSummary
        auto summary = svs::lib::load_from_disk<lvq::DatasetSummary>(dir);
        CATCH_REQUIRE(summary.kind == lvq::DatasetSchema::Compressed);
        CATCH_REQUIRE(summary.is_signed == std::is_same_v<Sign, lvq::Signed>);
        CATCH_REQUIRE(summary.dims == dims);
        CATCH_REQUIRE(summary.bits == Bits);

        // Dynamic resizing.
        if constexpr (Dataset::is_resizeable) {
            test_dynamic(*this, make_copy(dataset));
        }
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

    CompressedReference copy_last(size_t n) {
        return CompressedReference{get_last(reference_, n)};
    }

    void put_back(const CompressedReference& other) {
        const auto& values = other.reference_;
        reference_.insert(reference_.end(), values.begin(), values.end());
    }

    CompressedReference compact(std::span<const size_t> indices) {
        return CompressedReference{compact_vector(reference_, indices)};
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

    // Book Keeping values.
    // We keep track of these variables to ensure we hit all branches of a rather large
    // if-constexpr chain for saving and loading.
    size_t reload_static_as_dynamic_ = 0;
    size_t reload_sequential_as_turbo_ = 0;
    size_t reload_turbo_as_sequential_ = 0;

  public:
    ScaledBiasedReference() = default;
    ScaledBiasedReference(
        std::vector<std::vector<int32_t>> reference,
        std::vector<svs::Float16> scales,
        std::vector<svs::Float16> biases
    )
        : reference_{std::move(reference)}
        , scales_{std::move(scales)}
        , biases_{std::move(biases)} {}

    ///
    /// Reallocate reference data to have `size` vectors each with `ndims` dimensions.
    ///
    void configure(size_t ndims, size_t size) {
        reference_.resize(size, std::vector<int32_t>(ndims));
        scales_.resize(size);
        biases_.resize(size);
    }

    size_t size() const { return reference_.size(); }
    void resize(size_t new_size) {
        reference_.resize(new_size);
        biases_.resize(new_size);
        scales_.resize(new_size);
    }

    template <size_t Bits, size_t Extent, lvq::LVQPackingStrategy Strategy, typename Alloc>
    void
    assign(lvq::ScaledBiasedDataset<Bits, Extent, Strategy, Alloc>& data, size_t start) {
        // Make sure we're filling to the end.
        CATCH_REQUIRE(data.size() == start + size());
        for (size_t i = 0, imax = size(); i < imax; ++i) {
            data.set_datum(i + start, scales_.at(i), biases_.at(i), 0, reference_.at(i));
        }
    }

    template <size_t Bits, size_t Extent, lvq::LVQPackingStrategy Strategy, typename Alloc>
    void populate(
        size_t size,
        MaybeStatic<Extent> dims = {},
        size_t alignment = 0,
        const Alloc& allocator = {}
    ) {
        configure(dims, size);
        using Dataset = lvq::ScaledBiasedDataset<Bits, Extent, Strategy, Alloc>;
        auto generator = test_q::create_generator<lvq::Unsigned, Bits>();
        auto float_generator = svs_test::make_generator<float>(0, 100);

        auto dataset = Dataset(size, dims, alignment, allocator);
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
            dataset.set_datum(i, scale, bias, 0, v);
        }
        // Make sure the dataset faithfully compresses the result.
        test_comparison(*this, dataset);
        test_comparison(*this, make_copy(dataset));

        // Make sure saving and loading works correctly.
        svs_test::prepare_temp_directory();
        auto dir = svs_test::temp_directory();

        test_save_load(dataset, dir);

        // Dataset Summary
        auto summary = svs::lib::load_from_disk<lvq::DatasetSummary>(dir);
        CATCH_REQUIRE(summary.kind == lvq::DatasetSchema::ScaledBiased);
        CATCH_REQUIRE(summary.is_signed == false);
        CATCH_REQUIRE(summary.dims == dims);
        CATCH_REQUIRE(summary.bits == Bits);

        // Resizing
        if constexpr (Dataset::is_resizeable) {
            test_dynamic(*this, make_copy(dataset));
        }
    }

    template <size_t Bits, size_t Extent, lvq::LVQPackingStrategy Strategy, typename Alloc>
    void test_save_load(
        const lvq::ScaledBiasedDataset<Bits, Extent, Strategy, Alloc>& dataset,
        const std::filesystem::path& dir
    ) {
        svs::lib::save_to_disk(dataset, dir);

        // Copy the original container's allocator to inherit any stateful properties.
        //
        // This is mostly needed when running in debug mode with the blocked allocator as
        // the default blocking parameter is quite high.
        //
        // Trivial constructors and destructors are not elided in debug mode which results
        // in run-time bloat.
        Alloc allocator = dataset.get_allocator();

        ///// Same Stratetgy
        // Load with different paddings.
        {
            using T = lvq::ScaledBiasedDataset<Bits, Extent, Strategy, Alloc>;
            auto other = svs::lib::load_from_disk<T>(dir, 0, allocator);
            test_comparison(*this, other);
            other = svs::lib::load_from_disk<T>(dir, 32, allocator);
            test_comparison(*this, other);
        }

        // Load with dynamic extent.
        if constexpr (Extent != svs::Dynamic) {
            using T = lvq::ScaledBiasedDataset<Bits, svs::Dynamic, Strategy, Alloc>;
            auto other = svs::lib::load_from_disk<T>(dir, 0, allocator);
            test_comparison(*this, other);
            other = svs::lib::load_from_disk<T>(dir, 32, allocator);
            test_comparison(*this, other);
            ++reload_static_as_dynamic_;
        }

        ///// Reload as turbo.
        if constexpr (Bits == 4 && std::is_same_v<Strategy, lvq::Sequential>) {
            using T =
                lvq::ScaledBiasedDataset<Bits, svs::Dynamic, lvq::Turbo<16, 8>, Alloc>;
            auto other = svs::lib::load_from_disk<T>(dir, 0, allocator);
            test_comparison(*this, other);
            other = svs::lib::load_from_disk<T>(dir, 32, allocator);
            test_comparison(*this, other);
            ++reload_sequential_as_turbo_;
        }

        ///// Reload as sequential.
        if constexpr (lvq::TurboLike<Strategy>) {
            using T = lvq::ScaledBiasedDataset<Bits, svs::Dynamic, lvq::Sequential, Alloc>;
            auto other = svs::lib::load_from_disk<T>(dir, 0, allocator);
            test_comparison(*this, other);
            other = svs::lib::load_from_disk<T>(dir, 32, allocator);
            test_comparison(*this, other);
            ++reload_turbo_as_sequential_;
        }
    }

    template <size_t Bits, size_t Extent, lvq::LVQPackingStrategy Strategy>
    bool compare(size_t i, lvq::ScaledBiasedVector<Bits, Extent, Strategy> v) const {
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

    ScaledBiasedReference copy_last(size_t n) {
        return ScaledBiasedReference{
            get_last(reference_, n), get_last(scales_, n), get_last(biases_, n)};
    }

    void put_back(const ScaledBiasedReference& other) {
        reference_.insert(
            reference_.end(), other.reference_.begin(), other.reference_.end()
        );
        scales_.insert(scales_.end(), other.scales_.begin(), other.scales_.end());
        biases_.insert(biases_.end(), other.biases_.begin(), other.biases_.end());
    }

    ScaledBiasedReference compact(std::span<const size_t> indices) {
        return ScaledBiasedReference(
            compact_vector(reference_, indices),
            compact_vector(scales_, indices),
            compact_vector(biases_, indices)
        );
    }

    // Make sure code-paths we care about were hit.
    void check_code_paths() const {
        CATCH_REQUIRE(reload_static_as_dynamic_ > 0);
        CATCH_REQUIRE(reload_sequential_as_turbo_ > 0);
        CATCH_REQUIRE(reload_turbo_as_sequential_ > 0);
    }
};

// Alias to reduce visual clutter.
template <size_t N> using Val = svs::lib::Val<N>;

} // namespace

CATCH_TEST_CASE("Compressed Dataset", "[quantization][lvq][lvq_datasets]") {
    // Use a weird size of the test dimensions to ensure that odd-sized edge cases
    // are handled appropriately.
    constexpr size_t TEST_DIM = 37;
    const size_t dataset_size = 100;
    auto bits = std::make_tuple(Val<8>(), Val<7>(), Val<6>(), Val<5>(), Val<4>(), Val<3>());
    CATCH_SECTION("Layout Helpers") {
        auto d = MaybeStatic<svs::Dynamic>(TEST_DIM);
        svs::lib::foreach (bits, [d]<size_t N>(Val<N> /*unused*/) {
            test_sbv_layout<N, TEST_DIM>();
            test_sbv_layout<N, svs::Dynamic>(d);
        });

        // Special case Turbo Strategies.
        test_sbv_layout<4, TEST_DIM, lvq::Turbo<16, 8>>();
        test_sbv_layout<4, svs::Dynamic, lvq::Turbo<16, 8>>(d);
    }

    CATCH_SECTION("Canonicalizer") {
        constexpr size_t CANONICAL_TEST_DIM = 133;
        // Sequential
        test_canonicalizer<8, CANONICAL_TEST_DIM, lvq::Sequential>();
        test_canonicalizer<7, CANONICAL_TEST_DIM, lvq::Sequential>();
        test_canonicalizer<6, CANONICAL_TEST_DIM, lvq::Sequential>();
        test_canonicalizer<5, CANONICAL_TEST_DIM, lvq::Sequential>();
        test_canonicalizer<4, CANONICAL_TEST_DIM, lvq::Sequential>();
        test_canonicalizer<3, CANONICAL_TEST_DIM, lvq::Sequential>();

        // Turbo
        test_canonicalizer<4, CANONICAL_TEST_DIM, lvq::Turbo<16, 8>>();
    }

    CATCH_SECTION("Compressed Dataset") {
        auto tester = CompressedReference();
        namespace lvq = svs::quantization::lvq;

        auto allocator = svs::lib::Allocator<std::byte>{};
        auto blocking_parameters =
            svs::data::BlockingParameters{.blocksize_bytes = svs::lib::PowerOfTwo(12)};
        auto blocked = svs::data::Blocked{blocking_parameters, allocator};

        svs::lib::foreach (bits, [&]<size_t N>(Val<N> /*unused*/) {
            tester.template populate<lvq::Signed, N, TEST_DIM>(
                dataset_size, MaybeStatic<TEST_DIM>(), allocator
            );
            tester.template populate<lvq::Unsigned, N, TEST_DIM>(
                dataset_size, MaybeStatic<TEST_DIM>(), allocator
            );

            tester.template populate<lvq::Signed, N, TEST_DIM>(
                dataset_size, MaybeStatic<TEST_DIM>(), blocked
            );
            tester.template populate<lvq::Unsigned, N, TEST_DIM>(
                dataset_size, MaybeStatic<TEST_DIM>(), blocked
            );

            tester.template populate<lvq::Signed, N, svs::Dynamic>(
                dataset_size, MaybeStatic<svs::Dynamic>(TEST_DIM), allocator
            );
            tester.template populate<lvq::Unsigned, N, svs::Dynamic>(
                dataset_size, MaybeStatic<svs::Dynamic>(TEST_DIM), allocator
            );
        });
    }

    CATCH_SECTION("Scaled Biased Dataset") {
        auto tester = ScaledBiasedReference();
        namespace lvq = svs::quantization::lvq;

        auto allocator = svs::lib::Allocator<std::byte>();
        auto blocking_parameters =
            svs::data::BlockingParameters{.blocksize_bytes = svs::lib::PowerOfTwo(12)};
        auto blocked = svs::data::Blocked{blocking_parameters, allocator};

        auto test_strategies = [&]<size_t N, lvq::LVQPackingStrategy Strategy>() {
            for (size_t alignment : {0, 32}) {
                auto s = MaybeStatic<TEST_DIM>();
                auto d = MaybeStatic<svs::Dynamic>(TEST_DIM);
                tester.template populate<N, TEST_DIM, Strategy>(
                    dataset_size, s, alignment, allocator
                );
                tester.template populate<N, TEST_DIM, Strategy>(
                    dataset_size, s, alignment, blocked
                );
                tester.template populate<N, svs::Dynamic, Strategy>(
                    dataset_size, d, alignment, allocator
                );
                tester.template populate<N, svs::Dynamic, Strategy>(
                    dataset_size, d, alignment, blocked
                );
            }
        };

        // clang-13 doesn't like it if we try to use another lambda to apply the number
        // of bits using `svs::lib::foreach` - so manually unroll.
        test_strategies.template operator()<3, lvq::Sequential>();
        test_strategies.template operator()<4, lvq::Sequential>();
        test_strategies.template operator()<5, lvq::Sequential>();
        test_strategies.template operator()<6, lvq::Sequential>();
        test_strategies.template operator()<7, lvq::Sequential>();
        test_strategies.template operator()<8, lvq::Sequential>();

        // Turbo Strategies
        test_strategies.template operator()<4, lvq::Turbo<16, 8>>();
    }
}
