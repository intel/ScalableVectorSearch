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

// header under test
#include "svs/core/data/simple.h"

// svs
#include "svs/core/data.h"

// test utilities
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stdlib
#include <span>

namespace {

// Allocator class which records allocated bytes
template <typename T> class RecordingAllocator {
  public:
    using value_type = T;

    RecordingAllocator() = default;

    T* allocate(size_t n) {
        *allocated_bytes += n * sizeof(T);
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }

    void deallocate(T* p, size_t n) {
        *allocated_bytes -= n * sizeof(T);
        ::operator delete(p);
    }

    template <typename U> bool operator==(const RecordingAllocator<U>& other) const {
        return allocated_bytes == other.allocated_bytes;
    }
    template <typename U> bool operator!=(const RecordingAllocator<U>& other) const {
        return !(*this == other);
    }

    template <typename U>
    RecordingAllocator(const RecordingAllocator<U>& other)
        : allocated_bytes(other.allocated_bytes) {}

    size_t& allocated() { return *allocated_bytes; }

    std::shared_ptr<size_t> allocated_bytes = std::make_shared<size_t>(0);
};

template <typename T> bool is_blocked(const T&) { return false; }
template <typename T, size_t N> bool is_blocked(const svs::data::BlockedData<T, N>&) {
    return true;
}

template <typename Left, typename Right>
bool data_equal(const Left& left, const Right& right) {
    auto lsize = left.size();
    auto ldims = left.dimensions();

    auto rsize = right.size();
    auto rdims = right.dimensions();

    if (lsize != rsize || ldims != rdims) {
        return false;
    }

    for (size_t i = 0; i < lsize; ++i) {
        auto l = left.get_datum(i);
        auto r = right.get_datum(i);
        if (!std::equal(l.begin(), l.end(), r.begin())) {
            return false;
        }
    }
    return true;
}

template <size_t Extent = svs::Dynamic> void test_blocked() {
    // Use a small block size so we can test the block bridging logic.
    size_t blocksize_bytes = 4096;
    size_t num_elements = 2000;
    size_t dimensions = 5;

    // Sanity check to prevent future changes from messing up this test.
    if constexpr (Extent != svs::Dynamic) {
        CATCH_REQUIRE(Extent == dimensions);
    }
    CATCH_REQUIRE(!is_blocked(10));

    size_t expected_blocksize = 128;

    auto parameters = svs::data::BlockingParameters{
        .blocksize_bytes = svs::lib::prevpow2(blocksize_bytes)};
    auto allocator = svs::data::Blocked<svs::lib::Allocator<float>>(parameters);
    auto data = svs::data::BlockedData<float, Extent>(num_elements, dimensions, allocator);
    CATCH_REQUIRE(is_blocked(data));
    CATCH_REQUIRE(data.dimensions() == 5);
    CATCH_REQUIRE(data.blocksize_bytes().value() == blocksize_bytes);
    CATCH_REQUIRE(data.blocksize().value() == expected_blocksize);
    CATCH_REQUIRE(data.size() == num_elements);

    auto set_contents = [dimensions](auto& data) {
        std::vector<float> values(dimensions);
        for (size_t i = 0; i < data.size(); ++i) {
            std::fill(values.begin(), values.end(), i);
            data.set_datum(i, std::span<float, Extent>{values.data(), values.size()});
        }
    };

    auto check_contents = [dimensions](const auto& this_data) {
        for (size_t i : this_data.eachindex()) {
            // Make sure prefetching at least works.
            this_data.prefetch(i);

            // Make sure that our data assignment was propagated correctly.
            auto datum = this_data.get_datum(i);
            CATCH_REQUIRE(datum.size() == dimensions);
            CATCH_REQUIRE(std::all_of(datum.begin(), datum.end(), [&](float v) {
                return v == i;
            }));
        }
    };

    set_contents(data);
    check_contents(data);
    auto copy = data.copy();
    check_contents(data.copy());
    CATCH_REQUIRE(is_blocked(copy));
    CATCH_REQUIRE(data_equal(data, data.copy()));

    ///// Resizing
    CATCH_REQUIRE(data.num_blocks() == 16);

    // Increase in size
    data.resize(4000);
    CATCH_REQUIRE(data.capacity() > 4000);
    CATCH_REQUIRE(data.num_blocks() == 32);

    set_contents(data);
    check_contents(data);
    check_contents(data.copy());

    // Decrease in size
    data.resize(2000);
    CATCH_REQUIRE(data.capacity() < 4000);
    CATCH_REQUIRE(data.num_blocks() == 16);
    check_contents(data);
    check_contents(data.copy());

    ///// Saving and Loading.
    svs_test::prepare_temp_directory();
    auto temp = svs_test::temp_directory();
    svs::lib::save_to_disk(data, temp);
    auto simple_data = svs::VectorDataLoader<float>(temp).load();
    check_contents(simple_data);
    CATCH_REQUIRE(!is_blocked(simple_data));
    CATCH_REQUIRE(data_equal(simple_data, data));

    // Reload as a blocked dataset.
    auto reloaded = svs::lib::load_from_disk<svs::data::BlockedData<float>>(temp);
    check_contents(reloaded);
    CATCH_REQUIRE(is_blocked(reloaded));
    CATCH_REQUIRE(data_equal(reloaded, data));
}
} // namespace

CATCH_TEST_CASE("Testing Blocked Data", "[core][data][blocked]") {
    CATCH_SECTION("BlockingParameters") {
        using T = svs::data::BlockingParameters;
        auto p = T{};
        CATCH_REQUIRE(p.blocksize_bytes == T::default_blocksize_bytes);
        CATCH_REQUIRE(p.blocksize_elements == std::nullopt);

        p = T{.blocksize_bytes = svs::lib::PowerOfTwo(10)};
        CATCH_REQUIRE(p.blocksize_bytes == svs::lib::PowerOfTwo(10));
        CATCH_REQUIRE(p.blocksize_elements == std::nullopt);

        p = T{.blocksize_elements = svs::lib::PowerOfTwo(9)};
        CATCH_REQUIRE(p.blocksize_bytes == T::default_blocksize_bytes);
        CATCH_REQUIRE(p.blocksize_elements == svs::lib::PowerOfTwo(9));
    }

    CATCH_SECTION("Blocked Allocator") {
        // Use an integer for the "allocator" to test value propagation.
        // Since the `Blocked` class doesn't actually use the allocator, this is okay
        // for functionality testing.
        using T = svs::data::Blocked<int>;
        using P = svs::data::BlockingParameters;
        auto x = T();
        CATCH_REQUIRE(x.get_allocator() == 0); // Default constructed integer.
        CATCH_REQUIRE(x.parameters() == P{});

        x = T(10);
        CATCH_REQUIRE(x.get_allocator() == 10);
        CATCH_REQUIRE(x.parameters() == P{});

        auto p = P{.blocksize_bytes = svs::lib::PowerOfTwo(10)};
        x = T(p);
        CATCH_REQUIRE(x.get_allocator() == 0);
        CATCH_REQUIRE(x.parameters() == p);

        x = T(p, 10);
        CATCH_REQUIRE(x.get_allocator() == 10);
        CATCH_REQUIRE(x.parameters() == p);
    }

    CATCH_SECTION("Basic Functionality") {
        test_blocked();
        test_blocked<5>();
    }

    CATCH_SECTION("Different Blocksizes for blocksize_bytes") {
        // When BlockingParameters::blocksize_bytes is used (no explicit
        // blocksize_elements), the computed blocksize() depends on the per-element byte
        // size, i.e. sizeof(T) * dimensions. The same BlockingParameters therefore
        // produces very different blocksize_ values for datasets with different element
        // types and dimensionalities.

        // 1 MiB block
        auto parameters =
            svs::data::BlockingParameters{.blocksize_bytes = svs::lib::PowerOfTwo(20)};

        size_t num_elements = 10;
        size_t vector_dims = 1024 + 1 + sizeof(float) * 2; // quantized vectors
        size_t graph_degree = 32;
        size_t graph_dims = graph_degree + 1; // +1 for edges counter

        RecordingAllocator<std::byte> byte_alloc;
        RecordingAllocator<std::uint32_t> int_alloc;

        auto vec_alloc = svs::data::Blocked(parameters, byte_alloc);
        auto graph_alloc = svs::data::Blocked(parameters, int_alloc);

        auto vec_data = svs::data::SimpleData<std::byte, svs::Dynamic, decltype(vec_alloc)>(
            num_elements, vector_dims, vec_alloc
        );
        auto graph_data =
            svs::data::SimpleData<uint32_t, svs::Dynamic, decltype(graph_alloc)>(
                num_elements, graph_dims, graph_alloc
            );

        // Both datasets are configured with the same blocksize_bytes (1 MiB).
        CATCH_REQUIRE(vec_data.blocksize_bytes() == graph_data.blocksize_bytes());
        CATCH_REQUIRE(vec_data.blocksize_bytes().value() == (size_t(1) << 20));

        // Per-element byte sizes differ by 64x (1033 vs 132).
        CATCH_REQUIRE(vec_data.element_size() == sizeof(std::byte) * vector_dims); // 1033
        CATCH_REQUIRE(graph_data.element_size() == sizeof(uint32_t) * graph_dims); //  132

        // Imagine that we are going to predict memory consumption based on element size and
        // a blocksize.
        auto index_element_size = graph_data.element_size() + vec_data.element_size();
        // We have just 10 vectors - this should trigger allocation of 1 block for graph and
        // 1 block for vectors, since both blocksizes are larger than 1. We are assuming
        // that the blocksize in elements looks like:
        auto blocksize_elements = parameters.blocksize_bytes.value() /
                                  vec_data.element_size(); // 1048576 / 1033 = 1014
        // This is not correct, because the actual block size in elements is computed as the
        // previous power of two of this value, which is 512 for vectors and 4096 for
        // graphs. So, if we use the same blocksize_bytes to predict memory consumption, we
        // will get different results for different element sizes, which is not what we
        // want.
        auto expected_memory_consumption =
            blocksize_elements *
            index_element_size; // 1014 * (1033 + 132) = 1014 * 1165 = 1,180,310

        auto actual_memory_consumption = byte_alloc.allocated() + int_alloc.allocated();
        CATCH_REQUIRE(expected_memory_consumption != actual_memory_consumption);

        // So, if we add 520 vectors to an index, we will get 1 block for graph and 2 blocks
        // for vectors. Which means, we have different numbers of blocks for the same number
        // of elements, even though the same BlockingParameters were used.
        vec_data.resize(520);
        graph_data.resize(520);
        CATCH_REQUIRE(vec_data.num_blocks() != graph_data.num_blocks());

        // It is because the graph blocksize_ is 8x larger than the vector blocksize_.
        CATCH_REQUIRE(graph_data.blocksize().value() == 8 * vec_data.blocksize().value());
    }

    // This is why, to properly predict memory consumption of blocked datasets, we should
    // directly manage blocksize_elements instead of blocksize_bytes, since the former
    // directly controls the number of elements per block, while the latter only indirectly
    // controls it through the element size.
    CATCH_SECTION("Explicit blocksize_elements") {
        auto parameters = svs::data::BlockingParameters{
            .blocksize_elements = svs::lib::PowerOfTwo(9) // 512 elements per block
        };

        size_t num_elements = 10;
        size_t vector_dims = 1024 + 1 + sizeof(float) * 2; // quantized vectors
        size_t graph_degree = 32;
        size_t graph_dims = graph_degree + 1; // +1 for edges counter

        RecordingAllocator<std::byte> byte_alloc;
        RecordingAllocator<std::uint32_t> int_alloc;

        auto vec_alloc = svs::data::Blocked(parameters, byte_alloc);
        auto graph_alloc = svs::data::Blocked(parameters, int_alloc);

        auto vec_data = svs::data::SimpleData<std::byte, svs::Dynamic, decltype(vec_alloc)>(
            num_elements, vector_dims, vec_alloc
        );
        auto graph_data =
            svs::data::SimpleData<uint32_t, svs::Dynamic, decltype(graph_alloc)>(
                num_elements, graph_dims, graph_alloc
            );

        // Per-element byte sizes differ by 64x (1033 vs 132).
        CATCH_REQUIRE(vec_data.element_size() == sizeof(std::byte) * vector_dims); // 1033
        CATCH_REQUIRE(graph_data.element_size() == sizeof(uint32_t) * graph_dims); //  132

        // Both datasets are configured with the same blocksize_elements (512).
        CATCH_REQUIRE(vec_data.blocksize().value() == graph_data.blocksize().value());

        // Imagine that we are going to predict memory consumption based on element size and
        // a blocksize.
        auto index_element_size = graph_data.element_size() + vec_data.element_size();
        // We have just 10 vectors - this should trigger allocation of 1 block for graph and
        // 1 block for vectors, since both blocksizes are larger than 10. So we can expect
        // the memory consumption for 1 block in both datasets is:
        auto expected_memory_consumption =
            parameters.blocksize_elements.value() * index_element_size;

        auto actual_memory_consumption = byte_alloc.allocated() + int_alloc.allocated();
        CATCH_REQUIRE(expected_memory_consumption == actual_memory_consumption);

        // So, if we add 520 vectors to an index, we will get 2 blocks for graph and 2
        // blocks for vectors. Which means, we have same number of blocks for the same
        // number of elements.
        vec_data.resize(520);
        graph_data.resize(520);
        CATCH_REQUIRE(vec_data.num_blocks() == graph_data.num_blocks());
    }
}
