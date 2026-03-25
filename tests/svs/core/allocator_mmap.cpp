/*
 * Copyright 2026 Intel Corporation
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

// Test file-backed MMapAllocator
#include "svs/core/allocator_mmap.h"
#include "svs/core/data.h"

#include "catch2/catch_test_macros.hpp"

#include <filesystem>
#include <vector>

namespace {

CATCH_TEST_CASE("MMapAllocator Basic Operations", "[allocator][mmap]") {
    auto temp_dir = std::filesystem::temp_directory_path() / "svs_mmap_test";
    std::filesystem::create_directories(temp_dir);

    CATCH_SECTION("Allocate and deallocate") {
        svs::MMapAllocator<float> alloc(temp_dir);

        constexpr size_t n = 1000;
        float* ptr = alloc.allocate(n);
        CATCH_REQUIRE(ptr != nullptr);

        // Write to the allocated memory
        for (size_t i = 0; i < n; ++i) {
            ptr[i] = static_cast<float>(i);
        }

        // Read back and verify
        for (size_t i = 0; i < n; ++i) {
            CATCH_REQUIRE(ptr[i] == static_cast<float>(i));
        }

        // Deallocate
        alloc.deallocate(ptr, n);
    }

    CATCH_SECTION("Multiple allocations") {
        svs::MMapAllocator<int> alloc(temp_dir);

        std::vector<int*> ptrs;
        constexpr size_t num_allocs = 5;
        constexpr size_t alloc_size = 100;

        // Allocate multiple blocks
        for (size_t i = 0; i < num_allocs; ++i) {
            int* ptr = alloc.allocate(alloc_size);
            CATCH_REQUIRE(ptr != nullptr);
            ptrs.push_back(ptr);

            // Initialize
            for (size_t j = 0; j < alloc_size; ++j) {
                ptr[j] = static_cast<int>(i * 1000 + j);
            }
        }

        // Verify all allocations
        for (size_t i = 0; i < num_allocs; ++i) {
            for (size_t j = 0; j < alloc_size; ++j) {
                CATCH_REQUIRE(ptrs[i][j] == static_cast<int>(i * 1000 + j));
            }
        }

        // Deallocate all
        for (size_t i = 0; i < num_allocs; ++i) {
            alloc.deallocate(ptrs[i], alloc_size);
        }
    }

    CATCH_SECTION("Large allocation") {
        svs::MMapAllocator<double> alloc(temp_dir);

        constexpr size_t n = 1'000'000; // 1 million doubles
        double* ptr = alloc.allocate(n);
        CATCH_REQUIRE(ptr != nullptr);

        // Spot check some values
        ptr[0] = 1.0;
        ptr[n / 2] = 2.0;
        ptr[n - 1] = 3.0;

        CATCH_REQUIRE(ptr[0] == 1.0);
        CATCH_REQUIRE(ptr[n / 2] == 2.0);
        CATCH_REQUIRE(ptr[n - 1] == 3.0);

        alloc.deallocate(ptr, n);
    }

    CATCH_SECTION("Default path (temp directory)") {
        svs::MMapAllocator<int> alloc; // No path specified

        constexpr size_t n = 50;
        int* ptr = alloc.allocate(n);
        CATCH_REQUIRE(ptr != nullptr);

        for (size_t i = 0; i < n; ++i) {
            ptr[i] = static_cast<int>(i * 2);
        }

        for (size_t i = 0; i < n; ++i) {
            CATCH_REQUIRE(ptr[i] == static_cast<int>(i * 2));
        }

        alloc.deallocate(ptr, n);
    }

    // Cleanup
    std::filesystem::remove_all(temp_dir);
}

CATCH_TEST_CASE("MMapAllocator with SimpleData", "[allocator][mmap][integration]") {
    auto temp_dir = std::filesystem::temp_directory_path() / "svs_mmap_simpledata_test";
    std::filesystem::create_directories(temp_dir);

    CATCH_SECTION("SimpleData with MMapAllocator") {
        constexpr size_t num_vectors = 100;
        constexpr size_t dims = 128;

        using Alloc = svs::MMapAllocator<float>;
        using Data = svs::data::SimpleData<float, svs::Dynamic, Alloc>;

        // Create data with file-backed allocator
        Data data(num_vectors, dims, Alloc{temp_dir});

        // Write data
        for (size_t i = 0; i < num_vectors; ++i) {
            auto datum = data.get_datum(i);
            for (size_t j = 0; j < dims; ++j) {
                datum[j] = static_cast<float>(i * dims + j);
            }
        }

        // Verify data
        for (size_t i = 0; i < num_vectors; ++i) {
            auto datum = data.get_datum(i);
            for (size_t j = 0; j < dims; ++j) {
                CATCH_REQUIRE(datum[j] == static_cast<float>(i * dims + j));
            }
        }

        CATCH_REQUIRE(data.size() == num_vectors);
        CATCH_REQUIRE(data.dimensions() == dims);
    }

    // Cleanup
    std::filesystem::remove_all(temp_dir);
}

CATCH_TEST_CASE("MMapAllocator Rebinding", "[allocator][mmap]") {
    auto temp_dir = std::filesystem::temp_directory_path() / "svs_mmap_rebind_test";
    std::filesystem::create_directories(temp_dir);

    CATCH_SECTION("Rebind allocator") {
        svs::MMapAllocator<float> float_alloc(temp_dir);
        svs::MMapAllocator<int> int_alloc(float_alloc);

        // Both should use the same path
        CATCH_REQUIRE(float_alloc.get_base_path() == int_alloc.get_base_path());

        // Test allocations with rebound allocator
        int* ptr = int_alloc.allocate(10);
        CATCH_REQUIRE(ptr != nullptr);
        ptr[0] = 42;
        CATCH_REQUIRE(ptr[0] == 42);
        int_alloc.deallocate(ptr, 10);
    }

    // Cleanup
    std::filesystem::remove_all(temp_dir);
}

CATCH_TEST_CASE("MMapAllocator Equality", "[allocator][mmap]") {
    auto temp_dir1 = std::filesystem::temp_directory_path() / "svs_mmap_eq1";
    auto temp_dir2 = std::filesystem::temp_directory_path() / "svs_mmap_eq2";

    CATCH_SECTION("Same path allocators are equal") {
        svs::MMapAllocator<float> alloc1(temp_dir1);
        svs::MMapAllocator<float> alloc2(temp_dir1);

        CATCH_REQUIRE(alloc1 == alloc2);
    }

    CATCH_SECTION("Different path allocators are not equal") {
        svs::MMapAllocator<float> alloc1(temp_dir1);
        svs::MMapAllocator<float> alloc2(temp_dir2);

        CATCH_REQUIRE_FALSE(alloc1 == alloc2);
    }

    CATCH_SECTION("Rebound allocators with same path are equal") {
        svs::MMapAllocator<float> float_alloc(temp_dir1);
        svs::MMapAllocator<int> int_alloc(float_alloc);

        CATCH_REQUIRE(float_alloc == int_alloc);
    }
}

} // anonymous namespace
