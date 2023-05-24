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

// stdlib
#include <filesystem>
#include <memory>
#include <vector>

// svs
#include "svs/core/allocator.h"
#include "svs/lib/memory.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// tests
#include "tests/utils/utils.h"

CATCH_TEST_CASE("Testing Allocator", "[allocators]") {
    CATCH_SECTION("Testing `VectorAllocator`") {
        auto allocator = svs::lib::VectorAllocator{};
        std::vector<float> v = svs::lib::allocate_managed<float>(allocator, 100);
        CATCH_REQUIRE(v.size() == 100);
        CATCH_REQUIRE(v.capacity() >= 100);
        auto maybe_filename = svs::lib::memory::filename(v);
        CATCH_REQUIRE(maybe_filename.has_value() == false);
    }

    CATCH_SECTION("Testing `HugepageAllocator`") {
        auto allocator = svs::HugepageAllocator{};
        constexpr size_t num_elements = 1024;
        CATCH_SECTION("Basic Behavior") {
            svs::MMapPtr<float> ptr =
                svs::lib::allocate_managed<float>(allocator, num_elements);
            CATCH_REQUIRE(ptr);
            CATCH_REQUIRE(ptr.size() >= sizeof(float) * num_elements);

            // Make sure we can write to each element.
            for (size_t i = 0; i < num_elements; ++i) {
                *(ptr.data() + i) = 1.0f;
            }

            void* actual_pointer = ptr.base();
            CATCH_REQUIRE(actual_pointer == ptr.data());

            // First off, test the un-mapping logic.
            // The idea is that after un-mapping, the resource should be left in a null
            // state.
            CATCH_REQUIRE(ptr != svs::MMapPtr<float>{});
            ptr.unmap();
            CATCH_REQUIRE(ptr == svs::MMapPtr<float>{});
        }

        CATCH_SECTION("Destructors") {
            auto ptr = svs::lib::allocate_managed<float>(allocator, num_elements);
            CATCH_REQUIRE(ptr);
            for (size_t i = 0; i < num_elements; ++i) {
                *(ptr.data() + i) = 1.0f;
            }
            CATCH_REQUIRE(ptr != nullptr);
        }

        // Move Constructor
        CATCH_SECTION("Move Constructor") {
            auto ptr = svs::lib::allocate_managed<float>(allocator, num_elements);
            CATCH_REQUIRE(ptr);
            for (size_t i = 0; i < num_elements; ++i) {
                *(ptr.data() + i) = i;
            }
            auto other{std::move(ptr)};
            // Make sure the other one was "moved out" correctly.
            CATCH_REQUIRE(ptr == nullptr);
            CATCH_REQUIRE(other);
            // Everything in the range of the new pointer should be initialized correctly.
            for (size_t i = 0; i < num_elements; ++i) {
                CATCH_REQUIRE(*(other.data() + i) == i);
            }
        }

        // Move Assignment Operator.
        CATCH_SECTION("Move Constructor") {
            auto ptr = svs::lib::allocate_managed<float>(allocator, num_elements);
            CATCH_REQUIRE(ptr);
            for (size_t i = 0; i < num_elements; ++i) {
                *(ptr.data() + i) = i;
            }
            auto other = decltype(ptr){};
            CATCH_REQUIRE(other == nullptr);
            other = std::move(ptr);
            // Make sure the other one was "moved out" correctly.
            CATCH_REQUIRE(ptr == nullptr);
            CATCH_REQUIRE(other);
            // Everything in the range of the new pointer should be initialized correctly.
            for (size_t i = 0; i < num_elements; ++i) {
                CATCH_REQUIRE(*(other.data() + i) == i);
            }
        }
    }

    CATCH_SECTION("Testing `MemoryMapper`") {
        CATCH_REQUIRE(svs_test::prepare_temp_directory());
        auto temp_dir = svs_test::temp_directory();
        CATCH_SECTION("Test Set 1") {
            using T = float;
            const size_t nelements = 100;
            const auto bytes = svs::lib::Bytes(nelements * sizeof(T));
            auto temp_file = temp_dir / "file1.bin";
            // Make sure we get an error when trying to map an existing file that doesn't
            // exist.
            auto mapper = svs::MemoryMapper();
            CATCH_REQUIRE(mapper.policy() == svs::MemoryMapper::MustUseExisting);
            CATCH_REQUIRE(mapper.permission() == svs::MemoryMapper::ReadOnly);
            CATCH_REQUIRE_THROWS_AS(mapper.mmap(temp_file, bytes), svs::ANNException);
            mapper.setpolicy(svs::MemoryMapper::MayCreate);
            mapper.setpermission(svs::MemoryMapper::ReadWrite);
            {
                svs::MMapPtr<float> ptr = mapper.mmap(temp_file, bytes);

                // Check flags - should mark file as created.
                CATCH_REQUIRE(std::filesystem::exists(temp_file));
                CATCH_REQUIRE(
                    std::filesystem::file_size(temp_file) >= sizeof(float) * nelements
                );
                // Write to each elements.
                auto base = svs::lib::memory::access_storage(ptr);
                for (size_t i = 0; i < nelements; ++i) {
                    *(base + i) = i;
                }
                // Desctructor for `ptr` runs here.
            }
            // Load the file again.
            // This time, mark the policy as `MustCreate` to make sure we get an error
            // because the file already exists.
            CATCH_REQUIRE(std::filesystem::exists(temp_file));
            mapper.setpolicy(svs::MemoryMapper::MustCreate);
            CATCH_REQUIRE_THROWS_AS(mapper.mmap(temp_file, bytes), svs::ANNException);

            // Set the policy back to `MustUseExisting`.
            // Make sure the file maps and has the contents we set earlier.
            mapper.setpolicy(svs::MemoryMapper::MustUseExisting);
            mapper.setpermission(svs::MemoryMapper::ReadOnly);
            svs::MMapPtr<float> ptr = mapper.mmap(temp_file, bytes);
            auto base = svs::lib::memory::access_storage(ptr);
            for (size_t i = 0; i < nelements; ++i) {
                CATCH_REQUIRE(*(base + i) == i);
            }

            // Finally, make sure we get an error if trying to use an existing file
            // that is too small.
            CATCH_REQUIRE_THROWS_AS(
                mapper.mmap(temp_file, svs::lib::Bytes(10 * nelements * sizeof(T))),
                svs::ANNException
            );
            // Make sure we can still allocate with the correct number of elements.
            mapper.setpolicy(svs::MemoryMapper::MayCreate);
            ptr = mapper.mmap(temp_file, bytes);
            base = svs::lib::memory::access_storage(ptr);
            for (size_t i = 0; i < nelements; ++i) {
                CATCH_REQUIRE(*(base + i) == i);
            }
        }
    }
}
