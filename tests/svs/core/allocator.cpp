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

// Compile-time tests
namespace {
using Alloc = svs::HugepageAllocator<float>;
using Traits = std::allocator_traits<Alloc>;

#define SVS_SAME(left, right) std::is_same_v<typename left, right>

static_assert(std::is_same_v<typename Traits::allocator_type, Alloc>);
static_assert(std::is_same_v<typename Traits::value_type, float>);
static_assert(std::is_same_v<typename Traits::pointer, float*>);
static_assert(std::is_same_v<typename Traits::const_pointer, const float*>);
static_assert(std::is_same_v<typename Traits::void_pointer, void*>);
static_assert(std::is_same_v<typename Traits::const_void_pointer, const void*>);
static_assert(std::is_same_v<typename Traits::difference_type, int64_t>);
static_assert(std::is_same_v<typename Traits::size_type, size_t>);
static_assert(std::is_same_v<
              typename Traits::propagate_on_container_copy_assignment,
              std::true_type>);
static_assert(std::is_same_v<
              typename Traits::propagate_on_container_move_assignment,
              std::true_type>);
static_assert(std::is_same_v<typename Traits::propagate_on_container_swap, std::true_type>);
static_assert(std::is_same_v<typename Traits::is_always_equal, std::true_type>);

} // namespace

CATCH_TEST_CASE("Testing Allocator", "[allocators]") {
    CATCH_SECTION("Testing `HugepageAllocator`") {
        constexpr size_t num_elements = 1024;
        CATCH_SECTION("Basic Behavior") {
            {
                auto v = std::vector<size_t, svs::HugepageAllocator<size_t>>(num_elements);
                CATCH_REQUIRE(v.size() == num_elements);
                // We should have an entry for this allocation.
                auto allocations = svs::detail::GenericHugepageAllocator::get_allocations();
                CATCH_REQUIRE(allocations.size() == 1);
                auto* ptr = v.data();
                CATCH_REQUIRE(allocations.contains(ptr));
                CATCH_REQUIRE(allocations.at(ptr) >= sizeof(size_t) * num_elements);
                // Destructor runs - allocations should get unmapped.
            }
            CATCH_REQUIRE(svs::detail::GenericHugepageAllocator::get_allocations().empty());
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
            // Make sure we get an error when trying to map an existing file that
            // doesn't exist.
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
                auto* base = ptr.data();
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
            auto* base = ptr.data();
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
            base = ptr.data();
            for (size_t i = 0; i < nelements; ++i) {
                CATCH_REQUIRE(*(base + i) == i);
            }
        }
    }
}
