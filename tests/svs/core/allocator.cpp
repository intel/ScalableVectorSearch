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

// stdlib
#include <algorithm>
#include <filesystem>
#include <memory>
#include <vector>

// svs
#include "svs/core/allocator.h"
#include "svs/core/data.h"
#include "svs/core/graph.h"
#include "svs/lib/memory.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// tests
#include "tests/utils/test_dataset.h"
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
#if defined(__APPLE__)
static_assert(std::is_same_v<typename Traits::difference_type, long>);
#else
static_assert(std::is_same_v<typename Traits::difference_type, int64_t>);
#endif // __APPLE__
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
                // Destructor for `ptr` runs here.
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

    CATCH_SECTION("Testing `AllocatorHandle`") {
        size_t num_elements = 1024;
        CATCH_SECTION("Allocator") {
            auto alloc = svs::make_allocator_handle(svs::lib::Allocator<float>());
            auto* ptr = alloc.allocate(num_elements);

            alloc.deallocate(ptr, num_elements);

            CATCH_STATIC_REQUIRE(std::is_same_v<decltype(ptr), float*>);
        }
        CATCH_SECTION("HugepageAllocator - std::byte") {
            auto alloc = svs::make_allocator_handle(svs::HugepageAllocator<std::byte>());
            auto* ptr = alloc.allocate(num_elements);

            auto allocations = svs::detail::GenericHugepageAllocator::get_allocations();
            CATCH_REQUIRE(allocations.size() == 1);
            CATCH_REQUIRE(allocations.contains(ptr));
            CATCH_REQUIRE(allocations.at(ptr) >= sizeof(std::byte) * num_elements);

            alloc.deallocate(ptr, num_elements);
            allocations = svs::detail::GenericHugepageAllocator::get_allocations();
            CATCH_REQUIRE(allocations.size() == 0);
            CATCH_REQUIRE(!allocations.contains(ptr));

            CATCH_STATIC_REQUIRE(std::is_same_v<decltype(ptr), std::byte*>);
        }
        CATCH_SECTION("HugepageAllocator - int8_t") {
            auto alloc = svs::make_allocator_handle(svs::HugepageAllocator<int8_t>());
            auto* ptr = alloc.allocate(num_elements);

            auto allocations = svs::detail::GenericHugepageAllocator::get_allocations();
            CATCH_REQUIRE(allocations.size() == 1);
            CATCH_REQUIRE(allocations.contains(ptr));
            CATCH_REQUIRE(allocations.at(ptr) >= sizeof(int8_t) * num_elements);

            alloc.deallocate(ptr, num_elements);
            allocations = svs::detail::GenericHugepageAllocator::get_allocations();
            CATCH_REQUIRE(allocations.size() == 0);
            CATCH_REQUIRE(!allocations.contains(ptr));

            CATCH_STATIC_REQUIRE(std::is_same_v<decltype(ptr), int8_t*>);
        }
        CATCH_SECTION("HugepageAllocator - svs::Float16") {
            auto alloc = svs::make_allocator_handle(svs::HugepageAllocator<svs::Float16>());
            auto* ptr = alloc.allocate(num_elements);

            auto allocations = svs::detail::GenericHugepageAllocator::get_allocations();
            CATCH_REQUIRE(allocations.size() == 1);
            CATCH_REQUIRE(allocations.contains(ptr));
            CATCH_REQUIRE(allocations.at(ptr) >= sizeof(svs::Float16) * num_elements);

            alloc.deallocate(ptr, num_elements);
            allocations = svs::detail::GenericHugepageAllocator::get_allocations();
            CATCH_REQUIRE(allocations.size() == 0);
            CATCH_REQUIRE(!allocations.contains(ptr));

            CATCH_STATIC_REQUIRE(std::is_same_v<decltype(ptr), svs::Float16*>);
        }
        CATCH_SECTION("Rebind") {
            auto alloc = svs::make_allocator_handle(svs::lib::Allocator<int>());
            svs::lib::rebind_allocator_t<svs::Float16, decltype(alloc)> rebound_alloc{
                alloc};
            auto* ptr = rebound_alloc.allocate(num_elements);
            rebound_alloc.deallocate(ptr, num_elements);
            CATCH_STATIC_REQUIRE(std::is_same_v<decltype(ptr), svs::Float16*>);

            svs::lib::rebind_allocator_t<float, decltype(alloc)> rebound_alloc2{
                rebound_alloc};
            auto* ptr2 = rebound_alloc2.allocate(num_elements);
            rebound_alloc2.deallocate(ptr2, num_elements);
            CATCH_STATIC_REQUIRE(std::is_same_v<decltype(ptr2), float*>);
        }
    }

    CATCH_SECTION("Testing MMapAllocator") {
        auto temp_dir = svs_test::prepare_temp_directory_v2();

        CATCH_SECTION("Basic Behavior") {
            using T = float;
            constexpr size_t nelements = 256;
            const size_t bytes = nelements * sizeof(T);

            auto list_regular_files = [](const std::filesystem::path& dir) {
                std::vector<std::filesystem::path> paths;
                for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                    if (entry.is_regular_file()) {
                        paths.push_back(entry.path());
                    }
                }
                return paths;
            };

            auto alloc = svs::MMapAllocator<T>(temp_dir, svs::MMapAccessHint::Sequential);
            CATCH_REQUIRE(alloc.get_base_path() == temp_dir);
            CATCH_REQUIRE(alloc.get_access_hint() == svs::MMapAccessHint::Sequential);

            alloc.set_access_hint(svs::MMapAccessHint::Random);
            CATCH_REQUIRE(alloc.get_access_hint() == svs::MMapAccessHint::Random);

            auto files_before = list_regular_files(temp_dir);
            auto count_before =
                svs::detail::MMapAllocationRegistry::instance().allocation_count();
            auto* ptr = alloc.allocate(nelements);
            CATCH_REQUIRE(ptr != nullptr);
            CATCH_REQUIRE(
                svs::detail::MMapAllocationRegistry::instance().allocation_count() ==
                count_before + 1
            );

            auto files_after = list_regular_files(temp_dir);
            CATCH_REQUIRE(files_after.size() == files_before.size() + 1);

            std::optional<std::filesystem::path> allocation_file;
            for (const auto& candidate : files_after) {
                if (std::find(files_before.begin(), files_before.end(), candidate) ==
                    files_before.end()) {
                    allocation_file = candidate;
                    break;
                }
            }

            CATCH_REQUIRE(allocation_file.has_value());
            CATCH_REQUIRE(std::filesystem::exists(*allocation_file));
            CATCH_REQUIRE(std::filesystem::file_size(*allocation_file) == bytes);

            for (size_t i = 0; i < nelements; ++i) {
                ptr[i] = static_cast<T>(i);
            }
            for (size_t i = 0; i < nelements; ++i) {
                CATCH_REQUIRE(ptr[i] == static_cast<T>(i));
            }

            alloc.deallocate(ptr, nelements);
            CATCH_REQUIRE(
                svs::detail::MMapAllocationRegistry::instance().allocation_count() ==
                count_before
            );
        }

        CATCH_SECTION("Rebind and Equality") {
            auto int_alloc = svs::MMapAllocator<int>(temp_dir, svs::MMapAccessHint::Normal);
            auto float_alloc = svs::MMapAllocator<float>(int_alloc);

            CATCH_REQUIRE(float_alloc.get_base_path() == int_alloc.get_base_path());
            CATCH_REQUIRE(float_alloc.get_access_hint() == int_alloc.get_access_hint());
            CATCH_REQUIRE(float_alloc == svs::MMapAllocator<float>(temp_dir));
        }

        CATCH_SECTION("MMapAllocator with SimpleData (non-blocked)") {
            using DataType =
                svs::data::SimpleData<float, svs::Dynamic, svs::MMapAllocator<float>>;

            // Load reference data
            auto original_data = test_dataset::data_f32();
            CATCH_REQUIRE(original_data.size() > 0);

            // Create a SimpleData container with MMapAllocator
            auto alloc =
                svs::MMapAllocator<float>(temp_dir, svs::MMapAccessHint::Sequential);
            auto count_before =
                svs::detail::MMapAllocationRegistry::instance().allocation_count();

            // Construct SimpleData with MMapAllocator as template parameter
            auto mmap_data =
                DataType(original_data.size(), original_data.dimensions(), alloc);
            CATCH_REQUIRE(mmap_data.size() == original_data.size());
            CATCH_REQUIRE(mmap_data.dimensions() == original_data.dimensions());
            CATCH_REQUIRE(
                svs::detail::MMapAllocationRegistry::instance().allocation_count() ==
                count_before + 1
            );

            // Copy original data to mmap'd data
            for (size_t i = 0; i < original_data.size(); ++i) {
                mmap_data.set_datum(i, original_data.get_datum(i));
            }

            // Verify data was copied correctly
            CATCH_REQUIRE(mmap_data == original_data);

            // Load the data directly from the file
            auto mmap_data2 = DataType::load(test_dataset::data_svs_file(), alloc);
            // Verify loaded data matches original data
            CATCH_REQUIRE(mmap_data2 == original_data);
        }

        CATCH_SECTION("MMapAllocator with BlockedData (using Blocked wrapper)") {
            using DataType =
                svs::data::BlockedData<float, svs::Dynamic, svs::MMapAllocator<float>>;

            // Load reference blocked data
            auto original_data = test_dataset::data_blocked_f32();
            CATCH_REQUIRE(original_data.size() > 0);

            // Create an allocator
            auto alloc = svs::MMapAllocator<float>(temp_dir, svs::MMapAccessHint::Normal);
            auto count_before =
                svs::detail::MMapAllocationRegistry::instance().allocation_count();

            // Compute blocking parameters based on original data size and dimensions
            // to ensure we allocate 2 blocks for the test dataset.
            auto blocking_params = svs::data::BlockingParameters{
                .blocksize_bytes = svs::lib::prevpow2(
                    sizeof(float) * original_data.dimensions() * original_data.size() - 1
                )};

            // Construct BlockedData with MMapAllocator via Blocked wrapper
            auto blocked_alloc =
                svs::data::Blocked<svs::MMapAllocator<float>>(blocking_params, alloc);

            auto mmap_data = DataType(1, original_data.dimensions(), blocked_alloc);
            mmap_data.resize(original_data.size());
            CATCH_REQUIRE(mmap_data.size() == original_data.size());
            CATCH_REQUIRE(mmap_data.dimensions() == original_data.dimensions());
            CATCH_REQUIRE(
                svs::detail::MMapAllocationRegistry::instance().allocation_count() ==
                count_before + 2
            );

            // Copy original data to mmap'd data
            for (size_t i = 0; i < original_data.size(); ++i) {
                mmap_data.set_datum(i, original_data.get_datum(i));
            }

            CATCH_REQUIRE(mmap_data == original_data);

            // Load the data directly from the file
            auto mmap_data2 = DataType::load(test_dataset::data_svs_file(), blocked_alloc);
            // Verify loaded data matches original data
            CATCH_REQUIRE(mmap_data2 == original_data);
        }

        CATCH_SECTION("MMapAllocator with SimpleGraph") {
            using GraphType =
                svs::graphs::SimpleGraph<uint32_t, svs::MMapAllocator<uint32_t>>;

            // Load reference graph
            auto original_graph = test_dataset::graph();
            CATCH_REQUIRE(original_graph.n_nodes() > 0);

            // Create an allocator for graph nodes (uint32_t)
            auto alloc =
                svs::MMapAllocator<uint32_t>(temp_dir, svs::MMapAccessHint::Random);
            auto count_before =
                svs::detail::MMapAllocationRegistry::instance().allocation_count();

            // Construct SimpleGraph with MMapAllocator as template parameter
            auto mmap_graph =
                GraphType(original_graph.n_nodes(), original_graph.max_degree(), alloc);
            CATCH_REQUIRE(mmap_graph.n_nodes() == original_graph.n_nodes());
            CATCH_REQUIRE(mmap_graph.max_degree() == original_graph.max_degree());
            CATCH_REQUIRE(
                svs::detail::MMapAllocationRegistry::instance().allocation_count() ==
                count_before + 1
            );

            // Copy edges from original to mmap'd graph
            for (size_t i = 0; i < original_graph.n_nodes(); ++i) {
                mmap_graph.replace_node(i, original_graph.get_node(i));
            }

            // Verify edges were copied correctly
            CATCH_REQUIRE(mmap_graph == original_graph);

            // Load the graph directly from the file
            auto mmap_graph2 = GraphType::load(test_dataset::graph_file(), alloc);
            // Verify loaded graph matches original graph
            CATCH_REQUIRE(mmap_graph2 == original_graph);
        }

        CATCH_SECTION("MMapAllocator with SimpleBlockedGraph (underlying data)") {
            // Note: SimpleBlockedGraph is hardcoded to use HugepageAllocator, but we can
            // test that using SimpleGraph with Blocked wrapper
            using GraphType = svs::graphs::
                SimpleGraph<uint32_t, svs::data::Blocked<svs::MMapAllocator<uint32_t>>>;

            auto original_graph = test_dataset::graph();
            CATCH_REQUIRE(original_graph.n_nodes() > 0);

            // Create an allocator
            auto alloc =
                svs::MMapAllocator<uint32_t>(temp_dir, svs::MMapAccessHint::Random);
            auto count_before =
                svs::detail::MMapAllocationRegistry::instance().allocation_count();

            // Compute blocking parameters based on original graph size and dimensions
            // to ensure we allocate at least 2 blocks for the test dataset.
            auto blocking_params = svs::data::BlockingParameters{
                .blocksize_bytes = svs::lib::prevpow2(
                    sizeof(uint32_t) * original_graph.n_nodes() *
                        original_graph.max_degree() -
                    1
                )};

            // Construct Graph with MMapAllocator via Blocked wrapper
            auto blocked_alloc =
                svs::data::Blocked<svs::MMapAllocator<uint32_t>>(blocking_params, alloc);
            auto mmap_graph = GraphType(
                original_graph.n_nodes(), original_graph.max_degree(), blocked_alloc
            );

            CATCH_REQUIRE(
                svs::detail::MMapAllocationRegistry::instance().allocation_count() >=
                count_before + 2
            );

            // Copy edges from original to mmap'd graph
            for (size_t i = 0; i < original_graph.n_nodes(); ++i) {
                mmap_graph.replace_node(i, original_graph.get_node(i));
            }

            // Verify edges were copied correctly
            CATCH_REQUIRE(mmap_graph == original_graph);

            // Load the graph directly from the file
            auto mmap_graph2 = GraphType::load(test_dataset::graph_file(), blocked_alloc);
            // Verify loaded graph matches original graph
            CATCH_REQUIRE(mmap_graph2 == original_graph);
        }
    }
}
