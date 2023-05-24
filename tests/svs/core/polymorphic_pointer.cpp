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

// svs
#include "svs/core/polymorphic_pointer.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stdlib
#include <atomic>
#include <vector>

// Implement a test storage imlementation to ensure destructors work correctly.
template <typename T> class VectorSnitch {
  private:
    std::vector<T> base;
    std::atomic<size_t>& destructor_count;

  public:
    T* data() { return base.data(); }
    const T* data() const { return base.data(); }
    VectorSnitch(std::vector<T>&& base, std::atomic<size_t>& destructor_count)
        : base{std::move(base)}
        , destructor_count{destructor_count} {}

    VectorSnitch(VectorSnitch&& other) = default;
    VectorSnitch& operator=(VectorSnitch&& other) = default;
    ~VectorSnitch() {
        if (data()) {
            ++destructor_count;
        }
    }
};

class VectorSnitchAllocator {
  private:
    std::atomic<size_t> counter = 0;

  public:
    template <typename T> auto allocate_managed(svs::lib::Bytes n) {
        size_t count = svs::lib::value(n) / sizeof(T);
        return VectorSnitch<T>(std::vector<T>(count), counter);
    }

    size_t getcount() { return counter.load(); }
};

namespace svs::lib::memory {
template <> struct IsAllocator<VectorSnitchAllocator> {
    static constexpr bool value = true;
};

template <typename T> struct PointerTraits<VectorSnitch<T>> {
    using value_type = T;
    using allocator = VectorSnitchAllocator;
    static constexpr bool is_storage = true;
    static constexpr bool writable = true;
    // N.B.: We're lying and saying that this is persistent.
    // That's mostly so we can test that this is propagated correctly to the
    // `PolymorphicPointer`.
    static constexpr bool is_persistent = true;

    static T* access(VectorSnitch<T>& x) { return x.data(); }
    static const T* access(const VectorSnitch<T>& x) { return x.data(); }
};
} // namespace svs::lib::memory

CATCH_TEST_CASE("Polymorphic Pointer", "[core][polymorphic_pointer]") {
    namespace lib = svs::lib;
    namespace memory = lib::memory;
    CATCH_SECTION("Vector Allocator") {
        auto allocator = lib::VectorAllocator{};
        size_t num_elements = 200;
        std::vector<float> storage = lib::allocate_managed<float>(allocator, num_elements);
        float* base_ptr = storage.data();
        for (auto& i : storage) {
            i = 2.0;
        }

        auto test = svs::PolymorphicPointer<float>(std::move(storage));
        CATCH_REQUIRE(memory::access_storage(test) == base_ptr);

        // Try the move operator
        auto other = std::move(test);
        CATCH_REQUIRE(memory::access_storage(other) == base_ptr);

        // Is our data still set?
        auto base = memory::access_storage(other);
        for (size_t i = 0; i < num_elements; ++i) {
            CATCH_REQUIRE(*(base + i) == 2.0);
        }
    }

    CATCH_SECTION("Destructor Monitor") {
        auto allocator = VectorSnitchAllocator{};
        CATCH_SECTION("Test 1") {
            CATCH_REQUIRE(allocator.getcount() == 0);
            {
                const size_t num_elements = 200;
                auto base = lib::allocate_managed<float>(allocator, num_elements);
                float* base_ptr = memory::access_storage(base);
                for (size_t i = 0; i < num_elements; ++i) {
                    *(base_ptr + i) = 5.0f;
                }
                auto erased = svs::PolymorphicPointer(std::move(base));
                CATCH_REQUIRE(allocator.getcount() == 0);
                CATCH_REQUIRE(memory::access_storage(erased) == base_ptr);
                for (size_t i = 0; i < num_elements; ++i) {
                    CATCH_REQUIRE(*(base_ptr + i) == 5.0f);
                }

                auto other = std::move(erased);
                CATCH_REQUIRE(allocator.getcount() == 0);
                CATCH_REQUIRE(memory::access_storage(other) == base_ptr);
                // Destructor for storage runs.
            }
            CATCH_REQUIRE(allocator.getcount() == 1);
        }

        CATCH_SECTION("Test 2") {
            CATCH_REQUIRE(allocator.getcount() == 0);
            {
                const size_t num_elements = 200;
                auto base_1 = lib::allocate_managed<float>(allocator, num_elements);
                auto base_2 = lib::allocate_managed<float>(allocator, num_elements);
                auto base_1_ptr = memory::access_storage(base_1);
                auto base_2_ptr = memory::access_storage(base_2);
                CATCH_REQUIRE(base_1_ptr != base_2_ptr);
                for (size_t i = 0; i < num_elements; ++i) {
                    *(base_1_ptr + i) = 5.0f;
                    *(base_2_ptr + i) = 10.0f;
                }

                auto erased_1 = svs::PolymorphicPointer(std::move(base_1));
                auto erased_2 = svs::PolymorphicPointer(std::move(base_2));
                // Now, move 1 into 2.
                erased_2 = std::move(erased_1);
                CATCH_REQUIRE(memory::access_storage(erased_2) == base_1_ptr);
                for (size_t i = 0; i < num_elements; ++i) {
                    *(base_1_ptr + i) = 5.0f;
                }

                // Destructors run here.
            }
            CATCH_REQUIRE(allocator.getcount() == 2);
        }
    }
}
