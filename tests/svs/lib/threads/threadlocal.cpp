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
#include <algorithm>
#include <vector>

// svs
#include "svs/lib/threads/threadlocal.h"

// catch2
#include "catch2/catch_test_macros.hpp"

namespace {

struct NoShallowCopy {
    static constexpr int default_value() { return 10; }

    NoShallowCopy(size_t count, int val = default_value())
        : data(count, val) {}
    size_t size() const { return data.size(); }

    // Members
    std::vector<int> data;
};

// For the shallow copy version - we change the element type.
// This is how we can detect that the shallow copy code path was taken.
struct ShallowCopy {
    static constexpr int default_value() { return 20; }
    static constexpr int shallow_value() { return 0; }
    ShallowCopy(size_t count, int val = default_value())
        : data(count, val) {}
    size_t size() const { return data.size(); }
    ShallowCopy shallow_copy() const { return ShallowCopy{data.size(), shallow_value()}; }

    // Members
    std::vector<int> data;
};

template <typename T> int64_t address_offset(const T* a, const T* b) {
    return reinterpret_cast<const std::byte*>(a) - reinterpret_cast<const std::byte*>(b);
}

// Static tests
using Allocator = svs::threads::AlignedAllocator<float, 4096>;
using Traits = std::allocator_traits<Allocator>;

#define SVSTEST_CHECK(lhs, rhs) static_assert(std::is_same_v<typename Traits::lhs, rhs>);

SVSTEST_CHECK(allocator_type, Allocator);
SVSTEST_CHECK(value_type, float);
SVSTEST_CHECK(pointer, float*);
SVSTEST_CHECK(const_pointer, const float*);
SVSTEST_CHECK(void_pointer, void*);
SVSTEST_CHECK(const_void_pointer, const void*);
SVSTEST_CHECK(difference_type, std::ptrdiff_t);
SVSTEST_CHECK(size_type, std::size_t);
SVSTEST_CHECK(propagate_on_container_copy_assignment, std::false_type);
SVSTEST_CHECK(propagate_on_container_move_assignment, std::false_type);
SVSTEST_CHECK(propagate_on_container_swap, std::false_type);
SVSTEST_CHECK(is_always_equal, std::true_type);

// Rebind
static_assert(std::is_same_v<
              typename Traits::rebind_alloc<size_t>,
              svs::threads::AlignedAllocator<size_t, 4096>>);

} // namespace

CATCH_TEST_CASE("Utils", "[core][util]") {
    CATCH_SECTION("Shallow Copy") {
        CATCH_REQUIRE(svs::threads::shallow_copyable_v<NoShallowCopy> == false);
        CATCH_REQUIRE(svs::threads::shallow_copyable_v<ShallowCopy> == true);

        /// Version without a shallow copy
        NoShallowCopy x{5};
        CATCH_REQUIRE(x.size() == 5);
        CATCH_REQUIRE(std::all_of(x.data.begin(), x.data.end(), [&](auto&& v) {
            return v == x.default_value();
        }));

        // Shallow copy should call the copy constructor.
        auto y = svs::threads::shallow_copy(x);
        CATCH_REQUIRE(y.size() == 5);
        CATCH_REQUIRE(std::all_of(y.data.begin(), y.data.end(), [&](auto&& v) {
            return v == y.default_value();
        }));

        /// Version with a shallow copy
        ShallowCopy z{5};
        CATCH_REQUIRE(z.shallow_value() != z.default_value());
        CATCH_REQUIRE(std::all_of(z.data.begin(), z.data.end(), [&](auto&& v) {
            return v == z.default_value();
        }));

        auto zz = svs::threads::shallow_copy(z);
        CATCH_REQUIRE(std::all_of(zz.data.begin(), zz.data.end(), [&](auto&& v) {
            return v == zz.shallow_value();
        }));
    }

    CATCH_SECTION("CacheAlignedAllocator") {
        // Try the allocator on its own.
        svs::threads::CacheAlignedAllocator<size_t> alloc{};
        CATCH_REQUIRE(decltype(alloc)::alignment == svs::threads::CACHE_LINE_BYTES);
        for (size_t i = 1; i < 100; ++i) {
            size_t* ptr = alloc.allocate(i);
            intptr_t address = reinterpret_cast<intptr_t>(ptr);
            CATCH_REQUIRE((address % svs::threads::CACHE_LINE_BYTES) == 0);
            alloc.deallocate(ptr, i);
        }

        // Put it in a vector.
        std::vector<int, svs::threads::CacheAlignedAllocator<int>> v{};
        for (int i = 0; i < 100; ++i) {
            v.push_back(i);
            CATCH_REQUIRE(v.at(i) == i);
            CATCH_REQUIRE(
                reinterpret_cast<intptr_t>(v.data()) % svs::threads::CACHE_LINE_BYTES == 0
            );
        }
    }

    CATCH_SECTION("PageAlignedAllocator") {
        // Try the allocator on its own.
        constexpr size_t alignment = 4096;
        svs::threads::AlignedAllocator<size_t, alignment> alloc{};
        CATCH_REQUIRE(decltype(alloc)::alignment == alignment);
        for (size_t i = 1; i < 100; ++i) {
            size_t* ptr = alloc.allocate(i);
            intptr_t address = reinterpret_cast<intptr_t>(ptr);
            CATCH_REQUIRE((address % alignment) == 0);
            alloc.deallocate(ptr, i);
        }

        // Put it in a vector.
        std::vector<int, svs::threads::AlignedAllocator<int, alignment>> v{};
        for (int i = 0; i < 100; ++i) {
            v.push_back(i);
            CATCH_REQUIRE(v.at(i) == i);
            CATCH_REQUIRE(reinterpret_cast<intptr_t>(v.data()) % alignment == 0);
        }
    }

    CATCH_SECTION("Padded") {
        CATCH_SECTION("Basic") {
            auto x = svs::threads::Padded(size_t{10});
            CATCH_REQUIRE(x.value == 10);
            CATCH_REQUIRE(sizeof(decltype(x)) == svs::threads::CACHE_LINE_BYTES);

            auto v = std::vector<int>(10);
            for (size_t i = 0, imax = v.size(); i < imax; ++i) {
                v.at(i) = i;
            }

            auto y = svs::threads::Padded(v);
            CATCH_REQUIRE(sizeof(decltype(y)) == svs::threads::CACHE_LINE_BYTES);
            auto& vv = y.value;
            CATCH_REQUIRE(vv.size() == v.size());
            for (int i = 0, imax = vv.size(); i < imax; ++i) {
                CATCH_REQUIRE(vv.at(i) == i);
            }
        }

        // Shallow copy compatibility
        CATCH_SECTION("Shallow Copy") {
            // The padded type should always be `shallow_copyable` because it defines
            // the `shallow_copy` method.
            CATCH_REQUIRE(
                svs::threads::shallow_copyable_v<svs::threads::Padded<NoShallowCopy>> ==
                true
            );
            CATCH_REQUIRE(
                svs::threads::shallow_copyable_v<svs::threads::Padded<ShallowCopy>> == true
            );

            /// Version without a shallow copy
            auto x = svs::threads::make_padded<NoShallowCopy>(size_t{5});
            CATCH_REQUIRE(x.unwrap().size() == 5);
            auto x_data = x.unwrap().data;
            CATCH_REQUIRE(std::all_of(x_data.begin(), x_data.end(), [&](auto&& v) {
                return v == NoShallowCopy::default_value();
            }));

            // Shallow copy should call the copy constructor.
            auto y = svs::threads::shallow_copy(x);
            CATCH_REQUIRE(y.unwrap().size() == 5);
            auto y_data = y.unwrap().data;
            CATCH_REQUIRE(std::all_of(y_data.begin(), y_data.end(), [&](auto&& v) {
                return v == NoShallowCopy::default_value();
            }));

            /// Version with a shallow copy
            auto z = svs::threads::make_padded<ShallowCopy>(size_t{5});
            CATCH_REQUIRE(ShallowCopy::shallow_value() != ShallowCopy::default_value());
            auto z_data = z.unwrap().data;
            CATCH_REQUIRE(std::all_of(z_data.begin(), z_data.end(), [&](auto&& v) {
                return v == ShallowCopy::default_value();
            }));

            auto zz = svs::threads::shallow_copy(z);
            auto zz_data = zz.unwrap().data;
            CATCH_REQUIRE(std::all_of(zz_data.begin(), zz_data.end(), [&](auto&& v) {
                return v == ShallowCopy::shallow_value();
            }));
        }
    }
}

CATCH_TEST_CASE("Sequential TLS", "[core][util]") {
    svs::threads::SequentialTLS<size_t> tls{0, 4};
    auto a = std::addressof(tls.at(0));
    auto b = std::addressof(tls.at(1));
    CATCH_REQUIRE(address_offset(b, a) == svs::threads::CACHE_LINE_BYTES);
}
