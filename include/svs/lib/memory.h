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

#pragma once

///
/// @ingroup lib_public
/// @defgroup lib_public_memory Polymorphic Memory Resource
///

// svs
#include "svs/lib/misc.h"

// stl
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace svs {
namespace lib {

///
/// @brief A default initializing version of `std::allocator`.
///
template <typename T> struct Allocator {
  public:
    // Type Aliases
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;

    // Constructor
    constexpr Allocator() = default;

    // Allocation and Deallocation.
    [[nodiscard]] constexpr value_type* allocate(std::size_t n) {
        return static_cast<value_type*>(
            ::operator new(n * sizeof(T), std::align_val_t(alignof(T)))
        );
    }

    constexpr void deallocate(value_type* ptr, size_t count) noexcept {
        ::operator delete(ptr, count);
    }

    // Intercept zero-argument construction to do default initialization.
    template <typename U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(p)) U;
    }
};

// For the allocator entry point, we usually want to pass a type and a count.
// However, users of allocators may want to be more explicit and pass an actual
// number of bytes.
//
// We support that by using a `Bytes` struct to precisely specify the unit of allocation.
struct Bytes {
  public:
    Bytes() = default;
    explicit Bytes(size_t value)
        : value_{value} {}

    Bytes operator+(size_t increment) const { return Bytes(value_ + increment); }

    size_t value() const { return value_; }

  public:
    size_t value_;
};
} // namespace lib
} // namespace svs
