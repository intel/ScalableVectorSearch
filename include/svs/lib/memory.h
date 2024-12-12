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

    // Construct from another value type allocator.
    template <typename Other>
    explicit constexpr Allocator(const Allocator<Other>& SVS_UNUSED(other)) {}

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
