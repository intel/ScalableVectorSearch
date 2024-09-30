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
#pragma once

// svs
#include "svs/lib/misc.h"

// stl
#include <array>
#include <concepts>
#include <cstddef>
#include <iostream>
#include <new>
#include <vector>

namespace svs::threads {

///
/// Shallow Copies
///

// clang-format off
template <typename T>
concept ShallowCopyable = requires(const T& x) {
    { x.shallow_copy() } -> std::same_as<T>;
};
// clang-format on

template <typename T> constexpr bool shallow_copyable_v = false;
template <ShallowCopyable T> constexpr bool shallow_copyable_v<T> = true;

template <typename T> constexpr T shallow_copy(const T& x) { return T{x}; }
template <ShallowCopyable T> constexpr T shallow_copy(const T& x) {
    return x.shallow_copy();
}

///
/// Cache Aligned Allocator
///

const size_t CACHE_LINE_BYTES = 64;
///
/// A minimal allocator that allocates memory aligned to cache line boundaries and in
/// multiples of the cache line size.
///
/// Useful for allocating containers that are meant to be used per-thread.
///
template <typename T, size_t Alignment = CACHE_LINE_BYTES> struct AlignedAllocator {
    using value_type = T;
    constexpr static size_t alignment = Alignment;

    // Need to explicitly define `rebind` since the trailing parameters for this allocator
    // are value parameters rather than type parameters.
    template <typename U> struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    // Constructors
    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>& /*other*/) {}

    // Methods
    value_type* allocate(size_t count) {
        size_t bytes = alignment * lib::div_round_up(sizeof(T) * count, alignment);
        return static_cast<value_type*>(::operator new(bytes, std::align_val_t{alignment}));
    }

    void deallocate(value_type* ptr, size_t count) noexcept {
        ::operator delete(ptr, count, std::align_val_t{alignment});
    }
};

// Default to the cacheline granularity for x86 base platforms.
template <typename T> using CacheAlignedAllocator = AlignedAllocator<T, CACHE_LINE_BYTES>;

template <typename T, typename U, size_t N>
constexpr bool operator==(
    const AlignedAllocator<T, N>& /*unused*/, const AlignedAllocator<U, N>& /*unused*/
) {
    return true;
}

template <typename T, typename U, size_t N>
constexpr bool
operator!=(const AlignedAllocator<T, N>& x, const AlignedAllocator<U, N>& y) {
    return !(x == y);
}

///
/// Pad types to a multiple of the underlying cache size.
/// Helpful for creating thread local storage objects.
///
template <typename T> struct Padded {
    // Constructor
    constexpr explicit Padded(T v)
        : value{std::move(v)} {}

    template <typename... Args>
    constexpr explicit Padded(Args&&... args)
        : value{std::forward<Args>(args)...} {}

    // Shallow copy API
    constexpr Padded shallow_copy() const {
        return Padded{svs::threads::shallow_copy(value)};
    }

    // Get the wrapped value.
    constexpr T& unwrap() { return value; }
    constexpr const T& unwrap() const { return value; }

    /// Members
    alignas(CACHE_LINE_BYTES) T value;
};

template <typename T, typename... Args> constexpr Padded<T> make_padded(Args&&... args) {
    return Padded<T>{std::forward<Args>(args)...};
}

template <typename T> class SequentialTLS {
  public:
    using value_type = T;
    using const_value_type = const T;
    using reference = value_type&;
    using const_reference = const value_type&;

    using padded_value_type = Padded<value_type>;
    using allocator = CacheAlignedAllocator<padded_value_type>;
    using container_type = std::vector<padded_value_type, allocator>;

    ///// Constructors
    explicit SequentialTLS(size_t count)
        : values_(count){};

    explicit SequentialTLS(T base, size_t count = 1)
        : values_(count, Padded{std::move(base)}) {}

    reference operator[](size_t i) { return values_[i].unwrap(); }
    const_reference operator[](size_t i) const { return values_[i].unwrap(); }

    reference at(size_t i) { return values_.at(i).unwrap(); }
    const_reference at(size_t i) const { return values_.at(i).unwrap(); }

    size_t size() const { return values_.size(); }

    void resize(size_t new_size) {
        new_size = std::max(size_t{1}, new_size);
        values_.resize(new_size, values_.front());
    }

    const void* data() const { return static_cast<const void*>(values_.data()); }

    // Iterator
    template <typename F> void visit(F&& f) {
        for (auto& v : values_) {
            f(v.unwrap());
        }
    }

    template <typename F> void visit(F&& f) const {
        for (const auto& v : values_) {
            f(v.unwrap());
        }
    }

  private:
    container_type values_;
};
} // namespace svs::threads
