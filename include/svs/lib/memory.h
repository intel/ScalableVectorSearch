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
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace svs {
namespace lib {

// Enum to select which kind of memory will be used.
namespace memory {

///
/// Default struct for types that aren't dense storage.
///
template <typename T> struct PointerTraits {
    static constexpr bool is_storage = false;
};

// Traits for objects returned from allocators.
struct PointerTraitsBase {
    static constexpr bool is_persistent = false;
    static constexpr bool is_storage = true;
    static constexpr bool is_owning = true;
    static constexpr bool writable = true;
    static constexpr bool disable_implicit_copy = false;
};

///
/// Evaluates to `true` if the type `T` specializes `PointerTraits` providing access
/// information.
///
template <typename T> inline constexpr bool is_storage_v = PointerTraits<T>::is_storage;

///
/// Conecpt for `is_storage_v`.
///
template <typename T>
concept Storage = is_storage_v<T>;

///
/// Evaluates to `true` if type `T` owns the storage it references. In this context,
/// ownership implies that `T` will deallocate the backing memory when its destructor runs.
///
template <Storage T> inline constexpr bool is_owning_v = PointerTraits<T>::is_owning;

///
/// The value obtained when dereferencing a pointer to memory within the storage range.
///
template <Storage T> using storage_value_type_t = typename PointerTraits<T>::value_type;

///
/// Evaluate the type of the allocator for the storage type `T`.
///
template <Storage T> using allocator_type_t = typename PointerTraits<T>::allocator;

///
/// Evaluates to `true` if storage type `T` allows implicit copies.
/// Cases where implicit copy might be disallowed would be for memory mapped pointers.
/// In this situation, one would likely not want implicit copying because the memory mapped
/// data is persistent and changes to copies would not be observed.
///
template <Storage T>
inline constexpr bool implicit_copy_enabled_v = !PointerTraits<T>::disable_implicit_copy;

///
/// Extendable trait to determine trivial constructability of Allocators.
/// Storage objects that do not have an associated allocator may return `void` as their
/// associated allocator.
///
template <typename T>
inline constexpr bool may_trivially_construct = std::is_trivially_constructible_v<T>;
template <> inline constexpr bool may_trivially_construct<void> = false;

template <typename T> struct IsAllocator {
    static constexpr bool value = false;
};
template <typename T> inline constexpr bool is_allocator_v = IsAllocator<T>::value;

template <typename T>
concept MemoryAllocator = is_allocator_v<T>;

/////
///// Generic Access
/////

// TODO: Constrain return type to a pointer?
template <typename T> auto access_storage(T&& x) {
    return PointerTraits<std::remove_cvref_t<T>>::access(std::forward<T>(x));
}

// By default, storage does not have an associated filename.
template <typename T> std::optional<std::string> filename(const T& /*unused*/) {
    return std::optional<std::string>{};
}

/////
///// Pointer
/////

// Storage traits for pointers.
template <typename T> struct PointerTraits<T*> : PointerTraitsBase {
    using value_type = T;
    using allocator = void;
    static constexpr bool is_owning = false;

    // Data Access
    static constexpr T* access(T* x) { return x; }
    static constexpr const T* access(const T* x) { return x; }
};

template <typename T> struct PointerTraits<const T*> : PointerTraitsBase {
    using value_type = const T;
    using allocator = void;
    static constexpr bool is_owning = false;
    static constexpr bool writable = false;

    // Data Access
    static constexpr const T* access(const T* x) { return x; }
};
} // namespace memory

// For the allocator entry point, we usually want to pass a type and a count.
// However, users of allocators may want to be more explicit and pass an actual
// number of bytes.
//
// We support that by using a `Bytes` struct to precisely specify the unit of allocation.
struct Bytes {
    Bytes() = default;
    explicit Bytes(size_t value)
        : value{value} {}

    Bytes operator+(size_t increment) const { return Bytes(value + increment); }

    size_t value;
};

inline size_t value(Bytes x) { return x.value; }

/////
///// allocate_managed
/////

// Required method for allocators.
template <typename T, typename Allocator>
[[nodiscard]] auto allocate_managed(Allocator&& allocator, Bytes bytes) {
    return allocator.template allocate_managed<T>(bytes);
}

// This is a helper method call the member function without the need for the
// ugly ".template" syntax.
template <typename T, typename Allocator>
[[nodiscard]] auto allocate_managed(Allocator&& allocator, size_t count) {
    return allocate_managed<T>(
        std::forward<Allocator>(allocator), Bytes(sizeof(T) * count)
    );
}

/////
///// Default Allocator
/////

// Forward Declare
class UniquePtrAllocator;

using DefaultAllocator = UniquePtrAllocator;
template <typename T>
using DefaultStorage = decltype(allocate_managed<T>(std::declval<DefaultAllocator>(), 0));

/////
///// std::vector allocator
/////

class VectorAllocator {
  public:
    VectorAllocator() = default;
    template <typename T> auto allocate_managed(Bytes bytes) const {
        return std::vector<T>(lib::div_round_up(value(bytes), sizeof(T)));
    }
};

namespace memory {
template <> struct IsAllocator<VectorAllocator> {
    static constexpr bool value = true;
};
template <typename T> struct PointerTraits<std::vector<T>> : PointerTraitsBase {
    using value_type = T;
    using allocator = VectorAllocator;
    // Data Access
    static T* access(std::vector<T>& x) { return x.data(); }
    static const T* access(const std::vector<T>& x) { return x.data(); }
};
} // namespace memory

/////
///// `ptr allocator`
/////

class UniquePtrAllocator {
  public:
    UniquePtrAllocator() = default;
    template <typename T> [[nodiscard]] auto allocate_managed(Bytes bytes) const {
        size_t count = lib::div_round_up(value(bytes), sizeof(T));
        return std::unique_ptr<T[]>(new T[count]);
    }
};

namespace memory {
template <> struct IsAllocator<UniquePtrAllocator> {
    static constexpr bool value = true;
};
template <typename T> struct PointerTraits<std::unique_ptr<T[]>> : PointerTraitsBase {
    using value_type = T;
    using allocator = UniquePtrAllocator;
    // Data Access
    static T* access(std::unique_ptr<T[]>& x) { return x.get(); }
    static const T* access(const std::unique_ptr<T[]>& x) { return x.get(); }
};
} // namespace memory
} // namespace lib
} // namespace svs
