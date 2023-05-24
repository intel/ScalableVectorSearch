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

#include "svs/core/allocator.h"
#include "svs/lib/array.h"
#include "svs/lib/memory.h"

namespace svs {

/////
///// Polymorphic Pointer supporting Memory Map.
/////

///
/// @ingroup lib_public_memory
/// @brief A polymorphic smart-pointer for memory management.
///
/// @tparam T The type pointed to by the base pointer.
///
/// The library supports multiple allocators to allow different ways of obtaining memory.
/// Often, this allocated memory requires special routines for deallocation.
/// However, propagating full type information for these resources is problematic as it
/// requires template propagation and extra code generation.
///
/// The PolymorphicPointer class implements a type-erased smart pointer to enable
/// smart-pointer propagation through the library while providing a unified type.
///
template <typename T> class PolymorphicPointer {
  private:
    class Interface {
      public:
        Interface() = default;
        virtual T* data() = 0;
        virtual const T* data() const = 0;

        // Delete copy constructors and assignment operators.
        Interface(const Interface&) = delete;
        Interface& operator=(const Interface&) = delete;
        // Default move operators.
        Interface(Interface&&) noexcept = default;
        Interface& operator=(Interface&&) noexcept = default;
        // Virtual destructor.
        virtual ~Interface() = default;
    };

    template <typename Ptr> class Impl : public Interface {
      private:
        Ptr ptr_;

      public:
        // Type Aliases
        using traits = typename lib::memory::PointerTraits<Ptr>;
        static constexpr bool is_persistent_impl = traits::is_persistent;

        // Constructor
        explicit Impl(Ptr&& ptr)
            : ptr_{std::move(ptr)} {}

        // Accessors
        T* data() override { return lib::memory::access_storage(ptr_); }
        const T* data() const override { return lib::memory::access_storage(ptr_); }

        // Special members
        Impl(const Impl&) noexcept = delete;
        Impl& operator=(const Impl&) noexcept = delete;
        Impl(Impl&&) noexcept = default;
        Impl& operator=(Impl&&) noexcept = default;
        ~Impl() override = default;
    };

    // This is the data owned by the implementation, hoisted out of the actual storage
    // for non-virtual access.
    //
    // As such, this class is not directly responsible for cleaning up the raw pointer,
    // That will be delegated to `GenericStorage`.
    T* data_;
    std::unique_ptr<Interface> impl_;

  public:
    // clang-format off

    ///
    /// @brief Take ownership and type-erase the provided smart pointer.
    ///
    /// @param storage - The (smart) pointer to take ownership of.
    ///
    template <typename Storage>
    explicit PolymorphicPointer(Storage storage)
        : data_{nullptr},
          impl_{std::make_unique<Impl<Storage>>(std::move(storage))} {
        data_ = impl_->data();
    }
    // clang-format on

    /// @brief Return a pointer to the beginning of the memory owned by this smart pointer.
    const T* data() const { return data_; }
    /// @copydoc data() const
    T* data() { return data_; }
};

namespace lib::memory {
template <typename T> struct PointerTraits<PolymorphicPointer<T>> : PointerTraitsBase {
    using value_type = T;
    using allocator = void;
    static constexpr bool disable_implicit_copy = true;
    // Data Access
    static T* access(PolymorphicPointer<T>& x) { return x.data(); }
    static const T* access(const PolymorphicPointer<T>& x) { return x.data(); }
};
} // namespace lib::memory

template <typename T, typename Dims, typename Base>
DenseArray<T, Dims, PolymorphicPointer<T>> polymorph(DenseArray<T, Dims, Base> array) {
    return make_dense_array(
        PolymorphicPointer<T>(array.acquire_base()), array.static_dims()
    );
}

// Deduction Guides
template <lib::memory::Storage Base>
PolymorphicPointer(Base) -> PolymorphicPointer<lib::memory::storage_value_type_t<Base>>;

} // namespace svs
