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
/// @file
/// @brief Implements common large-scale allocators used by the many data structures.
///

///
/// @ingroup core
/// @defgroup core_allocators_entry Core Allocators
/// @brief Memory allocators used for large allocations.
///

///
/// @ingroup core
/// @defgroup core_allocators_public Public API for Allocators
///

#include "svs/lib/memory.h"
#include "svs/lib/misc.h"
#include "svs/lib/narrow.h"

#include "tsl/robin_map.h"

#include <array>
#include <fcntl.h>
#include <filesystem>

// <linux/mman.h> provides some linux-specific flags like
// MAP_POPULATE, MAP_NORESERVE, MAP_HUGETLB.
// They can be used only with linux, and are not available
// for MacOS

#if defined(__linux__)
#include <linux/mman.h>
#endif // __linux__

#include <memory>
#include <optional>
#include <string>
#include <sys/mman.h>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <variant>
#include <vector>

namespace svs {

/////
///// hugepage allocator
/////

// Pagesize and memory map flags for x86 architectures.
// It probably makes sense to put these into their own struct if we want to migrate
// to different architectures/operating systems with different page granulatiries.
//
// That will likely require some more significant changes in the logic here, but at
// least there should be enough flexibility built in to allow for that.
struct HugepageX86Parameters {
    constexpr HugepageX86Parameters(size_t pagesize, int mmap_flags)
        : pagesize{pagesize}
        , mmap_flags{mmap_flags} {};

    // Members
    size_t pagesize;
    int mmap_flags;

    // Check Equality
    friend bool operator==(HugepageX86Parameters l, HugepageX86Parameters r) = default;
};

// Hugepage Allocation will happen in the order given below.
//
// MAP_HUGETLB, MAP_HUGE_1GB, MAP_HUGE_2MB are linux-specific
//
#if defined(__linux__)
static constexpr std::array<HugepageX86Parameters, 3> hugepage_x86_options{
    HugepageX86Parameters{1 << 30, MAP_HUGETLB | MAP_HUGE_1GB},
    HugepageX86Parameters{1 << 21, MAP_HUGETLB | MAP_HUGE_2MB},
    HugepageX86Parameters{1 << 12, 0},
};
#else
static constexpr std::array<HugepageX86Parameters, 1> hugepage_x86_options{
    HugepageX86Parameters{1 << 12, 0}};
#endif // __linux__

namespace detail {

struct HugepageAllocation {
    void* ptr;
    size_t sz;
};

[[nodiscard]] inline HugepageAllocation hugepage_mmap(size_t bytes, bool force = false) {
    assert(bytes != 0);
    void* ptr = MAP_FAILED;
    size_t sz = 0;
    for (auto params : hugepage_x86_options) {
        // Don't fallback to huge pages if `force == true`.
        if (force && params == hugepage_x86_options.back()) {
            break;
        }

        auto pagesize = params.pagesize;
        auto flags = params.mmap_flags;
        sz = lib::round_up_to_multiple_of(bytes, pagesize);

        int mmap_flags = MAP_PRIVATE;
#if defined(MAP_ANONYMOUS)
        mmap_flags |= MAP_ANONYMOUS;
#elif defined(MAP_ANON)
        mmap_flags |= MAP_ANON;
#else
#error "System does not support anonymous mmap (no MAP_ANONYMOUS or MAP_ANON)"
#endif
// Add Linux-specific flags
#ifdef __linux__
        mmap_flags |= MAP_POPULATE;
#endif // __linux__

        ptr = mmap(nullptr, sz, PROT_READ | PROT_WRITE, mmap_flags | flags, -1, 0);

        if (ptr != MAP_FAILED) {
            break;
        }
    }

    if (ptr == MAP_FAILED) {
        throw ANNEXCEPTION("Hugepage memory map allocation of size {} failed!", bytes);
    }
    return HugepageAllocation{.ptr = ptr, .sz = sz};
}

[[nodiscard]] inline bool hugepage_unmap(void* ptr, size_t sz) {
    return (munmap(ptr, sz) == 0);
}

class GenericHugepageAllocator {
  public:
    GenericHugepageAllocator() = default;

    template <typename... Args> [[nodiscard]] void* allocate(size_t bytes, Args&&... args) {
        auto [ptr, sz] = hugepage_mmap(bytes, SVS_FWD(args)...);
        {
            std::lock_guard lock{mutex_};
            ptr_to_size_.insert({ptr, sz});
        }
        return ptr;
    }

    static void deallocate(void* ptr) {
        std::lock_guard lock{mutex_};
        auto itr = ptr_to_size_.find(ptr);
        if (itr == ptr_to_size_.end()) {
            throw ANNEXCEPTION("Could not find a corresponding size of unmap pointer!");
        }
        size_t sz = itr->second;
        ptr_to_size_.erase(itr);
        if (!hugepage_unmap(ptr, sz)) {
            throw ANNEXCEPTION("Unmap failed!");
        }
    }

    static tsl::robin_map<void*, size_t> get_allocations() {
        std::lock_guard lock{mutex_};
        return ptr_to_size_;
    }

  private:
    inline static std::mutex mutex_{};
    inline static tsl::robin_map<void*, size_t> ptr_to_size_{};
};
} // namespace detail

///
/// @ingroup core_allocators_entry
/// @brief Allocator class to use hugepages to back memory allocations.
///
template <typename T> class HugepageAllocator {
  private:
    bool force_ = false;

  public:
    // Allocator type aliases.
    using value_type = T;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::true_type;

    ///
    /// @brief Construct a new allocator
    ///
    /// When default construction is used, the resulting allocator will be configured to
    /// use normal pages if sufficient huge pages are not available to fulfill an
    /// allocation.
    ///
    HugepageAllocator() = default;

    ///
    /// @brief Construct a new HugepageAllocator
    ///
    /// @param force - If ``true``, ensure that memory allocations are fulfilled with
    ///     hugepages, throwing an exception if not enough pages are available.
    ///
    ///     If ``false``, than normal 4 KiB will be used if huge pages are not available.
    ///
    explicit HugepageAllocator(bool force)
        : force_{force} {}

    // Enable rebinding of allocators.
    template <typename U> friend class HugepageAllocator;

    template <typename U>
    HugepageAllocator(const HugepageAllocator<U>& other)
        : force_{other.force_} {}

    template <typename U> bool operator==(const HugepageAllocator<U>& SVS_UNUSED(other)) {
        return true;
    }

    [[nodiscard]] T* allocate(size_t n) {
        return static_cast<T*>(
            detail::GenericHugepageAllocator().allocate(sizeof(T) * n, force_)
        );
    }

    void deallocate(void* ptr, size_t SVS_UNUSED(n)) {
        detail::GenericHugepageAllocator::deallocate(ptr);
    }

    // Perform default initialization.
    void construct(T* ptr) { ::new (static_cast<void*>(ptr)) T; }
};

/////
///// Memory Mapper
/////

///
/// @ingroup core_allocators_public
/// @brief A memory mapped smart pointer.
///
/// Contains the memory for an array of objects of type `T`.
/// The memory pointed to will be freed when this object's destructor is run.
/// Objects of type `T` must be trivially default constructible and trivially copyable.
///
/// @tparam T The element type of the array.
///
template <typename T> class MMapPtr {
  private:
    T* ptr_{nullptr};
    void* base_{nullptr};
    size_t size_{0};

    // Reset the contents of this type.
    void reset() {
        auto null = MMapPtr<T>{};
        ptr_ = null.ptr_;
        base_ = null.base_;
        size_ = null.size_;
    }

    /// Allow for conversion from untyped pointers to typed pointers.
    template <typename U> friend class MMapPtr;

  public:
    /// Return a pointer to the start of valid memory held by the pointer.
    T* data() noexcept { return ptr_; }
    /// @copydoc MMapPtr::data()
    const T* data() const noexcept { return ptr_; }

    ///
    /// @brief Return the first addresses owned by the pointer.
    ///
    /// This is not necessarily the start of the data region as instances are free to use
    /// an arbitrary amount of leading space in memory maps.
    ///
    void* base() noexcept { return base_; }
    /// @copydoc MMapPtr::base()
    const void* base() const noexcept { return base_; }

    ///
    /// @brief Return the size (in bytes) of the total memory mapping.
    ///
    size_t size() const noexcept { return size_; }

    ///
    /// @brief Configure the offset.
    ///
    /// This sets the difference between ``data()`` and ``base()``.
    ///
    void setoffset(size_t offset) {
        ptr_ = reinterpret_cast<T*>(reinterpret_cast<char*>(base_) + offset);
    }

    ///
    /// @brief Construct a new `MMapPtr` around raw pointers.
    ///
    /// @param ptr A pointer to the start of the valid (non-header) data for the mapping.
    /// @param base The pointer returned by the memory mapping.
    /// @param size The size (in bytes) of the total memory mapping (measured from `base`).
    ///
    /// The base pointer must have been obtained from a memory mapping.
    /// In other words, calling `munmap` on it must be valid.
    ///
    MMapPtr(T* ptr, void* base, size_t size)
        : ptr_{ptr}
        , base_{base}
        , size_{size} {};

    ///
    /// @brief Construct a new pointer with no data/base offset.
    ///
    MMapPtr(void* base, size_t size)
        : MMapPtr(reinterpret_cast<T*>(base), base, size) {}

    // Implement what's required by `NullablePointer` in the standard library
    MMapPtr() = default;
    MMapPtr(std::nullptr_t) {}
    explicit operator bool() const { return ptr_ != nullptr; }
    friend bool operator==(const MMapPtr& l, const MMapPtr& r) = default;

    // This is a dangerous thing to call and generally shouldn't be called externally.
    // TODO: Is it okay from a testing stand-point to make this private?
    void unmap() noexcept {
        if (*this) {
            int code = munmap(base_, size_);
            // Something has gone horribly wrong if we double-free a pointer.
            // It should be considered an unrecoverable bug and we should exit hard.
            if (code != 0) {
                std::terminate();
            }
            reset();
        }
    }

    // Special members
    MMapPtr(const MMapPtr& other) = delete;
    MMapPtr& operator=(const MMapPtr& other) = delete;

    // For the move constructor, we don't hold onto any resources, so no need to release
    // anything.
    //
    // Make the conversion from "void" to non-void "explicit" to avoid nasty unexpected
    // implicit conversions.
    template <typename OtherT>
        requires(std::is_same_v<OtherT, T> || std::is_same_v<OtherT, void>)
    MMapPtr(MMapPtr<OtherT>&& other) noexcept
        : ptr_{reinterpret_cast<T*>(other.ptr_)}
        , base_{other.base_}
        , size_{other.size_} {
        other.reset();
    }

    MMapPtr& operator=(MMapPtr&& other) noexcept {
        // Free any resources we're holding onto.
        unmap();
        // Copy over fields.
        ptr_ = other.ptr_;
        base_ = other.base_;
        size_ = other.size_;
        // Clear the other.
        other.reset();
        return *this;
    }

    ~MMapPtr() { unmap(); }
};

///
/// @ingroup core_allocators_entry
/// @brief Allocate memory for a data collection directly from a file using memory mapping.
///
/// Used by more file-aware allocators. Shouldn't be invoked directly.
///
class MemoryMapper {
  public:
    /// @brief Policy for file memory mapping.
    enum Policy {
        /// Memory mapping must use an existing file. A new file will not be created.
        MustUseExisting,
        /// Memory mapping must create a new file. An existing file cannot be used.
        MustCreate,
        /// Memory mapping may either create a new file or use an existing one.
        MayCreate
    };

    /// @brief Permissions for the memory map.
    enum Permission { ReadOnly, ReadWrite };

    static constexpr int open_permissions(Permission permission) {
        switch (permission) {
            case ReadOnly: {
                return O_RDONLY;
            }
            case ReadWrite: {
                return O_RDWR;
            }
        }
        throw ANNEXCEPTION("Unreachable");
    }

    static constexpr int mmap_permissions(Permission permission) {
        switch (permission) {
            case ReadOnly: {
                return PROT_READ;
            }
            case ReadWrite: {
                return PROT_READ | PROT_WRITE;
            }
        }
        throw ANNEXCEPTION("Unreachable");
    };

  private:
    Permission permission_{ReadOnly};
    Policy policy_{MustUseExisting};

  public:
    ///
    /// @brief Construct a new MemoryMapper
    ///
    /// @param permission The permission for the file
    /// @param policy The policy to use when memory mapping. If a situation arises that
    ///     goes against the policy, an ANNException will be thrown when allocation is
    ///     performed.
    ///
    explicit MemoryMapper(Permission permission = ReadOnly, Policy policy = MustUseExisting)
        : permission_{permission}
        , policy_{policy} {}

    // Accessors
    Policy policy() const { return policy_; }
    void setpolicy(Policy policy) { policy_ = policy; }

    Permission permission() const { return permission_; }
    void setpermission(Permission permission) { permission_ = permission; }

    // Allocation
    MMapPtr<void> mmap(const std::filesystem::path& filename, lib::Bytes bytes) const {
        bool exists = std::filesystem::exists(filename);

        // Check policy against the filesystem state.
        if (policy_ == MustCreate && exists) {
            throw ANNEXCEPTION(
                "Memory Map Allocator is configured to create a file ({}) that already "
                "exists!",
                filename
            );
        }
        if (policy_ == MustUseExisting && !exists) {
            throw ANNEXCEPTION(
                "Memory Map Allocator is configured to use an existing file ({}) that does "
                "not exist!",
                filename
            );
        }

        // Now that have some logic checks out of the way, we need to ensure that the
        // existing mapping is large enough.
        int fd = -1;
        if (!exists) {
            // Try to create and resize the file.
            auto create_flags = O_RDWR | O_CREAT | O_TRUNC;
            fd = open(filename.c_str(), create_flags, static_cast<mode_t>(0600));
            if (fd == -1) {
                throw ANNEXCEPTION("Could not create file {}!", filename);
            }
            auto result = lseek(fd, lib::narrow<int64_t>(bytes.value() - 1), SEEK_SET);
            if (result == -1) {
                close(fd);
                throw ANNEXCEPTION("Cannot resize mmap file");
            }
            result = write(fd, "", 1);
            if (result != 1) {
                close(fd);
                throw ANNEXCEPTION("Error writing last byte of the file");
            }
        } else {
            auto filesize = std::filesystem::file_size(filename);
            if (filesize < bytes.value()) {
                throw ANNEXCEPTION(
                    "The size of file ({}) to memory map is {} which is less than the "
                    "number of bytes ({}) requested!",
                    filesize,
                    filename,
                    bytes.value()
                );
            }
            // By this point, we found our file and it looks large enough.
            // Time to perform the memory mapping and set our flags appropriately.
            fd = open(filename.c_str(), open_permissions(permission_));
            if (fd == -1) {
                throw ANNEXCEPTION("Could not open file {}!", filename);
            }
        }
        lseek(fd, 0, SEEK_SET);

        int mmap_flags = MAP_SHARED; // Accessible from all processes
// Add Linux-specific flags
#ifdef __linux__
        mmap_flags |= MAP_NORESERVE; // Don't reserve space in DRAM for this until used
        mmap_flags |= MAP_POPULATE;  // Populate page table entries in the DRAM
#endif                               // __linux__

        void* base = ::mmap(
            nullptr, bytes.value(), mmap_permissions(permission_), mmap_flags, fd, 0
        );
        close(fd);

        if (base == nullptr || base == MAP_FAILED) {
            throw ANNEXCEPTION("Memory Map Failed!");
        }
        return MMapPtr<void>(base, bytes.value());
    }
};

namespace detail {

template <typename Alloc>
concept HasValueType = requires { typename Alloc::value_type; };

// clang-format off
template <typename Alloc>
concept Allocator = HasValueType<Alloc> && std::is_copy_constructible_v<Alloc>
&& requires(Alloc& alloc, typename Alloc::value_type* ptr, size_t n) {
    { alloc.allocate(n) } -> std::same_as<typename Alloc::value_type*>;

    { alloc.deallocate(ptr, n) };
};
// clang-format on

} // namespace detail

/////
///// Type erasure allocator handle implementation
/////

class AllocatorInterface {
  public:
    virtual ~AllocatorInterface() = default;
    virtual void* allocate(size_t n) = 0;
    virtual void deallocate(void* ptr, size_t n) = 0;

    // covariant return type
    virtual AllocatorInterface* clone() const = 0;
    virtual AllocatorInterface* rebind_float() const = 0;
    virtual AllocatorInterface* rebind_float16() const = 0;
};

template <detail::Allocator Impl> class AllocatorImpl : public AllocatorInterface {
  public:
    using rebind_allocator_float = lib::rebind_allocator_t<float, Impl>;
    using rebind_allocator_float16 = lib::rebind_allocator_t<Float16, Impl>;

    // pass by value due to clone()
    explicit AllocatorImpl(Impl impl)
        : AllocatorInterface{}
        , impl_{std::move(impl)} {}

    void* allocate(size_t n) override { return static_cast<void*>(impl_.allocate(n)); }

    void deallocate(void* ptr, size_t n) override {
        impl_.deallocate(static_cast<typename Impl::value_type*>(ptr), n);
    }

    AllocatorImpl<Impl>* clone() const override { return new AllocatorImpl(impl_); }

    AllocatorImpl<rebind_allocator_float>* rebind_float() const override {
        return new AllocatorImpl<rebind_allocator_float>(rebind_allocator_float{impl_});
    }

    AllocatorImpl<rebind_allocator_float16>* rebind_float16() const override {
        return new AllocatorImpl<rebind_allocator_float16>(rebind_allocator_float16{impl_});
    }

  private:
    Impl impl_;
};

template <typename T> class AllocatorHandle {
  public:
    using value_type = T;

    template <detail::Allocator Impl>
    explicit AllocatorHandle(Impl&& impl)
        requires(!std::is_same_v<Impl, AllocatorHandle>) &&
                std::is_rvalue_reference_v<Impl&&> &&
                std::is_same_v<typename Impl::value_type, T>
        : impl_{new AllocatorImpl(std::move(impl))} {}

    AllocatorHandle() {}
    AllocatorHandle(const AllocatorHandle& other)
        : impl_{other.impl_->clone()} {}
    AllocatorHandle(AllocatorHandle&&) = default;
    AllocatorHandle& operator=(const AllocatorHandle& other) {
        impl_.reset(other.impl_->clone());
        return *this;
    }
    AllocatorHandle& operator=(AllocatorHandle&&) = default;
    ~AllocatorHandle() = default;

    // Enable rebinding of allocators.
    template <typename U> friend class AllocatorHandle;

    template <typename U>
    AllocatorHandle(const AllocatorHandle<U>& other)
        requires std::is_same_v<T, float> && (!std::is_same_v<U, T>)
        : impl_{other.impl_->rebind_float()} {}
    template <typename U>
    AllocatorHandle(const AllocatorHandle<U>& other)
        requires std::is_same_v<T, Float16> && (!std::is_same_v<U, T>)
        : impl_{other.impl_->rebind_float16()} {}

    template <typename U>
    AllocatorHandle& operator=(const AllocatorHandle<U>& other)
        requires std::is_same_v<T, float> && (!std::is_same_v<U, T>)
    {
        impl_.reset(other.impl_->rebind_float());
        return *this;
    }
    template <typename U>
    AllocatorHandle& operator=(const AllocatorHandle<U>& other)
        requires std::is_same_v<T, Float16> && (!std::is_same_v<U, T>)
    {
        impl_.reset(other.impl_->rebind_float16());
        return *this;
    }

    T* allocate(size_t n) {
        if (impl_.get() == nullptr) {
            throw ANNEXCEPTION("Empty allocator handle!");
        }
        return static_cast<T*>(impl_->allocate(n));
    }

    void deallocate(T* ptr, size_t n) {
        if (impl_.get() == nullptr) {
            throw ANNEXCEPTION("Empty allocator handle!");
        }
        impl_->deallocate(static_cast<void*>(ptr), n);
    }

  private:
    std::unique_ptr<AllocatorInterface> impl_;
};

///
/// @class allocator_handle
///
/// AllocatorHandle
/// ================
/// `AllocatorHandle` is designed to support a custom allocator that works with the
/// shared library. It offers an interface that enables the
/// implementation of a custom allocator not included in the shared library. Which custom
/// allocator to use is determined at runtime.
///

///
/// @brief Creates an `AllocatorHandle` from a given allocator.
///
/// @tparam Alloc The type of the allocator, which must satisfy the `Allocator` concept.
//
/// @param alloc The allocator to be wrapped in an `AllocatorHandle`.
/// @return An `AllocatorHandle` that manages the given allocator.
///
/// @copydoc allocator_handle
///
/// @sa make_blocked_allocator_handle
///
template <detail::Allocator Alloc>
AllocatorHandle<typename Alloc::value_type> make_allocator_handle(Alloc alloc) {
    return AllocatorHandle<typename Alloc::value_type>{std::move(alloc)};
}

} // namespace svs
