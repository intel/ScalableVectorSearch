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
/// @file
/// @brief Implements commont large-scale allocators used by the many data structures.
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

#include <array>
#include <fcntl.h>
#include <filesystem>
#include <linux/mman.h>
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

// Forward declarations.
class HugepageAllocator;

///
/// @brief Standard allocators used for large internal data structures.
///
/// Algorithms accepting this variant will use the provided alternative in the variant as
/// the memory provider for the class and use some form of type-erasure to provide a unified
/// return type.
///
using StandardAllocators = std::variant<HugepageAllocator>;

///
/// @ingroup core_allocators_entry
/// @brief Most common large-scale allocator for DRAM.
///
using DRAM = HugepageAllocator;

/////
///// hugepage allocator
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
static constexpr std::array<HugepageX86Parameters, 3> hugepage_x86_options{
    HugepageX86Parameters{1 << 30, MAP_HUGETLB | MAP_HUGE_1GB},
    HugepageX86Parameters{1 << 21, MAP_HUGETLB | MAP_HUGE_2MB},
    HugepageX86Parameters{1 << 12, 0},
};

///
/// @ingroup core_allocators_entry
/// @brief Allocator class to use hugepages to back memory allocations.
///
class HugepageAllocator {
  private:
    bool force_ = false;

  public:
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
        : force_{force} {};

    // TODO: What kind of type restrictions are there actually on `T`?
    template <typename T> MMapPtr<T> allocate_managed(lib::Bytes bytes) const {
        // Try to allocate sing huge pages.
        // First 1 GiB, then 2 MiB.
        void* mmap_ptr_void = MAP_FAILED;
        size_t requested_bytes = value(bytes);
        size_t allocated_bytes = 0;
        for (auto params : hugepage_x86_options) {
            // Forcing logic.
            // If the constructor of the allocator really want's huge pages, don't
            // fallback to 4K pages.
            if (force_ && params == hugepage_x86_options.back()) {
                break;
            }

            // Unpack parameters.
            auto pagesize = params.pagesize;
            auto mmap_flags = params.mmap_flags;

            // Round up to the page size.
            allocated_bytes = lib::round_up_to_multiple_of(requested_bytes, pagesize);
            mmap_ptr_void = mmap(
                nullptr,
                allocated_bytes,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE | mmap_flags,
                -1,
                0
            );

            if (mmap_ptr_void != MAP_FAILED) {
                break;
            }
        }

        if (mmap_ptr_void == MAP_FAILED) {
            throw ANNEXCEPTION(
                "Hugepage memory map allocation of size ", requested_bytes, " failed!"
            );
        }

        return MMapPtr<T>{mmap_ptr_void, allocated_bytes};
    }
};

namespace lib::memory {
// Allocator specifier
template <> struct IsAllocator<HugepageAllocator> {
    static constexpr bool value = true;
};

// Storage Traits
template <typename T> struct PointerTraits<MMapPtr<T>> : PointerTraitsBase {
    using value_type = T;
    using allocator = HugepageAllocator;
    static constexpr bool disable_implicit_copy = true;
    // Data Access
    static T* access(MMapPtr<T>& ptr) { return ptr.data(); }
    static const T* access(const MMapPtr<T>& ptr) { return ptr.data(); }
};
} // namespace lib::memory

/////
///// Memory Mapper
/////

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
                "Memory Map Allocator is configured to create a file (",
                filename,
                ") that already exists!"
            );
        } else if (policy_ == MustUseExisting && !exists) {
            throw ANNEXCEPTION(
                "Memory Map Allocator is configured to use an existing file (",
                filename,
                ") that does not exist!"
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
                throw ANNEXCEPTION("Could not create file ", filename, "!");
            }
            auto result = lseek(fd, lib::narrow<int64_t>(value(bytes) - 1), SEEK_SET);
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
            if (filesize < value(bytes)) {
                throw ANNEXCEPTION(
                    "The size of file (",
                    filename,
                    ") to memory map is ",
                    filesize,
                    " which is less than the number of bytes (",
                    value(bytes),
                    " requested!"
                );
            }
            // By this point, we found our file and it looks large enough.
            // Time to perform the memory mapping and set our flags appropriately.
            fd = open(filename.c_str(), open_permissions(permission_));
            if (fd == -1) {
                throw ANNEXCEPTION("Could not open file ", filename, "!");
            }
        }
        lseek(fd, 0, SEEK_SET);
        void* base = ::mmap(
            nullptr,
            value(bytes),
            mmap_permissions(permission_),
            MAP_NORESERVE       // Don't reserve space in DRAM for this until used
                | MAP_SHARED    // Accessible from all processes
                | MAP_POPULATE, // Populate page table entries in the DRAM
            fd,
            0
        );
        close(fd);

        if (base == nullptr || base == MAP_FAILED) {
            throw ANNEXCEPTION("Memory Map Failed!");
        }
        return MMapPtr<void>(base, value(bytes));
    }
};

/////
///// Parse a string into an allocator type.
/////

inline StandardAllocators select_memory_style(const std::string& memory_style_str) {
    if (memory_style_str == "dram") {
        return HugepageAllocator{};
    }
    throw ANNEXCEPTION(
        "Unsupported memory option (", memory_style_str, "). Use dram/memmap."
    );
}

} // namespace svs
