/*
 * Copyright 2026 Intel Corporation
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

#include "svs/core/allocator.h"
#include "svs/lib/exception.h"
#include "svs/lib/memory.h"

#include "fmt/core.h"
#include "tsl/robin_map.h"

#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <type_traits>

namespace svs {

namespace detail {

///
/// @brief Manager for file-backed memory mapped allocations.
///
/// Tracks active memory-mapped allocations by keeping MMapPtr objects alive
/// in a process-wide registry. All members are static; the class is non-
/// instantiable and acts as a namespaced collection of helper functions over
/// the shared registry. Thread-safe for concurrent allocations.
///
class MMapAllocationManager {
  public:
    MMapAllocationManager() = delete;
    ~MMapAllocationManager() = delete;
    MMapAllocationManager(const MMapAllocationManager&) = delete;
    MMapAllocationManager& operator=(const MMapAllocationManager&) = delete;

    ///
    /// @brief Allocate memory mapped to a freshly created (or extended) file.
    ///
    /// @param bytes Number of bytes to allocate
    /// @param file_path Path to the file for backing storage
    /// @return Pointer to the allocated memory
    ///
    [[nodiscard]] static void*
    allocate(size_t bytes, const std::filesystem::path& file_path) {
        MemoryMapper mapper{MemoryMapper::ReadWrite, MemoryMapper::MayCreate};
        auto mmap_ptr = mapper.mmap(file_path, lib::Bytes(bytes));

        void* ptr = mmap_ptr.data();

        // Store the MMapPtr to keep the mapping alive
        {
            std::lock_guard lock{mutex_};
            allocations_.insert({ptr, std::move(mmap_ptr)});
        }

        return ptr;
    }

    ///
    /// @brief Map an existing file read-only, returning a pointer offset into the mapping.
    ///
    /// This is used for zero-copy loading: the returned pointer points to
    /// `base + offset` within the mmap'd region. The underlying mapping covers
    /// the entire file so that munmap and madvise operate on the full range.
    ///
    /// @param data_bytes Number of bytes of data expected after the offset.
    /// @param file_path Path to an existing file.
    /// @param offset Byte offset into the file where data starts (e.g., header size).
    /// @return Pointer to data at `base + offset`.
    ///
    [[nodiscard]] static void* map_existing_at_offset(
        size_t data_bytes, const std::filesystem::path& file_path, size_t offset
    ) {
        auto file_size = std::filesystem::file_size(file_path);
        if (file_size < offset + data_bytes) {
            throw ANNEXCEPTION(
                "File {} is {} bytes, need at least {} (offset={} + data={})",
                file_path,
                file_size,
                offset + data_bytes,
                offset,
                data_bytes
            );
        }

        MemoryMapper mapper{MemoryMapper::ReadOnly, MemoryMapper::MustUseExisting};
        auto mmap_ptr = mapper.mmap(file_path, lib::Bytes(file_size));

        void* data_ptr = static_cast<std::byte*>(mmap_ptr.data()) + offset;

        {
            std::lock_guard lock{mutex_};
            allocations_.insert({data_ptr, std::move(mmap_ptr)});
        }

        return data_ptr;
    }

    ///
    /// @brief Deallocate a memory-mapped allocation.
    ///
    /// Removes the MMapPtr, which triggers munmap in its destructor.
    ///
    /// @param ptr Pointer previously returned by allocate() or
    ///        map_existing_at_offset().
    ///
    static void deallocate(void* ptr) {
        std::lock_guard lock{mutex_};
        auto itr = allocations_.find(ptr);
        if (itr == allocations_.end()) {
            throw ANNEXCEPTION("Could not find memory-mapped allocation to deallocate!");
        }

        // Erasing will destroy the MMapPtr, which calls munmap
        allocations_.erase(itr);
    }

    ///
    /// @brief Get count of current allocations (for debugging/testing)
    ///
    static size_t allocation_count() {
        std::lock_guard lock{mutex_};
        return allocations_.size();
    }

    ///
    /// @brief Evict all mmap'd pages from memory using madvise(MADV_DONTNEED).
    ///
    /// This tells the kernel to discard the pages backing all active mmap allocations.
    /// The pages will be re-faulted from the backing files on next access.
    /// Useful for benchmarking to simulate truly cold cache access.
    ///
    static void evict_pages() {
#ifdef __linux__
        std::lock_guard lock{mutex_};
        for (auto& [ptr, mmap_ptr] : allocations_) {
            void* base = const_cast<void*>(mmap_ptr.base());
            size_t size = mmap_ptr.size();
            if (base != nullptr && size > 0) {
                (void)madvise(base, size, MADV_DONTNEED);
            }
        }
#endif
    }

    ///
    /// @brief Evict the pages backing a single allocation.
    ///
    /// Calls madvise(MADV_DONTNEED) on the full underlying mapping for the
    /// allocation registered at `ptr`. Useful for selectively dropping just-
    /// faulted pages after a one-shot read of an existing file.
    ///
    static void evict_pages_for(void* ptr) {
#ifdef __linux__
        std::lock_guard lock{mutex_};
        auto itr = allocations_.find(ptr);
        if (itr == allocations_.end()) {
            return;
        }
        void* base = const_cast<void*>(itr->second.base());
        size_t size = itr->second.size();
        if (base != nullptr && size > 0) {
            (void)madvise(base, size, MADV_DONTNEED);
        }
#else
        (void)ptr;
#endif
    }

  private:
    inline static std::mutex mutex_{};
    inline static tsl::robin_map<void*, MMapPtr<void>> allocations_{};
};

} // namespace detail

///
/// @brief Access pattern hint for memory-mapped allocations
///
enum class MMapAccessHint {
    Normal,     ///< Default access pattern
    Sequential, ///< Data will be accessed sequentially
    Random      ///< Data will be accessed randomly
};

namespace detail {

///
/// @brief Apply a madvise() access-pattern hint to an mmap'd region.
///
/// No-op on non-Linux platforms or for null/empty regions. madvise() is a
/// hint, so any error is ignored.
///
inline void apply_mmap_access_hint(void* ptr, size_t bytes, MMapAccessHint hint) {
#ifdef __linux__
    if (ptr == nullptr || bytes == 0) {
        return;
    }
    int advice = MADV_NORMAL;
    switch (hint) {
        case MMapAccessHint::Normal:
            advice = MADV_NORMAL;
            break;
        case MMapAccessHint::Sequential:
            advice = MADV_SEQUENTIAL;
            break;
        case MMapAccessHint::Random:
            advice = MADV_RANDOM;
            break;
    }
    (void)madvise(ptr, bytes, advice);
#else
    (void)ptr;
    (void)bytes;
    (void)hint;
#endif
}

} // namespace detail

///
/// @brief File-backed, writable memory-mapped allocator.
///
/// Each allocate() call creates a fresh temp file under @c base_path_ and
/// returns a writable mmap of that file. Intended for storing data that is
/// produced at runtime (e.g. an index's secondary, full-dimension dataset)
/// in file-backed pages instead of anonymous RAM.
///
/// For zero-copy loading from a pre-existing file, use
/// @ref MMapFileViewAllocator instead.
///
/// @tparam T The value type for the allocator. Must be trivially default-
///         constructible: construction is a no-op (storage is either kernel-
///         zeroed for new files, or already-valid bytes for existing files
///         via the read-only sibling allocator).
///
template <typename T> class MMapAllocator {
  private:
    std::filesystem::path base_path_;
    size_t allocation_counter_ = 0;
    MMapAccessHint access_hint_ = MMapAccessHint::Normal;

  public:
    // C++ allocator type aliases
    using value_type = T;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal =
        std::false_type; // Allocators with different paths are different

    ///
    /// @brief Construct a new MMapAllocator
    ///
    /// @param base_path Directory path for storing memory-mapped files.
    ///        If empty, will use /tmp with generated names.
    /// @param access_hint Hint about how the data will be accessed
    ///
    explicit MMapAllocator(
        std::filesystem::path base_path = {},
        MMapAccessHint access_hint = MMapAccessHint::Normal
    )
        : base_path_{std::move(base_path)}
        , access_hint_{access_hint} {
        if (!base_path_.empty() && !std::filesystem::exists(base_path_)) {
            std::filesystem::create_directories(base_path_);
        }
    }

    // Enable rebinding of allocators
    template <typename U> friend class MMapAllocator;

    template <typename U>
    MMapAllocator(const MMapAllocator<U>& other)
        : base_path_{other.base_path_}
        , allocation_counter_{other.allocation_counter_}
        , access_hint_{other.access_hint_} {}

    ///
    /// @brief Compare allocators
    ///
    /// Two allocators are equal if they use the same base path and access hint
    ///
    template <typename U> bool operator==(const MMapAllocator<U>& other) const {
        return base_path_ == other.base_path_ && access_hint_ == other.access_hint_;
    }

    ///
    /// @brief Allocate a writable file-backed mmap of @c n elements.
    ///
    /// Creates a fresh temp file under base_path_ sized to @c sizeof(T) * n,
    /// maps it ReadWrite, and applies the configured madvise() hint.
    ///
    /// @param n Number of elements to allocate
    /// @return Pointer to allocated memory
    ///
    [[nodiscard]] T* allocate(size_t n) {
        size_t bytes = sizeof(T) * n;
        auto file_path = generate_file_path(bytes);
        void* ptr = detail::MMapAllocationManager::allocate(bytes, file_path);
        detail::apply_mmap_access_hint(ptr, bytes, access_hint_);
        return static_cast<T*>(ptr);
    }

    ///
    /// @brief Deallocate memory previously returned by allocate().
    ///
    void deallocate(void* ptr, size_t SVS_UNUSED(n)) {
        detail::MMapAllocationManager::deallocate(ptr);
    }

    ///
    /// @brief Construct an object (no-op).
    ///
    /// Storage returned by allocate() is a freshly created file-backed mapping;
    /// its bytes are kernel-zeroed and there is no further work to do for the
    /// trivially-default-constructible element types this allocator supports.
    /// Suppressing the default `std::allocator_traits::construct` placement-new
    /// keeps this allocator's behaviour symmetric with @ref MMapFileViewAllocator,
    /// whose backing mapping is read-only and where placement-new would be
    /// invalid.
    ///
    static_assert(
        std::is_trivially_default_constructible_v<T>,
        "MMapAllocator only supports trivially default-constructible types."
    );
    void construct(T* /*ptr*/) noexcept {}

    /// @brief Get the base path for allocations.
    const std::filesystem::path& get_base_path() const { return base_path_; }

    /// @brief Get the access hint.
    MMapAccessHint get_access_hint() const { return access_hint_; }

    /// @brief Set the access hint for future allocations.
    void set_access_hint(MMapAccessHint hint) { access_hint_ = hint; }

    ///
    /// @brief Evict all mmap'd pages from memory.
    ///
    /// Calls madvise(MADV_DONTNEED) on all active mmap allocations
    /// (including those from MMapFileViewAllocator), forcing pages to be
    /// re-faulted from disk on next access.
    ///
    static void evict_pages() { detail::MMapAllocationManager::evict_pages(); }

  private:
    /// @brief Generate a unique file path for an allocation.
    std::filesystem::path generate_file_path(size_t bytes) {
        auto filename = fmt::format(
            "mmap_alloc_{}_{}_{}.dat",
            std::this_thread::get_id(),
            allocation_counter_++,
            bytes
        );

        if (base_path_.empty()) {
            return std::filesystem::temp_directory_path() / filename;
        }
        return base_path_ / filename;
    }
};

///
/// @brief Read-only memory-mapped view over an existing on-disk file.
///
/// This allocator is the zero-copy counterpart to @ref MMapAllocator. Each
/// instance is bound at construction time to a specific @c (path, offset)
/// pair; calls to allocate() return a read-only mapping into that file at
/// @c base + offset. The Allocator concept is supported so that data
/// containers can be loaded directly over a file backing store, but the
/// returned memory must never be written to: the underlying mapping is
/// PROT_READ.
///
/// One MMapFileViewAllocator instance backs at most one allocation. After
/// allocate() succeeds, subsequent allocate() calls on the same instance
/// throw, so each container component (primary/secondary/...) requires its
/// own instance.
///
/// Two construction modes are supported:
///   * **Bound**: caller knows the file path and offset up front.
///   * **Config-only**: caller knows just the access hint and eviction policy
///     (e.g. for the loader API where the actual file path is resolved later
///     by UUID lookup). Use @ref with_file() to produce a bound copy before
///     allocate() is called.
///
/// @tparam T The value type. Must be trivially default-constructible:
///         construction is a no-op because the file bytes already represent
///         valid T objects, and the mapping is not writable.
///
template <typename T> class MMapFileViewAllocator {
  private:
    std::filesystem::path file_path_{};
    size_t offset_ = 0;
    MMapAccessHint access_hint_ = MMapAccessHint::Normal;
    bool evict_on_load_ = true;
    bool consumed_ = false;

  public:
    // C++ allocator type aliases
    using value_type = T;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    ///
    /// @brief Construct an unbound (config-only) file-view allocator.
    ///
    /// allocate() will throw until the allocator is bound to a file via
    /// @ref with_file().
    ///
    /// @param access_hint madvise() access pattern hint applied after mapping.
    /// @param evict_on_load If true, immediately advise the kernel to drop
    ///        any pages prefaulted by the mapping (SSD-residency simulation).
    ///
    explicit MMapFileViewAllocator(
        MMapAccessHint access_hint = MMapAccessHint::Normal, bool evict_on_load = true
    )
        : access_hint_{access_hint}
        , evict_on_load_{evict_on_load} {}

    ///
    /// @brief Construct a file-view allocator bound to a specific file.
    ///
    /// @param file_path Path to the existing file to map.
    /// @param offset Byte offset into the file where the view starts (e.g.
    ///        to skip a file header). Defaults to 0.
    /// @param access_hint madvise() access pattern hint applied after mapping.
    /// @param evict_on_load If true, immediately advise the kernel to drop
    ///        any pages prefaulted by the mapping, so they fault from disk
    ///        on first access (SSD-residency simulation). If false, pages
    ///        stay resident after mapping (RAM-residency).
    ///
    MMapFileViewAllocator(
        std::filesystem::path file_path,
        size_t offset,
        MMapAccessHint access_hint = MMapAccessHint::Normal,
        bool evict_on_load = true
    )
        : file_path_{std::move(file_path)}
        , offset_{offset}
        , access_hint_{access_hint}
        , evict_on_load_{evict_on_load} {}

    ///
    /// @brief Return a copy of this allocator bound to @c (path, offset).
    ///
    /// Used by loader code that knows access policy at user time but only
    /// learns the on-disk file path after parsing the saved table.
    ///
    [[nodiscard]] MMapFileViewAllocator
    with_file(std::filesystem::path path, size_t offset = 0) const {
        MMapFileViewAllocator copy{access_hint_, evict_on_load_};
        copy.file_path_ = std::move(path);
        copy.offset_ = offset;
        return copy;
    }

    // Enable rebinding of allocators
    template <typename U> friend class MMapFileViewAllocator;

    template <typename U>
    MMapFileViewAllocator(const MMapFileViewAllocator<U>& other)
        : file_path_{other.file_path_}
        , offset_{other.offset_}
        , access_hint_{other.access_hint_}
        , evict_on_load_{other.evict_on_load_}
        , consumed_{other.consumed_} {}

    ///
    /// @brief Compare allocators.
    ///
    /// Two allocators are equal if they target the same file/offset and use
    /// the same access hint and eviction policy.
    ///
    template <typename U> bool operator==(const MMapFileViewAllocator<U>& other) const {
        return file_path_ == other.file_path_ && offset_ == other.offset_ &&
               access_hint_ == other.access_hint_ && evict_on_load_ == other.evict_on_load_;
    }

    ///
    /// @brief Map the bound file read-only and return a pointer at offset.
    ///
    /// The returned pointer must be treated as read-only memory: writing
    /// through it will fault. May only be called once per instance.
    ///
    /// @param n Number of elements expected after the offset.
    /// @return Pointer to data at @c base + offset.
    ///
    [[nodiscard]] T* allocate(size_t n) {
        if (file_path_.empty()) {
            throw ANNEXCEPTION(
                "MMapFileViewAllocator::allocate() called on an unbound allocator; "
                "use with_file(path, offset) before allocating."
            );
        }
        if (consumed_) {
            throw ANNEXCEPTION(
                "MMapFileViewAllocator::allocate() called more than once on the same "
                "instance for file {}",
                file_path_
            );
        }
        consumed_ = true;

        size_t bytes = sizeof(T) * n;
        void* ptr = detail::MMapAllocationManager::map_existing_at_offset(
            bytes, file_path_, offset_
        );

        if (evict_on_load_) {
            // Drop the prefaulted pages so first access faults from disk.
            detail::MMapAllocationManager::evict_pages_for(ptr);
        }

        detail::apply_mmap_access_hint(ptr, bytes, access_hint_);
        return static_cast<T*>(ptr);
    }

    ///
    /// @brief Deallocate memory previously returned by allocate().
    ///
    void deallocate(void* ptr, size_t SVS_UNUSED(n)) {
        detail::MMapAllocationManager::deallocate(ptr);
    }

    ///
    /// @brief Construct an object (no-op).
    ///
    /// The backing mapping is PROT_READ; placement-new would attempt to write
    /// to it. Bytes on disk already represent valid T objects, so no
    /// construction is required. Restricting T to trivially default-
    /// constructible types makes this safe.
    ///
    static_assert(
        std::is_trivially_default_constructible_v<T>,
        "MMapFileViewAllocator only supports trivially default-constructible types; "
        "non-trivial construction would write to a read-only mapping."
    );
    void construct(T* /*ptr*/) noexcept {}

    /// @brief Get the bound file path.
    const std::filesystem::path& get_file_path() const { return file_path_; }

    /// @brief Get the byte offset into the bound file.
    size_t get_offset() const { return offset_; }

    /// @brief Get the access hint.
    MMapAccessHint get_access_hint() const { return access_hint_; }

    /// @brief Whether prefaulted pages are evicted after mapping.
    bool evict_on_load() const { return evict_on_load_; }
};

///
/// @brief Type trait: true iff @c A is a (cv-qualified) MMapFileViewAllocator.
///
/// Used by zero-copy load specializations to detect the read-only file-view
/// allocator and dispatch to the mmap-existing-file path.
///
template <typename A> struct is_mmap_file_view_allocator : std::false_type {};
template <typename T>
struct is_mmap_file_view_allocator<MMapFileViewAllocator<T>> : std::true_type {};
template <typename A>
inline constexpr bool is_mmap_file_view_allocator_v =
    is_mmap_file_view_allocator<std::remove_cv_t<A>>::value;

} // namespace svs
