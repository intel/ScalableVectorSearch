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

namespace svs {

namespace detail {

///
/// @brief Manager for file-backed memory mapped allocations
///
/// Tracks memory-mapped allocations by keeping MMapPtr objects alive.
/// Thread-safe for concurrent allocations.
///
class MMapAllocationManager {
  public:
    MMapAllocationManager() = default;

    ///
    /// @brief Allocate memory mapped to a file
    ///
    /// @param bytes Number of bytes to allocate
    /// @param file_path Path to the file for backing storage
    /// @return Pointer to the allocated memory
    ///
    [[nodiscard]] void* allocate(size_t bytes, const std::filesystem::path& file_path) {
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
    [[nodiscard]] void* map_existing_at_offset(
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
    /// @brief Deallocate memory mapped allocation
    ///
    /// Removes the MMapPtr, which triggers munmap in its destructor
    ///
    /// @param ptr Pointer to deallocate
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

  private:
    inline static std::mutex mutex_{};
    inline static tsl::robin_map<void*, MMapPtr<void>> allocations_{};
};

} // namespace detail

///
/// @brief File-backed memory-mapped allocator for LeanVec secondary data
///
/// This allocator uses memory-mapped files to store data on SSD rather than RAM.
/// It's particularly useful for the secondary (full-dimension) dataset in LeanVec,
/// which is accessed less frequently during search.
///
/// @tparam T The value type for the allocator
///
///
/// @brief Access pattern hint for memory-mapped allocations
///
enum class MMapAccessHint {
    Normal,     ///< Default access pattern
    Sequential, ///< Data will be accessed sequentially
    Random      ///< Data will be accessed randomly
};

template <typename T> class MMapAllocator {
  private:
    std::filesystem::path base_path_;
    size_t allocation_counter_ = 0;
    MMapAccessHint access_hint_ = MMapAccessHint::Normal;
    std::optional<std::filesystem::path> override_file_{};
    size_t override_offset_ = 0;

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
        , access_hint_{other.access_hint_}
        , override_file_{other.override_file_}
        , override_offset_{other.override_offset_} {}

    ///
    /// @brief Compare allocators
    ///
    /// Two allocators are equal if they use the same base path and access hint
    ///
    template <typename U> bool operator==(const MMapAllocator<U>& other) const {
        return base_path_ == other.base_path_ && access_hint_ == other.access_hint_;
    }

    ///
    /// @brief Configure the next allocate() to map an existing file read-only.
    ///
    /// Instead of creating a new temp file, the next allocate() call will
    /// memory-map the given file at the specified byte offset (e.g., to skip
    /// a file header). This is a one-shot override: consumed by the first
    /// allocate() call after setting it.
    ///
    /// @param path Path to the existing binary file.
    /// @param offset Byte offset into the file where data begins.
    ///
    void use_existing_file(std::filesystem::path path, size_t offset = 0) {
        override_file_ = std::move(path);
        override_offset_ = offset;
    }

    ///
    /// @brief Allocate memory
    ///
    /// Creates a memory-mapped file and returns a pointer to it.
    /// If use_existing_file() was called, maps that file read-only instead.
    /// Applies madvise hints based on the access hint.
    ///
    /// @param n Number of elements to allocate
    /// @return Pointer to allocated memory
    ///
    [[nodiscard]] T* allocate(size_t n) {
        size_t bytes = sizeof(T) * n;
        void* ptr;

        if (override_file_) {
            // Zero-copy path: map the existing file read-only at the given offset.
            auto path = std::move(*override_file_);
            auto offset = override_offset_;
            override_file_.reset();
            override_offset_ = 0;
            ptr =
                detail::MMapAllocationManager{}.map_existing_at_offset(bytes, path, offset);
        } else {
            // Normal path: create a new temp file.
            auto file_path = generate_file_path(bytes);
            ptr = detail::MMapAllocationManager{}.allocate(bytes, file_path);
        }

        // Apply madvise hint if on Linux
        apply_access_hint(ptr, bytes);

        return static_cast<T*>(ptr);
    }

    ///
    /// @brief Deallocate memory
    ///
    /// Unmaps the memory-mapped file and cleans up.
    ///
    /// @param ptr Pointer to deallocate
    /// @param n Number of elements (unused but required by allocator interface)
    ///
    void deallocate(void* ptr, size_t SVS_UNUSED(n)) {
        detail::MMapAllocationManager::deallocate(ptr);
    }

    ///
    /// @brief Construct an object
    ///
    /// Performs default initialization of the object.
    ///
    void construct(T* ptr) { ::new (static_cast<void*>(ptr)) T; }

    ///
    /// @brief Get the base path for allocations
    ///
    const std::filesystem::path& get_base_path() const { return base_path_; }

    ///
    /// @brief Get the access hint
    ///
    MMapAccessHint get_access_hint() const { return access_hint_; }

    ///
    /// @brief Set the access hint for future allocations
    ///
    void set_access_hint(MMapAccessHint hint) { access_hint_ = hint; }

    ///
    /// @brief Evict all mmap'd pages from memory.
    ///
    /// Calls madvise(MADV_DONTNEED) on all active mmap allocations,
    /// forcing pages to be re-faulted from disk on next access.
    ///
    static void evict_pages() { detail::MMapAllocationManager::evict_pages(); }

  private:
    ///
    /// @brief Apply madvise hint based on access pattern
    ///
    void apply_access_hint(void* ptr, size_t bytes) const {
#ifdef __linux__
        if (ptr == nullptr || bytes == 0) {
            return;
        }

        int advice = MADV_NORMAL;
        switch (access_hint_) {
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

        // madvise is a hint, so ignore errors
        (void)madvise(ptr, bytes, advice);
#else
        (void)ptr;
        (void)bytes;
#endif
    }
    ///
    /// @brief Generate a unique file path for an allocation
    ///
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

} // namespace svs
