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

#include "svs/c/svs_c.h"

#include "error.hpp"

#include <svs/concurrent/blocked_data.h>
#include <svs/core/allocator.h>
#include <svs/core/data/simple.h>
#include <svs/lib/float16.h>
#include <svs/lib/memory.h>
#include <svs/lib/meta.h>

namespace svs {
namespace c_runtime {

/// Storage layout of a dataset or graph.
enum class BlockKind {
    None,     ///< Single contiguous allocation (static index).
    Blocked,  ///< data::Blocked (regular dynamic index).
    Segmented ///< concurrent::SegmentedBlocked (concurrent dynamic index).
};

namespace detail {
template <BlockKind Kind, typename Allocator> struct BlockedAllocSelector {
    using type = Allocator;
};
template <typename Allocator> struct BlockedAllocSelector<BlockKind::Blocked, Allocator> {
    using type = svs::data::Blocked<Allocator>;
};
template <typename Allocator> struct BlockedAllocSelector<BlockKind::Segmented, Allocator> {
    using type = svs::index::vamana::concurrent::SegmentedBlocked<Allocator>;
};
} // namespace detail

template <typename T, BlockKind Kind, typename Allocator = svs::lib::Allocator<T>>
using MaybeBlockedAlloc = typename detail::BlockedAllocSelector<Kind, Allocator>::type;

// Accepts a BlockKind, or a bool meaning Blocked (true) / None (false).
template <auto Kind>
inline constexpr BlockKind block_kind_v = [] {
    if constexpr (std::is_same_v<decltype(Kind), bool>) {
        return Kind ? BlockKind::Blocked : BlockKind::None;
    } else {
        return Kind;
    }
}();

class AllocatorBuilder {
    template <typename T> class CustomAllocator {
      public:
        using value_type = T;

        static void validate(const svs_allocator_i allocator) {
            if (allocator == nullptr) {
                throw std::invalid_argument("Custom allocator pointer cannot be null.");
            }
            if (allocator->ops == nullptr) {
                throw std::invalid_argument("Custom allocator interface is not initialized."
                );
            }
            if (allocator->ops->version > svs_get_version()) {
                throw std::invalid_argument(
                    "Custom allocator interface version is not supported."
                );
            }
            if (allocator->ops->struct_size < sizeof(svs_allocator_ops_t)) {
                throw std::invalid_argument(
                    "Incompatible custom allocator interface struct size."
                );
            }
            if (allocator->ops->allocate == nullptr ||
                allocator->ops->deallocate == nullptr) {
                throw std::invalid_argument(
                    "Custom allocator interface has null function pointers."
                );
            }
        }

        CustomAllocator(const svs_allocator_ops_t& ops, void* self)
            : ops_(ops)
            , self_(self) {}

        template <typename U> friend class CustomAllocator;

        template <typename U>
        CustomAllocator(const CustomAllocator<U>& other)
            : ops_(other.ops_)
            , self_(other.self_) {}

        template <typename U> CustomAllocator& operator=(const CustomAllocator<U>& other) {
            ops_ = other.ops_;
            self_ = other.self_;
            return *this;
        }

        value_type* allocate(size_t n) {
            svs_error_desc err{
                SVS_ERROR_UNKNOWN, "Unknown error in custom allocator allocate"};

            auto result = ops_.allocate(self_, n * sizeof(T), alignof(T), &err);
            if (result == nullptr) {
                throw svs::c_runtime::out_of_memory(
                    "Custom allocator failed to allocate memory: (" +
                    std::to_string(err.code) + ") " + err.message
                );
            }
            return static_cast<value_type*>(result);
        }

        void deallocate(value_type* p, size_t n) {
            ops_.deallocate(self_, p, n * sizeof(T), alignof(T));
        }

      private:
        svs_allocator_ops_t ops_;
        void* self_;
    };

    svs_allocator_kind kind_;
    // Owned copy of the user's allocator vtable; `self_` is referenced only.
    svs_allocator_ops_t user_ops_{};
    void* user_self_ = nullptr;

  public:
    AllocatorBuilder(svs_allocator_kind kind = SVS_ALLOCATOR_KIND_DEFAULT)
        : kind_(kind) {
        switch (kind_) {
            case SVS_ALLOCATOR_KIND_DEFAULT:
            case SVS_ALLOCATOR_KIND_SIMPLE:
            case SVS_ALLOCATOR_KIND_HUGE_PAGE:
                break;
            case SVS_ALLOCATOR_KIND_CUSTOM:
                throw std::invalid_argument(
                    "SVS_ALLOCATOR_KIND_CUSTOM requires a custom allocator interface."
                );
            default:
                throw std::invalid_argument("Unknown allocator kind.");
        }
    }

    AllocatorBuilder(svs_allocator_i allocator)
        : kind_(SVS_ALLOCATOR_KIND_CUSTOM) {
        CustomAllocator<std::byte>::validate(allocator);
        // Copy the 'vtable' since the user may free the original.
        user_ops_ = *allocator->ops;
        user_self_ = allocator->self;
    }

    svs_allocator_kind kind() const { return kind_; }

    template <typename T = std::byte> svs::AllocatorHandle<T> build() const {
        // For now, default allocator is simple allocator - can be changed in the future.
        switch (kind_) {
            case SVS_ALLOCATOR_KIND_DEFAULT:
            case SVS_ALLOCATOR_KIND_SIMPLE:
                return svs::make_allocator_handle(svs::lib::Allocator<T>{});
            case SVS_ALLOCATOR_KIND_HUGE_PAGE:
                return svs::make_allocator_handle(svs::HugepageAllocator<T>{});
            case SVS_ALLOCATOR_KIND_CUSTOM:
                return svs::make_allocator_handle(CustomAllocator<T>{user_ops_, user_self_}
                );
            default:
                throw std::invalid_argument("Unknown allocator kind.");
        }
    }

    template <typename T = uint32_t> svs::AllocatorHandle<T> build_for_graph() const {
        // Default graph allocator is hugepage allocator - can be changed in the future.
        switch (kind_) {
            case SVS_ALLOCATOR_KIND_SIMPLE:
                return svs::make_allocator_handle(svs::lib::Allocator<T>{});
            case SVS_ALLOCATOR_KIND_DEFAULT:
            case SVS_ALLOCATOR_KIND_HUGE_PAGE:
                return svs::make_allocator_handle(svs::HugepageAllocator<T>{});
            case SVS_ALLOCATOR_KIND_CUSTOM:
                return svs::make_allocator_handle(CustomAllocator<T>{user_ops_, user_self_}
                );
            default:
                throw std::invalid_argument("Unknown allocator kind.");
        }
    }
};
} // namespace c_runtime
} // namespace svs
