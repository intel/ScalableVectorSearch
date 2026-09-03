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

#include <svs/core/allocator.h>
#include <svs/core/data/simple.h>
#include <svs/lib/float16.h>
#include <svs/lib/memory.h>
#include <svs/lib/meta.h>

namespace svs {
namespace c_runtime {

template <typename T, bool UseBlocked, typename Allocator = svs::lib::Allocator<T>>
using MaybeBlockedAlloc =
    std::conditional_t<UseBlocked, svs::data::Blocked<Allocator>, Allocator>;

template <typename T> class CustomAllocator : public svs::AllocatorInterface {
  public:
    using value_type = T;

    static void validate(const svs_allocator_i allocator) {
        if (allocator == nullptr) {
            throw std::invalid_argument("Custom allocator pointer cannot be null.");
        }
        if (allocator->ops == nullptr) {
            throw std::invalid_argument("Custom allocator interface is not initialized.");
        }
        if (allocator->ops->allocate == nullptr || allocator->ops->deallocate == nullptr) {
            throw std::invalid_argument(
                "Custom allocator interface has null function pointers."
            );
        }
    }

    CustomAllocator(const svs_allocator_ops_t& ops, void* self)
        : ops_(ops)
        , self_(self) {}

    void* allocate(size_t n) override {
        svs_error_desc err{SVS_ERROR_UNKNOWN, "Unknown error in custom allocator allocate"};

        auto result = ops_.allocate(self_, n * sizeof(T), alignof(T), &err);
        if (result == nullptr) {
            throw std::runtime_error(
                "Custom allocator failed to allocate memory: (" + std::to_string(err.code) +
                ") " + err.message
            );
        }
        return result;
    }

    void deallocate(void* p, size_t n) override {
        ops_.deallocate(self_, p, n * sizeof(T), alignof(T));
    }

    AllocatorInterface* clone() const override {
        return new CustomAllocator<T>(ops_, self_);
    }

    AllocatorInterface* rebind_to(DataType type) const override {
        return svs::lib::match(
            AllocatorInterface::rebind_types{},
            type,
            [this]<typename Tag>(svs::lib::Type<Tag>) -> AllocatorInterface* {
                return new CustomAllocator<Tag>(ops_, self_);
            }
        );
    }

  private:
    svs_allocator_ops_t ops_;
    void* self_;
};

template <typename T = std::byte>
AllocatorHandle<T> make_custom_allocator_handle(const svs_allocator_i allocator) {
    CustomAllocator<T>::validate(allocator);
    return AllocatorHandle<T>{
        std::make_unique<CustomAllocator<T>>(*allocator->ops, allocator->self)};
}
} // namespace c_runtime
} // namespace svs
