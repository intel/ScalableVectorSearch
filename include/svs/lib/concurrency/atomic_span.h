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

#include <atomic>
#include <cstddef>
#include <iterator>

namespace svs {

///
/// @brief A non-owning, zero-copy view over a contiguous range of ``T`` that performs
/// atomic loads on every element access.
///
/// Each dereference uses ``std::atomic_ref<const T>::load(std::memory_order_relaxed)``.
/// On x86, this compiles to a plain MOV instruction — identical to non-atomic access.
///
/// This type is designed to be used as a drop-in replacement for ``std::span<const T>``
/// when concurrent reads and writes are possible, ensuring no undefined behavior
/// while maintaining zero-copy semantics.
///
template <typename T> class AtomicSpan {
  public:
    using value_type = std::remove_const_t<T>;

    class iterator {
      public:
        using value_type = AtomicSpan::value_type;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

        explicit iterator(const T* p)
            : ptr_(p) {}

        value_type operator*() const {
            return std::atomic_ref<const value_type>(*ptr_).load(std::memory_order_relaxed);
        }

        iterator& operator++() {
            ++ptr_;
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++ptr_;
            return tmp;
        }

        bool operator==(const iterator& other) const { return ptr_ == other.ptr_; }
        bool operator!=(const iterator& other) const { return ptr_ != other.ptr_; }

      private:
        const T* ptr_;
    };

    AtomicSpan(const T* data, size_t size)
        : data_(data)
        , size_(size) {}

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    const T* data() const { return data_; }

    value_type operator[](size_t i) const {
        return std::atomic_ref<const value_type>(data_[i]).load(std::memory_order_relaxed);
    }

    iterator begin() const { return iterator{data_}; }
    iterator end() const { return iterator{data_ + size_}; }

  private:
    const T* data_;
    size_t size_;
};

} // namespace svs
