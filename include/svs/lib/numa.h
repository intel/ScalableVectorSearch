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

SVS_VALIDATE_BOOL_ENV(SVS_ENABLE_NUMA);
#if SVS_ENABLE_NUMA

// stdlib
#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

// c-deps
#include <numa.h>
#include <sys/syscall.h>
#include <unistd.h>

// local deps
#include "svs/lib/exception.h"

namespace svs {
namespace numa {

struct CPUMask {};
struct NodeMask {};

namespace detail {

inline pid_t gettid() { return syscall(SYS_gettid); }

template <bool V = false> void do_not_call() { static_assert(V); }
template <typename T> struct bitmask* allocate() {
    do_not_call();
    return nullptr;
}
template <typename T> void free(struct bitmask* /*unused*/) { do_not_call(); }
template <typename T> size_t max_count() {
    do_not_call();
    return 0;
}

// CPUMask
template <> inline struct bitmask* allocate<CPUMask>() { return numa_allocate_cpumask(); }
template <> inline void free<CPUMask>(struct bitmask* mask) { numa_free_cpumask(mask); }
template <> inline size_t max_count<CPUMask>() { return numa_num_configured_cpus(); }

// NodeMask
template <> inline struct bitmask* allocate<NodeMask>() { return numa_allocate_nodemask(); }
template <> inline void free<NodeMask>(struct bitmask* mask) { numa_free_nodemask(mask); }
template <> inline size_t max_count<NodeMask>() { return numa_num_configured_nodes(); }
} // namespace detail

template <typename Kind> class BitMask {
  public:
    BitMask()
        : mask_{detail::allocate<Kind>()} {}

    // Doesn't take ownership
    explicit BitMask(struct bitmask* mask)
        : mask_{detail::allocate<Kind>()} {
        copy_bitmask_to_bitmask(mask, mask_);
    }

    bool get(size_t i) const { return numa_bitmask_isbitset(mask_, i) != 0; }
    void set(size_t i, bool value) {
        value ? (numa_bitmask_setbit(mask_, i)) : (numa_bitmask_clearbit(mask_, i));
    }

    static size_t capacity() { return detail::max_count<Kind>(); }

    ///
    /// Get the number of the `nth` set bit in the mask.
    /// Return `capacity()` if less than `n` bits are set.
    /// @param n The set bit to get.
    ///
    size_t get_nth(size_t n) const {
        size_t seen = 0;
        for (size_t i = 0; i < capacity(); ++i) {
            if (get(i)) {
                if (seen == n) {
                    return i;
                } else {
                    ++seen;
                }
            }
        }
        return capacity();
    }

    size_t count() const {
        size_t count = 0;
        for (size_t i = 0; i < capacity(); ++i) {
            if (get(i)) {
                ++count;
            }
        }
        return count;
    }

    void setnode(size_t node) { numa_node_to_cpus(node, mask_); }

    // For passing to C-functions.
    struct bitmask* ptr() { return mask_; }

    // Special Member Functions
    BitMask(const BitMask& other)
        : mask_{other.mask_ == nullptr ? nullptr : detail::allocate<Kind>()} {
        if (other.mask_ != nullptr) {
            copy_bitmask_to_bitmask(other.mask_, mask_);
        }
    }

    BitMask& operator=(const BitMask& other) {
        if (this != &other) {
            free();
            if (other.mask_ == nullptr) {
                mask_ = nullptr;
            } else {
                mask_ = detail::allocate<Kind>();
                copy_bitmask_to_bitmask(other.mask_, mask_);
            }
        }
        return *this;
    }
    BitMask(BitMask&& other) noexcept
        : mask_{std::exchange(other.mask_, nullptr)} {}
    BitMask& operator=(BitMask&& other) noexcept {
        if (this != &other) {
            free();
            mask_ = std::exchange(other.mask_, nullptr);
        }
        return *this;
    }

    ~BitMask() { free(); }

  private:
    void free() {
        if (mask_ != nullptr) {
            detail::free<Kind>(mask_);
        }
    }
    struct bitmask* mask_;
};

using CPUBitMask = BitMask<CPUMask>;
using NodeBitMask = BitMask<NodeMask>;

/// Printing
inline std::ostream& operator<<(std::ostream& stream, CPUMask /*unused*/) {
    return stream << "CPUMask";
}
inline std::ostream& operator<<(std::ostream& stream, NodeMask /*unused*/) {
    return stream << "NodeMask";
}

template <typename Kind>
std::ostream& operator<<(std::ostream& stream, const BitMask<Kind>& bitmask) {
    stream << Kind{} << '[';
    bool printed_first = false;
    for (size_t i = 0; i < bitmask.capacity(); ++i) {
        if (bitmask.get(i)) {
            if (!printed_first) {
                printed_first = true;
            } else {
                stream << ' ';
            }
            stream << i;
        }
    }
    return stream << ']';
}

/////
///// Queries
/////

inline size_t num_nodes() { return NodeBitMask::capacity(); }
inline size_t num_cpus() { return CPUBitMask::capacity(); }

///
/// Return the number of CPUs on the given NUMA node.
/// @param node The NUMA node to query.
///
inline size_t cpus_on_node(size_t node) {
    size_t nnodes = num_nodes();
    if (node > nnodes) {
        throw std::out_of_range(
            "Node " + std::to_string(node) +
            " is larget than the number of nodes on the system (" + std::to_string(nnodes) +
            ")."
        );
    }

    auto cpumask = CPUBitMask{};
    cpumask.setnode(node);
    return cpumask.count();
}

/////
///// NumaLocal
/////

namespace tls {
static thread_local size_t assigned_node = std::numeric_limits<size_t>::max();
inline bool is_assigned() { return assigned_node != std::numeric_limits<size_t>::max(); }
} // namespace tls

template <typename T> class NumaLocal {
  public:
    // Type Aliases
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    template <typename F>
    NumaLocal(size_t ncopies, F&& f)
        : copies_{} {
        // Requirements:
        // Generally speaking, the objects stored inside a NumaLocal container
        // are going to be large. This means:
        // 1. We don't want to copy them.
        // 2. We don't want to require them to be default constructible.
        // 3. The entity constructing the container might want to construct multiple
        //    elements at a time to improve performance.
        //
        // Strategy:
        // - Use an intermediate `std::vector<std::optional<T>>` to allow default
        //   initialization.
        // - Pass the constructed temporary vector to a lambda to perform the
        //   actual construction of the elements.
        // - Move the constructed elements into the final vector after verifying
        //   that everything has been constructed.
        std::vector<std::optional<T>> temp_copies(ncopies);
        f(temp_copies);
        bool all_initialized =
            std::all_of(temp_copies.begin(), temp_copies.end(), [](const auto& opt) {
                return opt.has_value();
            });
        if (!all_initialized) {
            throw ANNEXCEPTION(
                "Not all entries in a numa local class have been initialized!"
            );
        }

        // Unpack and move the optional values into the actual copies.
        std::transform(
            std::make_move_iterator(std::begin(temp_copies)),
            std::make_move_iterator(std::end(temp_copies)),
            std::back_inserter(copies_),
            [](std::optional<T>&& v) -> T&& { return std::move(v.value()); }
        );
    }

    size_t size() const { return copies_.size(); }

    const T& get() const { return get_checked(); }
    T& get() {
        // Const-cast is okay because we're guarenteed to be in a non-const context.
        return const_cast<T&>(get_checked());
    }

    const T& get_checked() const {
        if (!tls::is_assigned()) {
            throw ANNEXCEPTION("Trying to access NUMA local container without assigning "
                               "thread-based node assignment!");
        }
        return get_direct(tls::assigned_node);
    }

    const T& get_direct(size_t i) const { return copies_.at(i); }
    T& get_direct(size_t i) { return copies_.at(i); }

    // Iterators
    const_iterator cbegin() const { return copies_.cbegin(); }
    const_iterator cend() const { return copies_.cend(); }

    const_iterator begin() const { return cbegin(); }
    const_iterator end() const { return end(); }

    iterator begin() { return copies_.begin(); }
    iterator end() { return copies_.end(); }

  private:
    // Members
    std::vector<T> copies_;
};

/////
///// Binding
/////

class NodeBind {
  public:
    explicit NodeBind(size_t node)
        : affinity_{} {
        affinity_.set(node, true);
        numa_bind(affinity_.ptr());
        tls::assigned_node = node;
    }

    // Since this is global and has side-effects.
    // Make it non-copyable or moveable.
    NodeBind(const NodeBind& /*unused*/) = delete;
    NodeBind& operator=(const NodeBind& /*unused*/) = delete;

    NodeBind(NodeBind&& /*unused*/) = delete;
    NodeBind& operator=(NodeBind&& /*unused*/) = delete;

    // Don't unbind affinity when deleting.
    // I'm honestly not sure what exactly the right semantics here are.
    ~NodeBind() = default;

  private:
    NodeBitMask affinity_;
};
} // namespace numa
} // namespace svs

#endif
