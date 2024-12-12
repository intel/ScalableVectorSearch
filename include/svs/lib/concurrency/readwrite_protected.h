/*
 * Copyright 2024 Intel Corporation
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

#include <mutex>
#include <shared_mutex>

namespace svs::lib {

///
/// @brief A class that provides thread-safe copying and writing to underlying data.
///
/// This class favors read-only data and should not require exclusive access to a lock while
/// copying the contained data.
///
/// A `std::shared_mutex` is used internally to provide synchronized access. Depending
/// on the implementation, writer starvation may occur. If this is a problem, a more
/// complex solution will be needed.
///
template <typename T> class ReadWriteProtected {
  private:
    T x_{};
    mutable std::shared_mutex mutex_{};

  public:
    /// @brief Construct a `ReadWriteProtected` with a default constructed contained object.
    ReadWriteProtected() = default;
    /// @brief Construct a `ReadWriteProtected` by copy-constructing the shared object.
    ReadWriteProtected(const T& x)
        : x_{x} {}
    /// @brief Construct a `ReadWriteProtected` by move-constructing the shared object.
    ReadWriteProtected(T&& x)
        : x_{std::move(x)} {}

    /// @brief Copy construct by performing a copy-construction and move assignment of the
    ///     shared object.
    ReadWriteProtected(const ReadWriteProtected& other) {
        // Even though we're constructing and thus our local mutex can never be contended,
        // we still need to acquire the write mutex so that our store to `x_` is correctly
        // ordered with down-stream readers.
        set(other.get());
    }

    /// @brief Copy assignment by copy-constructing and move assigning the shared object.
    ReadWriteProtected& operator=(const ReadWriteProtected& other) {
        if (this == &other) {
            return *this;
        }
        set(other.get());
    }

    /// @brief Move construction by move-assigning the shared object.
    ReadWriteProtected(ReadWriteProtected&& other) { set(std::move(other).get()); }

    /// @brief Move assignment by move-assigning the shared object.
    ReadWriteProtected& operator=(ReadWriteProtected&& other) {
        if (this == &other) {
            return *this;
        }
        set(std::move(other).get());
        return *this;
    }

    /// @brief Default destructor.
    ~ReadWriteProtected() = default;

    /// @brief Return a copy of the shared object.
    ///
    /// This function is safe to call in a multi-threaded context with multiple readers and
    /// writers.
    ///
    /// However, doing so *may* block for an unspecified amount of time.
    T get() const& {
        std::shared_lock lock{mutex_};
        // Note: The return value is constructed before "lock"s destructor runs.
        // So we're holding the lock until the copy is complete.
        return x_;
    }

    /// @brief Pilfer the shared object using a return by move-construction.
    T get() && {
        // Subtle: We need to acquire the `unique_lock` since we are updating the internal
        // state.
        //
        // Further, we return by value rather than r-value reference (though the returned
        // object should be constructed by its move consructor).
        //
        // This is because if we return an r-value reference, then we no longer hold the
        // lock after returning and that r-value could be modified.
        std::unique_lock lock{mutex_};
        return std::move(x_);
    }

    /// @brief Set the shared object.
    ///
    /// This function is safe to call in a multi-threaded context with multiple readers and
    /// writers.
    ///
    /// However, doing so *may* block for an unspecified amount of time.
    void set(const T& x) {
        std::unique_lock lock{mutex_};
        x_ = x;
    }

    /// @brief Set the shared object.
    ///
    /// This function is safe to call in a multi-threaded context with multiple readers and
    /// writers.
    ///
    /// However, doing so *may* block for an unspecified amount of time.
    void set(T&& x) {
        std::unique_lock lock{mutex_};
        x_ = std::move(x);
    }
};

} // namespace svs::lib
