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

#include "svs/lib/spinlock.h"

#include <atomic>

namespace svs::index::vamana::concurrent {

///
/// @brief ``svs::SpinLock`` plus value-initializing copy/move operations.
///
/// ``lib::SegmentedVector<SpinLock>`` constructs each segment's elements and, on
/// ``resize``, needs the element type to be constructible from an existing element. A
/// ``std::atomic<bool>`` member makes ``svs::SpinLock`` neither copyable nor movable, so
/// this subclass supplies those operations. A copied or moved lock is always born
/// *unlocked*: the operations exist to satisfy container requirements, never to transfer
/// ownership of a held lock, and a container is only ever grown while its existing
/// elements are untouched.
///
class SpinLock : public svs::SpinLock {
  public:
    SpinLock() = default;

    SpinLock(const SpinLock& /*unused*/)
        : svs::SpinLock{} {}
    SpinLock& operator=(const SpinLock& /*unused*/) { return *this; }
    SpinLock(SpinLock&& /*unused*/) noexcept
        : svs::SpinLock{} {}
    SpinLock& operator=(SpinLock&& /*unused*/) noexcept { return *this; }
    ~SpinLock() = default;
};

} // namespace svs::index::vamana::concurrent
