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

#include <svs/core/data/simple.h>
#include <svs/lib/memory.h>

namespace svs {
namespace c_runtime {

template <typename T, bool UseBlocked, typename Allocator = svs::lib::Allocator<T>>
using MaybeBlockedAlloc =
    std::conditional_t<UseBlocked, svs::data::Blocked<Allocator>, Allocator>;

} // namespace c_runtime
} // namespace svs
