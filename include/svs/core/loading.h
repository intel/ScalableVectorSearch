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

#include "svs/lib/misc.h"
#include "svs/lib/threads.h"

namespace svs::detail {

///
/// @brief Generic function for possibly threaded object loading.
///
/// The result of the first compilable action described below will be returned:
///
/// 1. ``f.load(threadpool)``.
/// 2. ``f.load()``.
/// 3. If ``f`` is an instance of ``svs::lib::Lazy`` and accepts the given threadpool, then
///    ``f(threadpool)``.
/// 4. If ``f`` is an instance of ``svs::lib::Lazy`` that is invocable with no arguments,
///    then ``f()``.
/// 5. ``f`` itself.
///
template <typename T, threads::ThreadPool Pool>
auto dispatch_load(T&& f, [[maybe_unused]] Pool& threadpool) {
    using TDecay = std::decay_t<T>;
    if constexpr (lib::HasLoad<TDecay, Pool&>) {
        return f.load(threadpool);
    } else if constexpr (lib::HasLoad<TDecay>) {
        return f.load();
    } else if constexpr (lib::LazyInvocable<TDecay, Pool&>) {
        return f(threadpool);
    } else if constexpr (lib::LazyInvocable<TDecay>) {
        return f();
    } else {
        return f;
    }
}

///
/// @brief Generic function for object loading.
///
/// The result of the first compilable action described below will be returned:
///
/// 2. ``f.load()``.
/// 4. If ``f`` is an instance of ``svs::lib::Lazy`` that is invocable with no arguments,
///    then ``f()``.
/// 5. ``f`` itself.
///
template <typename T> auto dispatch_load(T&& f) {
    using TDecay = std::decay_t<T>;
    if constexpr (lib::HasLoad<TDecay>) {
        return f.load();
    } else if constexpr (lib::LazyInvocable<TDecay>) {
        return f();
    } else {
        return f;
    }
}

} // namespace svs::detail
