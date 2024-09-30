/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
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
