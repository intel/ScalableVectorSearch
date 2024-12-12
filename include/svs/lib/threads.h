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

#include "svs/lib/threads/thread.h"
#include "svs/lib/threads/threadlocal.h"
#include "svs/lib/threads/threadpool.h"
#include "svs/lib/threads/thunks.h"
#include "svs/lib/threads/types.h"

namespace svs::threads {

///
/// @brief Construct a default threadpool containing the requested number of threads.
///
/// @param num_threads The number of threads to use in the threadpool.
///
/// This function returns a default generic threadpool. If more specialized behavior is
/// desired, manually construct the necessary thread pool.
///
inline NativeThreadPool as_threadpool(size_t num_threads) {
    return NativeThreadPool(num_threads);
}

///
/// @brief Pass through the provided thread pool.
///
/// @param threadpool The thread pool to forward.
///
template <threads::ThreadPool Pool> Pool&& as_threadpool(Pool&& threadpool) {
    return std::forward<Pool>(threadpool);
}

} // namespace svs::threads
