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
