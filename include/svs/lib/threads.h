/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
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
