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

// Mark functions as "noinline"
#define SVS_NOINLINE [[gnu::noinline]]

// Selectively apply the `[[gnu::noinline]]` attribute.
#if defined(NDEBUG) && defined(__clang__)
#define CLANG_NDEBUG_NOINLINE SVS_NOINLINE
#else
#define CLANG_NDEBUG_NOINLINE
#endif

#define SVS_UNUSED_ATTRIBUTE __attribute__((unused))
// Prefix the variable with "unused" to that if it ever *IS* used by the function that
// we get a compiler error.
//
// Doxygen needs to explicilty define this macro in order to process the code correctly.
#define SVS_UNUSED(x) unused_##x SVS_UNUSED_ATTRIBUTE

// Force inline
#if defined(__DOXYGEN__)
// Doxygen gets confused by `__attribute__((always_inline))`, so disable it when we're
// building to documentation to remove a warning.
#define SVS_FORCE_INLINE
#else
#define SVS_FORCE_INLINE __attribute__((always_inline))
#endif

#define SVS_FWD(x) static_cast<decltype(x)&&>(x)
