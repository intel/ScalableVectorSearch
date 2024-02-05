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

namespace svs::preprocessor::detail {

// consteval functions for working with preprocessor defines.
// See the discussion here: https://discourse.llvm.org/t/rfc-llvm-libc-tuning/67980
// for some context.

// char* string is not null and is non-empty.
consteval bool is_valid(const char* ptr) { return (ptr != nullptr) && (*ptr != '\0'); }
consteval bool is_one_or_zero(const char* ptr) {
    return is_valid(ptr) && ((*ptr == '0') || *ptr == '1');
}

} // namespace svs::preprocessor::detail

#define SVS_EVALUATE_AND_STRINGIZE(expr) #expr

// Ensure that the macro `macro` has been defined externally and is either 0 or 1.
#define SVS_VALIDATE_BOOL_ENV(macro)                                                  \
    static_assert(                                                                    \
        svs::preprocessor::detail::is_one_or_zero(SVS_EVALUATE_AND_STRINGIZE(macro)), \
        #macro " should be either 0 or 1"                                             \
    );

// Mark functions as "noinline"
#define SVS_NOINLINE [[gnu::noinline]]

// Selectively apply the `[[gnu::noinline]]` attribute.
#if defined(__clang__)
#define SVS_CLANG_NOINLINE SVS_NOINLINE
#else
#define SVS_CLANG_NOINLINE
#endif

#if defined(__GNUC__)
#define SVS_GCC_NOINLINE SVS_NOINLINE
#else
#define SVS_GCC_NOINLINE
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
#define SVS_FORCE_INLINE __attribute__((always_inline)) inline
#endif

#define SVS_FWD(x) static_cast<decltype(x)&&>(x)

#define SVS_CHAIN_SETTER_(type, name)      \
    type& name(decltype(name##_) arg)& {   \
        name##_ = arg;                     \
        return *this;                      \
    }                                      \
    type&& name(decltype(name##_) arg)&& { \
        name##_ = arg;                     \
        return std::move(*this);           \
    }

#define SVS_CHAIN_SETTER_TYPED_(type, argtype, name) \
    type& name(argtype arg)& {                       \
        name##_ = SVS_FWD(arg);                      \
        return *this;                                \
    }                                                \
    type&& name(argtype arg)&& {                     \
        name##_ = SVS_FWD(arg);                      \
        return std::move(*this);                     \
    }
