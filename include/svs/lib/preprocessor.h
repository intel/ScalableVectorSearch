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

namespace svs::preprocessor::detail {

// consteval functions for working with preprocessor defines.
// See the discussion here: https://discourse.llvm.org/t/rfc-llvm-libc-tuning/67980
// for some context.

// Compile time string length.
// Provided pointer must not be null.
consteval long long strlen(const char* ptr) {
    const auto* head = ptr;
    while (*head != '\0') {
        ++head;
    }
    return head - ptr;
}
// char* string is not null and has length 1.
consteval bool is_valid(const char* ptr) { return (ptr != nullptr) && (strlen(ptr) == 1); }
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

/////
///// Intel(R) AVX extensions
/////

namespace svs::arch {
// Switching ifdefs to boolean defs helps reduce the probability of mistyping.

// Most 32-bit and 64-bit Intel(R) AVX instructions.
// - 512 bit registers
// - operation masks
// - broadcasting
// - embedded rounding and exception control.
#if defined(__AVX512F__)
#define SVS_AVX512_F 1
inline constexpr bool have_avx512_f = true;
#else
#define SVS_AVX512_F 0
inline constexpr bool have_avx512_f = false;
#endif

// Extends Intel(R) AVX-512 operations to 128-bit and 256-bit registers.
#if defined(__AVX512VL__)
#define SVS_AVX512_VL 1
inline constexpr bool have_avx512_vl = true;
#else
#define SVS_AVX512_VL 0
inline constexpr bool have_avx512_vl = true;
#endif

// 8-bit and 16-bit integer operations for Intel(R) AVX-512.
#if defined(__AVX512BW__)
#define SVS_AVX512_BW 1
inline constexpr bool have_avx512_bw = true;
#else
#define SVS_AVX512_BW 0
inline constexpr bool have_avx512_bw = false;
#endif

// Vector instruction for deep learning.
#if defined(__AVX512VNNI__)
#define SVS_AVX512_VNNI 1
inline constexpr bool have_avx512_vnni = true;
#else
#define SVS_AVX512_VNNI 0
inline constexpr bool have_avx512_vnni = false;
#endif

// 256-bit Intel(R) AVX instruction set.
#if defined(__AVX2__)
#define SVS_AVX2 1
inline constexpr bool have_avx512_avx2 = true;
#else
#define SVS_AVX2 0
inline constexpr bool have_avx512_avx2 = true;
#endif

} // namespace svs::arch
