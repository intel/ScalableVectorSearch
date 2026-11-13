/*
 * Copyright 2025 Intel Corporation
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

#include <cstdint>
#include <dlfcn.h>

#if defined(__x86_64__) && defined(__linux__)
#include <asm/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace svs::detail {

#ifdef __x86_64__
struct AVXRuntimeFlags {
    AVXRuntimeFlags() {
        unsigned int eax, ebx, ecx, edx;

        __asm__ __volatile__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(0));

        bool extended_features = eax >= 7;

        __asm__ __volatile__("cpuid"
                             : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                             : "a"(7), "c"(0));

        avx2 = extended_features && ((ebx & (1 << 5)) != 0);
        avx512f = extended_features && ((ebx & (1 << 16)) != 0);
        avx512vnni = extended_features && ((ecx & (1 << 11)) != 0);
        amx_tile_bf16_cpu =
            extended_features && ((edx & (1 << 24)) != 0) && ((edx & (1 << 22)) != 0);
    }

    bool is_avx2_supported() const noexcept { return avx2; }
    bool is_avx512f_supported() const noexcept { return avx512f; }
    bool is_avx512vnni_supported() const noexcept { return avx512vnni; }

    bool avx2;
    bool avx512f;
    bool avx512vnni;
    bool amx_tile_bf16_cpu;
};
#else
struct AVXRuntimeFlags {
    bool is_avx2_supported() const noexcept { return false; }
    bool is_avx512f_supported() const noexcept { return false; }
    bool is_avx512vnni_supported() const noexcept { return false; }

    bool amx_tile_bf16_cpu = false;
};
#endif

inline const AVXRuntimeFlags avx_runtime_flags = {};

#if defined(__x86_64__) && defined(__linux__)
// Kernel XFEATURE_XTILEDATA index for ARCH_REQ_XCOMP_PERM; see amx_tile_bf16_available's
// documentation below for where this number comes from and how it was confirmed.
inline constexpr unsigned long SVS_XFEATURE_XTILEDATA = 18;

/// Returns true the first time, and every time after, that this process may run AMX
/// tile instructions without SIGILL: the CPU has AMX-TILE and AMX-BF16, and the one-time
/// `ARCH_REQ_XCOMP_PERM` request below succeeded. Any "no" here is silent, not an error:
/// dispatch simply does not pick AMX.
///
/// Requests AMX tile-data OS permission exactly once per process, the first time any
/// thread calls this function. Requesting at library load would run in whatever process
/// SVS is linked into before that process has decided it wants AMX at all; requesting
/// per-call would repeat a syscall this function's caching exists to avoid. The request
/// must run unconditionally, every time this initializes: XCR0 (read via `xgetbv`)
/// reports which xstate components the CPU/OS architecturally support, not whether this
/// process has been granted permission to use tile data, so an "already looks enabled"
/// shortcut on XCR0 would skip the one call that actually grants it (confirmed by
/// experiment: a process that never calls `ARCH_REQ_XCOMP_PERM` still reads XCR0 bits
/// 17/18 as set, then SIGILLs on the first real tile instruction).
///
/// Thread-safety and the once-ness come from the function-local static's initializer
/// guarantee (C++ [stmt.dcl]): concurrent first callers block on the same guard variable
/// and only one of them runs the body; every call after that is the "one predictable
/// branch on a cached value" the dispatch call site needs.
///
/// The feature number passed to `ARCH_REQ_XCOMP_PERM` (18, `SVS_XFEATURE_XTILEDATA`
/// above) is not in any header on this machine: glibc's `asm/prctl.h` defines
/// `ARCH_REQ_XCOMP_PERM` itself but not the per-feature indices it takes. 18 is
/// `XFEATURE_XTILEDATA` per the kernel's xstate ABI (`Documentation/arch/x86/xstate.rst`
/// in the Linux source tree); confirmed by experiment on this machine (Linux, x86-64,
/// see `/tmp/amx-study/t2c_detect.c`) that requesting permission with this value is what
/// lets tile instructions run without SIGILL, not merely a documented-but-untested
/// number.
///
/// This syscall is process-wide, not just for the calling thread: once granted, tile
/// state (the eight tmm registers and the tile-config register) becomes part of every
/// thread's extended processor state in this process, saved and restored across context
/// switches like any other xstate component from then on. A process running under a
/// seccomp policy that kills on an unlisted syscall will be terminated right here, at
/// this call, if `arch_prctl` is not on that policy's allow list. Taken from kernel
/// documentation, not verified on this machine: the kernel may refuse the request if an
/// installed alternate signal stack (`sigaltstack`) is smaller than the now-larger
/// xstate save area requires.
inline bool amx_tile_bf16_available() {
    static const bool available = [] {
        // amx.cpp is compiled at exactly the AVX-512 level's instruction set (plus the
        // two AMX extensions, see cmake/multi-arch.cmake) and its non-tile code path
        // forwards to that level's kernels directly, so AMX is only usable where AVX-512
        // already is: a CPU could in principle expose the AMX bits without AVX-512F
        // (e.g. a hypervisor masking features independently), and that combination is
        // not one this file's code was built for.
        if (!avx_runtime_flags.amx_tile_bf16_cpu ||
            !avx_runtime_flags.is_avx512f_supported()) {
            return false;
        }
        long rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, SVS_XFEATURE_XTILEDATA);
        return rc == 0;
    }();
    return available;
}
#else
inline bool amx_tile_bf16_available() { return false; }
#endif

} // namespace svs::detail
