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

#include "svs/lib/cpuid.h"

// helper for IDE C++ language support
// #define SVS_CPUARCH_NATIVE 1

namespace svs::arch {

enum class CPUArch {
#if defined(SVS_CPUARCH_NATIVE)
    native,
#elif defined(__x86_64__)
    // Refer to the GCC docs for the list of targeted architectures:
    // https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html
    x86_64_v3,
    // Q: are aliases needed in future?
    haswell = x86_64_v3,
    core_avx2 = x86_64_v3,
    broadwell,
    skylake,
    skylake_avx512,
    cascadelake,
    // TODO: Uncomment once supported on python bindings side
    // cooperlake,
    // icelake_server,
    sapphirerapids,
    emeraldrapids = sapphirerapids,
    // graniterapids,
    // graniterapids_d,
#elif defined(__aarch64__)
    neoverse_n1,
    neoverse_v1,
#endif
    baseline = 0,
};

inline bool arch_is_supported(CPUArch arch) {
    switch (arch) {
#if defined(SVS_CPUARCH_NATIVE)
        case CPUArch::native:
            return true;
#elif defined(__x86_64__)
        case CPUArch::x86_64_v3:
            return check_extensions(std::vector<ISAExt>{
                ISAExt::MMX, ISAExt::SSE, ISAExt::SSE2, ISAExt::SSE3, ISAExt::SSSE3,
                ISAExt::SSE4_1, ISAExt::SSE4_2, ISAExt::POPCNT, ISAExt::CX16, ISAExt::SAHF,
                ISAExt::FXSR, ISAExt::AVX, ISAExt::XSAVE, ISAExt::PCLMUL, ISAExt::FSGSBASE,
                ISAExt::RDRND, ISAExt::F16C, ISAExt::AVX2, ISAExt::BMI, ISAExt::BMI2,
                ISAExt::LZCNT, ISAExt::FMA, ISAExt::MOVBE
            });
        case CPUArch::broadwell:
            return arch_is_supported(CPUArch::x86_64_v3) && check_extensions(std::vector<ISAExt>{
                ISAExt::RDSEED, ISAExt::ADCX, ISAExt::PREFETCHW
            });
        case CPUArch::skylake:
            return arch_is_supported(CPUArch::broadwell) && check_extensions(std::vector<ISAExt>{
                ISAExt::AES, ISAExt::CLFLUSHOPT, ISAExt::XSAVEC, ISAExt::XSAVES, ISAExt::SGX
            });
        case CPUArch::skylake_avx512:
            return arch_is_supported(CPUArch::skylake) && check_extensions(std::vector<ISAExt>{
                ISAExt::AVX512_F, ISAExt::CLWB, ISAExt::AVX512_VL, ISAExt::AVX512_BW,
                ISAExt::AVX512_DQ, ISAExt::AVX512_CD
            });
        case CPUArch::cascadelake:
            return arch_is_supported(CPUArch::skylake_avx512) && check_extensions(std::vector<ISAExt>{
                ISAExt::AVX512_VNNI
            });
        // case CPUArch::cooperlake:
        //     return arch_is_supported(CPUArch::cascadelake) && check_extensions(std::vector<ISAExt>{
        //         ISAExt::AVX512_BF16
        //     });
        // case CPUArch::icelake_server:
        //     return arch_is_supported(CPUArch::cooperlake) && check_extensions(std::vector<ISAExt>{
        //         ISAExt::PKU, ISAExt::AVX512_VBMI, ISAExt::AVX512_IFMA, ISAExt::SHA,
        //         ISAExt::GFNI, ISAExt::VAES, ISAExt::AVX512_VBMI2, ISAExt::VPCLMULQDQ,
        //         ISAExt::AVX512_BITALG, ISAExt::RDPID, ISAExt::AVX512_VPOPCNTDQ, ISAExt::PCONFIG,
        //         ISAExt::WBNOINVD, ISAExt::CLWB
        //     });
        case CPUArch::sapphirerapids:
            // return arch_is_supported(CPUArch::icelake_server) && check_extensions(std::vector<ISAExt>{
            return arch_is_supported(CPUArch::cascadelake) && check_extensions(std::vector<ISAExt>{
                ISAExt::MOVDIRI, ISAExt::MOVDIR64B, ISAExt::ENQCMD, ISAExt::CLDEMOTE,
                ISAExt::PTWRITE, ISAExt::WAITPKG, ISAExt::SERIALIZE, ISAExt::TSXLDTRK,
                ISAExt::UINTR, ISAExt::AMX_BF16, ISAExt::AMX_TILE, ISAExt::AMX_INT8,
                ISAExt::AVX_VNNI, ISAExt::AVX512_FP16, ISAExt::AVX512_BF16
            });
        // case CPUArch::graniterapids:
        //     return arch_is_supported(CPUArch::sapphirerapids) && check_extensions(std::vector<ISAExt>{
        //         ISAExt::AMX_FP16, ISAExt::PREFETCHI
        //     });
        // case CPUArch::graniterapids_d:
        //     return arch_is_supported(CPUArch::graniterapids) && check_extensions(std::vector<ISAExt>{
        //         ISAExt::AMX_COMPLEX
        //     });
#elif defined(__aarch64__)
        // TODO: complete lists of supported extensions
        case CPUArch::neoverse_n1:
            return check_extension(ISAExt::SVE);
        case CPUArch::neoverse_v1:
            return arch_is_supported(CPUArch::neoverse_n1) && check_extensions(std::vector<ISAExt>{
                ISAExt::SVE2
            });
#endif
        default:
            return false;
    }
}

class CPUArchEnvironment {
public:
    static CPUArchEnvironment& get_instance() {
        // TODO: ensure thread safety
        static CPUArchEnvironment instance;
        return instance;
    }
    CPUArch get_cpu_arch() const {
        return max_arch_;
    }
private:
    CPUArchEnvironment() {
        const std::vector<CPUArch> compiled_archs = {
#if defined(SVS_CPUARCH_NATIVE)
            CPUArch::native,
#elif defined(__x86_64__)
            // TODO: add support for dynamic list of compiled archs
            CPUArch::x86_64_v3,
            CPUArch::broadwell,
            CPUArch::skylake,
            CPUArch::skylake_avx512,
            CPUArch::cascadelake,
            CPUArch::sapphirerapids,
#endif
        };
        compiled_archs_ = compiled_archs;
        max_arch_ = CPUArch::baseline;
        for (const auto& arch : compiled_archs_) {
            if (arch_is_supported(arch)) {
                supported_archs_.push_back(arch);
                if (static_cast<int>(arch) > static_cast<int>(max_arch_)) {
                    max_arch_ = arch;
                }
            }
        }
    }

    std::vector<CPUArch> compiled_archs_;
    std::vector<CPUArch> supported_archs_;
    CPUArch max_arch_;
};

#define SVS_PACK_ARGS(...) __VA_ARGS__
#define SVS_CLASS_METHOD_CPUARCH_CASE(cpuarch, cls, method, args) \
    case svs::arch::CPUArch::cpuarch: \
        return cls<svs::arch::CPUArch::cpuarch>::method(args); \
        break;
#if defined(SVS_CPUARCH_NATIVE)
    #define SVS_TARGET_CPUARCH svs::arch::CPUArch::native

    #define SVS_DISPATCH_CLASS_BY_CPUARCH(cls, method, args) \
        return cls<svs::arch::CPUArch::native>::method(args);
#elif defined(__x86_64__)
    #define SVS_TARGET_CPUARCH svs::arch::CPUArch::SVS_TUNE_TARGET

    #define SVS_DISPATCH_CLASS_BY_CPUARCH(cls, method, args) \
        svs::arch::CPUArch cpu_arch = svs::arch::CPUArchEnvironment::get_instance().get_cpu_arch(); \
        switch (cpu_arch) { \
            SVS_CLASS_METHOD_CPUARCH_CASE(x86_64_v3, cls, method, SVS_PACK_ARGS(args)) \
            SVS_CLASS_METHOD_CPUARCH_CASE(broadwell, cls, method, SVS_PACK_ARGS(args)) \
            SVS_CLASS_METHOD_CPUARCH_CASE(skylake, cls, method, SVS_PACK_ARGS(args)) \
            SVS_CLASS_METHOD_CPUARCH_CASE(skylake_avx512, cls, method, SVS_PACK_ARGS(args)) \
            SVS_CLASS_METHOD_CPUARCH_CASE(cascadelake, cls, method, SVS_PACK_ARGS(args)) \
            SVS_CLASS_METHOD_CPUARCH_CASE(sapphirerapids, cls, method, SVS_PACK_ARGS(args)) \
            default: \
                return cls<svs::arch::CPUArch::baseline>::method(args); \
                break; \
        }
#elif defined(__aarch64__)
    #define SVS_TARGET_CPUARCH svs::arch::CPUArch::SVS_TUNE_TARGET

    #define SVS_DISPATCH_CLASS_BY_CPUARCH(cls, method, args) \
        svs::arch::CPUArch cpu_arch = svs::arch::CPUArchEnvironment::get_instance().get_cpu_arch(); \
        switch (cpu_arch) { \
            SVS_CLASS_METHOD_CPUARCH_CASE(neoverse_n1, cls, method, SVS_PACK_ARGS(args)) \
            SVS_CLASS_METHOD_CPUARCH_CASE(neoverse_v1, cls, method, SVS_PACK_ARGS(args)) \
            default: \
                return cls<svs::arch::CPUArch::baseline>::method(args); \
                break; \
        }
#endif

#define SVS_INST_CLASS_METHOD_TMPL_BY_CPUARCH(return_type, cls, method, template_args, args) \
    template return_type cls<SVS_TARGET_CPUARCH>::method<template_args>(args);
// Generic distance dispatching macro
#define SVS_INST_DISTANCE_CLASS_BY_CPUARCH_AND_TYPENAMES(cls, a_type, b_type) \
    SVS_INST_CLASS_METHOD_TMPL_BY_CPUARCH(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, unsigned long))
// Cosine distance dispatching macro
#define SVS_INST_COSINE_DISTANCE_CLASS_BY_CPUARCH_AND_TYPENAMES(cls, a_type, b_type) \
    SVS_INST_CLASS_METHOD_TMPL_BY_CPUARCH(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long))

} // namespace svs::arch
