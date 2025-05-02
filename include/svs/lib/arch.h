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

#include "svs/lib/cpuid.h"

namespace svs::arch {

enum class MicroArch {
#if defined(__x86_64__)
    // Refer to the GCC docs for the list of targeted architectures:
    // https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html
    nehalem,
    x86_64_v2 = nehalem,
    westmere,
    sandybridge,
    ivybridge,
    haswell,
    x86_64_v3 = haswell,
    broadwell,
    skylake,
    skylake_avx512,
    x86_64_v4 = skylake_avx512,
    cascadelake,
    cooperlake,
    icelake_client,
    icelake_server,
    sapphirerapids,
    emeraldrapids = sapphirerapids,
    graniterapids,
    graniterapids_d,
#elif defined(__aarch64__)
#if defined(__APPLE__)
    m1,
    m2,
#else
    neoverse_v1,
    neoverse_n2,
#endif
#endif
    baseline = 0,
};

inline bool arch_is_supported(MicroArch arch) {
    switch (arch) {
#if defined(__x86_64__)
        case MicroArch::nehalem:
            return check_extensions(std::vector<ISAExt>{
                ISAExt::MMX,
                ISAExt::SSE,
                ISAExt::SSE2,
                ISAExt::SSE3,
                ISAExt::SSSE3,
                ISAExt::SSE4_1,
                ISAExt::SSE4_2,
                ISAExt::POPCNT,
                ISAExt::CX16,
                ISAExt::SAHF,
                ISAExt::FXSR});
        case MicroArch::westmere:
            return arch_is_supported(MicroArch::nehalem) &&
                   check_extensions(std::vector<ISAExt>{ISAExt::PCLMUL});
        case MicroArch::sandybridge:
            return arch_is_supported(MicroArch::westmere) &&
                   check_extensions(std::vector<ISAExt>{ISAExt::AVX, ISAExt::XSAVE});
        case MicroArch::ivybridge:
            return arch_is_supported(MicroArch::sandybridge) &&
                   check_extensions(std::vector<ISAExt>{
                       ISAExt::FSGSBASE, ISAExt::RDRND, ISAExt::F16C});
        case MicroArch::haswell:
            return arch_is_supported(MicroArch::ivybridge) &&
                   check_extensions(std::vector<ISAExt>{
                       ISAExt::AVX2,
                       ISAExt::BMI,
                       ISAExt::BMI2,
                       ISAExt::LZCNT,
                       ISAExt::FMA,
                       ISAExt::MOVBE});
        case MicroArch::broadwell:
            return arch_is_supported(MicroArch::haswell) &&
                   check_extensions(std::vector<ISAExt>{
                       ISAExt::RDSEED, ISAExt::ADCX, ISAExt::PREFETCHW});
        case MicroArch::skylake:
            return arch_is_supported(MicroArch::broadwell) &&
                   check_extensions(std::vector<ISAExt>{
                       ISAExt::AES,
                       ISAExt::CLFLUSHOPT,
                       ISAExt::XSAVEC,
                       ISAExt::XSAVES,
                       ISAExt::SGX});
        case MicroArch::skylake_avx512:
            return arch_is_supported(MicroArch::skylake) &&
                   check_extensions(std::vector<ISAExt>{
                       ISAExt::AVX512_F,
                       ISAExt::CLWB,
                       ISAExt::AVX512_VL,
                       ISAExt::AVX512_BW,
                       ISAExt::AVX512_DQ,
                       ISAExt::AVX512_CD});
        case MicroArch::cascadelake:
            return arch_is_supported(MicroArch::skylake_avx512) &&
                   check_extensions(std::vector<ISAExt>{ISAExt::AVX512_VNNI});
        case MicroArch::cooperlake:
            // N.B.: Cooper Lake supports AVX512_BF16, Ice Lake - doesn't, Sapphire Rapids
            // and newer - do
            return arch_is_supported(MicroArch::cascadelake) &&
                   check_extensions(std::vector<ISAExt>{ISAExt::AVX512_BF16});
        case MicroArch::icelake_client:
            return arch_is_supported(MicroArch::cascadelake) &&
                   check_extensions(std::vector<ISAExt>{
                       ISAExt::PKU,
                       ISAExt::AVX512_VBMI,
                       ISAExt::AVX512_IFMA,
                       ISAExt::SHA,
                       ISAExt::GFNI,
                       ISAExt::VAES,
                       ISAExt::AVX512_VBMI2,
                       ISAExt::VPCLMULQDQ,
                       ISAExt::AVX512_BITALG,
                       ISAExt::RDPID,
                       ISAExt::AVX512_VPOPCNTDQ});
        case MicroArch::icelake_server:
            return arch_is_supported(MicroArch::icelake_client) &&
                   check_extensions(std::vector<ISAExt>{
                       ISAExt::PCONFIG, ISAExt::WBNOINVD, ISAExt::CLWB});
        case MicroArch::sapphirerapids:
            return arch_is_supported(MicroArch::icelake_server) &&
                   check_extensions(std::vector<ISAExt>{
                       ISAExt::MOVDIRI,
                       ISAExt::MOVDIR64B,
                       ISAExt::ENQCMD,
                       ISAExt::CLDEMOTE,
                       ISAExt::PTWRITE,
                       ISAExt::WAITPKG,
                       ISAExt::SERIALIZE,
                       ISAExt::TSXLDTRK,
                       ISAExt::UINTR,
                       ISAExt::AMX_BF16,
                       ISAExt::AMX_TILE,
                       ISAExt::AMX_INT8,
                       ISAExt::AVX_VNNI,
                       ISAExt::AVX512_FP16,
                       ISAExt::AVX512_BF16});
        case MicroArch::graniterapids:
            return arch_is_supported(MicroArch::sapphirerapids) &&
                   check_extensions(std::vector<ISAExt>{ISAExt::AMX_FP16, ISAExt::PREFETCHI}
                   );
        case MicroArch::graniterapids_d:
            return arch_is_supported(MicroArch::graniterapids) &&
                   check_extensions(std::vector<ISAExt>{ISAExt::AMX_COMPLEX});
#elif defined(__aarch64__)
#if defined(__APPLE__)
        case MicroArch::m1:
            return check_extensions(std::vector<ISAExt>{ISAExt::M1});
        case MicroArch::m2:
            return check_extensions(std::vector<ISAExt>{ISAExt::M2});
#else
        case MicroArch::neoverse_v1:
            return check_extensions(std::vector<ISAExt>{ISAExt::SVE});
        case MicroArch::neoverse_n2:
            return arch_is_supported(MicroArch::neoverse_v1) &&
                   check_extensions(std::vector<ISAExt>{ISAExt::SVE2});
#endif
#endif
        default:
            return false;
    }
}

// Function to convert MicroArch enum to string
inline std::string microarch_to_string(MicroArch arch) {
    switch (arch) {
#if defined(__x86_64__)
        case MicroArch::nehalem:
            return "nehalem";
        case MicroArch::westmere:
            return "westmere";
        case MicroArch::sandybridge:
            return "sandybridge";
        case MicroArch::ivybridge:
            return "ivybridge";
        case MicroArch::haswell:
            return "haswell";
        case MicroArch::broadwell:
            return "broadwell";
        case MicroArch::skylake:
            return "skylake";
        case MicroArch::skylake_avx512:
            return "skylake_avx512";
        case MicroArch::cascadelake:
            return "cascadelake";
        case MicroArch::cooperlake:
            return "cooperlake";
        case MicroArch::icelake_client:
            return "icelake_client";
        case MicroArch::icelake_server:
            return "icelake_server";
        case MicroArch::sapphirerapids:
            return "sapphirerapids";
        case MicroArch::graniterapids:
            return "graniterapids";
        case MicroArch::graniterapids_d:
            return "graniterapids_d";
#elif defined(__aarch64__)
#if defined(__APPLE__)
        case MicroArch::m1:
            return "m1";
        case MicroArch::m2:
            return "m2";
#else
        case MicroArch::neoverse_v1:
            return "neoverse_v1";
        case MicroArch::neoverse_n2:
            return "neoverse_n2";
#endif
#endif
        default:
            return "unknown";
    }
}

class MicroArchEnvironment {
  public:
    static MicroArchEnvironment& get_instance() {
        // TODO: ensure thread safety
        static MicroArchEnvironment instance;
        return instance;
    }
    MicroArch get_microarch() const { return max_arch_; }

    const std::vector<MicroArch>& get_supported_microarchs() const {
        return supported_archs_;
    }

  private:
    MicroArchEnvironment() {
        const std::vector<MicroArch> compiled_archs = {
#if defined(__x86_64__)
#if defined(SVS_MICROARCH_SUPPORT_nehalem)
            MicroArch::nehalem,
#endif
#if defined(SVS_MICROARCH_SUPPORT_westmere)
            MicroArch::westmere,
#endif
#if defined(SVS_MICROARCH_SUPPORT_sandybridge)
            MicroArch::sandybridge,
#endif
#if defined(SVS_MICROARCH_SUPPORT_ivybridge)
            MicroArch::ivybridge,
#endif
#if defined(SVS_MICROARCH_SUPPORT_haswell)
            MicroArch::haswell,
#endif
#if defined(SVS_MICROARCH_SUPPORT_broadwell)
            MicroArch::broadwell,
#endif
#if defined(SVS_MICROARCH_SUPPORT_skylake)
            MicroArch::skylake,
#endif
#if defined(SVS_MICROARCH_SUPPORT_skylake_avx512)
            MicroArch::skylake_avx512,
#endif
#if defined(SVS_MICROARCH_SUPPORT_cascadelake)
            MicroArch::cascadelake,
#endif
#if defined(SVS_MICROARCH_SUPPORT_cooperlake)
            MicroArch::cooperlake,
#endif
#if defined(SVS_MICROARCH_SUPPORT_icelake_client)
            MicroArch::icelake_client,
#endif
#if defined(SVS_MICROARCH_SUPPORT_icelake_server)
            MicroArch::icelake_server,
#endif
#if defined(SVS_MICROARCH_SUPPORT_sapphirerapids)
            MicroArch::sapphirerapids,
#endif
#if defined(SVS_MICROARCH_SUPPORT_graniterapids)
            MicroArch::graniterapids,
#endif
#if defined(SVS_MICROARCH_SUPPORT_graniterapids_d)
            MicroArch::graniterapids_d,
#endif
#elif defined(__aarch64__)
#if defined(__APPLE__)
#if defined(SVS_MICROARCH_SUPPORT_m1)
            MicroArch::m1,
#endif
#if defined(SVS_MICROARCH_SUPPORT_m2)
            MicroArch::m2,
#endif
#else
#if defined(SVS_MICROARCH_SUPPORT_neoverse_v1)
            MicroArch::neoverse_v1,
#endif
#if defined(SVS_MICROARCH_SUPPORT_neoverse_n2)
            MicroArch::neoverse_n2,
#endif
#endif
#endif
        };
        compiled_archs_ = compiled_archs;
        max_arch_ = MicroArch::baseline;
        for (const auto& arch : compiled_archs_) {
            if (arch_is_supported(arch)) {
                supported_archs_.push_back(arch);
                if (static_cast<int>(arch) > static_cast<int>(max_arch_)) {
                    max_arch_ = arch;
                }
            }
        }
    }

    std::vector<MicroArch> compiled_archs_;
    std::vector<MicroArch> supported_archs_;
    MicroArch max_arch_;
};

#define SVS_PACK_ARGS(...) __VA_ARGS__
#define SVS_CLASS_METHOD_MICROARCH_CASE(microarch, cls, method, args) \
    case svs::arch::MicroArch::microarch:                             \
        return cls<svs::arch::MicroArch::microarch>::method(args);    \
        break;
#define SVS_TARGET_MICROARCH svs::arch::MicroArch::SVS_TUNE_TARGET

#if defined(__x86_64__)

#define SVS_DISPATCH_CLASS_BY_MICROARCH(cls, method, args)                                \
    svs::arch::MicroArch cpu_arch =                                                       \
        svs::arch::MicroArchEnvironment::get_instance().get_microarch();                  \
    switch (cpu_arch) {                                                                   \
        SVS_CLASS_METHOD_MICROARCH_CASE(nehalem, cls, method, SVS_PACK_ARGS(args))        \
        SVS_CLASS_METHOD_MICROARCH_CASE(haswell, cls, method, SVS_PACK_ARGS(args))        \
        SVS_CLASS_METHOD_MICROARCH_CASE(skylake_avx512, cls, method, SVS_PACK_ARGS(args)) \
        SVS_CLASS_METHOD_MICROARCH_CASE(cascadelake, cls, method, SVS_PACK_ARGS(args))    \
        SVS_CLASS_METHOD_MICROARCH_CASE(icelake_client, cls, method, SVS_PACK_ARGS(args)) \
        SVS_CLASS_METHOD_MICROARCH_CASE(sapphirerapids, cls, method, SVS_PACK_ARGS(args)) \
        default:                                                                          \
            return cls<svs::arch::MicroArch::baseline>::method(args);                     \
            break;                                                                        \
    }
#elif defined(__aarch64__)

#if defined(__APPLE__)

#define SVS_DISPATCH_CLASS_BY_MICROARCH(cls, method, args)                    \
    svs::arch::MicroArch cpu_arch =                                           \
        svs::arch::MicroArchEnvironment::get_instance().get_microarch();      \
    switch (cpu_arch) {                                                       \
        SVS_CLASS_METHOD_MICROARCH_CASE(m1, cls, method, SVS_PACK_ARGS(args)) \
        SVS_CLASS_METHOD_MICROARCH_CASE(m2, cls, method, SVS_PACK_ARGS(args)) \
        default:                                                              \
            return cls<svs::arch::MicroArch::baseline>::method(args);         \
            break;                                                            \
    }

#else

#define SVS_DISPATCH_CLASS_BY_MICROARCH(cls, method, args)                             \
    svs::arch::MicroArch cpu_arch =                                                    \
        svs::arch::MicroArchEnvironment::get_instance().get_microarch();               \
    switch (cpu_arch) {                                                                \
        SVS_CLASS_METHOD_MICROARCH_CASE(neoverse_v1, cls, method, SVS_PACK_ARGS(args)) \
        SVS_CLASS_METHOD_MICROARCH_CASE(neoverse_n2, cls, method, SVS_PACK_ARGS(args)) \
        default:                                                                       \
            return cls<svs::arch::MicroArch::baseline>::method(args);                  \
            break;                                                                     \
    }

#endif

#endif

#define SVS_INST_CLASS_METHOD_TMPL_BY_MICROARCH(  \
    return_type, cls, method, template_args, args \
)                                                 \
    template return_type cls<SVS_TARGET_MICROARCH>::method<template_args>(args);
// Generic distance dispatching macro
#define SVS_INST_DISTANCE_CLASS_BY_MICROARCH_AND_TYPENAMES(cls, a_type, b_type) \
    SVS_INST_CLASS_METHOD_TMPL_BY_MICROARCH(                                    \
        float,                                                                  \
        svs::distance::cls,                                                     \
        compute,                                                                \
        SVS_PACK_ARGS(a_type, b_type),                                          \
        SVS_PACK_ARGS(a_type const*, b_type const*, unsigned long)              \
    )
// Cosine distance dispatching macro
#define SVS_INST_COSINE_DISTANCE_CLASS_BY_MICROARCH_AND_TYPENAMES(cls, a_type, b_type) \
    SVS_INST_CLASS_METHOD_TMPL_BY_MICROARCH(                                           \
        float,                                                                         \
        svs::distance::cls,                                                            \
        compute,                                                                       \
        SVS_PACK_ARGS(a_type, b_type),                                                 \
        SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long)              \
    )

} // namespace svs::arch
