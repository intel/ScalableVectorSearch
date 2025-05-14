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

#include "svs/lib/arch_defines.h"
#include "svs/lib/cpuid.h"
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace svs::arch {

enum class MicroArch {
#if defined(__x86_64__)
    // Refer to the GCC docs for the list of targeted architectures:
    // https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html
    nehalem,
    westmere,
    sandybridge,
    ivybridge,
    haswell,
    broadwell,
    skylake,
    x86_64_v4,
    skylake_avx512,
    cascadelake,
    cooperlake,
    icelake_client,
    icelake_server,
    sapphirerapids,
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

struct MicroArchInfo {
    std::optional<MicroArch> parent;
    std::vector<ISAExt> extensions;
    std::string name;
};

// Unordered map with MicroArch to MicroArchInfo mapping
inline const std::unordered_map<MicroArch, MicroArchInfo>& get_microarch_info_map() {
    static const std::unordered_map<MicroArch, MicroArchInfo> microarch_info = {
#if defined(__x86_64__)
        {MicroArch::nehalem,
         {std::nullopt,
          {ISAExt::MMX,
           ISAExt::SSE,
           ISAExt::SSE2,
           ISAExt::SSE3,
           ISAExt::SSSE3,
           ISAExt::SSE4_1,
           ISAExt::SSE4_2,
           ISAExt::POPCNT,
           ISAExt::CX16,
           ISAExt::SAHF,
           ISAExt::FXSR},
          "nehalem"}},
        {MicroArch::westmere, {MicroArch::nehalem, {ISAExt::PCLMUL}, "westmere"}},
        {MicroArch::sandybridge,
         {MicroArch::westmere, {ISAExt::AVX, ISAExt::XSAVE}, "sandybridge"}},
        {MicroArch::ivybridge,
         {MicroArch::sandybridge,
          {ISAExt::FSGSBASE, ISAExt::RDRND, ISAExt::F16C},
          "ivybridge"}},
        {MicroArch::haswell,
         {MicroArch::sandybridge,
          {ISAExt::AVX2,
           ISAExt::BMI,
           ISAExt::BMI2,
           ISAExt::LZCNT,
           ISAExt::FMA,
           ISAExt::MOVBE},
          "haswell"}},
        {MicroArch::broadwell,
         {MicroArch::haswell,
          {ISAExt::RDSEED, ISAExt::ADCX, ISAExt::PREFETCHW},
          "broadwell"}},
        {MicroArch::skylake,
         {MicroArch::broadwell,
          {ISAExt::AES, ISAExt::CLFLUSHOPT, ISAExt::XSAVEC, ISAExt::XSAVES, ISAExt::SGX},
          "skylake"}},
        {MicroArch::x86_64_v4,
         {std::nullopt,
          {ISAExt::AVX512_F,
           ISAExt::AVX512_VL,
           ISAExt::AVX512_BW,
           ISAExt::AVX512_DQ,
           ISAExt::AVX512_CD},
          "x86_64_v4"}},
        {MicroArch::skylake_avx512,
         {MicroArch::skylake,
          {ISAExt::AVX512_F,
           ISAExt::CLWB,
           ISAExt::AVX512_VL,
           ISAExt::AVX512_BW,
           ISAExt::AVX512_DQ,
           ISAExt::AVX512_CD},
          "skylake_avx512"}},
        {MicroArch::cascadelake,
         {MicroArch::skylake_avx512, {ISAExt::AVX512_VNNI}, "cascadelake"}},
        {MicroArch::cooperlake,
         {MicroArch::cascadelake, {ISAExt::AVX512_BF16}, "cooperlake"}},
        {MicroArch::icelake_client,
         {MicroArch::cascadelake,
          {ISAExt::PKU,
           ISAExt::AVX512_VBMI,
           ISAExt::AVX512_IFMA,
           ISAExt::SHA,
           ISAExt::GFNI,
           ISAExt::VAES,
           ISAExt::AVX512_VBMI2,
           ISAExt::VPCLMULQDQ,
           ISAExt::AVX512_BITALG,
           ISAExt::RDPID,
           ISAExt::AVX512_VPOPCNTDQ},
          "icelake_client"}},
        {MicroArch::icelake_server,
         {MicroArch::icelake_client,
          {ISAExt::PCONFIG, ISAExt::WBNOINVD, ISAExt::CLWB},
          "icelake_server"}},
        {MicroArch::sapphirerapids,
         {MicroArch::icelake_server,
          {ISAExt::MOVDIRI,
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
           ISAExt::AVX512_BF16},
          "sapphirerapids"}},
        {MicroArch::graniterapids,
         {MicroArch::sapphirerapids,
          {ISAExt::AMX_FP16, ISAExt::PREFETCHI},
          "graniterapids"}},
        {MicroArch::graniterapids_d,
         {MicroArch::graniterapids, {ISAExt::AMX_COMPLEX}, "graniterapids_d"}},
#elif defined(__aarch64__)
#if defined(__APPLE__)
        {MicroArch::m1, {std::nullopt, {ISAExt::M1}, "m1"}},
        {MicroArch::m2, {std::nullopt, {ISAExt::M2}, "m2"}},
#else
        {MicroArch::neoverse_v1, {std::nullopt, {ISAExt::SVE}, "neoverse_v1"}},
        {MicroArch::neoverse_n2, {MicroArch::neoverse_v1, {ISAExt::SVE2}, "neoverse_n2"}},
#endif
#endif
        {MicroArch::baseline, {std::nullopt, {}, "baseline"}}
    };
    return microarch_info;
}

inline bool arch_is_supported(MicroArch arch) {
    const auto& info_map = get_microarch_info_map();
    auto it = info_map.find(arch);
    if (it == info_map.end()) {
        return false;
    }

    const auto& info = it->second;

    // First check if parent architecture is supported
    if (info.parent.has_value() && !arch_is_supported(info.parent.value())) {
        return false;
    }

    // Then check additional extensions
    return check_extensions(info.extensions);
}

inline std::string microarch_to_string(MicroArch arch) {
    const auto& info_map = get_microarch_info_map();
    auto it = info_map.find(arch);
    if (it != info_map.end()) {
        return it->second.name;
    }
    return "unknown";
}

inline MicroArch string_to_microarch(const std::string& arch_name) {
    const auto& info_map = get_microarch_info_map();
    for (const auto& [arch, info] : info_map) {
        if (info.name == arch_name) {
            return arch;
        }
    }
    throw std::invalid_argument("Unknown microarchitecture name: " + arch_name);
}

class MicroArchEnvironment {
  public:
    static MicroArchEnvironment& get_instance() {
        // TODO: ensure thread safety
        static MicroArchEnvironment instance;
        return instance;
    }
    MicroArch get_microarch() const { return max_arch_; }

    void set_microarch(MicroArch arch) {
        if (arch_is_supported(arch)) {
            max_arch_ = arch;
        } else {
            throw std::invalid_argument("Unsupported microarchitecture");
        }
    }

    const std::vector<MicroArch>& get_supported_microarchs() const {
        return supported_archs_;
    }

    const std::vector<MicroArch>& get_compiled_microarchs() const {
        return compiled_archs_;
    }

  private:
    MicroArchEnvironment() {
        const std::vector<MicroArch> compiled_archs = {
#if defined(__x86_64__)
            SVS_MICROARCH_COMPILED_nehalem
            SVS_MICROARCH_COMPILED_westmere
            SVS_MICROARCH_COMPILED_sandybridge
            SVS_MICROARCH_COMPILED_ivybridge
            SVS_MICROARCH_COMPILED_haswell
            SVS_MICROARCH_COMPILED_broadwell
            SVS_MICROARCH_COMPILED_skylake
            SVS_MICROARCH_COMPILED_x86_64_v4
            SVS_MICROARCH_COMPILED_skylake_avx512
            SVS_MICROARCH_COMPILED_cascadelake
            SVS_MICROARCH_COMPILED_cooperlake
            SVS_MICROARCH_COMPILED_icelake_client
            SVS_MICROARCH_COMPILED_icelake_server
            SVS_MICROARCH_COMPILED_sapphirerapids
            SVS_MICROARCH_COMPILED_graniterapids
            SVS_MICROARCH_COMPILED_graniterapids_d
#elif defined(__aarch64__)
#if defined(__APPLE__)
            SVS_MICROARCH_COMPILED_m1
            SVS_MICROARCH_COMPILED_m2
#else
            SVS_MICROARCH_COMPILED_neoverse_v1
            SVS_MICROARCH_COMPILED_neoverse_n2
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

#if defined(__x86_64__)

#define SVS_DISPATCH_CLASS_BY_MICROARCH(cls, method, args)                                 \
    svs::arch::MicroArch cpu_arch =                                                        \
        svs::arch::MicroArchEnvironment::get_instance().get_microarch();                   \
    switch (cpu_arch) {                                                                    \
        SVS_CLASS_METHOD_MICROARCH_CASE_nehalem(cls, method, SVS_PACK_ARGS(args))          \
        SVS_CLASS_METHOD_MICROARCH_CASE_westmere(cls, method, SVS_PACK_ARGS(args))         \
        SVS_CLASS_METHOD_MICROARCH_CASE_sandybridge(cls, method, SVS_PACK_ARGS(args))      \
        SVS_CLASS_METHOD_MICROARCH_CASE_ivybridge(cls, method, SVS_PACK_ARGS(args))        \
        SVS_CLASS_METHOD_MICROARCH_CASE_haswell(cls, method, SVS_PACK_ARGS(args))          \
        SVS_CLASS_METHOD_MICROARCH_CASE_broadwell(cls, method, SVS_PACK_ARGS(args))        \
        SVS_CLASS_METHOD_MICROARCH_CASE_skylake(cls, method, SVS_PACK_ARGS(args))          \
        SVS_CLASS_METHOD_MICROARCH_CASE_x86_64_v4(cls, method, SVS_PACK_ARGS(args))        \
        SVS_CLASS_METHOD_MICROARCH_CASE_skylake_avx512(cls, method, SVS_PACK_ARGS(args))   \
        SVS_CLASS_METHOD_MICROARCH_CASE_cascadelake(cls, method, SVS_PACK_ARGS(args))      \
        SVS_CLASS_METHOD_MICROARCH_CASE_cooperlake(cls, method, SVS_PACK_ARGS(args))       \
        SVS_CLASS_METHOD_MICROARCH_CASE_icelake_client(cls, method, SVS_PACK_ARGS(args))   \
        SVS_CLASS_METHOD_MICROARCH_CASE_icelake_server(cls, method, SVS_PACK_ARGS(args))   \
        SVS_CLASS_METHOD_MICROARCH_CASE_sapphirerapids(cls, method, SVS_PACK_ARGS(args))   \
        SVS_CLASS_METHOD_MICROARCH_CASE_graniterapids(cls, method, SVS_PACK_ARGS(args))    \
        SVS_CLASS_METHOD_MICROARCH_CASE_graniterapids_d(cls, method, SVS_PACK_ARGS(args))  \
        default:                                                                           \
            return cls<svs::arch::MicroArch::baseline>::method(args);                      \
            break;                                                                         \
    }
#elif defined(__aarch64__)

#if defined(__APPLE__)

#define SVS_DISPATCH_CLASS_BY_MICROARCH(cls, method, args)                    \
    svs::arch::MicroArch cpu_arch =                                           \
        svs::arch::MicroArchEnvironment::get_instance().get_microarch();      \
    switch (cpu_arch) {                                                       \
        SVS_CLASS_METHOD_MICROARCH_CASE_m1(cls, method, SVS_PACK_ARGS(args))  \
        SVS_CLASS_METHOD_MICROARCH_CASE_m2(cls, method, SVS_PACK_ARGS(args))  \
        default:                                                              \
            return cls<svs::arch::MicroArch::baseline>::method(args);         \
            break;                                                            \
    }

#else

#define SVS_DISPATCH_CLASS_BY_MICROARCH(cls, method, args)                               \
    svs::arch::MicroArch cpu_arch =                                                      \
        svs::arch::MicroArchEnvironment::get_instance().get_microarch();                 \
    switch (cpu_arch) {                                                                  \
        SVS_CLASS_METHOD_MICROARCH_CASE_neoverse_v1(cls, method, SVS_PACK_ARGS(args))    \
        SVS_CLASS_METHOD_MICROARCH_CASE_neoverse_n2(cls, method, SVS_PACK_ARGS(args))    \
        default:                                                                         \
            return cls<svs::arch::MicroArch::baseline>::method(args);                    \
            break;                                                                       \
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
// Generic distance extent macro
#if defined(__x86_64__)

#define SVS_EXTERN_DISTANCE_CLASS_BY_TYPENAMES(cls, a_type, b_type) \
    SVS_EXTERN_CLASS_METHOD_westmere(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_sandybridge(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_ivybridge(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_haswell(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_broadwell(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_skylake(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_x86_64_v4(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_skylake_avx512(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_cascadelake(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_cooperlake(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_icelake_client(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_icelake_server(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_sapphirerapids(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_graniterapids(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_graniterapids_d(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, unsigned long))

#elif defined(__aarch64__)

#if defined(__APPLE__)

#define SVS_EXTERN_DISTANCE_CLASS_BY_TYPENAMES(cls, a_type, b_type) \
    SVS_EXTERN_CLASS_METHOD_m2(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, unsigned long))

#else

#define SVS_EXTERN_DISTANCE_CLASS_BY_TYPENAMES(cls, a_type, b_type) \
    SVS_EXTERN_CLASS_METHOD_neoverse_n2(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, unsigned long))

#endif

#endif
// Cosine distance dispatching macro
#define SVS_INST_COSINE_DISTANCE_CLASS_BY_MICROARCH_AND_TYPENAMES(cls, a_type, b_type) \
    SVS_INST_CLASS_METHOD_TMPL_BY_MICROARCH(                                           \
        float,                                                                         \
        svs::distance::cls,                                                            \
        compute,                                                                       \
        SVS_PACK_ARGS(a_type, b_type),                                                 \
        SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long)              \
    )

// Cosine distance extent macro
#if defined(__x86_64__)

#define SVS_EXTERN_COSINE_DISTANCE_CLASS_BY_TYPENAMES(cls, a_type, b_type) \
    SVS_EXTERN_CLASS_METHOD_westmere(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_sandybridge(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_ivybridge(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_haswell(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_broadwell(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_skylake(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_x86_64_v4(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_skylake_avx512(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_cascadelake(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_cooperlake(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_icelake_client(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_icelake_server(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_sapphirerapids(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_graniterapids(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_graniterapids_d(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long))

#elif defined(__aarch64__)

#if defined(__APPLE__)

#define SVS_EXTERN_COSINE_DISTANCE_CLASS_BY_TYPENAMES(cls, a_type, b_type) \
    SVS_EXTERN_CLASS_METHOD_m1(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_m2(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long))

#else

#define SVS_EXTERN_COSINE_DISTANCE_CLASS_BY_TYPENAMES(cls, a_type, b_type) \
    SVS_EXTERN_CLASS_METHOD_neoverse_v1(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long)) \
    SVS_EXTERN_CLASS_METHOD_neoverse_n2(float, svs::distance::cls, compute, SVS_PACK_ARGS(a_type, b_type), SVS_PACK_ARGS(a_type const*, b_type const*, float, unsigned long))

#endif

#endif

} // namespace svs::arch
