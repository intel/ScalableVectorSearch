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
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace svs::arch {

// Refer to the GCC docs for the list of available uarch targets:
// https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html
// https://gcc.gnu.org/onlinedocs/gcc/AArch64-Options.html
enum class MicroArch {
// Use macros to list all uarch instead of duplicating the list from arch_defines.h
#define SVS_MICROARCH_FUNC(uarch) uarch,
    SVS_FOR_EACH_KNOWN_MICROARCH
#undef SVS_MICROARCH_FUNC
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

    void set_microarch(const std::string& arch) {
        set_microarch(string_to_microarch(arch));
    }

    const std::vector<MicroArch>& get_supported_microarchs() const {
        return supported_archs_;
    }

    const std::vector<MicroArch>& get_compiled_microarchs() const {
        return compiled_archs_;
    }

    void describe(std::ostream& out) const {
        write_extensions_status(out);

        out << "\nCurrent µarch: " << microarch_to_string(max_arch_) << std::endl;

        out << "\nSupported µarchs: ";
        for (const auto& arch : supported_archs_) {
            out << microarch_to_string(arch) << " ";
        }
        out << std::endl;

        out << "\nCompiled µarchs: ";
        for (const auto& arch : compiled_archs_) {
            out << microarch_to_string(arch) << " ";
        }
        out << std::endl;
    }

  private:
    MicroArchEnvironment() {
        const std::vector<MicroArch> compiled_archs = {
#define SVS_MICROARCH_FUNC(uarch) MicroArch::uarch,
            SVS_FOR_EACH_MICROARCH
#undef SVS_MICROARCH_FUNC
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

} // namespace svs::arch
