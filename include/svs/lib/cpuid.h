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

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__)
#include <cpuid.h>
#endif

namespace svs::arch {

#if defined(__x86_64__)

enum class ISAExt {
    // Common extensions
    MMX,
    SSE,
    SSE2,
    SSE3,
    SSSE3,
    SSE4_1,
    SSE4_2,
    POPCNT,
    CX16,
    SAHF,
    FXSR,
    AVX,
    XSAVE,
    PCLMUL,
    FSGSBASE,
    RDRND,
    F16C,
    AVX2,
    BMI,
    BMI2,
    LZCNT,
    FMA,
    MOVBE,
    RDSEED,
    ADCX,
    PREFETCHW,
    AES,
    CLFLUSHOPT,
    XSAVEC,
    XSAVES,
    SGX,
    CLWB,
    PKU,
    SHA,
    GFNI,
    VAES,
    VPCLMULQDQ,
    RDPID,
    PCONFIG,
    WBNOINVD,
    MOVDIRI,
    MOVDIR64B,
    ENQCMD,
    CLDEMOTE,
    PTWRITE,
    WAITPKG,
    SERIALIZE,
    TSXLDTRK,
    UINTR,
    PREFETCHI,

    // AVX family
    AVX_VNNI,

    // AVX512 family
    AVX512_F,
    AVX512_VL,
    AVX512_BW,
    AVX512_DQ,
    AVX512_CD,
    AVX512_VBMI,
    AVX512_IFMA,
    AVX512_VNNI,
    AVX512_VBMI2,
    AVX512_BITALG,
    AVX512_VPOPCNTDQ,
    AVX512_BF16,
    AVX512_FP16,

    // AMX family
    AMX_BF16,
    AMX_TILE,
    AMX_INT8,
    AMX_FP16,
    AMX_COMPLEX
};

struct CPUIDFlag {
    const uint32_t function;    // EAX input for CPUID
    const uint32_t subfunction; // ECX input for CPUID
    const uint32_t reg;         // Register index (0=EAX, 1=EBX, 2=ECX, 3=EDX)
    const uint32_t bit;         // Bit position in the register
    const char* name;

    bool get_value() const {
        std::array<int, 4> regs{};
        __cpuid_count(function, subfunction, regs[0], regs[1], regs[2], regs[3]);
        return (regs[reg] & (1 << bit)) != 0;
    }
};

inline const std::unordered_map<ISAExt, CPUIDFlag> ISAExtInfo = {
    // flags are sorted by function, subfunction, register and bit
    {ISAExt::MMX, {1, 0, 3, 23, "MMX"}},
    {ISAExt::FXSR, {1, 0, 3, 24, "FXSR"}},
    {ISAExt::SSE, {1, 0, 3, 25, "SSE"}},
    {ISAExt::SSE2, {1, 0, 3, 26, "SSE2"}},
    {ISAExt::SSE3, {1, 0, 2, 0, "SSE3"}},
    {ISAExt::PCLMUL, {1, 0, 2, 1, "PCLMUL"}},
    {ISAExt::SSSE3, {1, 0, 2, 9, "SSSE3"}},
    {ISAExt::FMA, {1, 0, 2, 12, "FMA"}},
    {ISAExt::CX16, {1, 0, 2, 13, "CX16"}},
    {ISAExt::SSE4_1, {1, 0, 2, 19, "SSE4_1"}},
    {ISAExt::SSE4_2, {1, 0, 2, 20, "SSE4_2"}},
    {ISAExt::MOVBE, {1, 0, 2, 22, "MOVBE"}},
    {ISAExt::POPCNT, {1, 0, 2, 23, "POPCNT"}},
    {ISAExt::AES, {1, 0, 2, 25, "AES"}},
    {ISAExt::XSAVE, {1, 0, 2, 26, "XSAVE"}},
    {ISAExt::AVX, {1, 0, 2, 28, "AVX"}},
    {ISAExt::F16C, {1, 0, 2, 29, "F16C"}},
    {ISAExt::RDRND, {1, 0, 2, 30, "RDRND"}},
    {ISAExt::FSGSBASE, {7, 0, 1, 0, "FSGSBASE"}},
    {ISAExt::SGX, {7, 0, 1, 2, "SGX"}},
    {ISAExt::BMI, {7, 0, 1, 3, "BMI"}},
    {ISAExt::AVX2, {7, 0, 1, 5, "AVX2"}},
    {ISAExt::BMI2, {7, 0, 1, 8, "BMI2"}},
    {ISAExt::AVX512_F, {7, 0, 1, 16, "AVX512_F"}},
    {ISAExt::AVX512_DQ, {7, 0, 1, 17, "AVX512_DQ"}},
    {ISAExt::RDSEED, {7, 0, 1, 18, "RDSEED"}},
    {ISAExt::ADCX, {7, 0, 1, 19, "ADCX"}},
    {ISAExt::AVX512_IFMA, {7, 0, 1, 21, "AVX512_IFMA"}},
    {ISAExt::CLFLUSHOPT, {7, 0, 1, 23, "CLFLUSHOPT"}},
    {ISAExt::CLWB, {7, 0, 1, 24, "CLWB"}},
    {ISAExt::AVX512_CD, {7, 0, 1, 28, "AVX512_CD"}},
    {ISAExt::SHA, {7, 0, 1, 29, "SHA"}},
    {ISAExt::AVX512_BW, {7, 0, 1, 30, "AVX512_BW"}},
    {ISAExt::AVX512_VL, {7, 0, 1, 31, "AVX512_VL"}},
    {ISAExt::AVX512_VBMI, {7, 0, 2, 1, "AVX512_VBMI"}},
    {ISAExt::PKU, {7, 0, 2, 3, "PKU"}},
    {ISAExt::WAITPKG, {7, 0, 2, 5, "WAITPKG"}},
    {ISAExt::AVX512_VBMI2, {7, 0, 2, 6, "AVX512_VBMI2"}},
    {ISAExt::GFNI, {7, 0, 2, 8, "GFNI"}},
    {ISAExt::VAES, {7, 0, 2, 9, "VAES"}},
    {ISAExt::VPCLMULQDQ, {7, 0, 2, 10, "VPCLMULQDQ"}},
    {ISAExt::AVX512_VNNI, {7, 0, 2, 11, "AVX512_VNNI"}},
    {ISAExt::AVX512_BITALG, {7, 0, 2, 12, "AVX512_BITALG"}},
    {ISAExt::AVX512_VPOPCNTDQ, {7, 0, 2, 14, "AVX512_VPOPCNTDQ"}},
    {ISAExt::RDPID, {7, 0, 2, 22, "RDPID"}},
    {ISAExt::CLDEMOTE, {7, 0, 2, 25, "CLDEMOTE"}},
    {ISAExt::MOVDIRI, {7, 0, 2, 27, "MOVDIRI"}},
    {ISAExt::MOVDIR64B, {7, 0, 2, 28, "MOVDIR64B"}},
    {ISAExt::ENQCMD, {7, 0, 2, 29, "ENQCMD"}},
    {ISAExt::UINTR, {7, 0, 3, 5, "UINTR"}},
    {ISAExt::SERIALIZE, {7, 0, 3, 14, "SERIALIZE"}},
    {ISAExt::TSXLDTRK, {7, 0, 3, 16, "TSXLDTRK"}},
    {ISAExt::PCONFIG, {7, 0, 3, 18, "PCONFIG"}},
    {ISAExt::AMX_BF16, {7, 0, 3, 22, "AMX_BF16"}},
    {ISAExt::AVX512_FP16, {7, 0, 3, 23, "AVX512_FP16"}},
    {ISAExt::AMX_TILE, {7, 0, 3, 24, "AMX_TILE"}},
    {ISAExt::AMX_INT8, {7, 0, 3, 25, "AMX_INT8"}},
    {ISAExt::AVX_VNNI, {7, 1, 0, 4, "AVX_VNNI"}},
    {ISAExt::AVX512_BF16, {7, 1, 0, 5, "AVX512_BF16"}},
    {ISAExt::AMX_FP16, {7, 1, 0, 21, "AMX_FP16"}},
    {ISAExt::AMX_COMPLEX, {7, 1, 3, 8, "AMX_COMPLEX"}},
    {ISAExt::PREFETCHI, {7, 1, 3, 14, "PREFETCHI"}},
    {ISAExt::XSAVEC, {0xD, 1, 0, 1, "XSAVEC"}},
    {ISAExt::XSAVES, {0xD, 1, 0, 3, "XSAVES"}},
    {ISAExt::PTWRITE, {0x14, 0, 1, 4, "PTWRITE"}},
    {ISAExt::WBNOINVD, {0x80000008, 0, 1, 9, "WBNOINVD"}},
    {ISAExt::SAHF, {0x80000001, 0, 2, 0, "SAHF"}},
    {ISAExt::LZCNT, {0x80000001, 0, 2, 5, "LZCNT"}},
    {ISAExt::PREFETCHW, {0x80000001, 0, 2, 8, "PREFETCHW"}},
};

#elif defined(__aarch64__)

enum class ISAExt {
    // SVE family
    SVE,
    SVE2,

    DOTPROD, // ARMv8.4-A
    RNG,     // ARMv8.5-A
    BF16,    // ARMv8.6-A
};

// Define register ID values for ARM features detection
#define ID_AA64PFR0_EL1 0
#define ID_AA64ISAR0_EL1 1
#define ID_AA64ISAR1_EL1 2
#define ID_AA64ZFR0_EL1 3

// Helper template to read system registers with mrs instruction
template <unsigned int ID> inline uint64_t read_system_reg() {
    uint64_t val;
    if constexpr (ID == ID_AA64PFR0_EL1) {
        asm("mrs %0, id_aa64pfr0_el1" : "=r"(val));
    } else if constexpr (ID == ID_AA64ISAR0_EL1) {
        asm("mrs %0, id_aa64isar0_el1" : "=r"(val));
    } else if constexpr (ID == ID_AA64ISAR1_EL1) {
        asm("mrs %0, id_aa64isar1_el1" : "=r"(val));
#if !(defined(__APPLE__))
    } else if constexpr (ID == ID_AA64ZFR0_EL1) {
        asm("mrs %0, id_aa64zfr0_el1" : "=r"(val));
#endif
    } else {
        val = 0;
    }
    return val;
}

// Extract bits from register value
inline uint64_t extract_bits(uint64_t val, int pos, int len) {
    return (val >> pos) & ((1ULL << len) - 1);
}

struct MSRFlag {
    unsigned int reg_id;   // System register ID
    int bit_pos;           // Bit position in the register
    int bit_len;           // Number of bits to check
    uint64_t expected_val; // Expected value for feature to be present
    const char* name;      // Feature name

    bool get_value() const {
        uint64_t reg_val = 0;

        try {
            switch (reg_id) {
                case ID_AA64PFR0_EL1:
                    reg_val = read_system_reg<ID_AA64PFR0_EL1>();
                    break;
                case ID_AA64ISAR0_EL1:
                    reg_val = read_system_reg<ID_AA64ISAR0_EL1>();
                    break;
                case ID_AA64ISAR1_EL1:
                    reg_val = read_system_reg<ID_AA64ISAR1_EL1>();
                    break;
#if !(defined(__APPLE__))
                case ID_AA64ZFR0_EL1:
                    if (extract_bits(read_system_reg<ID_AA64PFR0_EL1>(), 32, 4) != 0) {
                        reg_val = read_system_reg<ID_AA64ZFR0_EL1>();
                    }
                    break;
#endif
                default:
                    return false;
            }

            return extract_bits(reg_val, bit_pos, bit_len) == expected_val;
        } catch (...) {
            // If reading the register fails, the feature is not supported
            return false;
        }
    }
};

inline const std::unordered_map<ISAExt, MSRFlag> ISAExtInfo = {
    {ISAExt::SVE, {ID_AA64PFR0_EL1, 32, 4, 1, "sve"}},
#if !(defined(__APPLE__))
    {ISAExt::SVE2, {ID_AA64ZFR0_EL1, 0, 4, 1, "sve2"}},
#endif
    {ISAExt::DOTPROD, {ID_AA64ISAR0_EL1, 24, 4, 1, "dotprod"}},
    {ISAExt::RNG, {ID_AA64ISAR0_EL1, 60, 4, 1, "rng"}},
    {ISAExt::BF16, {ID_AA64ISAR1_EL1, 44, 4, 1, "bf16"}},
};

#endif

inline bool check_extension(ISAExt ext) { return ISAExtInfo.at(ext).get_value(); }

inline bool check_extensions(std::vector<ISAExt> exts) {
    for (const auto& ext : exts) {
        if (!check_extension(ext)) {
            return false;
        }
    }
    return true;
}

} // namespace svs::arch
