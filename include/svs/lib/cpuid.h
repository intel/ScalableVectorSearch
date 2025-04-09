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

#include <cpuid.h>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace svs::lib {

inline bool intel_enabled() {
    uint32_t eax, ebx, ecx, edx;
    __cpuid(0, eax, ebx, ecx, edx);
    std::string vendor_id = std::string((const char*)&ebx, 4) +
        std::string((const char*)&edx, 4) +
        std::string((const char*)&ecx, 4);
    return vendor_id == "GenuineIntel";
}

} // namespace svs::lib
