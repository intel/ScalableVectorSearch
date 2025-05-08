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

#include "svs/lib/arch.h"
#include "svs/lib/cpuid.h"
#include <iostream>

int main() {
    std::ostream& out = std::cout;
    auto& arch_env = svs::arch::MicroArchEnvironment::get_instance();

    // Print support status for all ISA extensions
    svs::arch::write_extensions_status(out);

    // Print current microarchitecture
    auto current_arch = arch_env.get_microarch();
    out << "\nCurrent µarch: " << svs::arch::microarch_to_string(current_arch) << std::endl;

    // Print all supported microarchitectures
    const auto& supported_archs = arch_env.get_supported_microarchs();
    out << "\nSupported µarchs: ";
    for (const auto& arch : supported_archs) {
        out << svs::arch::microarch_to_string(arch) << " ";
    }
    out << std::endl;

    // Print all compiled microarchitectures
    const auto& compiled_archs = arch_env.get_compiled_microarchs();
    out << "\nCompiled µarchs: ";
    for (const auto& arch : compiled_archs) {
        out << svs::arch::microarch_to_string(arch) << " ";
    }
    out << std::endl;

    return 0;
}
