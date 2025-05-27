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

#include "svs/lib/exception.h"
#include "svs/third-party/eve.h"
#include <dlfcn.h>
#include <filesystem>

namespace svs::detail {

inline bool is_avx2_supported() { return eve::is_supported(eve::avx2); }

inline bool is_avx512_supported() { return eve::is_supported(eve::avx512); }

inline auto load_shared_lib(const std::filesystem::path& f_dir = "./") {
    static auto selector = [&]() {
        void* handle = nullptr;
        if (!handle && is_avx512_supported())
            handle = dlopen((f_dir / "libsvs_shared_library_avx512.so").c_str(), RTLD_NOW);
        if (!handle && is_avx2_supported())
            handle = dlopen((f_dir / "libsvs_shared_library_avx2.so").c_str(), RTLD_NOW);
        if (!handle)
            handle = dlopen((f_dir / "libsvs_shared_library.so").c_str(), RTLD_NOW);

        if (!handle) {
            throw ANNEXCEPTION("Unable to load the shared library!");
        }
        return handle;
    }();
    return selector;
}

} // namespace svs::detail
