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

// header under test
#include "svs/lib/avx_detection.h"

// catch2
#include "catch2/catch_test_macros.hpp"
#include <iostream>

CATCH_TEST_CASE("AVX detection", "[lib][lib-avx-detection]") {
    std::cout << "Checking AVX availability...\n";
    std::cout << "AVX2: " << std::boolalpha
              << svs::detail::avx_runtime_flags.is_avx2_supported() << "\n";
    std::cout << "AVX512F: " << std::boolalpha
              << svs::detail::avx_runtime_flags.is_avx512f_supported() << "\n";
    std::cout << "AVX512VNNI: " << std::boolalpha
              << svs::detail::avx_runtime_flags.is_avx512vnni_supported() << "\n";
}
