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

#include <svs/runtime/version.h>
#include "catch2/catch_test_macros.hpp"

// Validate macro is defined (like FAISS does)
#ifndef FAISS_SVS_RUNTIME_VERSION
#error "FAISS_SVS_RUNTIME_VERSION is not defined"
#endif

// Create namespace alias (FAISS integration pattern)
SVS_RUNTIME_CREATE_API_ALIAS(svs_runtime, FAISS_SVS_RUNTIME_VERSION);

CATCH_TEST_CASE("Version Namespace Compatibility", "[runtime][version]") {
    CATCH_SECTION("Namespace alias resolves to v0") {
        // Both access methods should give same values
        CATCH_REQUIRE(svs_runtime::VersionInfo::major == svs::runtime::v0::VersionInfo::major);
        CATCH_REQUIRE(svs_runtime::VersionInfo::minor == svs::runtime::v0::VersionInfo::minor);
        CATCH_REQUIRE(svs_runtime::VersionInfo::patch == svs::runtime::v0::VersionInfo::patch);
    }
    
    CATCH_SECTION("Version compatibility check") {
        // Should be compatible with v0
        CATCH_REQUIRE(svs_runtime::VersionInfo::is_compatible_with_major(0));
        
        // Should not be compatible with v1
        CATCH_REQUIRE_FALSE(svs_runtime::VersionInfo::is_compatible_with_major(1));
    }
    
    CATCH_SECTION("Version string") {
        // Verify version string is accessible
        CATCH_REQUIRE(svs_runtime::VersionInfo::get_version() != nullptr);
        CATCH_REQUIRE(svs_runtime::VersionInfo::get_api_namespace() != nullptr);
    }
}
