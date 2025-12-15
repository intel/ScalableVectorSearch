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

///
/// @brief Version information and API versioning for SVS Runtime
///
/// This header defines the SVS Runtime API versioning scheme:
/// 1. Versioned namespaces (e.g., v0, v1) for API stability
/// 2. Using declarations to bring current version to parent namespace
/// 3. Clean integration points for external libraries
///
/// Usage:
/// - Users can access APIs via svs::runtime::FlatIndex (maps to current version)
/// - External integrators can use namespace aliases (e.g., namespace svs_api =
/// svs::runtime::v0)
/// - Specific versions can be accessed via svs::runtime::v0::FlatIndex
///

///// Version Numbers

#ifndef SVS_RUNTIME_VERSION_MAJOR
/// Major version number - incremented for breaking API changes
/// When this changes, a new version namespace (e.g., v0 -> v1) is created
#define SVS_RUNTIME_VERSION_MAJOR 0
#endif

#ifndef SVS_RUNTIME_VERSION_MINOR
/// Minor version number - incremented for backward-compatible feature additions
#define SVS_RUNTIME_VERSION_MINOR 1
#endif

#ifndef SVS_RUNTIME_VERSION_PATCH
/// Patch version number - incremented for backward-compatible bug fixes
#define SVS_RUNTIME_VERSION_PATCH 0
#endif

#ifndef SVS_RUNTIME_VERSION_STRING
/// Complete version string
#define SVS_RUNTIME_VERSION_STRING "0.1.0"
#endif

#ifndef SVS_RUNTIME_API_VERSION
/// Default to current major version if not specified by client
#define SVS_RUNTIME_API_VERSION SVS_RUNTIME_VERSION_MAJOR
#endif

#if (SVS_RUNTIME_API_VERSION == 0)
/// Use v0 API
/// Current API version namespace
#define SVS_RUNTIME_CURRENT_API_NAMESPACE v0
namespace svs {
namespace runtime {
/// Current API version namespace (v0)
/// All public runtime APIs live here and are accessible as svs::runtime::FunctionName
/// due to inline namespace
inline namespace v0 {
// Public runtime APIs will be defined in their respective headers
// IMPORTANT: include this header before other runtime headers to ensure proper versioning
}
} // namespace runtime
} // namespace svs
#else
#error "Unsupported SVS Runtime major version"
#endif

///// Integration Support

/// Helper macro to create namespace aliases for external integrators
/// Example: SVS_RUNTIME_CREATE_API_ALIAS(svs_runtime_api, v0)
/// creates: namespace svs_runtime_api = svs::runtime::v0;
#define SVS_RUNTIME_CREATE_API_ALIAS(alias_name, version_ns) \
    namespace alias_name = svs::runtime::version_ns

///
/// @brief Version information structure for runtime queries
///
namespace svs::runtime::v0 {

struct VersionInfo {
    static constexpr int major = SVS_RUNTIME_VERSION_MAJOR;
    static constexpr int minor = SVS_RUNTIME_VERSION_MINOR;
    static constexpr int patch = SVS_RUNTIME_VERSION_PATCH;
    static constexpr const char* version_string = SVS_RUNTIME_VERSION_STRING;
    static constexpr const char* api_namespace = "v0";

    /// Get the complete version as a string
    static const char* get_version() { return version_string; }

    /// Get the API namespace identifier
    static const char* get_api_namespace() { return api_namespace; }

    /// Check if this version is compatible with a requested major version
    static bool is_compatible_with_major(int requested_major) {
        return major == requested_major;
    }
};

} // namespace svs::runtime::v0
