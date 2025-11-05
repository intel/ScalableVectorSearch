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
/// @file version.hpp
/// @brief SVS API versioning support for integration with external libraries like Faiss
///
/// This header defines the SVS API versioning scheme that allows:
/// 1. Stable API versions (e.g., v0, v1) with inline namespace support
/// 2. Clean integration points for external libraries
/// 3. Gradual migration between API versions
///
/// Usage:
/// - Public APIs are wrapped in SVS_VERSIONED_NAMESPACE_BEGIN/END
/// - Users can access APIs via svs::ClassName (maps to current version)
/// - External integrators can use namespace aliases (e.g., namespace svs_api = svs::v0)
///

///// Version Numbers

/// Major version number - incremented for breaking API changes
/// When this changes, a new version namespace (e.g., v0 -> v1) is created
#define SVS_VERSION_MAJOR 0

/// Minor version number - incremented for backward-compatible feature additions
#define SVS_VERSION_MINOR 1

/// Patch version number - incremented for backward-compatible bug fixes
#define SVS_VERSION_PATCH 0

/// Complete version string
#define SVS_VERSION_STRING "0.1.0"

///// API Version Namespace

/// The current API version namespace identifier
/// This defines which API generation is currently active
/// Example: v0 for the first stable API, v1 for the next major version, etc.
#define SVS_VERSION_NAMESPACE v0

///// Namespace Macros

/// Begin a versioned namespace block for public APIs
/// Use this to wrap public classes, functions, and types that should be
/// stable across minor/patch releases within the same major version
#define SVS_VERSIONED_NAMESPACE_BEGIN \
    namespace svs { \
    inline namespace SVS_VERSION_NAMESPACE {

/// End a versioned namespace block
#define SVS_VERSIONED_NAMESPACE_END \
    } /* end inline namespace SVS_VERSION_NAMESPACE */ \
    } /* end namespace svs */

///// Internal Namespace

/// Internal namespace for implementation details that are not part of the stable API
/// Items in this namespace can change freely without version bumps
#define SVS_INTERNAL_NAMESPACE_BEGIN \
    namespace svs { \
    namespace internal {

#define SVS_INTERNAL_NAMESPACE_END \
    } /* end namespace internal */ \
    } /* end namespace svs */

///// Version Namespace Declaration

/// Declare the main SVS namespace with inline version namespace
/// This makes svs::Foo automatically resolve to svs::v0::Foo (or current version)
namespace svs {
    /// Current API version namespace
    /// All public APIs live here and are accessible as svs::ClassName
    inline namespace SVS_VERSION_NAMESPACE {
        // Public APIs will be defined here via SVS_VERSIONED_NAMESPACE_BEGIN/END
    }
    
    /// Internal implementation details
    /// Not part of the stable API - can change freely
    namespace internal {
        // Internal helpers and implementation details
    }
}

///// Integration Support

/// Helper macro to create namespace aliases for external integrators
/// Example: SVS_CREATE_API_ALIAS(svs_api, v0) creates: namespace svs_api = svs::v0;
#define SVS_CREATE_API_ALIAS(alias_name, version_ns) \
    namespace alias_name = svs::version_ns

///
/// @brief Version information structure for runtime queries
///
SVS_VERSIONED_NAMESPACE_BEGIN

struct VersionInfo {
    static constexpr int major = SVS_VERSION_MAJOR;
    static constexpr int minor = SVS_VERSION_MINOR; 
    static constexpr int patch = SVS_VERSION_PATCH;
    static constexpr const char* version_string = SVS_VERSION_STRING;
    static constexpr const char* api_namespace = "v0"; // Should match SVS_VERSION_NAMESPACE
    
    /// Get the complete version as a string
    static const char* get_version() { return version_string; }
    
    /// Get the API namespace identifier
    static const char* get_api_namespace() { return api_namespace; }
    
    /// Check if this version is compatible with a requested major version
    static bool is_compatible_with_major(int requested_major) {
        return major == requested_major;
    }
};

SVS_VERSIONED_NAMESPACE_END