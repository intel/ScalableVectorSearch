/*
 * Copyright 2026 Intel Corporation
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

// TODO: use cmakedefine and cmake configure_file
// #define SVS_C_API_VERSION ((MAJOR << 16) | (MINOR << 8) | PATCH)
#define SVS_C_API_VERSION ((0 << 16) | (4 << 8) | 0)
#define SVS_GET_VERSION_MAJOR(version) ((version >> 16) & 0xFF)
#define SVS_GET_VERSION_MINOR(version) ((version >> 8) & 0xFF)
#define SVS_GET_VERSION_PATCH(version) (version & 0xFF)

// All symbols shall be internal unless marked as SVS_API
#if defined _WIN32 || defined __CYGWIN__
#define SVS_HELPER_DLL_IMPORT __declspec(dllimport)
#define SVS_HELPER_DLL_EXPORT __declspec(dllexport)
#else
#if __GNUC__ >= 4
#define SVS_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#define SVS_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#else
#define SVS_HELPER_DLL_IMPORT
#define SVS_HELPER_DLL_EXPORT
#endif
#endif

#ifdef svs_c_api_EXPORTS
#define SVS_API SVS_HELPER_DLL_EXPORT
#else
#define SVS_API SVS_HELPER_DLL_IMPORT
#endif

// Mark an API as deprecated, optionally providing a message for callers.
#if defined _WIN32 || defined __CYGWIN__
#define SVS_DEPRECATED(msg) __declspec(deprecated(msg))
#elif defined __GNUC__ || defined __clang__
#define SVS_DEPRECATED(msg) __attribute__((deprecated(msg)))
#else
#define SVS_DEPRECATED(msg)
#endif
