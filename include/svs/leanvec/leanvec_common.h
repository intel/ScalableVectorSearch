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

// svs
#include "svs/core/data.h"

// stl
#include <optional>
#include <string>
#include <string_view>
#include <variant>

// third-party
#include "fmt/core.h"

namespace svs {
namespace leanvec {

// Sentinel type to select an LVQ dataset as either the primary or secondary
// dataset for `LeanVec`.
template <size_t Bits> struct UsingLVQ {};

// Hoist out schemas for reuse while auto-loading.
inline constexpr std::string_view lean_dataset_schema = "leanvec_dataset";
inline constexpr lib::Version lean_dataset_save_version = lib::Version(0, 0, 0);
inline constexpr std::string_view fallback_schema = "leanvec_fallback";
inline constexpr lib::Version fallback_save_version = lib::Version(0, 0, 0);

namespace detail {

template <typename T> inline constexpr bool is_using_lvq_tag_v = false;
template <size_t N> inline constexpr bool is_using_lvq_tag_v<UsingLVQ<N>> = true;

} // namespace detail

// Compatible type parameters for LeanDatasets
template <typename T>
concept LeanCompatible = has_datatype_v<T> || detail::is_using_lvq_tag_v<T>;

} // namespace leanvec
} // namespace svs
