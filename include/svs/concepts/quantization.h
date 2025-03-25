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
#include <cstddef>

#include <concepts>
#include <variant>
namespace svs::quantization {

namespace detail {

// Trait to identify whether a type has `uses_compressed_data`
template <typename T, typename = void> struct compressed_data_trait : std::false_type {};

// Specialization for types that have `uses_compressed_data == true`
template <typename T>
struct compressed_data_trait<T, std::void_t<decltype(T::uses_compressed_data)>>
    : std::bool_constant<T::uses_compressed_data> {};

template <typename T>
inline constexpr bool compressed_data_trait_v = compressed_data_trait<T>::value;

} // namespace detail

template <typename T>
concept IsCompressedData = detail::compressed_data_trait_v<T>;

} // namespace svs::quantization
