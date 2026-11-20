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

///
/// @ingroup concepts
/// @defgroup vector_set_concept_entry Concepts modeling late-interaction vector-set
/// datasets.
///

#include <concepts>
#include <cstddef>
#include <span>

namespace svs {
namespace data {

// clang-format off

///
/// @ingroup vector_set_concept_entry
/// @brief A variable-cardinality set of token vectors, as used by late-interaction
/// retrieval (e.g. one set per document, one token vector per word-piece embedding).
///
/// @code{.cpp}
/// template <typename S>
/// concept VectorSet = requires(const S& s, size_t i) {
///     // The element type of each token vector.
///     typename S::element_type;
///
///     // The number of token vectors in the set.
///     { s.cardinality() } -> std::convertible_to<size_t>;
///
///     // The dimensionality shared by every token vector in the set.
///     { s.dimensions() } -> std::convertible_to<size_t>;
///
///     // The token vector at index ``i``, with ``0 <= i < cardinality()``.
///     { s.token(i) } -> std::convertible_to<std::span<const typename S::element_type>>;
/// };
/// @endcode
///
template <typename S>
concept VectorSet = requires(const S& s, size_t i) {
    typename S::element_type;
    { s.cardinality() } -> std::convertible_to<size_t>;
    { s.dimensions() } -> std::convertible_to<size_t>;
    { s.token(i) } -> std::convertible_to<std::span<const typename S::element_type>>;
};

///
/// @ingroup vector_set_concept_entry
/// @brief Compatibility interface for a dataset whose elements are vector sets.
///
/// @code{.cpp}
/// template <typename D>
/// concept ImmutableVectorSetDataset = requires(const D& d, size_t i) {
///     // The value returned by ``get_datum`` must itself model ``VectorSet``.
///     requires VectorSet<typename D::const_value_type>;
///
///     // The number of vector sets in the dataset.
///     { d.size() } -> std::convertible_to<size_t>;
///
///     // The dimensionality shared by every token vector in the dataset.
///     { d.dimensions() } -> std::convertible_to<size_t>;
///
///     // A constant handle to the vector set at index ``i``.
///     { d.get_datum(i) } -> std::convertible_to<typename D::const_value_type>;
///
///     // Performance optimization. Prefetch the vector set at index ``i``.
///     d.prefetch(i);
/// };
/// @endcode
///
template <typename D>
concept ImmutableVectorSetDataset = requires(const D& d, size_t i) {
    requires VectorSet<typename D::const_value_type>;
    { d.size() } -> std::convertible_to<size_t>;
    { d.dimensions() } -> std::convertible_to<size_t>;
    { d.get_datum(i) } -> std::convertible_to<typename D::const_value_type>;
    d.prefetch(i);
};

///
/// @ingroup vector_set_concept_entry
/// @brief A ``VectorSet`` whose tokens are stored row-major and contiguous, reachable
/// from ``data()``.
///
template <typename S>
concept ContiguousVectorSet = VectorSet<S> && requires(const S& s) {
    { s.data() } -> std::convertible_to<const typename S::element_type*>;
};
// clang-format on

} // namespace data
} // namespace svs
