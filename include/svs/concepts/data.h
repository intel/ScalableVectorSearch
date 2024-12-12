/*
 * Copyright 2023 Intel Corporation
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

// !! NOTICE TO MAINTAINERS !!
//
// Due to limitations in the Doxygen -> Breathe -> Sphinx interface, support for documenting
// C++ 20 Concepts is limited.
//
// For now, we need to put the member-wise documentation in a code block in the main
// concept docstring.
//
// Hopefully this situation changes in future versions of these systems.

///
/// @ingroup concepts
/// @defgroup data_concept_public Helper utilities for dataset modeling.
///

///
/// @ingroup concepts
/// @defgroup data_concept_entry The main concepts modeling datasets.
///

#include "svs/lib/exception.h"
#include "svs/lib/type_traits.h"
#include "svs/third-party/fmt.h"

#include <concepts>
#include <cstddef>
#include <cstdint>

namespace svs {
namespace data {

// clang-format off

///
/// @ingroup data_concept_public
/// @brief Require the type aliases ``value_type`` and ``const_value_type``.
///
/// The members and inline documentation are given in the code snippet below.
///
/// @code{.cpp}
/// template<typename T>
/// concept HasValueType = requires {
///    // Require that ``T`` has the type alias ``T::value_type``.
///    //
///    // Note that the the alias does not necessarily need to be a "value_type" in the
///    // sence of C++ value type. In other words, it can (and probably should be for
///    // performance reasons be a reference).
///    typename T::value_type;
///
///    // Require that ``T`` has the type alias ``T::const_value_type``.
///    //
///    // Note that the the alias does not necessarily need to be a "value_type" in the
///    // sence of C++ value type. In other words, it can (and probably should be for
///    // performance reasons be a reference).
///    typename T::const_value_type;
/// };
/// @endcode
///
template<typename T>
concept HasValueType = requires {
    typename T::value_type;
    typename T::const_value_type;
};
// clang-format on

///
/// @ingroup data_concept_public
/// @brief Get the ``value_type`` of ``T``.
///
template <HasValueType T> using value_type_t = typename T::value_type;

///
/// @ingroup data_concept_public
/// @brief Get the ``const_value_type`` of ``T``.
///
template <HasValueType T> using const_value_type_t = typename T::const_value_type;

// clang-format off

///
/// @ingroup data_concept_entry
/// @brief Compatibility interface for methods working with datasets.
///
/// @code{.cpp}
/// template <typename T>
/// concept ImmutableMemoryDataset = HasValueType<T> && requires(const T a, size_t i) {
///     // Return the number of valid entries in the dataset.
///     { a.size() } -> std::convertible_to<size_t>;
///
///     // @brief Return the number of dimensions for each entry in the dataset.
///     //
///     // **Note**: The existence of this method is targeted for deprecation. It assumes
///     // that // all elements in the dataset have uniform dimensionality, which may not be
///     // the case in future workloads.
///     { a.dimensions() } -> std::convertible_to<size_t>;
///
///     // Return a constant handl to the element at index ``i``.
///     { a.get_datum(i) } -> std::same_as<const_value_type_t<T>>;
///
///     // Performance optimization. Prefetch the data at index ``i``.
///     //
///     // This may be implemented as a no-op if it is too difficult for a particular
///     // dataset instance. However, a correctly implemented ``prefetch`` can greatly
///     // improve performance.
///     a.prefetch(i);
/// }
/// @endcode
///
template <typename T>
concept ImmutableMemoryDataset = HasValueType<T> && requires(const T a, size_t i) {
    { a.size() } -> std::convertible_to<size_t>;
    { a.dimensions() } -> std::convertible_to<size_t>;
    { a.get_datum(i) } -> std::same_as<const_value_type_t<T>>;
    a.prefetch(i);
};

///
/// @ingroup data_concept_entry
/// @brief Compatibility interface for working with mutable datasets.
///
/// Mutable datasets aren't required to be resizeable. Mutability here simply means "the
/// element values may change".
///
/// @code{.cpp}
/// template <typename T>
/// concept MemoryDataset = ImmutableMemoryDataset<T>
/// && requires(T a, size_t i, const_value_type_t<T> v) {
///     // Return a (potentially) mutable handle to the entry at index ``i``.
///     //
///     // **Note**: This method is targeted for deprecation. This is because data-sets
///     // may require more contextual information for writing or overwriting contents
///     // and updating a mutable reference may not be sufficient.
///     //
///     // (From Mark): I don't **think** any functions in the library use this
///     // particular method, instead opting to use the more power ``set_datum``.
///     { a.get_datum(i) } -> std::same_as<value_type_t<T>>;
///
///     // Overwrite the contents of the index ``i`` with ``v``.
///     { a.set_datum(i, v) };
/// };
/// @endcode
///
template <typename T>
concept MemoryDataset = ImmutableMemoryDataset<T>
&& requires(T a, size_t i, const_value_type_t<T> v) {
    { a.get_datum(i) } -> std::same_as<value_type_t<T>>;
    { a.set_datum(i, v) };
};
// clang-format on

///
/// Copy the contents of one data to another.
///
template <ImmutableMemoryDataset Input, MemoryDataset Output>
void copy(const Input& input, Output& output) {
    auto isize = input.size();
    auto osize = output.size();
    if (isize != osize) {
        throw ANNEXCEPTION(
            "Source of copy has {} elements while the destination has {}", isize, osize
        );
    }

    for (size_t i = 0; i < isize; ++i) {
        output.set_datum(i, input.get_datum(i));
    }
}

struct GetDatumAccessor {
    template <ImmutableMemoryDataset Data, std::integral I>
    SVS_FORCE_INLINE auto operator()(const Data& data, I i) const
        -> decltype(data.get_datum(i)) {
        return data.get_datum(i);
    }

    template <ImmutableMemoryDataset Data, std::integral I>
    SVS_FORCE_INLINE void prefetch(const Data& data, I i) const {
        data.prefetch(i);
    }
};

template <typename Accessor, typename Data>
concept AccessorFor = requires(Accessor& accessor, const Data& data, size_t i) {
                          accessor(data, i);
                          accessor.prefetch(data, i);
                      };

} // namespace data
} // namespace svs
