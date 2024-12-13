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

///
/// @ingroup core
/// @defgroup core_recall_top Recall computation.
///

///
/// @ingroup core
/// @defgroup core_recall_public Recall computation.
///

#include "svs/core/data.h"
#include "svs/core/query_result.h"

#include "svs/lib/array.h"
#include "svs/lib/exception.h"
#include "svs/lib/narrow.h"

#include <numeric>

namespace svs {

//
// Preconditions:
// * `groundtruth.size() == results.size()`
// * `k <= n`
// * `k <= groundtruth.dimensions()`
// * `n <= results.dimensions()`
//
template <data::ImmutableMemoryDataset Groundtruth, data::ImmutableMemoryDataset Results>
double k_recall_at_n_impl_nocheck(
    const Groundtruth& groundtruth, const Results& results, size_t k, size_t n
) {
    const size_t npoints = groundtruth.size();
    size_t count = 0;
    for (size_t i = 0; i < npoints; ++i) {
        const auto lhs_begin = groundtruth.get_datum(i).begin();
        const auto rhs_begin = results.get_datum(i).begin();
        count += lib::count_intersect(lhs_begin, lhs_begin + k, rhs_begin, rhs_begin + n);
    }
    return lib::narrow_cast<double>(count) / (k * npoints);
}

template <data::ImmutableMemoryDataset Groundtruth, data::ImmutableMemoryDataset Results>
double k_recall_at_n_impl(
    const Groundtruth& groundtruth, const Results& results, size_t k = 0, size_t n = 0
) {
    const size_t nmax = results.dimensions();
    // N.B.: Explicitly using `nmax` here to default to `n-recall@n` where `n` is the number
    // of neighbors returned in the results.
    k = (k == 0) ? nmax /* <-- Not a typo */ : k;
    n = (n == 0) ? nmax : n;

    if (groundtruth.size() != results.size()) {
        throw ANNEXCEPTION(
            "Groundtruth contains {} points while the result has {} points!",
            groundtruth.size(),
            results.size()
        );
    }

    auto throw_exception = [](const char* aname, size_t a, const char* bname, size_t b) {
        throw ANNEXCEPTION(
            "Argument {} ({}) must be less than {} ({})", aname, a, bname, b
        );
    };

    // Check invariants.
    if (k > n) {
        throw_exception("k", k, "n", n);
    }

    const size_t kmax = groundtruth.dimensions();
    if (k > kmax) {
        throw_exception("k", k, "groundtruth entries", kmax);
    }
    if (n > nmax) {
        throw_exception("n", k, "result entries", kmax);
    }
    return k_recall_at_n_impl_nocheck(groundtruth, results, k, n);
}

///
/// @ingroup core_recall_public
/// @brief Placeholder for conversion to a ``svs::data::ImmutableMemoryDataset``.
///
/// Classes already implementing ``svs::data::ImmutableMemoryDataset`` are passed
/// through by default unless a more specialized overload exists.
///
/// The element types of the values inside the dataset must be integers. Otherwise, a
/// compile-time exception will be thrown.
///
template <data::ImmutableMemoryDataset Data>
// requires std::is_integral_v<typename data::value_type_t<Data>::value_type>
const Data& recall_convert(const Data& x) {
    using ValueValueType = typename data::value_type_t<Data>::value_type;
    static_assert(
        std::is_integral_v<ValueValueType>, "Data vector components must be integers!"
    );
    return x;
}

///
/// @ingroup core_recall_public
/// @brief Create a ``data::ConstSimpleDataView`` aliasing the contents of matrix ``x``.
///
/// @param x The matrix to alias.
///
/// @returns A ``data::ConstSimpleDataView`` aliasing the memory owned by ``x``.
///
template <std::integral T, typename Dims, typename Base>
    requires(std::tuple_size_v<Dims> == 2)
data::ConstSimpleDataView<T> recall_convert(const DenseArray<T, Dims, Base>& x) {
    return data::ConstSimpleDataView<T>(x.data(), getsize<0>(x), getsize<1>(x));
}

///
/// @ingroup core_recall_public
/// @brief Extract the `ids` in the query result as a ``data::ConstSimpleDataView``.
///
/// @param x The query result to view.
///
template <std::integral Idx, template <typename> typename Array>
data::ConstSimpleDataView<Idx> recall_convert(const QueryResultImpl<Idx, Array>& x) {
    return recall_convert(x.indices());
}

///
/// @ingroup core_recall_top
/// @brief Compute the ``k-recall@n`` for ``results`` with respect to ``groundtruth``
///
/// @param groundtruth The groundtruth of nearest neighbors.
/// @param results The actual computed results.
/// @param k The number of groundtruth neighbors to consider (optional).
///     Defaults to `results.dimensions()`.
/// @param n The number of results to consider (optional).
///     Defaults to `results.dimensions()`.
///
/// @returns The average `k-recall@n` for all entries pairwise matchings in ``groundtruth``
///     and ``results``.
///
/// Computes and returns the average of the ``k`` actual nearest neighbors found in the
/// groundtruth with the ``n`` top entries in ``results``.
///
/// If the parameters ``k`` and ``n`` are left off, this defaults to computing the
/// ``n-recall@n`` where ``n == results.dimensions()``.
///
/// **Preconditions:**
///
/// * ``groundtruth.size() == results.size()``. Same number of groundtruth elements as
///   result elements.
/// * ``k <= groundtruth.dimensions()``. If `k == 0`, then
///   ``results.dimensions() <= groundtruth.dimensions()``.
/// * ``n <= results.dimensions()``.
/// * ``k <= n``.
///
/// ### Type Restrictions
///
/// The types for ``groundtruth`` and ``results`` must be convertible to
/// ``svs::data::ImmutableMemoryDataset`` using the ``svs::recall_convert`` conversion
/// routine.
///
template <typename Groundtruth, typename Results>
double k_recall_at_n(
    const Groundtruth& groundtruth, const Results& results, size_t k = 0, size_t n = 0
) {
    return k_recall_at_n_impl(recall_convert(groundtruth), recall_convert(results), k, n);
}
} // namespace svs
