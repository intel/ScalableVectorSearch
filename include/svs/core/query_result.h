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

#include "svs/core/io/vecs.h"
#include "svs/lib/array.h"
#include "svs/lib/neighbor.h"

#include <cassert>
#include <type_traits>

namespace svs {

namespace detail {
template <std::integral I, typename Dims, typename Base>
void save_vecs(DenseArray<I, Dims, Base>& matrix, const std::string& filename) {
    auto writer = io::vecs::VecsWriter<uint32_t>{filename, getsize<1>(matrix)};
    for (size_t i = 0; i < getsize<0>(matrix); ++i) {
        writer << matrix.slice(i);
    }
}
} // namespace detail

///
/// @brief Struct containing result indices and distances for a query batch.
///
/// A struct holding the results of a Query, containing the indices of the nearest neighbors
/// and the distances.
///
/// The second type parameter is expected to be one of two arguments:
///
/// (1) `Matrix` - in this case `QueryResultImpl` owns its data. Alias: `QueryResult<Idx>`.
/// (2) `MatrixView` - in the case `QueryResultImpl` does not own its data. Alias:
///     `QueryResultView<Idx>`
///
template <typename Idx, template <typename> typename Array = Matrix> class QueryResultImpl {
  public:
    QueryResultImpl() = default;

    ///
    /// @brief Construct an uninitialized QueryResult of the given size.
    ///
    /// @param n_queries The number of queries to use.
    /// @param n_neighbors The number of neighbors to return for each query.
    ///
    /// Creates an uninitialized QueryResult with the capacity to hold the indices and
    /// distances for `n_neighbors` nearest neighbors for `n_queries` queries.
    ///
    QueryResultImpl(size_t n_queries, size_t n_neighbors)
        : distances_{make_dims(n_queries, n_neighbors)}
        , indices_{make_dims(n_queries, n_neighbors)} {}

    ///
    /// @brief Construct a QueryResultImpl from indices and distance directly.
    ///
    /// @param indices The indices array to use.
    /// @param distances The distances array to use.
    ///
    /// **Preconditions:**
    ///
    /// * `indices.ndims() == 2` (compiler time failure if not).
    /// * `distances.ndims() == 2` (compiler time failure if not).
    /// * `indices.dims() == distances.dims()`.
    ///
    /// The constructed QueryResultImpl will have the following properties.
    /// * `n_queries() == getsize<0>(indices) == getsize<0>(distances)`.
    /// * `n_neighbors() == getsize<1>(indices) == getsize<1>(distances)`.
    ///
    /// **Note**: This constructor provides a way using externally-supplied pointers for
    /// the contents of the QueryResultImpl by using array views.
    ///
    QueryResultImpl(Array<Idx> indices, Array<float> distances)
        : distances_{std::move(distances)}
        , indices_{std::move(indices)} {
        static_assert(indices.ndims() == 2, "Indices must be a 2-dimensional array!");
        static_assert(distances.ndims() == 2, "Indices must be a 2-dimensional array!");
    }

    /// Return the number of queries this instance has neighbors for.
    size_t n_queries() const { return getsize<0>(distances_); }
    /// Return the number of neighbors this instance can hold for each query.
    size_t n_neighbors() const { return getsize<1>(distances_); }

    /// Return the indices container directly.
    const Array<Idx>& indices() const { return indices_; }
    /// @copydoc indices() const
    Array<Idx>& indices() { return indices_; }

    /// Return the distances container directly.
    const Array<float>& distances() const { return distances_; }
    /// @copydoc distances() const
    Array<float>& distances() { return distances_; }

    ///
    /// @brief Return the ID for the requested position.
    ///
    /// @param query The query index to lookup. Must be in `[0, n_queries())`.
    /// @param neighbor The neighbor number to return. Must be in `[0, n_neighbors())`.
    ///
    const Idx& index(size_t query, size_t neighbor) const {
        return indices_.at(query, neighbor);
    }
    /// @copydoc QueryResultImpl::index(size_t,size_t) const
    Idx& index(size_t query, size_t neighbor) { return indices_.at(query, neighbor); }

    ///
    /// @brief Return the ID for the requested position.
    ///
    /// @param query The query index to lookup. Must be in `[0, n_queries())`.
    /// @param neighbor The neighbor number to return. Must be in `[0, n_neighbors())`.
    ///
    const float& distance(size_t query, size_t neighbor) const {
        return distances_.at(query, neighbor);
    }
    /// @copydoc distance(size_t,size_t) const
    float& distance(size_t query, size_t neighbor) {
        return distances_.at(query, neighbor);
    }

    template <NeighborLike Neighbor>
    void set(const Neighbor& neighbor, size_t query_index, size_t neighbor_index) {
        index(query_index, neighbor_index) = neighbor.id();
        distance(query_index, neighbor_index) = neighbor.distance();
    }

    ///
    /// @brief Return a non-owning view the underlying data structures.
    ///
    /// Often, especially when dealing with type-erased interfaces, it is useful to have a
    /// single concrete type as a parameter. The `view` mechanisms performs that for the
    /// QueryResult, taking any particular specialization (i.e., different allocators) and
    /// creating a non-owning view of the underlying arrays.
    ///
    QueryResultImpl<Idx, MatrixView> view() {
        return QueryResultImpl<Idx, MatrixView>(indices_.view(), distances_.view());
    }

    ///
    /// @brief Save the indices in `ivecs` form.
    ///
    /// @param filename The file path where the nearest neighbors will be saved.
    ///
    void save_vecs(const std::string& filename) { detail::save_vecs(indices_, filename); }

  private:
    Array<float> distances_;
    Array<Idx> indices_;
};

///
/// @brief Default type alias for general RAII style query results.
///
template <typename Idx> using QueryResult = QueryResultImpl<Idx, Matrix>;

///
/// @brief Default type alias for query results viewing a span of memory.
///
template <typename Idx> using QueryResultView = QueryResultImpl<Idx, MatrixView>;

} // namespace svs
