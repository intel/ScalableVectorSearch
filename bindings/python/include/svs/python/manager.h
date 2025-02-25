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

// SVS python bindings
#include "svs/python/common.h"

// SVS
#include "svs/orchestrators/manager.h"

// PyBind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// stdlib
#include <concepts>

namespace svs::python {
template <typename QueryType, typename Manager>
pybind11::tuple py_search(
    Manager& self,
    pybind11::array_t<QueryType, pybind11::array::c_style> queries,
    size_t n_neighbors
) {
    const auto query_data = data_view(queries, allow_vectors);
    size_t n_queries = query_data.size();
    auto result_idx = numpy_matrix<size_t>(n_queries, n_neighbors);
    auto result_dists = numpy_matrix<float>(n_queries, n_neighbors);
    svs::QueryResultView<size_t> q_result(
        matrix_view(result_idx), matrix_view(result_dists)
    );

    svs::index::search_batch_into(self, q_result, query_data.cview());
    return pybind11::make_tuple(result_idx, result_dists);
}

template <typename QueryType, typename Manager>
void add_search_specialization(pybind11::class_<Manager>& py_manager) {
    py_manager.def(
        "search",
        [](Manager& self,
           pybind11::array_t<QueryType, pybind11::array::c_style> queries,
           size_t n_neighbors) { return py_search<QueryType>(self, queries, n_neighbors); },
        pybind11::arg("queries"),
        pybind11::arg("n_neighbors"),
        R"(
Perform a search to return the `n_neighbors` approximate nearest neighbors to the query.

Args:
    queries: Numpy Vector or Matrix representing the queries.
        If the argument is a vector, it will be treated as a single query.
        If the argument is a matrix, individual queries are assumed to the rows of the
        matrix. Returned results will have a position-wise correspondence
        with the queries. That is, the `N`-th row of the returned IDs and distances will
        correspond to the `N`-th row in the query matrix.

    n_neighbors: The number of neighbors to return for this search job.

Returns:
    A tuple `(I, D)` where `I` contains the `n_neighbors` approximate (or exact) nearest
    neighbors to the queries and `D` contains the approximate distances.

    Note: This form is returned regardless of whether the given query was a vector or a
    matrix.
        )"
    );
}

template <typename Manager>
void add_threading_interface(pybind11::class_<Manager>& manager) {
    manager.def_property(
        "num_threads",
        &Manager::get_num_threads,
        [](Manager& self, int num_threads) {
            self.set_threadpool(svs::threads::DefaultThreadPool(num_threads));
        },
        "Read/Write (int): Get and set the number of threads used to process queries."
    );
}

template <typename Manager> void add_data_interface(pybind11::class_<Manager>& manager) {
    manager.def_property_readonly(
        "size", &Manager::size, "Return the number of elements in the indexed dataset."
    );
    manager.def_property_readonly(
        "dimensions",
        &Manager::dimensions,
        "Return the logical number of dimensions for each vector in the dataset."
    );
    manager.def_property_readonly(
        "query_types",
        &Manager::query_types,
        "Return the query element types this index is specialized for."
    );
}
namespace detail {

template <typename Index>
py_contiguous_array_t<float>
reconstruct(Index& index, py_contiguous_array_t<uint64_t> ids) {
    auto data_dims = index.dimensions();
    const size_t num_ids = ids.size();
    // Create a flat buffer for the destination.
    // We will reshape is appropriately before returning.
    auto destination = py_contiguous_array_t<float>({num_ids, data_dims});
    index.reconstruct_at(
        mutable_data_view(destination),
        std::span<const uint64_t>(
            ids.template mutable_unchecked<-1>().mutable_data(), num_ids
        )
    );

    // Reshape the destination to have the same shape as the original IDs (plus the extra
    // dimension for the data vectors themselves.
    auto final_shape = std::vector<size_t>{};
    size_t ndim = svs::lib::narrow<size_t>(ids.ndim());
    for (size_t i = 0; i < ndim; ++i) {
        final_shape.push_back(ids.shape(i));
    }
    final_shape.push_back(data_dims);
    return destination.reshape({std::move(final_shape)});
}

} // namespace detail

template <typename Manager>
void add_reconstruct_interface(pybind11::class_<Manager>& manager) {
    manager.def("reconstruct", &detail::reconstruct<Manager>, pybind11::arg("ids"));
}
} // namespace svs::python
