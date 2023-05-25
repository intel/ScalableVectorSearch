/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

#pragma once

// Bindings
#include "common.h"

// SVS
#include "svs/orchestrators/manager.h"

// PyBind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// stdlib
#include <concepts>

template <typename QueryType, typename Manager>
pybind11::tuple py_search(
    Manager& self,
    pybind11::array_t<QueryType, pybind11::array::c_style> queries,
    size_t n_neighbors
) {
    const size_t n_queries = queries.shape(0);
    const auto query_data = data_view(queries);
    auto result_idx = numpy_matrix<size_t>(n_queries, n_neighbors);
    auto result_dists = numpy_matrix<float>(n_queries, n_neighbors);
    svs::QueryResultView<size_t> q_result(
        matrix_view(result_idx), matrix_view(result_dists)
    );
    self.search(query_data, n_neighbors, q_result);
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
    queries: Numpy Matrix representing the query batch. Individual queries are assumed to
        the rows of the matrix. Returned results will have a position-wise correspondence
        with the queries. That is, the `N`-th row of the returned IDs and distances will
        correspond to the `N`-th row in the query matrix.

    n_neighbors: The number of neighbors to return for this search job.

Returns:
    A tuple `(I, D)` where `I` contains the `n_neighbors` approximate (or exact) nearest
    neighbors to the queries and `D` contains the approximate distances.
        )"
    );
}

template <typename Manager>
void add_threading_interface(pybind11::class_<Manager>& manager) {
    manager.def_property_readonly("can_change_threads", &Manager::can_change_threads);
    manager.def_property(
        "num_threads",
        &Manager::get_num_threads,
        &Manager::set_num_threads,
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
}
