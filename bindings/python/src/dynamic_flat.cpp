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

// svs python bindings
#include "svs/python/dynamic_flat.h"
#include "svs/python/common.h"
#include "svs/python/core.h"
#include "svs/python/flat.h"
#include "svs/python/manager.h"

// svs
#include "svs/lib/dispatcher.h"
#include "svs/orchestrators/dynamic_flat.h"

// pybind
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// stl
#include <span>

/////
///// DynamicFlat
/////

namespace py = pybind11;
namespace svs::python::dynamic_flat {

namespace {

template <typename F> void for_standard_specializations(F&& f) {
#define X(Q, T, N) f.template operator()<Q, T, N>()
    // Pattern:
    // QueryType, DataType, Dimensionality, Enable Building
    // clang-format off
    X(float,   float,        Dynamic);
    X(float,   svs::Float16, Dynamic);
    X(uint8_t, uint8_t,      Dynamic);
    X(int8_t,  int8_t,       Dynamic);
    // clang-format on
#undef X
}

template <typename ElementType>
void add_points(
    svs::DynamicFlat& index,
    const py_contiguous_array_t<ElementType>& py_data,
    const py_contiguous_array_t<size_t>& ids,
    bool reuse_empty = false
) {
    if (py_data.ndim() != 2) {
        throw ANNEXCEPTION("Expected points to have 2 dimensions!");
    }
    if (ids.ndim() != 1) {
        throw ANNEXCEPTION("Expected ids to have 1 dimension!");
    }
    if (py_data.shape(0) != ids.shape(0)) {
        throw ANNEXCEPTION(
            "Expected IDs to be the same length as the number of rows in points!"
        );
    }
    index.add_points(data_view(py_data), std::span(ids.data(), ids.size()), reuse_empty);
}

/////
///// Initialize DynamicFlat from Numpy Array
/////
template <typename Q, typename T, size_t N>
svs::DynamicFlat assemble_from_array(
    svs::data::ConstSimpleDataView<T, N> view,
    svs::DistanceType distance_type,
    size_t n_threads
) {
    // Create the data storage
    auto data =
        svs::data::SimpleData<T, N, RebindAllocator<T>>(view.size(), view.dimensions());
    svs::data::copy(view, data);

    // Use distance dispatcher to create the index with proper distance type
    auto dispatcher = svs::DistanceDispatcher(distance_type);
    return dispatcher([&](auto distance) {
        return svs::DynamicFlat::assemble<Q>(std::move(data), distance, n_threads);
    });
}
svs::DynamicFlat assemble_from_array(
    AnonymousVectorData data, svs::DistanceType distance_type, size_t n_threads
) {
    auto dispatcher = svs::lib::
        Dispatcher<svs::DynamicFlat, AnonymousVectorData, svs::DistanceType, size_t>();
    for_standard_specializations([&dispatcher]<typename Q, typename T, size_t N>() {
        dispatcher.register_target(assemble_from_array<Q, T, N>);
    });
    return dispatcher.invoke(data, distance_type, n_threads);
}

template <typename QueryType, typename ElementType>
void add_assemble_specialization(py::class_<svs::DynamicFlat>& flat) {
    flat.def(
        py::init([](py_contiguous_array_t<ElementType> py_data,
                    svs::DistanceType distance_type,
                    size_t num_threads) {
            return assemble_from_array(
                AnonymousVectorData(py_data), distance_type, num_threads
            );
        }),
        py::arg("data"),
        py::arg("distance"),
        py::arg("num_threads") = 1,
        R"(
Construct a DynamicFlat index over the given data, returning a searchable index.

Args:
    data: The dataset to index. **NOTE**: SVS will maintain an internal copy of the
        dataset. This may change in future releases.
    distance: The distance type to use for this dataset.
    num_threads: Number of threads for index construction.
)"
    );
}

const char* ADD_POINTS_DOCSTRING = R"(
Add every point in ``points`` to the index, assigning the element-wise corresponding ID to
each point.

Args:
    points: A matrix of data whose rows, corresponding to points in R^n, will be added to
        the index.
    ids: Vector of ids to assign to each row in ``points``. Must have the same number of
        elements as ``points`` has rows.
    reuse_empty: A flag that determines whether to reuse empty entries that may exist after deletion and consolidation. When enabled,
    scan from the beginning to find and fill these empty entries when adding new points.

Furthermore, all entries in ``ids`` must be unique and not already exist in the index.
If either of these does not hold, an exception will be thrown without mutating the
underlying index.

When ``delete_entries`` is called, a soft deletion is performed, marking the entries as ``deleted``.
When ``consolidate`` is called, the state of these deleted entries becomes ``empty``.
When ``add_points`` is called with the ``reuse_empty`` flag enabled, the memory is scanned from the beginning to locate and fill these empty entries with new points.
)";

template <typename ElementType>
void add_points_specialization(py::class_<svs::DynamicFlat>& index) {
    index.def(
        "add",
        &add_points<ElementType>,
        py::arg("points"),
        py::arg("ids"),
        py::arg("reuse_empty") = false,
        ADD_POINTS_DOCSTRING
    );
}

///// Docstrings
// Put docstrings here to hopefully make the implementation of `wrap` a bit less
// cluttered.
const char* CONSOLIDATE_DOCSTRING = R"(
Remove and patch around all deleted entries in the data.
Should be called after a sufficient number of deletions to avoid the memory consumption of
the index monotonically increasing.
)";

const char* COMPACT_DOCSTRING = R"(
Remove any holes created in the data by renumbering internal IDs.
Shrink the underlying data structures.
Following ``consolidate``, this can potentially reduce the memory footprint of the index
if a sufficient number of points were deleted.
)";

const char* DELETE_DOCSTRING = R"(
Soft delete the IDs from the index. Soft deletion does not remove the IDs from the data,
but prevents them from being returned from future searches.

Args:
    ids: The IDs to delete.

Each element in IDs must be unique and must correspond to a valid ID stored in the index.
Otherwise, an exception will be thrown. If an exception is thrown for this reason, the
index will be left unchanged from before the function call.
)";

const char* ALL_IDS_DOCSTRING = R"(
Return a Numpy vector of all IDs currently in the index.
)";

// Index saving.
void save_index(const svs::DynamicFlat& index, const std::string& data_dir) {
    index.save(data_dir);
}

/////
///// Assembly
/////

template <typename Q, typename T, size_t N>
svs::DynamicFlat assemble_uncompressed(
    svs::VectorDataLoader<T, N, RebindAllocator<T>> datafile,
    svs::DistanceType distance_type,
    size_t num_threads
) {
    auto dispatcher = svs::DistanceDispatcher(distance_type);
    return dispatcher([&](auto distance) {
        return svs::DynamicFlat::assemble<Q>(datafile, distance, num_threads);
    });
}

template <typename Dispatcher> void register_assembly(Dispatcher& dispatcher) {
    for_standard_specializations([&]<typename Q, typename T, size_t N>() {
        dispatcher.register_target(&assemble_uncompressed<Q, T, N>);
    });
}

using DynamicFlatAssembleTypes = std::variant<UnspecializedVectorDataLoader>;

svs::DynamicFlat assemble(
    DynamicFlatAssembleTypes data_loader,
    svs::DistanceType distance_type,
    svs::DataType SVS_UNUSED(query_type),
    bool SVS_UNUSED(enforce_dims),
    size_t num_threads
) {
    auto dispatcher = svs::lib::
        Dispatcher<svs::DynamicFlat, DynamicFlatAssembleTypes, svs::DistanceType, size_t>();

    register_assembly(dispatcher);
    return dispatcher.invoke(std::move(data_loader), distance_type, num_threads);
}

} // namespace

void wrap(py::module& m) {
    std::string name = "DynamicFlat";
    py::class_<svs::DynamicFlat> flat(
        m, name.c_str(), "Top level class for the dynamic Flat exhaustive search index."
    );

    add_search_specialization<float>(flat);
    add_threading_interface(flat);
    add_data_interface(flat);

    // Dynamic interface.
    flat.def("consolidate", &svs::DynamicFlat::consolidate, CONSOLIDATE_DOCSTRING);
    flat.def("compact", &svs::DynamicFlat::compact, COMPACT_DOCSTRING);

    // Reloading
    flat.def(
        py::init(&assemble),
        py::arg("data_loader"),
        py::arg("distance") = svs::L2,
        py::arg("query_type") = svs::DataType::float32,
        py::arg("enforce_dims") = false,
        py::arg("num_threads") = 1
    );

    // Array construction - similar to flat.cpp
    add_assemble_specialization<float, svs::Float16>(flat);
    add_assemble_specialization<float, float>(flat);
    add_assemble_specialization<uint8_t, uint8_t>(flat);
    add_assemble_specialization<int8_t, int8_t>(flat);

    // Index modification.
    add_points_specialization<float>(flat);

    // Index Deletion.
    flat.def(
        "delete",
        [](svs::DynamicFlat& index, const py_contiguous_array_t<size_t>& ids) {
            index.delete_points(as_span(ids));
        },
        py::arg("ids"),
        DELETE_DOCSTRING
    );

    // ID inspection
    flat.def(
        "has_id",
        &svs::DynamicFlat::has_id,
        py::arg("id"),
        "Return whether the ID exists in the index."
    );

    flat.def(
        "all_ids",
        [](const svs::DynamicFlat& index) {
            const auto& v = index.all_ids();
            // Populate a numpy-set
            auto npv = numpy_vector<size_t>(v.size());
            std::copy(v.begin(), v.end(), npv.mutable_unchecked().mutable_data());
            return npv;
        },
        ALL_IDS_DOCSTRING
    );

    // Saving
    flat.def(
        "save",
        &save_index,
        py::arg("data_directory"),
        R"(
Save a constructed index to disk (useful following index construction).

Args:
    data_directory: Directory where the dataset will be saved.

If the directory does not exist, it will be created if its parent exists.

It is the caller's responsibility to ensure that no existing data will be
overwritten when saving the index to this directory.
    )"
    );
}

} // namespace svs::python::dynamic_flat
