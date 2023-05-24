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

// project local
#include "dynamic_vamana.h"
#include "common.h"
#include "core.h"
#include "manager.h"
#include "vamana.h"

// svs
#include "svs/orchestrators/dynamic_vamana.h"

// pybind
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// stl
#include <span>

/////
///// DynamicVamana
/////

namespace py = pybind11;
namespace dynamic_vamana {

namespace {

template <typename ElementType>
svs::DynamicVamana build_from_array(
    const svs::index::vamana::VamanaBuildParameters& parameters,
    py_contiguous_array_t<ElementType> py_data,
    py_contiguous_array_t<size_t> py_ids,
    svs::DistanceType distance_type,
    size_t num_threads
) {
    auto dispatcher = svs::DistanceDispatcher(distance_type);
    return dispatcher([&](auto distance) {
        return svs::DynamicVamana::build<ElementType>(
            parameters,
            create_blocked_data(py_data),
            std::span(py_ids.data(), py_ids.size()),
            distance,
            num_threads
        );
    });
}

const char* BUILD_FROM_ARRAY_DOC = R"(
Construct a Vamana index over the given data, returning a searchable index.

Args:
    data: The dataset to index. **NOTE**: PySVS will maintain an internal copy of the
        dataset. This may change in future releases.
    parameters: Parameters controlling graph construction.
        See below for the documentation of this class.
    distance_type: The distance type to use for this dataset.
)";

template <typename ElementType>
void add_build_specialization(py::class_<svs::DynamicVamana>& index) {
    index.def_static(
        "build",
        &build_from_array<ElementType>,
        py::arg("parameters"),
        py::arg("data"),
        py::arg("ids"),
        py::arg("distance_type"),
        py::arg("num_threads"),
        BUILD_FROM_ARRAY_DOC
    );
}

template <typename ElementType>
void add_points(
    svs::DynamicVamana& index,
    const py_contiguous_array_t<ElementType>& py_data,
    const py_contiguous_array_t<size_t>& ids
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
    index.add_points(data_view(py_data), std::span(ids.data(), ids.size()));
}

const char* ADD_POINTS_DOCSTRING = R"(
Add every point in ``points`` to the index, assigning the element-wise corresponding ID to
each point.

Args:
    points: A matrix of data whose rows, corresponding to points in R^n, will be added to
        the index.
    ids: Vector of ids to assign to each row in ``points``. Must have the same number of
        elements as ``points`` has rows.

Furthermore, all entries in ``ids`` must be unique and not already exist in the index.
If either of these does not hold, an exception will be thrown without mutating the
underlying index.
)";

template <typename ElementType>
void add_points_specialization(py::class_<svs::DynamicVamana>& index) {
    index.def(
        "add",
        &add_points<ElementType>,
        py::arg("points"),
        py::arg("ids"),
        ADD_POINTS_DOCSTRING
    );
}

///// Docstrings
// Put docstrings heere to hopefully make the implementation of `wrap` a bit less
// cluttered.
const char* CONSOLIDATE_DOCSTRING = R"(
Remove and patch around all deleted entries in the graph.
Should be called after a sufficient number of deletions to avoid the memory consumption of
the index monotonically increasing.
)";

const char* COMPACT_DOCSTRING = R"(
Remove any holes created in the graph and data by renumbering internal IDs.
Shrink the underlying data structures.
Following ``consolidate``, this can potentialy reduce the memory footprint of the index
if a sufficient number of points were deleted.
)";

const char* DELETE_DOCSTRING = R"(
Soft delete the IDs from the index. Soft deletion does not remove the IDs from the graph,
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

} // namespace

void wrap(py::module& m) {
    std::string name = "DynamicVamana";
    py::class_<svs::DynamicVamana> vamana(
        m, name.c_str(), "Top level class for the dynamic Vamana graph index."
    );

    add_search_specialization<float>(vamana);
    add_threading_interface(vamana);
    add_data_interface(vamana);

    // Vamana specific extentions.
    vamana::add_interface(vamana);

    // Dynamic interface.
    vamana.def_property(
        "alpha",
        &svs::DynamicVamana::get_alpha,
        &svs::DynamicVamana::set_alpha,
        "Read/Write (float): Get/set the alpha value used when adding and deleting points."
    );

    vamana.def_property(
        "construction_window_size",
        &svs::DynamicVamana::get_construction_window_size,
        &svs::DynamicVamana::set_construction_window_size,
        "Read/Write (int): Get/set the window size used when adding and deleting points."
    );

    vamana.def("consolidate", &svs::DynamicVamana::consolidate, CONSOLIDATE_DOCSTRING);
    vamana.def("compact", &svs::DynamicVamana::compact, COMPACT_DOCSTRING);

    // Index building.
    add_build_specialization<float>(vamana);

    // Index modification.
    add_points_specialization<float>(vamana);

    // Index Deletion.
    vamana.def(
        "delete",
        [](svs::DynamicVamana& index, const py_contiguous_array_t<size_t>& ids) {
            index.delete_points(as_span(ids));
        },
        py::arg("ids"),
        DELETE_DOCSTRING
    );

    // ID inspection
    vamana.def(
        "has_id",
        &svs::DynamicVamana::has_id,
        py::arg("id"),
        "Return whether the ID exists in the index."
    );

    vamana.def(
        "all_ids",
        [](const svs::DynamicVamana& index) {
            const auto& v = index.all_ids();
            // Populate a numpy-set
            auto npv = numpy_vector<size_t>(v.size());
            std::copy(v.begin(), v.end(), npv.mutable_unchecked().mutable_data());
            return npv;
        },
        ALL_IDS_DOCSTRING
    );
}

} // namespace dynamic_vamana
