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
#include "svs/python/dynamic_ivf.h"
#include "svs/python/common.h"
#include "svs/python/core.h"
#include "svs/python/ivf.h"
#include "svs/python/manager.h"

// svs
#include "svs/lib/dispatcher.h"
#include "svs/orchestrators/dynamic_ivf.h"

// pybind
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// fmt
#include <fmt/format.h>

// stl
#include <span>

/////
///// DynamicIVF
/////

namespace py = pybind11;
namespace svs::python::dynamic_ivf {

namespace {

template <typename ElementType>
svs::DynamicIVF build_from_array(
    const svs::index::ivf::IVFBuildParameters& parameters,
    py_contiguous_array_t<ElementType> py_data,
    py_contiguous_array_t<size_t> py_ids,
    svs::DistanceType distance_type,
    size_t num_threads
) {
    auto dispatcher = svs::DistanceDispatcher(distance_type);
    return dispatcher([&](auto distance) {
        // Create a view for building - build_clustering needs immutable data
        // Note: Even though we use SimpleDataView (non-const), the data won't be modified
        // during clustering, and BlockedData created from it will have mutable element type
        auto data_view = data::SimpleDataView<ElementType>(
            const_cast<ElementType*>(py_data.data()), py_data.shape(0), py_data.shape(1)
        );
        return svs::DynamicIVF::build<ElementType>(
            parameters,
            data_view,
            std::span(py_ids.data(), py_ids.size()),
            distance,
            num_threads
        );
    });
}

const char* BUILD_FROM_ARRAY_DOC = R"(
Construct a DynamicIVF index over the given data, returning a searchable index.

Args:
    parameters: Parameters controlling IVF construction (clustering and search parameters).
        See below for the documentation of this class.
    data: The dataset to index. **NOTE**: SVS will maintain an internal copy of the
        dataset. This may change in future releases.
    ids: Vector of ids to assign to each row in the dataset; must match dataset length and contain unique values.
    distance_type: The distance type to use for this dataset.
    num_threads: Number of threads to use for index construction.
)";

template <typename ElementType>
void add_build_specialization(py::class_<svs::DynamicIVF>& index) {
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

/////
///// Build from file (data loader)
/////

template <typename Q, typename T, typename Dist, size_t N>
svs::DynamicIVF dynamic_ivf_build_uncompressed(
    const svs::index::ivf::IVFBuildParameters& parameters,
    svs::VectorDataLoader<T, N, RebindAllocator<T>> data_loader,
    std::span<const size_t> ids,
    svs::DistanceType distance_type,
    size_t num_threads
) {
    return svs::DynamicIVF::build<Q>(
        parameters, std::move(data_loader), ids, distance_type, num_threads
    );
}

using DynamicIVFBuildFromFileDispatcher = svs::lib::Dispatcher<
    svs::DynamicIVF,
    const svs::index::ivf::IVFBuildParameters&,
    UnspecializedVectorDataLoader,
    std::span<const size_t>,
    svs::DistanceType,
    size_t>;

DynamicIVFBuildFromFileDispatcher dynamic_ivf_build_from_file_dispatcher() {
    auto dispatcher = DynamicIVFBuildFromFileDispatcher{};
    // Register uncompressed specializations (Dynamic dimensionality only)
    for_standard_specializations([&]<typename Q, typename T, typename D, size_t N>() {
        auto method = &dynamic_ivf_build_uncompressed<Q, T, D, N>;
        dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
    });
    return dispatcher;
}

svs::DynamicIVF dynamic_ivf_build_from_file(
    const svs::index::ivf::IVFBuildParameters& parameters,
    UnspecializedVectorDataLoader data_loader,
    const py_contiguous_array_t<size_t>& py_ids,
    svs::DistanceType distance_type,
    size_t num_threads
) {
    auto ids = std::span<const size_t>(py_ids.data(), py_ids.size());
    return dynamic_ivf_build_from_file_dispatcher().invoke(
        parameters, std::move(data_loader), ids, distance_type, num_threads
    );
}

constexpr std::string_view DYNAMIC_IVF_BUILD_FROM_FILE_DOCSTRING_PROTO = R"(
Construct a DynamicIVF index using a data loader, returning the index.

Args:
    parameters: Build parameters controlling IVF construction (clustering and search parameters).
    data_loader: Data loader (e.g., a VectorDataLoader instance).
    ids: Vector of ids to assign to each row in the dataset; must match dataset length and contain unique values.
    distance_type: The similarity function to use for this index.
    num_threads: Number of threads to use for index construction. Default: 1.

Specializations compiled into the binary are listed below.

{}  # (Method listing auto-generated)
)";

template <typename ElementType>
void add_points(
    svs::DynamicIVF& index,
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

When ``delete`` is called, a soft deletion is performed, marking the entries as ``deleted``.
When ``consolidate`` is called, the state of these deleted entries becomes ``empty``.
When ``add_points`` is called with the ``reuse_empty`` flag enabled, the memory is scanned from the beginning to locate and fill these empty entries with new points.
)";

template <typename ElementType>
void add_points_specialization(py::class_<svs::DynamicIVF>& index) {
    index.def(
        "add",
        &add_points<ElementType>,
        py::arg("points"),
        py::arg("ids"),
        py::arg("reuse_empty") = false,
        ADD_POINTS_DOCSTRING
    );
}

const char* CONSOLIDATE_DOCSTRING = R"(
No-op method for compatibility with dynamic index interface.
For the IVF index, deletion marks entries as Empty and they are excluded from searches.
Empty slots can be reused when adding new points.
)";

const char* COMPACT_DOCSTRING = R"(
Remove any holes created in the data by renumbering internal IDs.
Shrink the underlying data structures.
This can potentially reduce the memory footprint of the index
if a sufficient number of points were deleted.
)";

const char* DELETE_DOCSTRING = R"(
Soft delete the IDs from the index. Soft deletion does not remove the IDs from the index,
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
void save_index(
    svs::DynamicIVF& index, const std::string& config_path, const std::string& data_dir
) {
    index.save(config_path, data_dir);
}

/////
///// Assembly
/////

template <typename Q, typename T, typename Dist, size_t N>
svs::DynamicIVF assemble_uncompressed(
    svs::VectorDataLoader<float, N, RebindAllocator<float>> centroids_loader,
    svs::VectorDataLoader<T, N, RebindAllocator<T>> datafile,
    std::span<const size_t> ids,
    Dist distance,
    size_t num_threads
) {
    using DataAlloc = RebindAllocator<T>;

    // Load centroids as SimpleData - they are immutable in IVF
    auto centroids = svs::data::SimpleData<float, N>::load(centroids_loader.path_);

    // Load data as BlockedData - it will grow/shrink with insertions/deletions
    auto data = svs::data::BlockedData<T, N, DataAlloc>::load(
        datafile.path_, as_blocked(DataAlloc(datafile.allocator_))
    );

    return svs::DynamicIVF::assemble<Q>(
        std::move(centroids), std::move(data), ids, distance, num_threads
    );
}

template <typename Dispatcher> void register_assembly(Dispatcher& dispatcher) {
    for_standard_specializations([&]<typename Q, typename T, typename D, size_t N>() {
        dispatcher.register_target(&assemble_uncompressed<Q, T, D, N>);
    });
}

using DynamicIVFAssembleTypes = std::variant<UnspecializedVectorDataLoader>;

svs::DynamicIVF assemble(
    DynamicIVFAssembleTypes centroids_loader,
    DynamicIVFAssembleTypes data_loader,
    const py_contiguous_array_t<size_t>& py_ids,
    svs::DistanceType distance_type,
    svs::DataType SVS_UNUSED(query_type),
    size_t num_threads
) {
    auto dispatcher = svs::lib::Dispatcher<
        svs::DynamicIVF,
        DynamicIVFAssembleTypes,
        DynamicIVFAssembleTypes,
        std::span<const size_t>,
        svs::DistanceType,
        size_t>();

    register_assembly(dispatcher);
    auto ids = std::span<const size_t>(py_ids.data(), py_ids.size());
    return dispatcher.invoke(
        std::move(centroids_loader), std::move(data_loader), ids, distance_type, num_threads
    );
}

} // namespace

void wrap(py::module& m) {
    std::string name = "DynamicIVF";
    py::class_<svs::DynamicIVF> ivf_index(
        m, name.c_str(), "Top level class for the dynamic IVF index."
    );

    add_search_specialization<float>(ivf_index);
    add_threading_interface(ivf_index);
    add_data_interface(ivf_index);

    // IVF specific extensions.
    ivf::add_interface(ivf_index);

    // Dynamic interface.
    ivf_index.def("consolidate", &svs::DynamicIVF::consolidate, CONSOLIDATE_DOCSTRING);
    ivf_index.def(
        "compact",
        &svs::DynamicIVF::compact,
        py::arg("batchsize") = 1'000'000,
        COMPACT_DOCSTRING
    );

    // Reloading/Assembly
    ivf_index.def(
        py::init(&assemble),
        py::arg("centroids_loader"),
        py::arg("data_loader"),
        py::arg("ids"),
        py::arg("distance") = svs::L2,
        py::arg("query_type") = svs::DataType::float32,
        py::arg("num_threads") = 1
    );

    // Index building.
    add_build_specialization<float>(ivf_index);

    // Build from file / data loader (dynamic docstring)
    {
        auto dispatcher = dynamic_ivf_build_from_file_dispatcher();
        std::string dynamic;
        for (size_t i = 0; i < dispatcher.size(); ++i) {
            fmt::format_to(
                std::back_inserter(dynamic),
                R"(Method {}:\n    - data_loader: {}\n    - distance: {}\n)",
                i,
                dispatcher.description(i, 1),
                dispatcher.description(i, 3)
            );
        }
        ivf_index.def_static(
            "build",
            &dynamic_ivf_build_from_file,
            py::arg("parameters"),
            py::arg("data_loader"),
            py::arg("ids"),
            py::arg("distance_type"),
            py::arg("num_threads") = 1,
            fmt::format(DYNAMIC_IVF_BUILD_FROM_FILE_DOCSTRING_PROTO, dynamic).c_str()
        );
    }

    // Index modification.
    add_points_specialization<float>(ivf_index);

    // Note: DynamicIVFIndex doesn't support reconstruct_at, so we don't add reconstruct
    // interface

    // Index Deletion.
    ivf_index.def(
        "delete",
        [](svs::DynamicIVF& index, const py_contiguous_array_t<size_t>& ids) {
            return index.delete_points(as_span(ids));
        },
        py::arg("ids"),
        DELETE_DOCSTRING
    );

    // ID inspection
    ivf_index.def(
        "has_id",
        &svs::DynamicIVF::has_id,
        py::arg("id"),
        "Return whether the ID exists in the index."
    );

    ivf_index.def(
        "all_ids",
        [](const svs::DynamicIVF& index) {
            const auto& v = index.all_ids();
            // Populate a numpy-set
            auto npv = numpy_vector<size_t>(v.size());
            std::copy(v.begin(), v.end(), npv.mutable_unchecked().mutable_data());
            return npv;
        },
        ALL_IDS_DOCSTRING
    );

    // Distance calculation
    ivf_index.def(
        "get_distance",
        [](const svs::DynamicIVF& index,
           size_t id,
           const py_contiguous_array_t<float>& query) {
            return index.get_distance(id, as_span(query));
        },
        py::arg("id"),
        py::arg("query"),
        R"(
        Compute the distance between a query vector and a vector in the index.

        Args:
            id: The external ID of the vector in the index.
            query: The query vector as a numpy array.

        Returns:
            The distance between the query and the indexed vector.

        Raises:
            RuntimeError: If the ID doesn't exist or dimensions don't match.
        )"
    );

    // Saving
    ivf_index.def(
        "save",
        &save_index,
        py::arg("config_directory"),
        py::arg("data_directory"),
        R"(
Save a constructed index to disk (useful following index construction).

Args:
    config_directory: Directory where index configuration information will be saved.
    data_directory: Directory where the dataset will be saved.

Note: All directories should be separate to avoid accidental name collision with any
auxiliary files that are needed when saving the various components of the index.

If the directory does not exist, it will be created if its parent exists.

It is the caller's responsibility to ensure that no existing data will be
overwritten when saving the index to this directory.
    )"
    );
}

} // namespace svs::python::dynamic_ivf
