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

// svs python bindings
#include "svs/python/dynamic_vamana.h"
#include "svs/python/common.h"
#include "svs/python/core.h"
#include "svs/python/manager.h"
#include "svs/python/vamana.h"
#include "svs/python/vamana_common.h"

// svs
#include "svs/lib/dispatcher.h"
#include "svs/orchestrators/dynamic_vamana.h"

#include "svs/fallback/fallback.h"

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
namespace svs::python::dynamic_vamana {

namespace {

namespace lvq = svs::quantization::lvq;

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
    data: The dataset to index. **NOTE**: SVS will maintain an internal copy of the
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

When ``delete_entries`` is called, a soft deletion is performed, marking the entries as ``deleted``.
When ``consolidate`` is called, the state of these deleted entries becomes ``empty``.
When ``add_points`` is called with the ``reuse_empty`` flag enabled, the memory is scanned from the beginning to locate and fill these empty entries with new points.
)";

template <typename ElementType>
void add_points_specialization(py::class_<svs::DynamicVamana>& index) {
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

// Index saving.
void save_index(
    svs::DynamicVamana& index,
    const std::string& config_path,
    const std::string& graph_dir,
    const std::string& data_dir
) {
    index.save(config_path, graph_dir, data_dir);
}

/////
///// Assembly
/////

template <typename Q, typename T, typename Dist, size_t N>
svs::DynamicVamana assemble_uncompressed(
    const std::filesystem::path& config_path,
    const UnspecializedGraphLoader& graph_loader,
    svs::VectorDataLoader<T, N, RebindAllocator<T>> datafile,
    Dist distance,
    size_t num_threads,
    bool debug_load_from_static
) {
    auto load_graph = svs::lib::Lazy([&]() {
        return svs::graphs::SimpleBlockedGraph<uint32_t>::load(graph_loader.path());
    });

    auto load_data = svs::lib::Lazy([&]() {
        // Forward the allocator we wish to use
        using A = RebindAllocator<T>;
        return svs::data::BlockedData<T, N, A>::load(
            datafile.path_, as_blocked(A(datafile.allocator_))
        );
    });

    return svs::DynamicVamana::assemble<Q>(
        config_path, load_graph, load_data, distance, num_threads, debug_load_from_static
    );
}

template <
    typename Dist,
    size_t Primary,
    size_t Residual,
    lvq::LVQPackingStrategy Strategy,
    size_t N>
svs::DynamicVamana assemble_lvq(
    const std::filesystem::path& config_path,
    const UnspecializedGraphLoader& graph_loader,
    svs::quantization::lvq::LVQLoader<Primary, Residual, N, Strategy, Allocator> loader,
    Dist distance,
    size_t num_threads,
    bool debug_load_from_static
) {
    auto load_graph = svs::lib::Lazy([&]() {
        return svs::graphs::SimpleBlockedGraph<uint32_t>::load(graph_loader.path());
    });

    return svs::DynamicVamana::assemble<float>(
        config_path,
        load_graph,
        loader.rebind_alloc(as_blocked),
        distance,
        num_threads,
        debug_load_from_static
    );
}

template <typename Dist, typename Primary, typename Secondary, size_t L, size_t N>
svs::DynamicVamana assemble_leanvec(
    const std::filesystem::path& config_path,
    const UnspecializedGraphLoader& graph_loader,
    svs::leanvec::LeanVecLoader<Primary, Secondary, L, N, Allocator> loader,
    Dist distance,
    size_t num_threads,
    bool debug_load_from_static
) {
    auto load_graph = svs::lib::Lazy([&]() {
        return svs::graphs::SimpleBlockedGraph<uint32_t>::load(graph_loader.path());
    });

    return svs::DynamicVamana::assemble<float>(
        config_path,
        load_graph,
        loader.rebind_alloc(as_blocked),
        distance,
        num_threads,
        debug_load_from_static
    );
}

template <typename Dispatcher> void register_assembly(Dispatcher& dispatcher) {
    for_standard_specializations([&]<typename Q, typename T, typename D, size_t N>() {
        dispatcher.register_target(&assemble_uncompressed<Q, T, D, N>);
    });

    for_compressed_specializations(
        [&]<typename D, size_t P, size_t R, lvq::LVQPackingStrategy S, size_t N>() {
            dispatcher.register_target(&assemble_lvq<D, P, R, S, N>);
        }
    );

    for_leanvec_specializations([&]<typename D, typename P, typename S, size_t L, size_t N>(
                                ) {
        dispatcher.register_target(&assemble_leanvec<D, P, S, L, N>);
    });
}

using DynamicVamanaAssembleTypes =
    std::variant<UnspecializedVectorDataLoader, LVQ, LeanVec>;

svs::DynamicVamana assemble(
    const std::string& config_path,
    const UnspecializedGraphLoader& graph_loader,
    DynamicVamanaAssembleTypes data_loader,
    svs::DistanceType distance_type,
    svs::DataType SVS_UNUSED(query_type),
    bool SVS_UNUSED(enforce_dims),
    size_t num_threads,
    bool debug_load_from_static
) {
    auto dispatcher = svs::lib::Dispatcher<
        svs::DynamicVamana,
        const std::filesystem::path&,
        const UnspecializedGraphLoader&,
        DynamicVamanaAssembleTypes,
        svs::DistanceType,
        size_t,
        bool>();

    register_assembly(dispatcher);
    return dispatcher.invoke(
        config_path,
        graph_loader,
        std::move(data_loader),
        distance_type,
        num_threads,
        debug_load_from_static
    );
}

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

    // Reloading
    vamana.def(
        py::init(&assemble),
        py::arg("config_path"),
        py::arg("graph_loader"),
        py::arg("data_loader"),
        py::arg("distance") = svs::L2,
        py::arg("query_type") = svs::DataType::float32,
        py::arg("enforce_dims") = false,
        py::arg("num_threads") = 1,
        py::arg("debug_load_from_static") = false
    );

    // Index building.
    add_build_specialization<float>(vamana);

    // Index modification.
    add_points_specialization<float>(vamana);

    // Vector Reconstruction
    add_reconstruct_interface(vamana);

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

    // Saving
    vamana.def(
        "save",
        &save_index,
        py::arg("config_directory"),
        py::arg("graph_directory"),
        py::arg("data_directory"),
        R"(
Save a constructed index to disk (useful following index construction).

Args:
    config_directory: Directory where index configuration information will be saved.
    graph_directory: Directory where graph will be saved.
    data_directory: Directory where the dataset will be saved.


Note: All directories should be separate to avoid accidental name collision with any
auxiliary files that are needed when saving the various components of the index.

If the directory does not exist, it will be created if its parent exists.

It is the caller's responsibilty to ensure that no existing data will be
overwritten when saving the index to this directory.
    )"
    );
}

} // namespace svs::python::dynamic_vamana
