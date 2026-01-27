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
#include "svs/python/ivf_loader.h"
#include "svs/python/manager.h"

// svs
#include "svs/index/ivf/data_traits.h"
#include "svs/lib/dispatcher.h"
#include "svs/lib/saveload.h"
#include "svs/orchestrators/dynamic_ivf.h"

// toml
#include <toml++/toml.h>

// pybind
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// fmt
#include <fmt/format.h>

// stl
#include <filesystem>
#include <span>

/////
///// DynamicIVF
/////

namespace py = pybind11;
namespace svs::python::dynamic_ivf {

// Reuse the Clustering type from static IVF since clustering is the same
using Clustering = svs::python::ivf::Clustering;

using IVFAssembleTypes =
    std::variant<UnspecializedVectorDataLoader, svs::lib::SerializedObject>;

/////
///// Dispatch Invocation
/////

/////
///// Assembly from Clustering
/////

template <typename Q, typename T, size_t N>
svs::DynamicIVF assemble_uncompressed(
    Clustering clustering,
    svs::VectorDataLoader<T, N, RebindAllocator<T>> data,
    std::span<const size_t> ids,
    svs::DistanceType distance_type,
    size_t num_threads,
    size_t intra_query_threads = 1
) {
    // Use std::visit to handle the variant clustering type
    return std::visit(
        [&](auto&& actual_clustering) {
            return svs::DynamicIVF::assemble_from_clustering<Q>(
                std::move(actual_clustering),
                std::move(data),
                ids,
                distance_type,
                num_threads,
                intra_query_threads
            );
        },
        std::move(clustering)
    );
}

template <typename Dispatcher>
void register_uncompressed_ivf_assemble(Dispatcher& dispatcher) {
    for_standard_specializations([&dispatcher]<typename Q, typename T, size_t N>() {
        auto method = &assemble_uncompressed<Q, T, N>;
        dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
    });
}

template <typename Dispatcher> void register_ivf_assembly(Dispatcher& dispatcher) {
    register_uncompressed_ivf_assemble(dispatcher);
}

/////
///// Assembly from File
/////
template <typename Q, typename T, size_t N>
svs::DynamicIVF assemble_from_file_uncompressed(
    const std::filesystem::path& cluster_path,
    svs::VectorDataLoader<T, N, RebindAllocator<T>> data,
    std::span<const size_t> ids,
    svs::DistanceType distance_type,
    size_t num_threads,
    size_t intra_query_threads = 1
) {
    return svs::DynamicIVF::assemble_from_file<Q, svs::BFloat16>(
        cluster_path, std::move(data), ids, distance_type, num_threads, intra_query_threads
    );
}

template <typename Dispatcher>
void register_uncompressed_ivf_assemble_from_file(Dispatcher& dispatcher) {
    for_standard_specializations([&dispatcher]<typename Q, typename T, size_t N>() {
        auto method = &assemble_from_file_uncompressed<Q, T, N>;
        dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
    });
}

template <typename Dispatcher>
void register_ivf_assembly_from_file(Dispatcher& dispatcher) {
    register_uncompressed_ivf_assemble_from_file(dispatcher);
}

using IVFAssembleTypes =
    std::variant<UnspecializedVectorDataLoader, svs::lib::SerializedObject>;

/////
///// Dispatch Invocation
/////

using AssemblyDispatcher = svs::lib::Dispatcher<
    svs::DynamicIVF,
    Clustering,
    IVFAssembleTypes,
    std::span<const size_t>,
    svs::DistanceType,
    size_t,
    size_t>;

AssemblyDispatcher assembly_dispatcher() {
    auto dispatcher = AssemblyDispatcher{};

    // Register available backend methods.
    register_ivf_assembly(dispatcher);
    return dispatcher;
}

// Assemble
svs::DynamicIVF assemble_from_clustering(
    Clustering clustering,
    IVFAssembleTypes data_kind,
    const py_contiguous_array_t<size_t>& py_ids,
    svs::DistanceType distance_type,
    svs::DataType SVS_UNUSED(query_type),
    bool SVS_UNUSED(enforce_dims),
    size_t num_threads,
    size_t intra_query_threads = 1
) {
    auto ids = std::span<const size_t>(py_ids.data(), py_ids.size());
    return assembly_dispatcher().invoke(
        std::move(clustering),
        std::move(data_kind),
        ids,
        distance_type,
        num_threads,
        intra_query_threads
    );
}

using AssemblyFromFileDispatcher = svs::lib::Dispatcher<
    svs::DynamicIVF,
    const std::filesystem::path&,
    IVFAssembleTypes,
    std::span<const size_t>,
    svs::DistanceType,
    size_t,
    size_t>;

AssemblyFromFileDispatcher assembly_from_file_dispatcher() {
    auto dispatcher = AssemblyFromFileDispatcher{};

    // Register available backend methods.
    register_ivf_assembly_from_file(dispatcher);
    return dispatcher;
}

// Assemble from file
svs::DynamicIVF assemble_from_file(
    const std::string& cluster_path,
    IVFAssembleTypes data_kind,
    const py_contiguous_array_t<size_t>& py_ids,
    svs::DistanceType distance_type,
    svs::DataType SVS_UNUSED(query_type),
    bool SVS_UNUSED(enforce_dims),
    size_t num_threads,
    size_t intra_query_threads = 1
) {
    auto ids = std::span<const size_t>(py_ids.data(), py_ids.size());
    return assembly_from_file_dispatcher().invoke(
        cluster_path,
        std::move(data_kind),
        ids,
        distance_type,
        num_threads,
        intra_query_threads
    );
}

constexpr std::string_view ASSEMBLE_DOCSTRING_PROTO = R"(
Assemble a searchable IVF index from provided clustering and data

Args:
    clustering_path/clustering: Path to the directory where the clustering was generated.
        OR directly provide the loaded Clustering.
    data_loader: The loader for the dataset. See comment below for accepted types.
    ids: External IDs for the vectors. Must match dataset length and contain unique values.
    distance: The distance function to use.
    query_type: The data type of the queries.
    enforce_dims: Require that the compiled dimensionality of the returned index matches
        the dimensionality provided in the ``data_loader`` argument. If a match is not
        found, an exception is thrown.

        This is meant to ensure that specialized dimensionality is provided without falling
        back to generic implementations. Leaving the ``dims`` out when constructing the
        ``data_loader`` will with `enable_dims = True` will always attempt to use a generic
        implementation.
    num_threads: The number of threads to use for queries (can't be changed after loading).
    intra_query_threads: (default: 1) these many threads work on a single query.
        Total number of threads required = ``query_batch_size`` * ``intra_query_threads``.
        Where ``query_batch_size`` is the number of queries processed in parallel.
        Use this parameter only when the ``query_batch_size`` is smaller and ensure your
        system has sufficient threads available. Set ``num_threads`` = ``query_batch_size``

The top level type is an abstract type backed by various specialized backends that will
be instantiated based on their applicability to the particular problem instance.

The arguments upon which specialization is conducted are:

* `data_loader`: Both kind (type of loader) and inner aspects of the loader like data type,
  quantization type, and number of dimensions.
* `distance`: The distance measure being used.

Specializations compiled into the binary are listed below.

{}
)";

/////
///// Add points
/////

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

// Load with auto-detection from saved config using common template dispatcher
svs::DynamicIVF load_index_auto(
    const std::string& config_path,
    const std::string& data_path,
    svs::DistanceType distance_type,
    size_t num_threads,
    size_t intra_query_threads = 1
) {
    // Generic loader that dispatches to DynamicIVF::assemble with the correct types
    // Using BlockedData for dynamic index to avoid 1GB hugepage allocations per cluster
    auto loader = []<typename DataType, typename CentroidType>(
                      const std::string& cfg,
                      const std::string& data,
                      svs::DistanceType dist,
                      size_t threads,
                      size_t intra_threads
                  ) {
        using data_storage =
            svs::data::BlockedData<DataType, svs::Dynamic, svs::data::Blocked<Allocator>>;
        return svs::DynamicIVF::assemble<float, CentroidType, data_storage>(
            cfg, data, dist, threads, intra_threads
        );
    };

    return svs::python::ivf_loader::load_index_auto<svs::DynamicIVF>(
        config_path, data_path, distance_type, num_threads, intra_query_threads, loader
    );
}

void wrap(py::module& m) {
    std::string name = "DynamicIVF";
    py::class_<svs::DynamicIVF> dynamic_ivf(
        m, name.c_str(), "Top level class for the dynamic IVF index."
    );

    add_search_specialization<float>(dynamic_ivf);
    add_threading_interface(dynamic_ivf);
    add_data_interface(dynamic_ivf);

    // IVF specific extensions.
    ivf::add_interface(dynamic_ivf);

    // Dynamic interface.
    dynamic_ivf.def("consolidate", &svs::DynamicIVF::consolidate, CONSOLIDATE_DOCSTRING);
    dynamic_ivf.def(
        "compact",
        &svs::DynamicIVF::compact,
        py::arg("batchsize") = 1'000'000,
        COMPACT_DOCSTRING
    );

    // Assemble interface
    {
        auto dispatcher = assembly_dispatcher();
        // Procedurally generate the dispatch string.
        auto dynamic = std::string{};
        for (size_t i = 0; i < dispatcher.size(); ++i) {
            fmt::format_to(
                std::back_inserter(dynamic),
                R"(
Method {}:
    - data_loader: {}
    - distance: {}
)",
                i,
                dispatcher.description(i, 2),
                dispatcher.description(i, 3)
            );
        }

        dynamic_ivf.def_static(
            "assemble_from_clustering",
            [](Clustering clustering,
               IVFAssembleTypes data_loader,
               const py_contiguous_array_t<size_t>& py_ids,
               svs::DistanceType distance,
               svs::DataType query_type,
               bool enforce_dims,
               size_t num_threads,
               size_t intra_query_threads) {
                return assemble_from_clustering(
                    std::move(clustering),
                    std::move(data_loader),
                    py_ids,
                    distance,
                    query_type,
                    enforce_dims,
                    num_threads,
                    intra_query_threads
                );
            },
            py::arg("clustering"),
            py::arg("data_loader"),
            py::arg("ids"),
            py::arg("distance") = svs::L2,
            py::arg("query_type") = svs::DataType::float32,
            py::arg("enforce_dims") = false,
            py::arg("num_threads") = 1,
            py::arg("intra_query_threads") = 1,
            fmt::format(ASSEMBLE_DOCSTRING_PROTO, dynamic).c_str()
        );
        dynamic_ivf.def_static(
            "assemble_from_file",
            [](const std::string& clustering_path,
               IVFAssembleTypes data_loader,
               const py_contiguous_array_t<size_t>& py_ids,
               svs::DistanceType distance,
               svs::DataType query_type,
               bool enforce_dims,
               size_t num_threads,
               size_t intra_query_threads) {
                return assemble_from_file(
                    clustering_path,
                    std::move(data_loader),
                    py_ids,
                    distance,
                    query_type,
                    enforce_dims,
                    num_threads,
                    intra_query_threads
                );
            },
            py::arg("clustering_path"),
            py::arg("data_loader"),
            py::arg("ids"),
            py::arg("distance") = svs::L2,
            py::arg("query_type") = svs::DataType::float32,
            py::arg("enforce_dims") = false,
            py::arg("num_threads") = 1,
            py::arg("intra_query_threads") = 1,
            fmt::format(ASSEMBLE_DOCSTRING_PROTO, dynamic).c_str()
        );
    }

    // Index modification.
    add_points_specialization<float>(dynamic_ivf);

    // Note: DynamicIVFIndex doesn't support reconstruct_at, so we don't add reconstruct
    // interface

    // Index Deletion.
    dynamic_ivf.def(
        "delete",
        [](svs::DynamicIVF& index, const py_contiguous_array_t<size_t>& ids) {
            return index.delete_points(as_span(ids));
        },
        py::arg("ids"),
        DELETE_DOCSTRING
    );

    // ID inspection
    dynamic_ivf.def(
        "has_id",
        &svs::DynamicIVF::has_id,
        py::arg("id"),
        "Return whether the ID exists in the index."
    );

    dynamic_ivf.def(
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
    dynamic_ivf.def(
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
    dynamic_ivf.def(
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

    // Loading
    dynamic_ivf.def_static(
        "load",
        &load_index_auto,
        py::arg("config_directory"),
        py::arg("data_directory"),
        py::arg("distance") = svs::L2,
        py::arg("num_threads") = 1,
        py::arg("intra_query_threads") = 1,
        R"(
Load a saved DynamicIVF index from disk.

The data type (uncompressed with float32 or float16) and centroid type (bfloat16 or float16)
are automatically detected from the saved configuration file.

Args:
    config_directory: Directory where index configuration was saved.
    data_directory: Directory where the dataset was saved.
    distance: The distance function to use.
    num_threads: The number of threads to use for queries.
    intra_query_threads: Number of threads for intra-query parallelism (default: 1).

Returns:
    A loaded DynamicIVF index ready for searching and modifications.

Note:
    This method auto-detects the data type from the saved configuration.
    The index must have been saved with a version that includes data type information.
    )"
    );
}

} // namespace svs::python::dynamic_ivf
