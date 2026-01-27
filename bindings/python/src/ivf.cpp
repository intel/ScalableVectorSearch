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
#include "svs/python/ivf.h"
#include "svs/python/common.h"
#include "svs/python/core.h"
#include "svs/python/dispatch.h"
#include "svs/python/ivf_loader.h"
#include "svs/python/manager.h"

// pybind11
#include <pybind11/stl.h> // For std::variant support

// svs
#include "svs/core/data/simple.h"
#include "svs/core/distance.h"
#include "svs/index/ivf/data_traits.h"
#include "svs/lib/array.h"
#include "svs/lib/datatype.h"
#include "svs/lib/dispatcher.h"
#include "svs/lib/float16.h"
#include "svs/lib/meta.h"
#include "svs/lib/saveload.h"
#include "svs/orchestrators/ivf.h"

// toml
#include <toml++/toml.h>

// pybind
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// stl
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <span>

/////
///// IVF
/////

namespace py = pybind11;
using namespace svs::python::ivf_specializations;

namespace svs::python::ivf {

namespace detail {

/////
///// Assembly
/////

template <typename Q, typename T, size_t N>
svs::IVF assemble_uncompressed(
    Clustering clustering,
    svs::VectorDataLoader<T, N, RebindAllocator<T>> data,
    svs::DistanceType distance_type,
    size_t num_threads,
    size_t intra_query_threads = 1
) {
    // Use std::visit to handle the variant clustering type
    return std::visit(
        [&](auto&& actual_clustering) {
            return svs::IVF::assemble_from_clustering<Q>(
                std::move(actual_clustering),
                std::move(data),
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
    for_standard_specializations(
        [&dispatcher]<typename Q, typename T, size_t N, EnableBuild B>() {
            auto method = &assemble_uncompressed<Q, T, N>;
            dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
        }
    );
}

template <typename Dispatcher> void register_ivf_assembly(Dispatcher& dispatcher) {
    register_uncompressed_ivf_assemble(dispatcher);
}

/////
///// Assembly from File
/////
// N.B (IB): quite a bit of repetition in Assemble and AssembleFromFile functions.
// Loading the cluster from file and then using the Assemble from clustering
// shows performance loss, mainly due to the threadpool used for loading.
// This needs to be revisited.
template <typename Q, typename T, size_t N>
svs::IVF assemble_from_file_uncompressed(
    const std::filesystem::path& cluster_path,
    svs::VectorDataLoader<T, N, RebindAllocator<T>> data,
    svs::DistanceType distance_type,
    size_t num_threads,
    size_t intra_query_threads = 1
) {
    return svs::IVF::assemble_from_file<Q, svs::BFloat16>(
        cluster_path, std::move(data), distance_type, num_threads, intra_query_threads
    );
}

template <typename Dispatcher>
void register_uncompressed_ivf_assemble_from_file(Dispatcher& dispatcher) {
    for_standard_specializations(
        [&dispatcher]<typename Q, typename T, size_t N, EnableBuild B>() {
            auto method = &assemble_from_file_uncompressed<Q, T, N>;
            dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
        }
    );
}

template <typename Dispatcher>
void register_ivf_assembly_from_file(Dispatcher& dispatcher) {
    register_uncompressed_ivf_assemble_from_file(dispatcher);
}

using IVFAssembleTypes =
    std::variant<UnspecializedVectorDataLoader, svs::lib::SerializedObject>;

/////
///// Build From File
/////

template <typename T, size_t N>
Clustering build_uncompressed(
    const svs::index::ivf::IVFBuildParameters& parameters,
    svs::VectorDataLoader<T, N, RebindAllocator<T>> data,
    svs::DistanceType distance_type,
    size_t num_threads
) {
    // Choose build type for clustering to leverage AMX instructions:
    // - Float32 data -> BFloat16 (AMX supports BFloat16)
    // - Float16 data -> Float16 (AMX supports Float16)
    // - BFloat16 data -> BFloat16 (already optimal)
    using BuildType = std::conditional_t<std::is_same_v<T, float>, svs::BFloat16, T>;
    auto clustering = svs::IVF::build_clustering<BuildType>(
        parameters, std::move(data), distance_type, num_threads
    );

    // Return as variant - Float16 or BFloat16 based on BuildType
    if constexpr (std::is_same_v<BuildType, svs::Float16>) {
        return Clustering(std::in_place_index<1>, std::move(clustering));
    } else {
        return Clustering(std::in_place_index<0>, std::move(clustering));
    }
}

template <typename Dispatcher>
void register_uncompressed_ivf_build_from_file(Dispatcher& dispatcher) {
    for_standard_specializations(
        [&dispatcher]<typename Q, typename T, size_t N, EnableBuild B>() {
            if constexpr (enable_build_from_file<B>) {
                auto method = &build_uncompressed<T, N>;
                dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
            }
        }
    );
}

template <typename Dispatcher> void register_ivf_build_from_file(Dispatcher& dispatcher) {
    register_uncompressed_ivf_build_from_file(dispatcher);
}

using IVFBuildTypes = std::variant<UnspecializedVectorDataLoader>;

/////
///// Build from Array
/////

template <typename T, size_t N>
Clustering uncompressed_build_from_array(
    const svs::index::ivf::IVFBuildParameters& parameters,
    svs::data::ConstSimpleDataView<T, N> view,
    svs::DistanceType distance_type,
    size_t num_threads
) {
    auto data =
        svs::data::SimpleData<T, N, RebindAllocator<T>>(view.size(), view.dimensions());
    svs::data::copy(view, data);
    // Choose build type for clustering to leverage AMX instructions:
    // - Float32 data -> BFloat16 (AMX supports BFloat16)
    // - Float16 data -> Float16 (AMX supports Float16)
    // - BFloat16 data -> BFloat16 (already optimal)
    using BuildType = std::conditional_t<std::is_same_v<T, float>, svs::BFloat16, T>;
    auto clustering = svs::IVF::build_clustering<BuildType>(
        parameters, std::move(data), distance_type, num_threads
    );

    // Return as variant - Float16 or BFloat16 based on BuildType
    if constexpr (std::is_same_v<BuildType, svs::Float16>) {
        return Clustering(std::in_place_index<1>, std::move(clustering));
    } else {
        return Clustering(std::in_place_index<0>, std::move(clustering));
    }
}

template <typename Dispatcher> void register_ivf_build_from_array(Dispatcher& dispatcher) {
    for_standard_specializations(
        [&dispatcher]<typename Q, typename T, size_t N, EnableBuild B>() {
            if constexpr (enable_build_from_array<B>) {
                auto method = &uncompressed_build_from_array<T, N>;
                dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
            }
        }
    );
}

/////
///// Dispatch Invocation
/////

using AssemblyDispatcher = svs::lib::
    Dispatcher<svs::IVF, Clustering, IVFAssembleTypes, svs::DistanceType, size_t, size_t>;

AssemblyDispatcher assembly_dispatcher() {
    auto dispatcher = AssemblyDispatcher{};

    // Register available backend methods.
    register_ivf_assembly(dispatcher);
    return dispatcher;
}

// Assemble
svs::IVF assemble_from_clustering(
    Clustering clustering,
    IVFAssembleTypes data_kind,
    svs::DistanceType distance_type,
    svs::DataType SVS_UNUSED(query_type),
    bool SVS_UNUSED(enforce_dims),
    size_t num_threads,
    size_t intra_query_threads = 1
) {
    return assembly_dispatcher().invoke(
        std::move(clustering),
        std::move(data_kind),
        distance_type,
        num_threads,
        intra_query_threads
    );
}

using AssemblyFromFileDispatcher = svs::lib::Dispatcher<
    svs::IVF,
    const std::filesystem::path&,
    IVFAssembleTypes,
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
svs::IVF assemble_from_file(
    const std::string& cluster_path,
    IVFAssembleTypes data_kind,
    svs::DistanceType distance_type,
    svs::DataType SVS_UNUSED(query_type),
    bool SVS_UNUSED(enforce_dims),
    size_t num_threads,
    size_t intra_query_threads = 1
) {
    return assembly_from_file_dispatcher().invoke(
        cluster_path, std::move(data_kind), distance_type, num_threads, intra_query_threads
    );
}

constexpr std::string_view ASSEMBLE_DOCSTRING_PROTO = R"(
Assemble a searchable IVF index from provided clustering and data

Args:
    clustering_path/clustering: Path to the directory where the clustering was generated.
        OR directly provide the loaded Clustering.
    data_loader: The loader for the dataset. See comment below for accepted types.
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

void wrap_assemble(py::class_<svs::IVF>& ivf) {
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

    ivf.def_static(
        "assemble_from_clustering",
        &detail::assemble_from_clustering,
        py::arg("clustering"),
        py::arg("data_loader"),
        py::arg("distance") = svs::L2,
        py::arg("query_type") = svs::DataType::float32,
        py::arg("enforce_dims") = false,
        py::arg("num_threads") = 1,
        py::arg("intra_query_threads") = 1,
        fmt::format(ASSEMBLE_DOCSTRING_PROTO, dynamic).c_str()
    );
    ivf.def_static(
        "assemble_from_file",
        &detail::assemble_from_file,
        py::arg("clustering_path"),
        py::arg("data_loader"),
        py::arg("distance") = svs::L2,
        py::arg("query_type") = svs::DataType::float32,
        py::arg("enforce_dims") = false,
        py::arg("num_threads") = 1,
        py::arg("intra_query_threads") = 1,
        fmt::format(ASSEMBLE_DOCSTRING_PROTO, dynamic).c_str()
    );
}

// Build from file
using BuildFromFileDispatcher = svs::lib::Dispatcher<
    Clustering,
    const svs::index::ivf::IVFBuildParameters&,
    IVFBuildTypes,
    svs::DistanceType,
    size_t>;

BuildFromFileDispatcher build_from_file_dispatcher() {
    auto dispatcher = BuildFromFileDispatcher{};
    register_ivf_build_from_file(dispatcher);
    return dispatcher;
}

Clustering build_from_file(
    const svs::index::ivf::IVFBuildParameters& parameters,
    IVFBuildTypes data_source,
    svs::DistanceType distance_type,
    size_t num_threads
) {
    return build_from_file_dispatcher().invoke(
        parameters, std::move(data_source), distance_type, num_threads
    );
}

// Build from array.
using BuildFromArrayDispatcher = svs::lib::Dispatcher<
    Clustering,
    const svs::index::ivf::IVFBuildParameters&,
    AnonymousVectorData,
    svs::DistanceType,
    size_t>;

BuildFromArrayDispatcher build_from_array_dispatcher() {
    auto dispatcher = BuildFromArrayDispatcher{};
    register_ivf_build_from_array(dispatcher);
    return dispatcher;
}

Clustering build_from_array(
    const svs::index::ivf::IVFBuildParameters& parameters,
    AnonymousVectorData py_data,
    svs::DistanceType distance_type,
    size_t num_threads
) {
    return build_from_array_dispatcher().invoke(
        parameters, py_data, distance_type, num_threads
    );
}

template <typename ElementType>
void add_build_specialization(py::class_<Clustering>& clustering) {
    clustering.def_static(
        "build",
        [](const svs::index::ivf::IVFBuildParameters& parameters,
           py_contiguous_array_t<ElementType> py_data,
           svs::DistanceType distance,
           size_t num_threads) {
            return build_from_array(
                parameters, AnonymousVectorData(py_data), distance, num_threads
            );
        },
        py::arg("build_parameters"),
        py::arg("py_data"),
        py::arg("distance"),
        py::arg("num_threads") = 1,
        R"(
 Build IVF clustering over the given data and return a sparse clustering.
 Use this clustering to assemble a searcheable IVF index.

 Args:
     parameters: Parameters controlling kmeans clustering.
     py_data: The dataset to index.
     distance: The distance type to use for this dataset.
     num_threads: The number of threads to use for index construction. Default: 1.
)"
    );
}

void wrap_build_from_file(py::class_<Clustering>& clustering) {
    constexpr std::string_view docstring_proto = R"(
 Build IVF clustering over the given data and return a sparse clustering.
 Use the returned clustering to assemble a searcheable IVF index.

 Args:
     build_parameters (:py:class:`svs.IVFBuildParameters`): Hyper-parameters
         controlling clustering build.
     data_loader: The source of the data on-disk. Can either be
         :py:class:`svs.DataFile` to represent a standard uncompressed dataset
     distance: The similarity-function to use for this index.
     num_threads: The number of threads to use for index construction. Default: 1.

 The top level type is an abstract type backed by various specialized backends that will
 be instantiated based on their applicability to the particular problem instance.

 The arguments upon which specialization is conducted are:

* `data_loader`: Only uncompressed data types are supported for IVF cluster building
* `distance`: The distance measure being used.

 Specializations compiled into the binary are listed below.

{}
)";

    auto dispatcher = build_from_file_dispatcher();
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
            dispatcher.description(i, 1),
            dispatcher.description(i, 2)
        );
    }

    clustering.def_static(
        "build",
        &detail::build_from_file,
        py::arg("build_parameters"),
        py::arg("data_loader"),
        py::arg("distance"),
        py::arg("num_threads") = 1,
        fmt::format(docstring_proto, dynamic).c_str()
    );
}

// Save the sparse clustering to a directory
void save_clustering(Clustering& clustering, const std::string& clustering_path) {
    std::visit(
        [&](auto&& actual_clustering) {
            svs::lib::save_to_disk(actual_clustering, clustering_path);
        },
        clustering
    );
}

// Load the sparse clustering from a directory
// Try loading as BFloat16 first, then Float16 if that fails
auto load_clustering(const std::string& clustering_path, size_t num_threads = 1) {
    auto threadpool = threads::as_threadpool(num_threads);
    try {
        auto bf16_clustering = svs::lib::load_from_disk<
            svs::index::ivf::Clustering<svs::data::SimpleData<svs::BFloat16>, uint32_t>>(
            clustering_path, threadpool
        );
        return Clustering(std::in_place_index<0>, std::move(bf16_clustering));
    } catch (...) {
        auto f16_clustering = svs::lib::load_from_disk<
            svs::index::ivf::Clustering<svs::data::SimpleData<svs::Float16>, uint32_t>>(
            clustering_path, threadpool
        );
        return Clustering(std::in_place_index<1>, std::move(f16_clustering));
    }
}

// Save the IVF index to directories
void save_index(
    svs::IVF& index, const std::string& config_path, const std::string& data_dir
) {
    index.save(config_path, data_dir);
}

// Load with auto-detection from saved config using common template dispatcher
svs::IVF load_index(
    const std::string& config_path,
    const std::string& data_path,
    svs::DistanceType distance_type,
    size_t num_threads,
    size_t intra_query_threads = 1
) {
    return svs::python::ivf_loader::load_index_auto<
        svs::IVF, svs::data::SimpleData, Allocator>(
        config_path,
        data_path,
        distance_type,
        num_threads,
        intra_query_threads
    );
}

} // namespace detail

void wrap(py::module& m) {
    // wrap_common(m);

    /// Build Parameters
    using IVFBuildParameters = svs::index::ivf::IVFBuildParameters;
    py::class_<svs::index::ivf::IVFBuildParameters> parameters(
        m, "IVFBuildParameters", "Build parameters for kmeans clustering."
    );

    parameters
        .def(
            py::init([](size_t num_centroids,
                        size_t minibatch_size,
                        size_t num_iterations,
                        bool is_hierarchical,
                        float training_fraction,
                        size_t hierarchical_level1_clusters,
                        size_t seed) {
                return svs::index::ivf::IVFBuildParameters{
                    num_centroids,
                    minibatch_size,
                    num_iterations,
                    is_hierarchical,
                    training_fraction,
                    hierarchical_level1_clusters,
                    seed};
            }),
            py::arg("num_centroids") = 1000,
            py::arg("minibatch_size") = 10'000,
            py::arg("num_iterations") = 10,
            py::arg("is_hierarchical") = true,
            py::arg("training_fraction") = 0.1,
            py::arg("hierarchical_level1_clusters") = 0,
            py::arg("seed") = 0xc0ffee,
            R"(
            Construct a new instance from keyword arguments.

            Args:
                num_centroids: The target number of clusters in the final result.
                minibatch_size: The size of each minibatch used to process data at a time.
                num_iterations: The number of iterations used in kmeans training.
                is_hierarchical: Use hierarchical Kmeans or not.
                training_fraction: Fraction of dataset used for training
                hierarchical_level1_clusters: Level1 clusters for hierarchical kmeans.
                    Use heuristic if 0.
                seed: The initial seed for the random number generator.
            )"
        )
        .def_readwrite("num_centroids", &IVFBuildParameters::num_centroids_)
        .def_readwrite("minibatch_size", &IVFBuildParameters::minibatch_size_)
        .def_readwrite("num_iterations", &IVFBuildParameters::num_iterations_)
        .def_readwrite("is_hierarchical", &IVFBuildParameters::is_hierarchical_)
        .def_readwrite("training_fraction", &IVFBuildParameters::training_fraction_)
        .def_readwrite(
            "hierarchical_level1_clusters",
            &IVFBuildParameters::hierarchical_level1_clusters_
        );

    /// Search Parameters
    using IVFSearchParameters = svs::index::ivf::IVFSearchParameters;
    auto params = py::class_<IVFSearchParameters>{
        m,
        "IVFSearchParameters",
        R"(
            Parameters controlling recall and performance of the IVF Index.
            Args:
                n_probes: The number of nearest clusters to be explored
                k_reorder: Level of reordering or reranking done when using compressed datasets
        )"};

    params
        .def(py::init<size_t, float>(), py::arg("n_probes") = 1, py::arg("k_reorder") = 1.0)
        .def_readwrite("n_probes", &IVFSearchParameters::n_probes_)
        .def_readwrite("k_reorder", &IVFSearchParameters::k_reorder_);

    ///
    /// IVF Static Module
    ///
    std::string name = "IVF";
    py::class_<svs::IVF> ivf(m, name.c_str(), "Top level class for the IVF index.");

    detail::wrap_assemble(ivf);

    // Make the IVF type searchable.
    add_search_specialization<svs::Float16>(ivf);
    add_search_specialization<float>(ivf);

    // Add threading layer.
    add_threading_interface(ivf);

    // Data layer
    add_data_interface(ivf);

    // IVF Specific Extensions.
    add_interface(ivf);

    // Index Saving.
    ivf.def(
        "save",
        &detail::save_index,
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

    // Index Loading.
    ivf.def_static(
        "load",
        &detail::load_index,
        py::arg("config_directory"),
        py::arg("data_directory"),
        py::arg("distance") = svs::L2,
        py::arg("num_threads") = 1,
        py::arg("intra_query_threads") = 1,
        R"(
Load a saved IVF index from disk.

The data type (uncompressed with float32 or float16) and centroid type (bfloat16 or float16)
are automatically detected from the saved configuration file.

Args:
    config_directory: Directory where index configuration was saved.
    data_directory: Directory where the dataset was saved.
    distance: The distance function to use.
    num_threads: The number of threads to use for queries.
    intra_query_threads: Number of threads for intra-query parallelism (default: 1).

Returns:
    A loaded IVF index ready for searching.

Note:
    This method auto-detects the data type from the saved configuration.
    The index must have been saved with a version that includes data type information.
    )"
    );

    // Reconstruction.
    // add_reconstruct_interface(ivf);

    // Register both clustering types that make up the variant
    name = "ClusteringBFloat16";
    py::class_<ClusteringBF16> clustering_bf16(m, name.c_str());
    clustering_bf16.def(
        "save",
        [](ClusteringBF16& clustering, const std::string& clustering_path) {
            svs::lib::save_to_disk(clustering, clustering_path);
        },
        py::arg("clustering_directory"),
        "Save a constructed IVF clustering to disk."
    );

    name = "ClusteringFloat16";
    py::class_<ClusteringF16> clustering_f16(m, name.c_str());
    clustering_f16.def(
        "save",
        [](ClusteringF16& clustering, const std::string& clustering_path) {
            svs::lib::save_to_disk(clustering, clustering_path);
        },
        py::arg("clustering_directory"),
        "Save a constructed IVF clustering to disk."
    );

    // Register the variant type as the main Clustering class
    name = "Clustering";
    py::class_<Clustering> clustering(
        m, name.c_str(), "Top level class for sparse IVF clustering"
    );

    /// Index building
    // Build from Numpy array.
    detail::add_build_specialization<svs::BFloat16>(clustering);
    detail::add_build_specialization<svs::Float16>(clustering);
    detail::add_build_specialization<float>(clustering);

    // Build from datasets on file.
    detail::wrap_build_from_file(clustering);

    /// Index Saving and Loading.
    clustering.def(
        "save_clustering",
        &detail::save_clustering,
        py::arg("clustering_directory"),
        R"(
             Save a constructed IVF clustering to disk (useful following build).

            Args:
                clustering_directory: Directory where clustering will be saved.

            Note: All directories should be separate to avoid accidental name collision
            with any auxiliary files that are needed when saving the various components of
            the index.

            If the directory does not exist, it will be created if its parent exists.

            It is the caller's responsibilty to ensure that no existing data will be
            overwritten when saving the index to this directory.
        )"
    );
    clustering.def_static(
        "load_clustering",
        &detail::load_clustering,
        py::arg("clustering_directory"),
        py::arg("num_threads") = 1,
        R"(
             Load IVF clustering from disk (maybe used before assembling).

            Args:
                clustering_directory: Directory from where to load the clustering.
                num_threads: Number of threads to use when loading (default: 1).
        )"
    );
}

} // namespace svs::python::ivf
