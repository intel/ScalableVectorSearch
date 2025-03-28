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
#include "svs/python/vamana.h"
#include "svs/python/common.h"
#include "svs/python/core.h"
#include "svs/python/dispatch.h"
#include "svs/python/manager.h"
#include "svs/python/vamana_common.h"

// svs
#include "svs/core/data/simple.h"
#include "svs/core/distance.h"
#include "svs/lib/array.h"
#include "svs/lib/datatype.h"
#include "svs/lib/dispatcher.h"
#include "svs/lib/float16.h"
#include "svs/lib/meta.h"
#include "svs/orchestrators/vamana.h"

// pybind
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// stl
#include <algorithm>
#include <iostream>
#include <map>
#include <span>

/////
///// Vamana
/////

namespace py = pybind11;
namespace lvq = svs::quantization::lvq;
namespace leanvec = svs::leanvec;

using namespace svs::python::vamana_specializations;

namespace svs::python::vamana {
namespace detail {

/////
///// Assembly
/////

// TODO: Go straight to the static function?
template <typename Q, typename T, size_t N>
svs::Vamana assemble_uncompressed(
    const std::filesystem::path& config_path,
    const UnspecializedGraphLoader& graph_loader,
    svs::VectorDataLoader<T, N, RebindAllocator<T>> data,
    svs::DistanceType distance_type,
    size_t num_threads
) {
    return svs::Vamana::assemble<Q>(
        config_path, graph_loader, std::move(data), distance_type, num_threads
    );
}

template <typename Dispatcher>
void register_uncompressed_vamana_assemble(Dispatcher& dispatcher) {
    for_standard_specializations(
        [&dispatcher]<typename Q, typename T, size_t N, EnableBuild B>() {
            auto method = &assemble_uncompressed<Q, T, N>;
            dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
        }
    );
}

template <
    size_t Primary,
    size_t Residual,
    size_t N,
    lvq::LVQPackingStrategy Strategy,
    typename D>
svs::Vamana assemble_lvq(
    const std::filesystem::path& config_path,
    const UnspecializedGraphLoader& graph_loader,
    lvq::LVQLoader<Primary, Residual, N, Strategy, Allocator> data,
    D distance,
    size_t num_threads
) {
    return svs::Vamana::assemble<float>(
        config_path, graph_loader, std::move(data), std::move(distance), num_threads
    );
}

template <typename Dispatcher> void register_lvq_vamana_assemble(Dispatcher& dispatcher) {
    compressed_specializations(
        [&dispatcher]<typename D, size_t P, size_t R, size_t N, typename S, bool B>() {
            auto method = &assemble_lvq<P, R, N, S, D>;
            dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
        }
    );
}

template <typename Primary, typename Secondary, size_t L, size_t N, typename D>
svs::Vamana assemble_leanvec(
    const std::filesystem::path& config_path,
    const UnspecializedGraphLoader& graph_loader,
    leanvec::LeanVecLoader<Primary, Secondary, L, N, Allocator> data,
    D distance,
    size_t num_threads
) {
    return svs::Vamana::assemble<float>(
        config_path, graph_loader, std::move(data), std::move(distance), num_threads
    );
}

template <typename Dispatcher>
void register_leanvec_vamana_assemble(Dispatcher& dispatcher) {
    leanvec_specializations(
        [&dispatcher]<typename P, typename S, size_t L, size_t N, typename D>() {
            auto method = &assemble_leanvec<P, S, L, N, D>;
            dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
        }
    );
}

template <typename Dispatcher> void register_vamana_assembly(Dispatcher& dispatcher) {
    register_uncompressed_vamana_assemble(dispatcher);
    register_lvq_vamana_assemble(dispatcher);
    register_leanvec_vamana_assemble(dispatcher);
}

using VamanaAssembleTypes =
    std::variant<UnspecializedVectorDataLoader, LVQ, LeanVec, svs::lib::SerializedObject>;

/////
///// Build From File
/////

template <typename Q, typename T, size_t N>
svs::Vamana build_uncompressed(
    const svs::index::vamana::VamanaBuildParameters& parameters,
    svs::VectorDataLoader<T, N, RebindAllocator<T>> data,
    svs::DistanceType distance_type,
    size_t num_threads
) {
    return svs::Vamana::build<Q>(parameters, std::move(data), distance_type, num_threads);
}

template <typename Dispatcher>
void register_uncompressed_vamana_build_from_file(Dispatcher& dispatcher) {
    for_standard_specializations(
        [&dispatcher]<typename Q, typename T, size_t N, EnableBuild B>() {
            if constexpr (enable_build_from_file<B>) {
                auto method = &build_uncompressed<Q, T, N>;
                dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
            }
        }
    );
}

template <size_t Primary, size_t Residual, size_t N, lvq::LVQPackingStrategy S, typename D>
svs::Vamana build_lvq_from_file(
    const svs::index::vamana::VamanaBuildParameters& parameters,
    lvq::LVQLoader<Primary, Residual, N, S, Allocator> data,
    D distance,
    size_t num_threads
) {
    return svs::Vamana::build<float>(
        parameters, std::move(data), std::move(distance), num_threads
    );
}

template <typename Dispatcher>
void register_lvq_vamana_build_from_file(Dispatcher& dispatcher) {
    compressed_specializations(
        [&dispatcher]<typename D, size_t P, size_t R, size_t N, typename S, bool B>() {
            if constexpr (B /* build-enabled*/) {
                auto method = &build_lvq_from_file<P, R, N, S, D>;
                dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
            }
        }
    );
}

template <typename Primary, typename Secondary, size_t L, size_t N, typename D>
svs::Vamana build_leanvec_from_file(
    const svs::index::vamana::VamanaBuildParameters& parameters,
    leanvec::LeanVecLoader<Primary, Secondary, L, N, Allocator> data,
    D distance,
    size_t num_threads
) {
    return svs::Vamana::build<float>(
        parameters, std::move(data), std::move(distance), num_threads
    );
}

template <typename Dispatcher>
void register_leanvec_vamana_build_from_file(Dispatcher& dispatcher) {
    leanvec_specializations(
        [&dispatcher]<typename P, typename S, size_t L, size_t N, typename D>() {
            auto method = &build_leanvec_from_file<P, S, L, N, D>;
            dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
        }
    );
}

template <typename Dispatcher>
void register_vamana_build_from_file(Dispatcher& dispatcher) {
    register_uncompressed_vamana_build_from_file(dispatcher);
    register_lvq_vamana_build_from_file(dispatcher);
    register_leanvec_vamana_build_from_file(dispatcher);
}

using VamanaBuildTypes = std::variant<UnspecializedVectorDataLoader, LVQ, LeanVec>;

/////
///// Build from Array
/////

template <typename Q, typename T, size_t N>
svs::Vamana uncompressed_build_from_array(
    const svs::index::vamana::VamanaBuildParameters& parameters,
    svs::data::ConstSimpleDataView<T, N> view,
    svs::DistanceType distance_type,
    size_t num_threads
) {
    auto data =
        svs::data::SimpleData<T, N, RebindAllocator<T>>(view.size(), view.dimensions());
    svs::data::copy(view, data);
    return svs::Vamana::build<Q>(parameters, std::move(data), distance_type, num_threads);
}

template <typename Dispatcher>
void register_vamana_build_from_array(Dispatcher& dispatcher) {
    for_standard_specializations(
        [&dispatcher]<typename Q, typename T, size_t N, EnableBuild B>() {
            if constexpr (enable_build_from_array<B>) {
                auto method = &uncompressed_build_from_array<Q, T, N>;
                dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
            }
        }
    );
}

/////
///// Dispatch Invocation
/////

using AssemblyDispatcher = svs::lib::Dispatcher<
    svs::Vamana,
    const std::filesystem::path&,
    const UnspecializedGraphLoader&,
    VamanaAssembleTypes,
    svs::DistanceType,
    size_t>;

AssemblyDispatcher assembly_dispatcher() {
    auto dispatcher = AssemblyDispatcher{};

    // Register available backend methods.
    register_vamana_assembly(dispatcher);
    return dispatcher;
}

// Assemble
svs::Vamana assemble(
    const std::string& config_path,
    const UnspecializedGraphLoader& graph_file,
    VamanaAssembleTypes data_kind,
    svs::DistanceType distance_type,
    svs::DataType SVS_UNUSED(query_type),
    bool SVS_UNUSED(enforce_dims),
    size_t num_threads
) {
    return assembly_dispatcher().invoke(
        config_path, graph_file, std::move(data_kind), distance_type, num_threads
    );
}

// Build from file
using BuildFromFileDispatcher = svs::lib::Dispatcher<
    svs::Vamana,
    const svs::index::vamana::VamanaBuildParameters&,
    VamanaBuildTypes,
    svs::DistanceType,
    size_t>;

BuildFromFileDispatcher build_from_file_dispatcher() {
    auto dispatcher = BuildFromFileDispatcher{};
    register_vamana_build_from_file(dispatcher);
    return dispatcher;
}

svs::Vamana build_from_file(
    const svs::index::vamana::VamanaBuildParameters& parameters,
    VamanaBuildTypes data_source,
    svs::DistanceType distance_type,
    size_t num_threads
) {
    return build_from_file_dispatcher().invoke(
        parameters, std::move(data_source), distance_type, num_threads
    );
}

// Build from array.
//
// We go through a dance of accepting a numpy-array, using type-erasure on the pointer
// obtained from the array, and then feeding that `AnonymousVectorData` object into our
// own internal dispatching infrastructures.
//
// This is largely because we can't communicate directly to Python which types are accepted
// by the backend - so we need to do it ourselves.
using BuildFromArrayDispatcher = svs::lib::Dispatcher<
    svs::Vamana,
    const svs::index::vamana::VamanaBuildParameters&,
    AnonymousVectorData,
    svs::DistanceType,
    size_t>;

BuildFromArrayDispatcher build_from_array_dispatcher() {
    auto dispatcher = BuildFromArrayDispatcher{};
    register_vamana_build_from_array(dispatcher);
    return dispatcher;
}

svs::Vamana build_from_array(
    const svs::index::vamana::VamanaBuildParameters& parameters,
    AnonymousVectorData py_data,
    svs::DistanceType distance_type,
    size_t num_threads
) {
    return build_from_array_dispatcher().invoke(
        parameters, py_data, distance_type, num_threads
    );
}

// Templatize at the top level
// Immediately tag the type of the incoming pointer, apply type erasure, and jump into
// internal dispatching.
template <typename ElementType>
void add_build_specialization(py::class_<svs::Vamana>& vamana) {
    vamana.def_static(
        "build",
        [](const svs::index::vamana::VamanaBuildParameters& parameters,
           py_contiguous_array_t<ElementType> py_data,
           svs::DistanceType distance_type,
           size_t num_threads) {
            return build_from_array(
                parameters, AnonymousVectorData(py_data), distance_type, num_threads
            );
        },
        py::arg("parameters"),
        py::arg("py_data"),
        py::arg("distance_type"),
        py::arg("num_threads") = 1,
        R"(
Construct a Vamana index over the given data, returning a searchable index.

Args:
    parameters: Parameters controlling graph construction.
        See the documentation of this class.
    py_data: The dataset to index. *NOTE*: SVS will maintain an internal copy
        of the dataset. This may change in future releases.
    distance_type: The distance type to use for this dataset.
    num_threads: The number of threads to use for index construction. Default: 1.
)"
    );
}

void save_index(
    svs::Vamana& index,
    const std::string& config_path,
    const std::string& graph_dir,
    const std::string& data_dir
) {
    index.save(config_path, graph_dir, data_dir);
}

constexpr std::string_view ASSEMBLE_DOCSTRING_PROTO = R"(
Load a Vamana style index from disk.

Args:
    config_path: Path to the directory where the index configuration file was generated.
    graph_loader: The loader class for the graph.
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
    num_threads: The number of threads to use for queries (can be changed after loading).

The top level type is an abstract type backed by various specialized backends that will
be instantiated based on their applicability to the particular problem instance.

The arguments upon which specialization is conducted are:

* `data_loader`: Both kind (type of loader) and inner aspects of the loader like data type,
  quantization type, and number of dimensions.
* `distance`: The distance measure being used.

Specializations compiled into the binary are listed below.

{}
)";

void wrap_assemble(py::class_<svs::Vamana>& vamana) {
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

    vamana.def(
        py::init(&detail::assemble),
        py::arg("config_path"),
        py::arg("graph_loader"),
        py::arg("data_loader"),
        py::arg("distance") = svs::L2,
        py::arg("query_type") = svs::DataType::float32,
        py::arg("enforce_dims") = false,
        py::arg("num_threads") = 1,
        fmt::format(ASSEMBLE_DOCSTRING_PROTO, dynamic).c_str()
    );
}

void wrap_build_from_file(py::class_<svs::Vamana>& vamana) {
    constexpr std::string_view docstring_proto = R"(
Construct a Vamana index over the given data file, returning a searchable index.

Args:
    build_parameters (:py:class:`svs.VamanaBuildParameters`): Hyper-parameters
        controlling index build.
    data_loader: The source of the data on-disk. Can either be
        :py:class:`svs.DataFile` to represent a standard uncompressed dataset, or a
        compressed loader.
    distance_type: The similarity-function to use for this index.
    num_threads: The number of threads to use for index construction. Default: 1.

The top level type is an abstract type backed by various specialized backends that will
be instantiated based on their applicability to the particular problem instance.

The arguments upon which specialization is conducted are:

* `data_loader`: Both kind (type of loader) and inner aspects of the loader like data type,
  quantization type, and number of dimensions.
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

    vamana.def_static(
        "build",
        &detail::build_from_file,
        py::arg("build_parameters"),
        py::arg("data_loader"),
        py::arg("distance_type"),
        py::arg("num_threads") = 1,
        fmt::format(docstring_proto, dynamic).c_str()
    );
}

} // namespace detail

void wrap(py::module& m) {
    wrap_common(m);

    /// Build Parameters
    py::class_<svs::index::vamana::VamanaBuildParameters> parameters(
        m, "VamanaBuildParameters", "Build parameters for Vamana index construction."
    );

    parameters
        .def(
            py::init([](float alpha,
                        size_t graph_max_degree,
                        size_t window_size,
                        size_t max_candidate_pool_size,
                        size_t prune_to,
                        size_t num_threads) {
                if (num_threads != std::numeric_limits<size_t>::max()) {
                    PyErr_WarnEx(
                        PyExc_DeprecationWarning,
                        "Constructing VamanaBuildParameters with the \"num_threads\" "
                        "keyword "
                        "argument is deprecated, no longer has any effect, and will be "
                        "removed "
                        "from future versions of the library. Use the \"num_threads\" "
                        "keyword "
                        "argument of \"svs.Vamana.build\" instead!",
                        1
                    );
                }

                // Default the `prune_to` argument appropriately.
                if (prune_to == std::numeric_limits<size_t>::max()) {
                    prune_to = graph_max_degree;
                }

                return svs::index::vamana::VamanaBuildParameters{
                    alpha,
                    graph_max_degree,
                    window_size,
                    max_candidate_pool_size,
                    prune_to,
                    true};
            }),
            py::arg("alpha") = 1.2,
            py::arg("graph_max_degree") = 32,
            py::arg("window_size") = 64,
            py::arg("max_candidate_pool_size") = 80,
            py::arg("prune_to") = std::numeric_limits<size_t>::max(),
            py::arg("num_threads") = std::numeric_limits<size_t>::max(),
            R"(
            Construct a new instance from keyword arguments.

            Args:
                alpha: Prune threshold degree for graph construction.
                    For distance types favoring minimization, set this to a number
                    greater than 1.0 (typically, 1.2 is sufficient). For distance types
                    preferring maximization, set to a value less than 1.0 (such as 0.95).
                graph_max_degree: The maximum out-degree in the final graph. Graphs with
                    a higher degree tend to yield better accuracy and performance at the cost
                    of a larger memory footprint.
                window_size: Parameter controlling the quality of graph construction. A
                    larger window size will yield a higher-quality index at the cost of
                    longer construction time. Should be larger than `graph_max_degree`.
                max_candidate_pool_size: Limit on the number of candidates to consider
                    for neighbor updates. Should be larger than `window_size`.
                prune_to: Amount candidate lists will be pruned to when exceeding the
                    target max degree. In general, setting this to slightly less than
                    `graph_max_degree` will yield faster index building times. Default:
                    `graph_max_degree`.
            )"
        )
        .def_readwrite("alpha", &svs::index::vamana::VamanaBuildParameters::alpha)
        .def_readwrite(
            "graph_max_degree", &svs::index::vamana::VamanaBuildParameters::graph_max_degree
        )
        .def_readwrite(
            "window_size", &svs::index::vamana::VamanaBuildParameters::window_size
        )
        .def_readwrite(
            "max_candidate_pool_size",
            &svs::index::vamana::VamanaBuildParameters::max_candidate_pool_size
        )
        .def_readwrite("prune_to", &svs::index::vamana::VamanaBuildParameters::prune_to)
        .def_readwrite(
            "use_full_search_history",
            &svs::index::vamana::VamanaBuildParameters::use_full_search_history
        );

    ///
    /// Vamana Static Module
    ///
    std::string name = "Vamana";
    py::class_<svs::Vamana> vamana(
        m, name.c_str(), "Top level class for the Vamana graph index."
    );

    detail::wrap_assemble(vamana);

    // Make the Vamana type searchable.
    add_search_specialization<svs::Float16>(vamana);
    add_search_specialization<float>(vamana);
    add_search_specialization<uint8_t>(vamana);
    add_search_specialization<int8_t>(vamana);

    // Add threading layer.
    add_threading_interface(vamana);

    // Data layer
    add_data_interface(vamana);

    // Vamana Specific Extensions.
    add_interface(vamana);

    // Reconstruction.
    add_reconstruct_interface(vamana);

    ///// Index building
    // Build from Numpy array.
    detail::add_build_specialization<svs::Float16>(vamana);
    detail::add_build_specialization<float>(vamana);
    detail::add_build_specialization<uint8_t>(vamana);
    detail::add_build_specialization<int8_t>(vamana);

    // Build from datasets on file.
    detail::wrap_build_from_file(vamana);

    ///// Index Saving.
    vamana.def(
        "save",
        &detail::save_index,
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
} // namespace svs::python::vamana
