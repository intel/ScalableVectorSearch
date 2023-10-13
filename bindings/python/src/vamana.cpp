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
#include "vamana.h"
#include "common.h"
#include "core.h"
#include "manager.h"

// svs
#include "svs/core/data/simple.h"
#include "svs/core/distance.h"
#include "svs/extensions/vamana/lvq.h"
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
using namespace vamana_specializations;

namespace vamana {
namespace detail {

/// Assemble an uncompressed index.
struct StandardAssemble_ {
    /// Keys:
    /// (0) - The type of the elements of the query vectors.
    /// (1) - The type of the elements of the data vectors.
    /// (2) - The compile-time dimensionality (or Dynamic).
    using key_type = std::tuple<svs::DataType, svs::DataType, size_t>;
    using mapped_type = std::function<svs::Vamana(
        const std::filesystem::path& /*config_path*/,
        const UnspecializedGraphLoader& /*graph_loader*/,
        const UnspecializedVectorDataLoader& /*data_loader*/,
        svs::DistanceType /*distance_type*/,
        size_t /*num_threads*/
    )>;

    // Specialize index assembling based on run-time values for:
    //
    // * Query Type: The element types of each component of the query vectors.
    // * Data Type: The element types for each component of the data base vectors.
    // * Data Dimensionality: The number of elements in each vector.
    template <typename Q, typename T, size_t N>
    static std::pair<key_type, mapped_type>
    specialize(Type<Q> query_type, Type<T> data_type, Val<N> ndims) {
        key_type key = {unwrap(query_type), unwrap(data_type), unwrap(ndims)};
        mapped_type fn = [=](const std::filesystem::path& config_path,
                             const UnspecializedGraphLoader& graph_loader,
                             const UnspecializedVectorDataLoader& data,
                             svs::DistanceType distance_type,
                             size_t num_threads) {
            return svs::Vamana::assemble<Q>(
                config_path,
                graph_loader,
                data.refine(data_type, ndims),
                distance_type,
                num_threads
            );
        };
        return std::make_pair(key, std::move(fn));
    }

    // Generate all requested specializations.
    template <typename F> static void fill(F&& f) {
        for_standard_specializations(
            [&f]<EnableBuild B>(
                auto query_type, auto data_type, auto ndims, Const<B> /*unused*/
            ) { f(specialize(query_type, data_type, ndims)); }
        );
    }
};
using StandardAssembler = svs::lib::Dispatcher<StandardAssemble_>;

/// Build an uncompressed index.
struct StandardBuild_ {
    // Keys: Data element type, Dimensionality.
    using key_type = std::tuple<svs::DataType, size_t>;
    using mapped_type = std::function<svs::Vamana(
        const svs::index::vamana::VamanaBuildParameters&,
        const UnspecializedVectorDataLoader&,
        svs::DistanceType,
        size_t
    )>;

    // Specialize index building based on run-time values for:
    //
    // * Query Type: The element types of each component of the query vectors.
    // * Data Type: The element types for each component of the data base vectors.
    // * Data Dimensionality: The number of elements in each vector.
    template <typename Q, typename T, size_t N>
    static std::pair<key_type, mapped_type>
    specialize(Type<Q> /*query_type*/, Type<T> data_type, Val<N> ndims) {
        auto key = key_type{unwrap(data_type), unwrap(ndims)};
        mapped_type fn = [=](const svs::index::vamana::VamanaBuildParameters& parameters,
                             const UnspecializedVectorDataLoader& data,
                             svs::DistanceType distance_type,
                             size_t num_threads) {
            return svs::Vamana::build<Q>(
                parameters, data.refine(data_type, ndims), distance_type, num_threads
            );
        };
        return std::make_pair(key, std::move(fn));
    }

    // Generate all requested specializations.
    template <typename F> static void fill(F&& f) {
        for_standard_specializations(
            [&f]<EnableBuild B>(
                auto query_type, auto data_type, auto ndims, Const<B> /*unused*/
            ) {
                // Only instantiate a specialization if building is explicitly enabled
                // for this combination of types/values.
                if constexpr (enable_build_from_file<B>) {
                    f(specialize(query_type, data_type, ndims));
                }
            }
        );
    }
};
using StandardBuilder = svs::lib::Dispatcher<StandardBuild_>;

///
/// Load a compressed dataset from files, optionally compressing on the fly.
///
template <typename Kind> struct CompressedAssemble_ {
    using key_type = std::tuple<svs::DistanceType, size_t>;
    using mapped_type = std::function<svs::Vamana(
        const std::filesystem::path& /*config_path*/,
        const UnspecializedGraphLoader& /*graph_loader*/,
        const Kind& /*data_loader*/,
        size_t /*n_threads*/
    )>;

    // Specializd index assembling based on run-time values.
    template <typename Distance, size_t N>
    static std::pair<key_type, mapped_type> specialize(Distance distance, Val<N> dims) {
        key_type key = {svs::distance_type_v<Distance>, unwrap(dims)};
        mapped_type fn = [=](const std::filesystem::path& config_path,
                             const UnspecializedGraphLoader& graph_loader,
                             const Kind& loader,
                             size_t num_threads) {
            return svs::Vamana::assemble<float>(
                config_path, graph_loader, loader.refine(dims), distance, num_threads
            );
        };
        return std::make_pair(key, std::move(fn));
    }

    template <typename F> static void fill(F&& f) {
        compressed_specializations([&f](auto distance, auto dims, auto /*enable_build*/) {
            f(specialize(distance, dims));
        });
    }
};

template <typename Kind>
using CompressedAssembler = svs::lib::Dispatcher<CompressedAssemble_<Kind>>;

/// Perform index construction using a compressed dataset.
template <typename Kind> struct CompressedBuild_ {
    // Type Aliases
    using key_type = std::tuple<svs::DistanceType, size_t>;
    using mapped_type = std::function<svs::Vamana(
        const svs::index::vamana::VamanaBuildParameters& /*build_parameters*/,
        const Kind& /*data_loader*/,
        size_t /*num_threads*/
    )>;

    template <typename Distance, size_t N>
    static std::pair<key_type, mapped_type> specialize(Distance distance, Val<N> dims) {
        key_type key = {svs::distance_type_v<Distance>, unwrap(dims)};
        mapped_type fn = [=](const svs::index::vamana::VamanaBuildParameters& parameters,
                             const Kind& loader,
                             size_t num_threads) {
            return svs::Vamana::build<float>(
                parameters, loader.refine(dims), distance, num_threads
            );
        };
        return std::make_pair(key, std::move(fn));
    }

    template <typename F> static void fill(F&& f) {
        compressed_specializations(
            [&f]<
                bool EnableBuild>(auto distance, auto dims, Const<EnableBuild> /*unused*/) {
                if constexpr (EnableBuild) {
                    f(specialize(distance, dims));
                }
            }
        );
    }
};
template <typename Kind>
using CompressedBuild = svs::lib::Dispatcher<CompressedBuild_<Kind>>;

// Dataset Load Dispatch Logic.

/// @brief Types used for index assembly.
using VamanaAssembleTypes =
    std::variant<UnspecializedVectorDataLoader, LVQ4, LVQ8, LVQ4x4, LVQ4x8, LVQ8x8>;

/// @brief Types used for index construction.
using VamanaBuildSourceTypes =
    std::variant<UnspecializedVectorDataLoader, LVQ8, LVQ4, LVQ4x4>;

svs::Vamana assemble(
    const std::string& config_path,
    const UnspecializedGraphLoader& graph_file,
    const VamanaAssembleTypes& data_kind,
    svs::DistanceType distance_type,
    svs::DataType query_type,
    bool enforce_dims,
    size_t num_threads
) {
    return std::visit<svs::Vamana>(
        [&](auto&& loader) {
            using T = std::decay_t<decltype(loader)>;
            if constexpr (std::is_same_v<T, UnspecializedVectorDataLoader>) {
                // Get the pre-dispatched dataset loader.
                // If `force_specialization == false`, then try the generic fallback in
                // dimensionality for the given types.
                const auto& f = StandardAssembler::lookup(
                    !enforce_dims, loader.dims_, query_type, loader.type_
                );

                return f(config_path, graph_file, loader, distance_type, num_threads);
            } else {
                const auto& f = CompressedAssembler<T>::lookup(
                    !enforce_dims, loader.dims_, distance_type
                );

                return f(config_path, graph_file, loader, num_threads);
            }
        },
        data_kind
    );
}

template <typename QueryType, typename ElementType>
svs::Vamana build_from_array(
    const svs::index::vamana::VamanaBuildParameters& parameters,
    py_contiguous_array_t<ElementType> py_data,
    svs::DistanceType distance_type,
    size_t num_threads
) {
    return svs::Vamana::build<QueryType>(
        parameters, create_data(py_data), distance_type, num_threads
    );
}

svs::Vamana build_from_file(
    const svs::index::vamana::VamanaBuildParameters& parameters,
    const VamanaBuildSourceTypes& data_source,
    svs::DistanceType distance_type,
    size_t num_threads
) {
    return std::visit<svs::Vamana>(
        [&](auto&& data_loader) {
            using T = std::decay_t<decltype(data_loader)>;
            // Loading a standard dataset.
            if constexpr (std::is_same_v<T, UnspecializedVectorDataLoader>) {
                const auto& f =
                    StandardBuilder::lookup(true, data_loader.dims_, data_loader.type_);
                return f(parameters, data_loader, distance_type, num_threads);
            } else {
                const auto& f =
                    CompressedBuild<T>::lookup(true, data_loader.dims_, distance_type);
                return f(parameters, data_loader, num_threads);
            }
        },
        data_source
    );
}

template <typename QueryType, typename ElementType>
void add_build_specialization(py::class_<svs::Vamana>& vamana) {
    vamana.def_static(
        "build",
        &build_from_array<QueryType, ElementType>,
        py::arg("parameters"),
        py::arg("py_data"),
        py::arg("distance_type"),
        py::arg("num_threads") = 1,
        R"(
Construct a Vamana index over the given data, returning a searchable index.

Args:
    parameters: Parameters controlling graph construction.
        See the documentation of this class.
    py_data: The dataset to index. **NOTE**: PySVS will maintain an internal copy of the
        dataset. This may change in future releases.
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
} // namespace detail

void wrap(py::module& m) {
    ///
    /// Build Parameters
    ///
    std::string build_name = "VamanaBuildParameters";
    py::class_<svs::index::vamana::VamanaBuildParameters> parameters(
        m, build_name.c_str(), "Build parameters for Vamana index construction."
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
                        "argument of \"pysvs.Vamana.build\" instead!",
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
        For distance types favoring minimization, set this to a number greater than 1.0
        (typically, 1.2 is sufficient). For distance types preferring maximization, set to a
        value less than 1.0 (such as 0.95).
    graph_max_degree: The maximum out-degree in the final graph. Graphs with a higher
        degree tend to yield better accuracy and performance at the cost of a larger memory
        footprint.
    window_size: Parameter controlling the quality of graph construction. A larger window
        size will yield a higher-quality index at the cost of longer construction time. Should
        be larger than `graph_max_degree`.
    max_candidate_pool_size: Limit on the number of candidates to consider for neighbor
        updates. Should be larger than `window_size`.
    prune_to: Amount candidate lists will be pruned to when exceeding the target max
        degree. In general, setting this to slightly less than `graph_max_degree` will
        yield faster index building times. Default: `graph_max_degree`.
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
        );

    ///
    /// Vamana Static Module
    ///
    std::string name = "Vamana";
    py::class_<svs::Vamana> vamana(
        m, name.c_str(), "Top level class for the Vamana graph index."
    );

    vamana.def(
        py::init(&detail::assemble),
        py::arg("config_path"),
        py::arg("graph_loader"),
        py::arg("data_loader"),
        py::arg("distance") = svs::L2,
        py::arg("query_type") = svs::DataType::float32,
        py::arg("enforce_dims") = false,
        py::arg("num_threads") = 1,
        R"(
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

Data types supported by the loader are the following:

    * ``pysvs.DataType.float32``
    * ``pysvs.DataType.float16``
    * ``pysvs.DataType.int8``
    * ``pysvs.DataType.uint8``
        )"
    );

    // Make the Vamana type searchable.
    add_search_specialization<float>(vamana);
    add_search_specialization<uint8_t>(vamana);
    add_search_specialization<int8_t>(vamana);

    // Add threading layer.
    add_threading_interface(vamana);

    // Data layer
    add_data_interface(vamana);

    // Vamana Specific Extensions.
    add_interface(vamana);

    ///// Index building
    // Build from Numpy array.
    detail::add_build_specialization<float, svs::Float16>(vamana);
    detail::add_build_specialization<float, float>(vamana);
    detail::add_build_specialization<uint8_t, uint8_t>(vamana);
    detail::add_build_specialization<int8_t, int8_t>(vamana);

    // Build from datasets on file.
    vamana.def_static(
        "build",
        &detail::build_from_file,
        py::arg("build_parameters"),
        py::arg("data_loader"),
        py::arg("distance_type"),
        py::arg("num_threads") = 1,
        R"(
Construct a Vamana index over the given data file, returning a searchable index.

Args:
    build_parameters (:py:class:`pysvs.VamanaBuildParameters`): Hyper-parameters controlling
        index build.
    data_loader: The source of the data on-disk. Can either be :py:class:`pysvs.DataFile` to
        represent a standard uncompressed dataset, or a compressed loader.
    distance_type: The similarity-function to use for this index.
    num_threads: The number of threads to use for index construction. Default: 1.
        )"
    );

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
} // namespace vamana
