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

// header
#include "core.h"

// svs
#include "svs/core/data.h"
#include "svs/lib/datatype.h"

// pybind
#include <pybind11/pybind11.h>

// stl
#include <string_view>

namespace py = pybind11;

namespace {

constexpr std::string_view compression_constructor_proto = R"(
Construct a loader that will lazily compress the results of the data loader.

Args:
    loader (:py:class:`pysvs.VectorDataLoader`): The uncompressed dataset to compress
        in-memory.
    padding (int): The value (in bytes) to align the beginning of each compressed vectors.
        Values of 32 or 64 may offer the best performance at the cost of a lower compression
        ratio. A value of 0 implies no special alignment.
)";

constexpr std::string_view reload_constructor_proto = R"(
Reload a compressed dataset from a previously saved dataset.

Args:
    directory (str): The directory where the dataset was previously saved.
    dims (int): The number of dimensions in the dataset. May provide a performance boost
      if given if a specialization has been compiled. Default: Dynamic (any dimension).
    padding (int): The value (in bytes) to align the beginning of each compressed vectors.
      Values of 32 or 64 may offer the best performance at the cost of a lower compression
      ratio. A value of 0 implies no special alignment. Default: 0.
)";

template <typename T>
void wrap_lvq_definition(
    py::module& m, std::string_view class_name, std::string_view docstring
) {
    auto class_def =
        py::class_<T>{m, std::string(class_name).c_str(), std::string(docstring).c_str()};
    class_def
        .def(
            py::init<UnspecializedVectorDataLoader, size_t>(),
            py::arg("datafile"),
            py::arg("padding") = 0,
            std::string(compression_constructor_proto).c_str()
        )
        .def(
            py::init([](const std::string& path, size_t dims, size_t padding) {
                return T{LVQReloader(path), dims, padding};
            }),
            py::arg("directory"),
            py::arg("dims") = svs::Dynamic,
            py::arg("padding") = 0,
            std::string(reload_constructor_proto).c_str()
        );
}

/// Generate bindings for LVQ compressors and loaders.
void wrap_lvq(py::module& m) {
    // Compression Sources
    wrap_lvq_definition<LVQ4>(m, "LVQ4", "Perform one level LVQ compression using 4-bits.");
    wrap_lvq_definition<LVQ8>(m, "LVQ8", "Perform one level LVQ compression using 8-bits.");
    wrap_lvq_definition<LVQ4x4>(
        m,
        "LVQ4x4",
        "Perform two level compression using 4 bits for the primary and residual."
    );
    wrap_lvq_definition<LVQ4x8>(
        m,
        "LVQ4x8",
        "Perform two level compression using 4 bits for the primary and 8 bits for the "
        "residual residual."
    );
    wrap_lvq_definition<LVQ8x8>(
        m,
        "LVQ8x8",
        "Perform two level compression using 8 bits for the primary and residual."
    );
}
} // namespace

namespace core {
void wrap(py::module& m) {
    ///// UnspecializedVectorDataLoader
    py::class_<UnspecializedVectorDataLoader> loader(
        m, "VectorDataLoader", "Handle representing an uncompressed vector data file."
    );
    loader
        .def(
            py::init<std::string, svs::DataType, size_t>(),
            py::arg("path"),
            py::arg("data_type"),
            py::arg("dims") = svs::Dynamic,
            R"(
Construct a new ``pysvs.VectorDataLoader``.

Args:
    path (str): The path to the file to load. This can either be:

        * The path to the directory where a previous vector dataset was saved (preferred).
        * The direct path to the vector data file itself. In this case, the type of the file
          will try to be inferred automatically. Recognized extensions: ".[b/i/f]vecs",
          ".bin", and ".svs".

    data_type (:py:class:`pysvs.DataType`): The native type of the elements in the dataset.
    dims (int): The expected dimsionality of the dataset. While this argument is generally
        optional, providing it may yield runtime speedups.
        )"
        )
        .def_readwrite(
            "filepath",
            &UnspecializedVectorDataLoader::path_,
            "Read/Write (str): Access the underlying file path."
        )
        .def_readwrite(
            "data_type",
            &UnspecializedVectorDataLoader::type_,
            "Read/Write (:py:class:`pysvs.DataType`): Access the assigned data type."
        )
        .def_readwrite(
            "dims",
            &UnspecializedVectorDataLoader::dims_,
            "Read/Write (int): Access the expected dimensionality."
        );

    ///// GraphLoader
    py::class_<UnspecializedGraphLoader> graph_loader(
        m, "GraphLoader", "Loader for graph files."
    );
    graph_loader.def(
        py::init<std::string>(),
        py::arg("directory"),
        R"(
Construct a new ``pysvs.GraphLoader``.

Args:
    directory (str): The path to the directory where the graph is stored.
        )"
    );

    ///// LVQ
    wrap_lvq(m);
}
} // namespace core
