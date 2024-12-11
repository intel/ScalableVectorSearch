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
#include "svs/python/core.h"

// svs
#include "svs/core/data.h"
#include "svs/core/logging.h"
#include "svs/lib/datatype.h"

// pybind
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

// spdlog
#include "spdlog/spdlog.h"

// stl
#include <optional>
#include <string_view>

namespace py = pybind11;

namespace svs::python {
namespace {
using MatrixType = float;
using MatrixAlloc = svs::lib::Allocator<MatrixType>;
using MatrixData = svs::data::SimpleData<MatrixType, Dynamic, MatrixAlloc>;
} // namespace

namespace core {
void wrap(py::module& m) {
    ///// UnspecializedVectorDataLoader
    py::class_<UnspecializedVectorDataLoader> loader(
        m, "VectorDataLoader", "Handle representing an uncompressed vector data file."
    );
    loader
        .def(
            py::init<std::string, std::optional<svs::DataType>, std::optional<size_t>>(),
            py::arg("path"),
            py::arg("data_type") = py::none(),
            py::arg("dims") = py::none(),
            R"(
Construct a new ``svs.VectorDataLoader``.

Args:
    path (str): The path to the file to load. This can either be:

        * The path to the directory where a previous vector dataset was saved (preferred).
        * The direct path to the vector data file itself. In this case, the type of the file
          will try to be inferred automatically. Recognized extensions: ".[b/i/f]vecs",
          ".bin", and ".svs".

    data_type (:py:class:`svs.DataType`): The native type of the elements in the dataset.
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
            "Read/Write (:py:class:`svs.DataType`): Access the assigned data type."
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
Construct a new ``svs.GraphLoader``.

Args:
    directory (str): The path to the directory where the graph is stored.
        )"
    );

    // Enable implicit conversion from `std::filesystem::path` to a GraphLoader, since
    // that's the only context we use the `GraphLoader` for anyways.
    py::implicitly_convertible<std::filesystem::path, UnspecializedGraphLoader>();

    //// SerializedObject
    // Allow implicit conversion from `std::filesystem::path`, to transitively enable
    // implicit construction from Python `PathLike` objects.
    py::class_<svs::lib::SerializedObject> serialized(
        m, "SerializedObject", "A handle to a SVS serialized object"
    );
    serialized.def(py::init([](const std::filesystem::path& path) {
        return svs::lib::begin_deserialization(path);
    }));
    py::implicitly_convertible<std::filesystem::path, svs::lib::SerializedObject>();

    ///// TOML Reconstructions
    m.def("__reformat_toml", [](const std::filesystem::path& path) {
        toml::table t = toml::parse_file(path.c_str());
        auto file = svs::lib::open_write(path, std::ios_base::out);
        file << t << "\n";
    });
}
} // namespace core
} // namespace svs::python
