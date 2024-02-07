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
#include <pybind11/stl.h>

// stl
#include <optional>
#include <string_view>

namespace py = pybind11;

namespace {

constexpr std::string_view compression_constructor_proto = R"(
Construct a loader that will lazily compress the results of the data loader.
Requires an appropriate back-end to be compiled for all combinations of primary and residual
bits.

Args:
    loader (:py:class:`pysvs.VectorDataLoader`): The uncompressed dataset to compress
        in-memory.
    primary (int): The number of bits to use for compression in the primary dataset.
    residual (int): The number of bits to use for compression in the residual dataset.
        Default: 0.
    padding (int): The value (in bytes) to align the beginning of each compressed vectors.
        Values of 32 or 64 may offer the best performance at the cost of a lower compression
        ratio. A value of 0 implies no special alignment.
    strategy (:py:class:`pysvs.LVQStrategy`): The packing strategy to use for the compressed
        codes. See the associated documenation for that enum.
)";

constexpr std::string_view reload_constructor_proto = R"(
Reload a compressed dataset from a previously saved dataset.
Requires an appropriate back-end to be compiled for all combinations of primary and residual
bits.

Args:
    directory (str): The directory where the dataset was previously saved.
    primary (int): The number of bits to use for compression in the primary dataset.
    residual (int): The number of bits to use for compression in the residual dataset.
        Default: 0>
    dims (int): The number of dimensions in the dataset. May provide a performance boost
        if given if a specialization has been compiled. Default: Dynamic (any dimension).
    padding (int): The value (in bytes) to align the beginning of each compressed vectors.
        Values of 32 or 64 may offer the best performance at the cost of a lower compression
        ratio. A value of 0 implies no special alignment. Default: 0.
    strategy (:py:class:`pysvs.LVQStrategy`): The packing strategy to use for the compressed
        codes. See the associated documenation for that enum.
)";

constexpr std::string_view leanvec_online_proto = R"(
Construct a loader that will lazily reduce the dimensionality of the data loader.
Requires an appropriate back-end to be compiled for all combinations of primary and
secondary types.

Args:
    loader (:py:class:`pysvs.VectorDataLoader`): The uncompressed original dataset.
    leanvec_dims (int): resulting value of reduced dimensionality
    primary (LeanVecKind): Type of dataset used for Primary (Default: LVQ8)
    secondary (LeanVecKind): Type of dataset used for Secondary (Default: LVQ8)
    data_matrix (Optional[numpy.ndarray[numpy.float32]]): Matrix for data transformation
        [see note 1] (Default: None).
    query_matrix (Optional[numpy.ndarray[numpy.float32]]): Matrix for query transformation
        [see note 1] (Default: None).
    alignment (int):  alignement/padding used in LVQ data types (Default: 32)

**Note 1**: The arguments ``data_matrix`` and ``data_matrix`` are optional and have the
following requirements for valid combinations:

    a) Neither matrix provided: Transform dataset and queries using a default PCA-based
       transformation.
    b) Only ``data_matrix`` provided: The provided matrix is used to transform both the
       queries and the original dataset.
    c) Both arguments are provided: Use the respective matrices for transformation.
)";

constexpr std::string_view leanvec_reload_proto = R"(
Reload a LeanVec dataset from a previously saved dataset.
Requires an appropriate back-end to be compiled for all combinations of primary and
secondary types.

Args:
    directory (str): The directory where the dataset was previously saved.
    leanvec_dims (int): resulting value of reduced dimensionality.
        Default: Dynamic (any dimension).
    dims (int): The number of dimensions in the original dataset.
        Default: Dynamic (any dimension).
    primary (LeanVecKind): Type of dataset used for Primary
        Default: ``pysvs.LeanVecKind.lvq8``.
    secondary (LeanVecKind): Type of dataset used for Secondary
        Default: ``pysvs.LeanVecKind.LVQ8``.
    alignment (int):  alignement/padding used in LVQ data types. Default: 32.
)";

// Legacy definitions.
template <size_t Primary, size_t Residual> struct LegacyLVQLoader {
  public:
    LegacyLVQLoader(UnspecializedVectorDataLoader loader, size_t padding)
        : loader_{std::move(loader), Primary, Residual, padding} {}

    LegacyLVQLoader(std::string path, size_t dims, size_t padding)
        : loader_{LVQReloader{std::move(path)}, Primary, Residual, dims, padding} {}

    // Implicitly convert to generic LVQ.
    operator LVQ() const { return loader_; }

  public:
    LVQ loader_;
};

template <size_t Primary, size_t Residual, typename Parent>
void wrap_lvq_alias(
    Parent& lvq_loader,
    py::module& m,
    std::string_view class_name,
    std::string_view docstring
) {
    auto class_def = py::class_<LegacyLVQLoader<Primary, Residual>>{
        m, std::string(class_name).c_str(), std::string(docstring).c_str()};

    // Define a converting constructor taking the legacy type.
    lvq_loader.def(
        py::init([](const LegacyLVQLoader<Primary, Residual>& legacy) { return legacy; }),
        py::arg("legacy")
    );

    // Allow implicit conversions from LegacyLVQLoader to LVQLoader.
    py::implicitly_convertible<LegacyLVQLoader<Primary, Residual>, LVQ>();

    // Alias the datafile constructor.
    class_def.def(
        py::init<UnspecializedVectorDataLoader, size_t>(),
        py::arg("datafile"),
        py::arg("padding") = 0,
        std::string(compression_constructor_proto).c_str()
    );

    // Alias the reload constructor
    class_def.def(
        py::init<std::string, size_t, size_t>(),
        py::arg("datafile"),
        py::arg("dims") = svs::Dynamic,
        py::arg("padding") = 0,
        std::string(reload_constructor_proto).c_str()
    );
}

/// Generate bindings for LVQ compressors and loaders.
void wrap_lvq(py::module& m) {
    using enum svs::quantization::lvq::LVQStrategyDispatch;

    // Strategy Dispatch enum.
    py::enum_<svs::quantization::lvq::LVQStrategyDispatch>(
        m, "LVQStrategy", "Select the packing mode for LVQ"
    )
        .value("Auto", Auto, "Let SVS decide the best strategy.")
        .value("Sequential", Sequential, "Use the Sequential packing strategy.")
        .value("Turbo", Turbo, "Use the best Turbo packing strategy for this architecture.")
        .export_values();

    // Wrap the base class.
    auto class_def = py::class_<LVQ>{m, "LVQLoader", "Generic LVQ Loader"};
    class_def
        .def(
            py::init<
                UnspecializedVectorDataLoader,
                size_t,
                size_t,
                size_t,
                svs::quantization::lvq::LVQStrategyDispatch>(),
            py::arg("datafile"),
            py::arg("primary"),
            py::arg("residual") = 0,
            py::arg("padding") = 0,
            py::arg("strategy") = Auto,
            std::string(compression_constructor_proto).c_str()
        )
        .def(
            py::init([](const std::string& path,
                        size_t primary,
                        size_t residual,
                        size_t dims,
                        size_t padding,
                        svs::quantization::lvq::LVQStrategyDispatch strategy) {
                return LVQ{LVQReloader(path), primary, residual, dims, padding, strategy};
            }),
            py::arg("directory"),
            py::arg("primary"),
            py::arg("residual") = 0,
            py::arg("dims") = svs::Dynamic,
            py::arg("padding") = 0,
            py::arg("strategy") = Auto,
            std::string(reload_constructor_proto).c_str()
        )
        .def(
            "reload_from",
            [](const LVQ& loader, const std::string& dir) {
                auto copy = loader;
                copy.source_ = LVQReloader{dir};
                return copy;
            },
            // py::arg("lvq_loader"),
            py::arg("directory"),
            R"(
Create a copy of the argument loader configured to reload a previously saved LVQ dataset
from the given directory.)"
        );

    // Compression Sources
    wrap_lvq_alias<4, 0>(
        class_def, m, "LVQ4", "Perform one level LVQ compression using 4-bits."
    );
    wrap_lvq_alias<8, 0>(
        class_def, m, "LVQ8", "Perform one level LVQ compression using 8-bits."
    );
    wrap_lvq_alias<4, 4>(
        class_def,
        m,
        "LVQ4x4",
        "Perform two level compression using 4 bits for the primary and residual."
    );
    wrap_lvq_alias<4, 8>(
        class_def,
        m,
        "LVQ4x8",
        "Perform two level compression using 4 bits for the primary and 8 bits for the "
        "residual residual."
    );
    wrap_lvq_alias<8, 8>(
        class_def,
        m,
        "LVQ8x8",
        "Perform two level compression using 8 bits for the primary and residual."
    );
}

using MatrixType = float;
using MatrixAlloc = svs::lib::Allocator<MatrixType>;
using MatrixData = svs::data::SimpleData<MatrixType, Dynamic, MatrixAlloc>;

// Helper function to convert leanvec Python matrices to SimpleData
// Bundles both the matrices in a tuple
template <typename M>
std::optional<svs::leanvec::LeanVecMatrices<Dynamic>> convert_leanvec_matrices(
    const std::optional<M>& data_matrix, const std::optional<M>& query_matrix
) {
    // Convert the matrices from Python arrays to SimpleData
    auto data_matrix_ =
        transform_optional(create_data<MatrixType, MatrixAlloc>, data_matrix);
    auto query_matrix_ =
        transform_optional(create_data<MatrixType, MatrixAlloc>, query_matrix);

    if (data_matrix_.has_value() && !query_matrix_.has_value()) {
        fmt::print("Warning: Query matrix not provided, using the Data matrix for both!");
        query_matrix_ = data_matrix_;
    } else if (query_matrix_.has_value() && !data_matrix_.has_value()) {
        throw ANNEXCEPTION("Invalid option: Query matrix provided but not the Data matrix!"
        );
    }

    if (!data_matrix_.has_value()) {
        return std::nullopt;
    }

    return std::optional<svs::leanvec::LeanVecMatrices<Dynamic>>(
        std::in_place, std::move(data_matrix_).value(), std::move(query_matrix_).value()
    );
}

/// Generate bindings for LeanVec compressors and loaders.
void wrap_leanvec(py::module& m) {
    using enum svs::leanvec::LeanVecKind;

    // Kind of data types used for primary and secondary.
    py::enum_<svs::leanvec::LeanVecKind>(
        m, "LeanVecKind", "LeanVec primary and secondary types"
    )
        .value("float32", float32, "Uncompressed float32")
        .value("float16", float16, "Uncompressed float16")
        .value("lvq8", lvq8, "Compressed with LVQ 8bits")
        .value("lvq4", lvq4, "Compressed with LVQ 4bits");

    // Wrap the base class.
    auto class_def = py::class_<LeanVec>{m, "LeanVecLoader", "Generic LeanVec Loader"};
    class_def
        .def(
            py::init([](UnspecializedVectorDataLoader datafile,
                        size_t leanvec_dims,
                        svs::leanvec::LeanVecKind primary_kind,
                        svs::leanvec::LeanVecKind secondary_kind,
                        const std::optional<py_contiguous_array_t<float>>& data_matrix,
                        const std::optional<py_contiguous_array_t<float>>& query_matrix,
                        size_t alignment) {
                return LeanVec{
                    datafile,
                    leanvec_dims,
                    primary_kind,
                    secondary_kind,
                    convert_leanvec_matrices(data_matrix, query_matrix),
                    alignment};
            }),
            py::arg("datafile"),
            py::arg("leanvec_dims"),
            py::arg("primary_kind") = lvq8,
            py::arg("secondary_kind") = lvq8,
            py::arg("data_matrix") = py::none(),
            py::arg("query_matrix") = py::none(),
            py::arg("alignment") = 32,
            std::string(leanvec_online_proto).c_str()
        )
        .def(
            py::init([](const std::string& path,
                        size_t leanvec_dims,
                        size_t dims,
                        svs::leanvec::LeanVecKind primary_kind,
                        svs::leanvec::LeanVecKind secondary_kind,
                        size_t alignment) {
                return LeanVec{
                    LeanVecReloader(path),
                    leanvec_dims,
                    dims,
                    primary_kind,
                    secondary_kind,
                    alignment};
            }),
            py::arg("directory"),
            py::arg("leanvec_dims") = svs::Dynamic,
            py::arg("dims") = svs::Dynamic,
            py::arg("primary_kind") = lvq8,
            py::arg("secondary_kind") = lvq8,
            py::arg("alignment") = 32,
            std::string(leanvec_reload_proto).c_str()
        )
        .def(
            "reload_from",
            [](const LeanVec& loader, const std::string& dir) {
                auto copy = loader;
                copy.source_ = LeanVecReloader{dir};
                return copy;
            },
            py::arg("directory"),
            R"(
Create a copy of the argument loader configured to reload a previously saved LeanVec dataset
from the given directory.)"
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

    ///// LeanVec
    wrap_leanvec(m);
}
} // namespace core
