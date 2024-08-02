/**
 *    Copyright (C) 2023, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
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

///// Logging
enum class LogStream { stdout_, stderr_, null };

void replace_logger_with_sink(svs::logging::sink_ptr sink) {
    auto current_logger = svs::logging::get();
    auto current_level = svs::logging::get_level(current_logger);
    const auto& name = current_logger->name();

    auto new_logger = std::make_shared<::spdlog::logger>(name, std::move(sink));
    svs::logging::set_level(new_logger, current_level);
    svs::logging::set(std::move(new_logger));
}

void set_log_stream(LogStream stream) {
    auto pick_sink = [stream]() {
        switch (stream) {
            using enum LogStream;
            case stdout_: {
                return svs::logging::stdout_sink();
            }
            case stderr_: {
                return svs::logging::stdout_sink();
            }
            case null: {
                return svs::logging::null_sink();
            }
        }
        throw ANNEXCEPTION("Unknown Stream: {}\n", static_cast<int64_t>(stream));
    };
    replace_logger_with_sink(pick_sink());
}

void wrap_logging(py::module& m) {
    auto logging = m.def_submodule("logging", "Logging API");

    // Wrap the logging levels.
    using Level = svs::logging::Level;
    const char* logging_enum_description = R"(
Log levels used by SVS listed in increasing level of severity.
Only messages equal to or more severe than the currently configured log level will be
reported.

See Also
--------
svs.logging.set_level, svs.logging.get_level
)";

    py::enum_<Level>(logging, "level", logging_enum_description)
        .value("trace", Level::Trace, "The most verbose logging")
        .value("debug", Level::Debug, "Log diagnostic debug information")
        .value(
            "info",
            Level::Info,
            "Report general information. Useful for long-running operations"
        )
        .value(
            "warn",
            Level::Warn,
            "Report information that is not immediately an error, but could be potentially "
            "problematic"
        )
        .value("error", Level::Error, "Report errors")
        .value(
            "critical",
            Level::Critical,
            "Report critical message that generall should not be suppressed"
        )
        .value("off", Level::Off, "Disable logging");

    py::enum_<LogStream>(logging, "stream", "Built-in Logging Stream")
        .value("stdout", LogStream::stdout_, "Route all logging to stdout")
        .value("stderr", LogStream::stderr_, "Route all logging to stderr")
        .value("null", LogStream::null, "Suppress all logging")
        .export_values();

    logging.def(
        "set_level",
        [](Level level) { svs::logging::set_level(level); },
        py::arg("level"),
        "Set logging to the specified level. Only messages more severe than the set level "
        "will be reported."
    );

    logging.def(
        "get_level",
        [&]() { return svs::logging::get_level(); },
        "Get the current logging level."
    );

    logging.def(
        "set_logging_stream",
        &set_log_stream,
        py::arg("stream"),
        R"(
Route logging to use the specified stream. Note that setting this will supersede
the default environment variable selection mechanism and all previous calls to
``svs.logging.set_logging_stream`` and ``svs.logging.set_logging_file``.
)"
    );

    logging.def(
        "set_logging_file",
        [](const std::filesystem::path& file) {
            replace_logger_with_sink(svs::logging::file_sink(file.native()));
        },
        py::arg("file"),
        R"(
Direct all logging message to the specified file. Caller must have sufficient permissions
to create the file.

Note that setting this will supersede the default environment variable selection mechanism
and all previous calls to ``svs.logging.set_logging_stream`` and
``svs.logging.set_logging_file``.
)"
    );

    logging.def(
        "log_message",
        [](Level level, const std::string& message) {
            svs::logging::log(level, "{}", message);
        },
        py::arg("level"),
        py::arg("message"),
        "Log the message with the given severity level."
    );
}

constexpr std::string_view compression_constructor_proto = R"(
Construct a loader that will lazily compress the results of the data loader.
Requires an appropriate back-end to be compiled for all combinations of primary and residual
bits.

Args:
    loader (:py:class:`svs.VectorDataLoader`): The uncompressed dataset to compress
        in-memory.
    primary (int): The number of bits to use for compression in the primary dataset.
    residual (int): The number of bits to use for compression in the residual dataset.
        Default: 0.
    padding (int): The value (in bytes) to align the beginning of each compressed vectors.
        Values of 32 or 64 may offer the best performance at the cost of a lower compression
        ratio. A value of 0 implies no special alignment.
    strategy (:py:class:`svs.LVQStrategy`): The packing strategy to use for the compressed
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
    strategy (:py:class:`svs.LVQStrategy`): The packing strategy to use for the compressed
        codes. See the associated documenation for that enum.
)";

constexpr std::string_view leanvec_online_proto = R"(
Construct a loader that will lazily reduce the dimensionality of the data loader.
Requires an appropriate back-end to be compiled for all combinations of primary and
secondary types.

Args:
    loader (:py:class:`svs.VectorDataLoader`): The uncompressed original dataset.
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
        Default: ``svs.LeanVecKind.lvq8``.
    secondary (LeanVecKind): Type of dataset used for Secondary
        Default: ``svs.LeanVecKind.LVQ8``.
    alignment (int):  alignement/padding used in LVQ data types. Default: 32.
)";

// Legacy definitions.
template <size_t Primary, size_t Residual> struct LegacyLVQLoader {
  public:
    LegacyLVQLoader(UnspecializedVectorDataLoader loader, size_t padding)
        : loader_{std::move(loader), Primary, Residual, padding} {}

    LegacyLVQLoader(std::string path, size_t dims, size_t padding)
        : loader_{LVQReloader{std::move(path)}, padding} {
        auto throw_err = [&](std::string_view kind, size_t has, size_t expected) {
            throw ANNEXCEPTION(
                "Reloaded dataset has {} {} but was expected to have {}!",
                kind,
                has,
                expected
            );
        };

        // Make sure the deduced results are correct.
        if (loader_.primary_ != Primary) {
            throw_err("primary bits", loader_.primary_, Primary);
        }

        if (loader_.residual_ != Residual) {
            throw_err("residual bits", loader_.residual_, Residual);
        }

        if (dims != Dynamic && dims != loader_.dims_) {
            throw_err("dimensions", loader_.dims_, dims);
        }
    }

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
                        size_t padding,
                        svs::quantization::lvq::LVQStrategyDispatch strategy) {
                return LVQ{LVQReloader(path), padding, strategy};
            }),
            py::arg("directory"),
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
            py::arg("directory"),
            R"(
Create a copy of the argument loader configured to reload a previously saved LVQ dataset
from the given directory.)"
        )
        .def_readonly(
            "primary_bits",
            &LVQ::primary_,
            "The number of bits used for the primary encoding."
        )
        .def_readonly(
            "residual_bits",
            &LVQ::residual_,
            "The number of bits used for the residual encoding."
        )
        .def_readonly("strategy", &LVQ::strategy_, "The packing strategy to use.")
        .def_readonly("dims", &LVQ::dims_, "The number of dimensions.");

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
    wrap_logging(m);

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
            py::init([](const std::string& path, size_t alignment) {
                return LeanVec{LeanVecReloader(path), alignment};
            }),
            py::arg("directory"),
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
        )
        .def_readonly(
            "leanvec_dims", &LeanVec::leanvec_dims_, "The reduced dimensionality."
        )
        .def_readonly("dims", &LeanVec::dims_, "The full-dimensionality.")
        .def_readonly(
            "primary_kind",
            &LeanVec::primary_kind_,
            "The encoding of the reduced dimensional dataset."
        )
        .def_readonly(
            "secondary_kind",
            &LeanVec::secondary_kind_,
            "The encoding of the full-dimensional dataset."
        )
        .def_readwrite(
            "alignment", &LeanVec::alignment_, "The alignment to use for LVQ encoded data."
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

    ///// LVQ
    wrap_lvq(m);

    ///// LeanVec
    wrap_leanvec(m);

    ///// TOML Reconstructions
    m.def("__reformat_toml", [](const std::filesystem::path& path) {
        toml::table t = toml::parse_file(path.c_str());
        auto file = svs::lib::open_write(path, std::ios_base::out);
        file << t << "\n";
    });
}
} // namespace core
} // namespace svs::python
