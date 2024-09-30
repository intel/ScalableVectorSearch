/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
 */
// Dependencies within the python SVS bindings directory.
#include "svs/python/allocator.h"
#include "svs/python/common.h"
#include "svs/python/conversion.h"
#include "svs/python/core.h"
#include "svs/python/dynamic_vamana.h"
#include "svs/python/flat.h"
#include "svs/python/svs_mkl.h"
#include "svs/python/vamana.h"

// SVS dependencies
#include "svs/core/distance.h"
#include "svs/core/io.h"
#include "svs/lib/array.h"
#include "svs/lib/datatype.h"
#include "svs/lib/float16.h"
#include "svs/third-party/toml.h"

// fmt
#include <fmt/core.h>

// Numpy
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
// Allow conversion of Python lists of strings to `std::vector<std::string>`.
#include <pybind11/stl.h>

// stl
#include <filesystem>
#include <optional>

// Get the expected name of the library
// Make sure CMake stays up to date with defining this parameter.
//
// The variable allows us to customize the name of the python module to support
// micro-architecture versioning.
#if !defined(SVS_MODULE_NAME)
#define SVS_MODULE_NAME _svs_native
#endif

namespace py = pybind11;

namespace {

// Convert fvecs to float16
void convert_fvecs_to_float16(
    const std::string& filename_f32, const std::string& filename_f16
) {
    auto reader = svs::io::vecs::VecsReader<float>{filename_f32};
    auto writer = svs::io::vecs::VecsWriter<svs::Float16>{filename_f16, reader.ndims()};
    for (auto i : reader) {
        writer << i;
    }
}

// Convert fvecs to svs - typed
template <typename Eltype>
void convert_vecs_to_svs_impl(const std::string& vecs_file, const std::string& svs_file) {
    auto reader = svs::io::vecs::VecsReader<Eltype>(vecs_file);
    auto writer = svs::io::NativeFile(svs_file).writer(reader.ndims());
    for (const auto& i : reader) {
        writer << i;
    }
}

const auto SUPPORTED_VECS_CONVERSION_TYPES =
    svs::lib::Types<float, svs::Float16, uint32_t, uint8_t>();
// Convert fvecs to svs - dynamic dispatch.
void convert_vecs_to_svs(
    const std::string& vecs_file, const std::string& svs_file, svs::DataType dtype
) {
    svs::lib::match(SUPPORTED_VECS_CONVERSION_TYPES, dtype, [&](auto type) {
        using T = typename decltype(type)::type;
        convert_vecs_to_svs_impl<T>(vecs_file, svs_file);
    });
}

void wrap_conversion(py::module& m) {
    auto supported_types = std::vector<svs::DataType>();
    svs::lib::for_each_type(SUPPORTED_VECS_CONVERSION_TYPES, [&](auto type) {
        supported_types.push_back(type);
    });

    constexpr const char* docstring_proto = R"(
Convert the vecs file (containing the specified element types) to the svs native format.

Args:
    vecs_file: The source [f/h/i/b]vecs file.
    svs_file: The destination native file.
    dtype: The svs.DataType of the vecs file. Supported types: ({}).
File extension type map:

* fvecs = svs.DataType.float32
* hvecs = svs.DataType.float16
* ivecs = svs.DataType.uint32
* bvecs = svs.DataType.uint8
)";

    m.def(
        "convert_vecs_to_svs",
        &convert_vecs_to_svs,
        py::arg("vecs_file"),
        py::arg("svs_file"),
        py::arg("dtype") = svs::DataType::float32,
        fmt::format(docstring_proto, svs::lib::format(supported_types)).c_str()
    );
}

/// Overrides the `__name__` of a module.  Classes defined by pybind11 use the
/// `__name__` of the module as of the time they are defined, which affects the
/// `__repr__` of the class type objects.
///
/// See: https://github.com/sphinx-doc/sphinx/issues/10199
/// and
/// https://github.com/google/tensorstore/blob/94be4f2e8715511bb60fc0a0eaf07335881673b3/python/tensorstore/tensorstore.cc#L75
class ScopedModuleNameOverride {
  public:
    explicit ScopedModuleNameOverride(py::module m, std::string name)
        : module_(std::move(m)) {
        original_name_ = module_.attr("__name__");
        module_.attr("__name__") = name;
    }
    ~ScopedModuleNameOverride() { module_.attr("__name__") = original_name_; }

  private:
    py::module module_;
    py::object original_name_;
};

} // namespace

PYBIND11_MODULE(SVS_MODULE_NAME, m) {
    // Internall, the top level `__init__.py` imports everything from the C++ module named
    // `_svs`.
    //
    // Performing the name override makes the definitions inside the C++ bindings
    // "first class" in the top level `svs` module>
    auto name_override = ScopedModuleNameOverride(m, "svs::python");
    m.doc() = "Python bindings";
    m.def(
        "library_version",
        []() { return svs::lib::svs_version.str(); },
        "Obtain the version string of the backing C++ library."
    );

    py::enum_<svs::DistanceType>(m, "DistanceType", "Select which distance function to use")
        .value("L2", svs::DistanceType::L2, "Euclidean Distance (minimize)")
        .value("MIP", svs::DistanceType::MIP, "Maximum Inner Product (maximize)")
        .value("Cosine", svs::DistanceType::Cosine, "Cosine similarity (maximize)")
        .export_values();

    py::enum_<svs::DataType>(m, "DataType", "Datatype Selector")
        .value("uint8", svs::DataType::uint8, "8-bit unsigned integer.")
        .value("uint16", svs::DataType::uint16, "16-bit unsigned integer.")
        .value("uint32", svs::DataType::uint32, "32-bit unsigned integer.")
        .value("uint64", svs::DataType::uint64, "64-bit unsigned integer.")
        .value("int8", svs::DataType::int8, "8-bit signed integer.")
        .value("int16", svs::DataType::int16, "16-bit signed integer.")
        .value("int32", svs::DataType::int32, "32-bit signed integer.")
        .value("int64", svs::DataType::int64, "64-bit signed integer.")
        .value("float16", svs::DataType::float16, "16-bit IEEE floating point.")
        .value("float32", svs::DataType::float32, "32-bit IEEE floating point.")
        .value("float64", svs::DataType::float64, "64-bit IEEE floating point.")
        .export_values();

    // Helper Functions
    m.def(
        "convert_fvecs_to_float16",
        &convert_fvecs_to_float16,
        py::arg("source_file"),
        py::arg("destination_file"),
        R"(
Convert the `fvecs` file on disk with 32-bit floating point entries to a `fvecs` file with
16-bit floating point entries.

Args:
    source_file: The source file path to convert.
    destination_file: The destination file to generate.
        )"
    );

    wrap_conversion(m);

    // Allocators
    svs::python::allocators::wrap(m);

    // Core data types
    svs::python::core::wrap(m);

    // Dataset conversion.
    svs::python::conversion::wrap(m);

    // Intel(R) MKL
    m.def(
        "have_mkl", &svs::python::have_mkl, "Return whether or not svs is linked with Intel(R) MKL."
    );
    m.def(
        "mkl_num_threads",
        &svs::python::mkl_num_threads,
        "Return the number of threads used by Intel(R) MKL, or None if svs is not linked with Intel(R) MKL."
    );

    ///// Indexes
    // Flat
    svs::python::flat::wrap(m);

    // Vamana
    svs::python::vamana::wrap(m);
    svs::python::dynamic_vamana::wrap(m);
}
