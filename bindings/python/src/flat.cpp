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
#include "svs/python/flat.h"
#include "svs/python/common.h"
#include "svs/python/core.h"
#include "svs/python/manager.h"

// svs
#include "svs/lib/datatype.h"
#include "svs/lib/dispatcher.h"
#include "svs/orchestrators/exhaustive.h"
#include "svs/quantization/lvq/lvq_concept.h"

// stl
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <functional>
#include <span>
#include <string>

/////
///// Flat
/////

namespace py = pybind11;
namespace lvq = svs::quantization::lvq;
namespace svs::python::flat {

template <typename F> void for_standard_specializations(F&& f) {
#define X(Q, T, N) f.template operator()<Q, T, N>()
    // Pattern:
    // QueryType, DataType, Dimensionality, Enable Building
    // clang-format off
    X(float,   float,        Dynamic);
    X(float,   svs::Float16, Dynamic);
    X(uint8_t, uint8_t,      Dynamic);
    X(int8_t,  int8_t,       Dynamic);
    // clang-format on
#undef X
}

// Compressed search specializations.
template <typename F> void for_lvq_specializations(F&& f) {
#define X(Dist, Primary, Residual, N) f.template operator()<Dist, Primary, Residual, N>()
    // Pattern:
    // DistanceType, Primary, Residual, Dimensionality
    X(DistanceL2, 4, 4, Dynamic);
    X(DistanceL2, 8, 0, Dynamic);

    X(DistanceIP, 4, 4, Dynamic);
    X(DistanceIP, 8, 0, Dynamic);
#undef X
}

namespace detail {

using FlatSourceTypes = std::variant<UnspecializedVectorDataLoader, LVQ>;

template <typename Q, typename T, size_t N>
svs::Flat assemble_uncompressed(
    svs::VectorDataLoader<T, N, RebindAllocator<T>> datafile,
    svs::DistanceType distance_type,
    size_t num_threads
) {
    return svs::Flat::assemble<Q>(std::move(datafile), distance_type, num_threads);
}

template <typename D, size_t Primary, size_t Residual, size_t N>
svs::Flat assemble_lvq(
    lvq::LVQLoader<Primary, Residual, N, lvq::Sequential, Allocator> loader,
    D distance,
    size_t num_threads
) {
    return svs::Flat::assemble<float>(std::move(loader), std::move(distance), num_threads);
}

using AssemblyDispatcher =
    svs::lib::Dispatcher<svs::Flat, FlatSourceTypes, svs::DistanceType, size_t>;

AssemblyDispatcher assembly_dispatcher() {
    auto dispatcher = AssemblyDispatcher();
    // Uncompressed instantiations.
    for_standard_specializations([&dispatcher]<typename Q, typename T, size_t N>() {
        auto method = &assemble_uncompressed<Q, T, N>;
        dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
    });

    // LVQ instantiations.
    for_lvq_specializations([&dispatcher]<typename D, size_t P, size_t R, size_t N>() {
        auto method = &assemble_lvq<D, P, R, N>;
        dispatcher.register_target(svs::lib::dispatcher_build_docs, method);
    });

    return dispatcher;
}
/////
///// Load Dataset from Files
/////

svs::Flat assemble(
    FlatSourceTypes source,
    svs::DistanceType distance_type,
    svs::DataType SVS_UNUSED(query_type),
    size_t n_threads
) {
    return assembly_dispatcher().invoke(std::move(source), distance_type, n_threads);
}

/////
///// Initialize from Numpy Array
/////
template <typename Q, typename T, size_t N>
svs::Flat assemble_from_array(
    svs::data::ConstSimpleDataView<T, N> view,
    svs::DistanceType distance_type,
    size_t n_threads
) {
    auto data =
        svs::data::SimpleData<T, N, RebindAllocator<T>>(view.size(), view.dimensions());
    svs::data::copy(view, data);
    return svs::Flat::assemble<Q>(std::move(data), distance_type, n_threads);
}

svs::Flat assemble_from_array(
    AnonymousVectorData data, svs::DistanceType distance_type, size_t n_threads
) {
    auto dispatcher =
        svs::lib::Dispatcher<svs::Flat, AnonymousVectorData, svs::DistanceType, size_t>();
    for_standard_specializations([&dispatcher]<typename Q, typename T, size_t N>() {
        dispatcher.register_target(assemble_from_array<Q, T, N>);
    });
    return dispatcher.invoke(data, distance_type, n_threads);
}

template <typename QueryType, typename ElementType>
void add_assemble_specialization(py::class_<svs::Flat>& flat) {
    flat.def(
        py::init([](py_contiguous_array_t<ElementType> py_data,
                    svs::DistanceType distance_type,
                    size_t num_threads) {
            return assemble_from_array(
                AnonymousVectorData(py_data), distance_type, num_threads
            );
        }),
        py::arg("data"),
        py::arg("distance"),
        py::arg("num_threads") = 1,
        R"(
Construct a Flat index over the given data, returning a searchable index.

Args:
    data: The dataset to index. **NOTE**: SVS will maintain an internal copy of the
        dataset. This may change in future releases.
    distance: The distance type to use for this dataset.
    num_threads: The number of threads to use for searching. This value can also be
        changed after the index is constructed.
       )"
    );
}

void wrap_assemble_from_file(py::class_<svs::Flat>& flat) {
    constexpr std::string_view docstring_proto = R"(
Load a Flat index from data stored on disk.

Args:
    data_loader: The loader for the dataset.
    distance: The distance function to use.
    query_type: The data type of the queries.
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
            dispatcher.description(i, 0),
            dispatcher.description(i, 1)
        );
    }

    flat.def(
        py::init(&detail::assemble),
        py::arg("data_loader"),
        py::arg("distance") = svs::L2,
        py::arg("query_type") = svs::DataType::float32,
        py::arg("num_threads") = 1,
        fmt::format(docstring_proto, dynamic).c_str()
    );
}

constexpr std::string_view flat_parameters_name = "FlatSearchParameters";

} // namespace detail

void wrap(py::module& m) {
    std::string name = "Flat";
    py::class_<svs::Flat> flat(m, name.c_str());

    // Build from file
    detail::wrap_assemble_from_file(flat);

    // Build from Array
    detail::add_assemble_specialization<float, svs::Float16>(flat);
    detail::add_assemble_specialization<float, float>(flat);
    detail::add_assemble_specialization<uint8_t, uint8_t>(flat);
    detail::add_assemble_specialization<int8_t, int8_t>(flat);

    // Make the flat index searchable.
    add_search_specialization<float>(flat);
    add_search_specialization<uint8_t>(flat);
    add_search_specialization<int8_t>(flat);

    // Add threading layer.
    add_threading_interface(flat);
    add_data_interface(flat);

    ///// Search Parameters
    py::class_<svs::index::flat::FlatParameters> flat_parameters(
        m,
        std::string(detail::flat_parameters_name).c_str(),
        R"(
Configuration parameters for the flat index.

Attributes:
    data_batch_size (int, read/write): The number of dataset elements to process at a time.
        This attempts to improve locality of dataset accesses. A value of 0 will use an
        implementation defined default. Default: 0.

    query_batch_size (int, read/write): The number of query elements to process at a time.
        This attempts to improve locality of query accesses. A value of 0 will use an
        implementation defined default. Default: 0.
)"
    );

    flat_parameters.def(py::init<>())
        .def_readwrite(
            "data_batch_size",
            &svs::index::flat::FlatParameters::data_batch_size_,
            "The batch-size to use over the dataset. A value of 0 means the implementation "
            "will choose."
        )
        .def_readwrite(
            "query_batch_size",
            &svs::index::flat::FlatParameters::query_batch_size_,
            "The batch-size to use over the queries. A value of 0 means the implementation "
            "will choose."
        )
        .def("__str__", [](const svs::index::flat::FlatParameters& p) {
            return fmt::format(
                "svs.{}(data_batch_size = {}, query_batch_size = {})",
                detail::flat_parameters_name,
                p.data_batch_size_,
                p.query_batch_size_
            );
        });

    flat.def_property(
        "search_parameters",
        &svs::Flat::get_search_parameters,
        &svs::Flat::set_search_parameters,
        R"(
"Read/Write (svs.FlatSearchParameters): Get/set the current search parameters for the
index. These parameters modify and non-algorthmic properties of search (affecting
queries-per-second).

See also: `svs.FlatSearchParameters`.)"
    );
}

} // namespace svs::python::flat
