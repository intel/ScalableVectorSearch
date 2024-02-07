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

#include "flat.h"
#include "common.h"
#include "core.h"
#include "manager.h"

#include "svs/extensions/flat/lvq.h"
#include "svs/lib/datatype.h"
#include "svs/lib/dispatcher.h"
#include "svs/orchestrators/exhaustive.h"

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
namespace flat {

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
    data: The dataset to index. **NOTE**: PySVS will maintain an internal copy of the
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

    // Flat index specific components.
    flat.def_property(
        "data_batch_size", &svs::Flat::get_data_batch_size, &svs::Flat::set_data_batch_size
    );
    flat.def_property(
        "query_batch_size",
        &svs::Flat::get_query_batch_size,
        &svs::Flat::set_query_batch_size
    );
}

} // namespace flat
