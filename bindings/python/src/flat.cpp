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
#define X(Q, T, N) f(Type<Q>(), Type<T>(), Val<N>())
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
template <typename F> void compressed_specializations(F&& f) {
#define X(Dist, N) f(Dist(), svs::meta::Val<N>())
    // Pattern:
    // DistanceType, Dimensionality
    X(DistanceL2, Dynamic);
    X(DistanceIP, Dynamic);
#undef X
}

namespace detail {
struct StandardAssemble_ {
    // QueryType, DataType, Dims
    using key_type = std::tuple<svs::DataType, svs::DataType, size_t>;
    using mapped_type = std::function<
        svs::Flat(const UnspecializedVectorDataLoader&, svs::DistanceType, size_t)>;

    template <typename Q, typename T, size_t N>
    static std::pair<key_type, mapped_type>
    specialize(Type<Q> query_type, Type<T> data_type, Val<N> ndims) {
        key_type key = {unwrap(query_type), unwrap(data_type), unwrap(ndims)};
        mapped_type fn = [=](const UnspecializedVectorDataLoader& datafile,
                             svs::DistanceType distance_type,
                             size_t num_threads) {
            return svs::Flat::assemble<Q>(
                datafile.refine(data_type, ndims), distance_type, num_threads
            );
        };
        return std::make_pair(key, std::move(fn));
    }

    template <typename F> static void fill(F&& f) {
        for_standard_specializations([&f](auto query_type, auto data_type, auto ndims) {
            return f(specialize(query_type, data_type, ndims));
        });
    }
};
using StandardAssembler = svs::lib::Dispatcher<StandardAssemble_>;

template <typename Kind> struct CompressedAssemble_ {
    using key_type = std::tuple<svs::DistanceType, size_t>;
    using mapped_type = std::function<svs::Flat(
        const Kind& /*loader*/, size_t /*num_threads*/
    )>;

    template <typename Distance, size_t N>
    static std::pair<key_type, mapped_type> specialize(Distance distance, Val<N> ndims) {
        key_type key = {svs::distance_type_v<Distance>, unwrap(ndims)};
        mapped_type fn = [=](const Kind& loader, size_t num_threads) {
            return svs::Flat::assemble<float>(
                loader.refine(Val<N>()), distance, num_threads
            );
        };
        return std::make_pair(key, std::move(fn));
    }

    template <typename F> static void fill(F&& f) {
        compressed_specializations([&f](auto distance, auto ndims) {
            f(specialize(distance, ndims));
        });
    }
};
template <typename Kind>
using CompressedAssembler = svs::lib::Dispatcher<CompressedAssemble_<Kind>>;

/////
///// Load Dataset from Files
/////

using FlatSourceTypes = std::variant<UnspecializedVectorDataLoader, LVQ8, LVQ4x4>;

svs::Flat assemble(
    const FlatSourceTypes& source,
    svs::DistanceType distance_type,
    svs::DataType query_type,
    size_t n_threads
) {
    return std::visit<svs::Flat>(
        [&](auto&& loader) {
            using T = std::decay_t<decltype(loader)>;
            if constexpr (std::is_same_v<T, UnspecializedVectorDataLoader>) {
                const auto& f =
                    StandardAssembler::lookup(true, loader.dims_, query_type, loader.type_);
                return f(loader, distance_type, n_threads);
            } else {
                const auto& f =
                    CompressedAssembler<T>::lookup(true, loader.dims_, distance_type);
                return f(loader, n_threads);
            }
        },
        source
    );
}

/////
///// Initialize from Numpy Array
/////
template <typename QueryType, typename ElementType>
svs::Flat assemble_from_array(
    py_contiguous_array_t<ElementType> py_data,
    svs::DistanceType distance_type,
    size_t n_threads
) {
    return svs::Flat::assemble<QueryType>(create_data(py_data), distance_type, n_threads);
}

template <typename QueryType, typename ElementType>
void add_assemble_specialization(py::class_<svs::Flat>& flat) {
    flat.def(
        py::init(&assemble_from_array<QueryType, ElementType>),
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

} // namespace detail

void wrap(py::module& m) {
    std::string name = "Flat";
    py::class_<svs::Flat> flat(m, name.c_str());
    flat.def(
        py::init(&detail::assemble),
        py::arg("data_loader"),
        py::arg("distance") = svs::L2,
        py::arg("query_type") = svs::DataType::float32,
        py::arg("num_threads") = 1,
        R"(
Load a Flat index from data stored on disk.

Args:
    data_loader: The loader for the dataset.
    distance: The distance function to use.
    query_type: The data type of the queries.
    num_threads: The number of threads to use for queries (can be changed after loading).
        )"
    );

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
