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

#pragma once

#include "svs/core/allocator.h"
#include "svs/core/data.h"
#include "svs/lib/array.h"
#include "svs/lib/float16.h"

#include <pybind11/numpy.h>

const size_t Dynamic = svs::Dynamic;

///
/// Ensure certain constexpr code-paths aren't reachable at compile time.
/// Helps ensure "constexpr" if-else chains are exhaustive.
///
template <bool T = false> void static_unreachable() { static_assert(T); }

///
/// Alias for the numpy arrays we support.
///
template <typename T>
using py_contiguous_array_t = pybind11::array_t<T, pybind11::array::c_style>;

///
/// @brief Construct a span view over the numpy array.
///
template <typename T> std::span<const T> as_span(const py_contiguous_array_t<T>& array) {
    const size_t ndim = array.ndim();
    if (ndim != 1) {
        throw ANNEXCEPTION(
            "Array to span conversion needs a vector. Instead, the provided array has ",
            ndim,
            " dimensions!"
        );
    }
    return std::span<const T>(array.data(), array.size());
}

///
/// Create a read-only data view over a numpy array.
///
/// @tparam Eltype The element type of the array.
///
/// @param data The numpy array to alias.
///
template <typename Eltype>
svs::data::ConstSimpleDataView<Eltype>
data_view(const pybind11::array_t<Eltype, pybind11::array::c_style>& data) {
    return svs::data::ConstSimpleDataView<Eltype>(
        data.template unchecked<2>().data(0, 0), data.shape(0), data.shape(1)
    );
}

///
/// Create a read-write MatrixView over a numpy array.
///
/// @tparam Eltype The element type of the numpy array and returned MatrixView.
///
/// @param data The numpy array to alias.
///
template <typename Eltype>
svs::MatrixView<Eltype>
matrix_view(pybind11::array_t<Eltype, pybind11::array::c_style>& data) {
    return svs::MatrixView<Eltype>{
        data.template mutable_unchecked<2>().mutable_data(0, 0),
        data.shape(0),
        data.shape(1),
    };
}

///
/// @brief Create a 1-dimensional Numpy vector with size `s`.
///
template <typename T>
pybind11::array_t<T, pybind11::array::c_style> numpy_vector(size_t s) {
    return pybind11::array_t<T, pybind11::array::c_style>{
        {svs::lib::narrow<pybind11::ssize_t>(s)}};
}

///
/// Create a 2-dimensional Numpy array with dimensions `(s0, s1)`.
///
template <typename T>
pybind11::array_t<T, pybind11::array::c_style> numpy_matrix(size_t s0, size_t s1) {
    return pybind11::array_t<T, pybind11::array::c_style>{{s0, s1}};
}

namespace detail {
template <typename T> struct NumpyMapping {
    using type = T;
    static constexpr type convert(T x) { return x; }
};
} // namespace detail

template <typename T> using numpy_mapped_type = typename detail::NumpyMapping<T>::type;
template <typename T> numpy_mapped_type<T> convert_numpy(T x) {
    return detail::NumpyMapping<T>::convert(x);
}

///
/// Construct a `SimplePolymorphicData` objects from a numpy array.
///
template <typename T>
svs::data::SimplePolymorphicData<numpy_mapped_type<T>, Dynamic>
create_data(const pybind11::array_t<T, pybind11::array::c_style>& data) {
    using MappedT = numpy_mapped_type<T>;
    auto poly = svs::data::SimplePolymorphicData<MappedT, Dynamic>{
        svs::HugepageAllocator(),
        svs::lib::narrow<size_t>(data.shape(0)),
        svs::lib::narrow<size_t>(data.shape(1)),
    };
    std::transform(data.data(), data.data() + data.size(), poly.data(), [](auto&& x) {
        return convert_numpy(x);
    });
    return poly;
}

template <typename T, size_t Extent = Dynamic>
svs::data::BlockedData<T, Extent>
create_blocked_data(const pybind11::array_t<T, pybind11::array::c_style>& py_data) {
    using MappedT = numpy_mapped_type<T>;
    if (py_data.ndim() != 2) {
        throw ANNEXCEPTION("Expected data to be a matrix!");
    }

    size_t count = py_data.shape(0);
    size_t dims = py_data.shape(1);

    if constexpr (Extent != Dynamic) {
        if (Extent != dims) {
            throw ANNEXCEPTION(
                "Trying to assign a numpy array with dynamic dimensionality (",
                dims,
                ") to a static blocked dataset with dimensionality ",
                Extent,
                '!'
            );
        }
    }

    auto data = svs::data::BlockedData<MappedT, Extent>(count, dims);
    auto direct_access = py_data.template unchecked<2>();
    std::vector<MappedT> buffer(dims);

    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < dims; ++j) {
            buffer[j] = convert_numpy(direct_access(i, j));
        }
        data.set_datum(i, buffer);
    }
    return data;
}

///
/// Helper type to assist in building dispatch tables for various index implementations.
/// A dispatch table consists of keys used for the dispatch, and mapped values which
/// contain type erased `std::functions` that call the implementing function using the
/// types used to create the key.
///
/// Dependent on a type `T` which must have the following properties:
///
/// -- Type Aliases
/// * `T::key_type` - A tuple consisting of runtime values corresponding to the unique
///   types used to construct the inner function.
///
///   For example, if the implemented function depends on an arithmetic type `T`` and a set
///   dimensionality `N`, the inner function would be constructed using the types
///   `svs::meta::Type<T>()` and `svs::meta::Val<N>()` while the key tuple would consist
///   of `(svs::datatype_v<T>, N)`.
///
/// * `T::mapped_type` - The type of the `std::function` implementing the dispatch.
///
/// -- Static Methods
/// ```
/// template<typename F>
/// static void T::fill(F&& f);
/// ```
/// Fill in a dispatch table. For each specialization of interest, call
/// `f(std::pair<T::key_type, T::mapped_type>);` to register the dispatched function.
///
/// -- API
/// The main API used to interact with the dispatcher is the `get()` static method, which
/// will return a dispatch table whose type can be queried using `map_type`.
///
/// **Implementation details**: The dispatch table is implemented using a magic static
/// variable. Thus, a new dispatch table is not constructed every time `get()` is called.
///
template <typename T> struct Dispatcher {
    using key_type = typename T::key_type;
    using mapped_type = typename T::mapped_type;
    using map_type = std::unordered_map<key_type, mapped_type, svs::lib::TupleHash>;

    static map_type populate() {
        auto map = map_type();
        T::fill([&map](std::pair<key_type, mapped_type> kv) { map.insert(kv); });
        return map;
    }

    static const map_type& get() {
        static map_type dispatcher = populate();
        return dispatcher;
    }
};

///
/// Perform the dynamic dispatch encoded in the `dispatcher`.
/// @param try_generic Boolean flag to indicate that generic (Dynamic) dimensionality can
///        be used. If `false`, then the dimensionality argument `ndims` must be used as
///        given.
/// @param ndims The dimensionality of the data involved. Usually passed to try and match
///        an implementation specialized for that dimension. If `try_generic == true`, then
///        the dispatch logic will try a fallback using dynamically sized dimensionality
///        as well.
/// @param args Any other arguments required the the dispatch tuple.
///
/// **Note**: The `ndims` argument will always be passed to the tail of the constructed
/// dispatch tuple.
///
template <typename Dispatcher, typename... Args>
const typename Dispatcher::mapped_type&
dispatch(const Dispatcher& dispatcher, bool try_generic, size_t ndims, Args&&... args) {
    // First, try finding an implementation with the given number of dimensions.
    auto key = std::make_tuple(args..., ndims);
    if (auto iter = dispatcher.find(key); iter != dispatcher.end()) {
        return iter->second;
    }

    // First attempt failed, try performing a generic lookup.
    if (try_generic && ndims != Dynamic) {
        key = std::make_tuple(args..., Dynamic);
        if (auto iter = dispatcher.find(key); iter != dispatcher.end()) {
            return iter->second;
        }
    }
    throw ANNEXCEPTION("Unimplemented Specialization!");
}
