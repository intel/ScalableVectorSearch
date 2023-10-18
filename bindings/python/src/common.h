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

// Implementation here is largely inspired by:
//
// https://github.com/pybind/pybind11/issues/1776#issuecomment-491514980W
//
// With updates to reflect current pybind11.
namespace pybind11::detail {
template <> struct npy_format_descriptor<svs::Float16> {
    static pybind11::dtype dtype() {
        // Obtaining the ID of the numpy float16 datatype:
        // ```
        // import numpy as np
        // print(np.datatype(np.float16).num)
        // ```
        handle ptr = npy_api::get().PyArray_DescrFromType_(23);
        return reinterpret_borrow<pybind11::dtype>(ptr);
    }
    static std::string format() {
        // following: https://docs.python.org/3/library/struct.html#format-characters
        return "e";
    }
    static constexpr auto name = _("float16");
};
} // namespace pybind11::detail

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
        svs::make_dims(data.shape(0), data.shape(1)),
        data.template mutable_unchecked<2>().mutable_data(0, 0),
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
svs::data::
    SimpleData<numpy_mapped_type<T>, Dynamic, svs::HugepageAllocator<numpy_mapped_type<T>>>
    create_data(const pybind11::array_t<T, pybind11::array::c_style>& data) {
    using MappedT = numpy_mapped_type<T>;
    auto out = svs::data::SimpleData<MappedT, Dynamic, svs::HugepageAllocator<MappedT>>{
        svs::lib::narrow<size_t>(data.shape(0)), svs::lib::narrow<size_t>(data.shape(1))};
    std::transform(data.data(), data.data() + data.size(), out.data(), [](auto&& x) {
        return convert_numpy(x);
    });
    return out;
}

template <typename T, size_t Extent = Dynamic>
svs::data::BlockedData<T, Extent, svs::HugepageAllocator<T>>
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

    auto data = svs::data::BlockedData<MappedT, Extent, svs::HugepageAllocator<MappedT>>(
        count, dims
    );
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
