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

namespace svs::python {

const size_t Dynamic = svs::Dynamic;

// Exposed Allocators
// N.B.: As more allocators get implemented, this can be switched to a ``std::variant`` of
// allocators that will get propagated throughout the code.
//
// Support for this might not be fully in place but should be relatively straight-forward
// to add.
using Allocator = svs::HugepageAllocator<std::byte>;

// Functor to wrap an allocator inside a blocked struct.
inline constexpr auto as_blocked = [](const auto& allocator) {
    return svs::data::Blocked<std::decay_t<decltype(allocator)>>{allocator};
};

template <typename T>
using RebindAllocator = typename std::allocator_traits<Allocator>::rebind_alloc<T>;

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

struct AllowVectorsTag {};

/// A property to pass to ``data_view`` to interpret a numpy vector as a 2D array with
/// the size of the first dimension equal to zero.
inline constexpr AllowVectorsTag allow_vectors{};

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
/// Create a read-only data view over a numpy matrix or vector.
///
/// @tparam Eltype The element type of the array.
///
/// @param data The numpy array to alias.
/// @param property Indicate that it is okay to promote numpy vectors to matrices.
///
template <typename Eltype>
svs::data::ConstSimpleDataView<Eltype> data_view(
    const pybind11::array_t<Eltype, pybind11::array::c_style>& data,
    AllowVectorsTag SVS_UNUSED(property)
) {
    size_t ndims = data.ndim();
    // If this is a vector, interpret is a batch of queries with size 1.
    // The type requirement `pybind11::array::c_style` means that the underlying data is
    // contiguous, so we can construct a view from its pointer.
    if (ndims == 1) {
        return svs::data::ConstSimpleDataView<Eltype>(
            data.template unchecked<1>().data(0), 1, data.shape(0)
        );
    }

    if (ndims != 2) {
        throw ANNEXCEPTION("This function can only accept numpy vectors or matrices.");
    }

    return svs::data::ConstSimpleDataView<Eltype>(
        data.template unchecked<2>().data(0, 0), data.shape(0), data.shape(1)
    );
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
/// Create a read-write data view over a numpy array.
///
/// @tparam Eltype The element type of the array.
///
/// @param data The numpy array to alias.
///
template <typename Eltype>
svs::data::SimpleDataView<Eltype>
mutable_data_view(pybind11::array_t<Eltype, pybind11::array::c_style>& data) {
    return svs::data::SimpleDataView<Eltype>(
        data.template mutable_unchecked<2>().mutable_data(0, 0),
        data.shape(0),
        data.shape(1)
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

///
/// Construct a `SimpleData` objects from a numpy array.
///
template <typename T, typename Alloc = RebindAllocator<T>>
svs::data::SimpleData<T, Dynamic, Alloc>
create_data(const pybind11::array_t<T, pybind11::array::c_style>& data) {
    auto src = data_view(data);
    auto dst = svs::data::SimpleData<T, Dynamic, Alloc>(src.size(), src.dimensions());
    svs::data::copy(src, dst);
    return dst;
}

template <typename T, size_t Extent = Dynamic>
svs::data::BlockedData<T, Extent, RebindAllocator<T>>
create_blocked_data(const pybind11::array_t<T, pybind11::array::c_style>& py_data) {
    auto src = data_view(py_data);
    if constexpr (Extent != Dynamic) {
        if (Extent != src.dimensions()) {
            throw ANNEXCEPTION(
                "Trying to assign a numpy array with dynamic dimensionality ({}) to a "
                "static blocked dataset with dimensionality {}!",
                src.dimensions(),
                Extent
            );
        }
    }

    auto dst =
        svs::data::BlockedData<T, Extent, RebindAllocator<T>>(src.size(), src.dimensions());
    svs::data::copy(src, dst);
    return dst;
}

namespace detail {
template <typename F, typename T>
using and_then_return_t = std::remove_cvref_t<std::invoke_result_t<F, T>>;
}

// Stand-in for C++23 `std::optional` monads.
template <typename F, typename T>
std::optional<detail::and_then_return_t<F, const T&>>
transform_optional(F&& f, const std::optional<T>& x) {
    if (!x) {
        return std::nullopt;
    }
    return std::optional<detail::and_then_return_t<F, const T&>>(f(*x));
}

}
