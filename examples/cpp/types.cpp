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

/// @example types.cpp
/// @copybrief example_types_cpp
/// > Annotated version: @ref example_types_cpp

/// @page example_types_cpp An example of using ``svs::lib::Types``.
/// In this example, we'll show usage of ``svs::lib::Types`` and
/// ``svs::lib::for_each_type``.
///
/// First, we include the appropriate headers:
/// @snippet types.cpp Includes
///
/// The "svs" headers include the following:
/// * ``datatype.h``: Type enums and naming for common C++ data types.
/// * ``meta.h``: Defines the ``Types`` class
/// * ``float16.h``: Support for 16-bit floating point.
///
/// With those included, we can define our main function.
/// @snippet types.cpp Main
///
/// We first construct the variable ``types`` as a type list with three parameters.
/// Then, we invoke ``svs::lib::for_each_type`` on the resulting object.
/// To this function, we also pass a lambda accepting a single argument of type 8
/// ``svs::lib::Type``.
/// This lambda is called for each type in ``types``' parameter pack.
/// The resulting output will be:
/// @code
/// int64
/// float32
/// float16
/// @endcode
///
/// The full code example is given below.
/// @snippet types.cpp Types All
///

//! [Types All]
//! [Includes]
#include "svs/lib/datatype.h"
#include "svs/lib/float16.h"
#include "svs/lib/meta.h"

#include <cstdint>
#include <iostream>
//! [Includes]

//! [Main]
int main() {
    auto types = svs::lib::Types<int64_t, float, svs::Float16>();
    // Print out the name of each type.
    svs::lib::for_each_type(types, []<typename T>(svs::lib::Type<T> /*svs_type*/) {
        // Print out the name of the data type.
        std::cout << svs::name<svs::datatype_v<T>>() << '\n';
    });
}
//! [Main]

//! [Types All]
