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
