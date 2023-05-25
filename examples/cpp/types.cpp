/// @example types.cpp
/// @copybrief example_types_cpp
/// > Annotated version: @ref example_types_cpp

/// @page example_types_cpp An example of using ``svs::meta::Types``.
/// In this example, we'll show usage of ``svs::meta::Types`` and
/// ``svs::meta::for_each_type``.
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
/// Then, we invoke ``svs::meta::for_each_type`` on the resulting object.
/// To this function, we also pass a lambda accepting a single argument of type 8
/// ``svs::meta::Type``.
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
    auto types = svs::meta::Types<int64_t, float, svs::Float16>();
    // Print out the name of each type.
    svs::meta::for_each_type(types, []<typename T>(svs::meta::Type<T> /*svs_type*/) {
        // Print out the name of the data type.
        std::cout << svs::name<svs::datatype_v<T>>() << '\n';
    });
}
//! [Main]

//! [Types All]
