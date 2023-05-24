/// @example hello_world.cpp
/// @copybrief hello_world_cpp
/// > Annotated version: @ref hello_world_cpp

/// @page hello_world_cpp This is an annotated for the documentation.
/// Here are some words.
///
/// @section hello_world_tutorial main function
/// We'll put some test text here.
///
/// @subsection hello_world_tutorial_sub1 Print Statement
///
/// Here, we print out everything to the whole world.
/// @snippet hello_world.cpp Print things
///
/// @subsection hello_world_tutorial_sub2 Return
///
/// We return zero to indicate success.
/// @snippet hello_world.cpp Return zero
///
/// @section wrapping_up Wrapping UP
/// That's basically it!
///
/// @section hello_world_all_code All Code
/// @snippet hello_world.cpp Hello World All

//! [Hello World All]
#include <exception>
#include <iostream>

int not_really_main() {
    //! [Print things]
    std::cout << "Hello World!\n";
    //! [Print things]

    //! [Return zero]
    // We return zero!
    return 0;
    //! [Return zero]
}

int main() { return not_really_main(); }
//! [Hello World All]
