cmake_minimum_required(VERSION 3.18) # keep in-sync with `pyproject.toml`
project(pysvs)

# Install pybind11 manually for version control.
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v2.10.1
)
FetchContent_MakeAvailable(pybind11)

# Try to find the Python executable.
#
# If it's given as part of the Cmake arguments given by "scikit build", then use that.
# Otherwise, fall back to using plain old "python".
# If *THAT* doesn't work, give up.
if(DEFINED PYTHON_EXECUTABLE)
    set(SVS_PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}")
else()
    set(SVS_PYTHON_EXECUTABLE "python")
endif()

# The micro architectures to compile for.
if(NOT DEFINED PYSVS_MICROARCHS)
    set(PYSVS_MICROARCHS native)
endif()

# Include the SVS library directly.
add_subdirectory("../.." "${CMAKE_CURRENT_BINARY_DIR}/svs")

# Run the python script to get optimization flags for the desired back-ends.
#
# FLAGS_SCRIPT - Path to the Python script that will take the compiler, compiler version,
#   and list of desired microarchitectures and generate optimization flags for each
#   microarchitecture.
#
# FLAGS_TEXT_FILE - List of optimization flags for each architecture.
#   Expected format:
#       -march=arch1,-mtune=arch1
#       -march=arch2,-mtune=arch2
#       ...
#       -march=archN,-mtune=archN
#
#   The number of lines should be equal to the number of microarchitectures.
#   NOTE: The entries within each line are separated by a comma on purpose to allow CMake
#   to read the whole file as a List and then use string replacement on the commas to turn
#   each line into a list in its own right.
#
# TEMP_JSON - JSON Manifest file describing the generated binaries. This is meant to be
#   included in the Python package to allow the Python code to reason about the packaged
#   libraries and select the correct one for loading.
#
set(FLAGS_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/microarch.py")
set(FLAGS_TEXT_FILE "${CMAKE_CURRENT_BINARY_DIR}/optimization_flags.txt")
set(FLAGS_MANIFEST_JSON "${CMAKE_CURRENT_BINARY_DIR}/flags_manifest.json")

execute_process(
    COMMAND
        ${SVS_PYTHON_EXECUTABLE}
        ${FLAGS_SCRIPT}
        ${FLAGS_TEXT_FILE}
        ${FLAGS_MANIFEST_JSON}
        --compiler ${CMAKE_CXX_COMPILER_ID}
        --compiler-version ${CMAKE_CXX_COMPILER_VERSION}
        --microarchitectures ${PYSVS_MICROARCHS}
    COMMAND_ERROR_IS_FATAL ANY
)

file(STRINGS "${FLAGS_TEXT_FILE}" OPTIMIZATION_FLAGS)
message("Flags: ${OPTIMIZATION_FLAGS}")
list(LENGTH OPTIMIZATION_FLAGS OPT_FLAGS_LENGTH)
message("Length of flags: ${OPT_FLAGS_LENGTH}")

# C++ files makind up the python bindings.
set(CPP_FILES
    src/allocator.cpp
    src/dynamic_vamana.cpp
    src/core.cpp
    src/flat.cpp
    src/python_bindings.cpp
    src/vamana.cpp
)

# Generate a shared library for each target microarchitecture.
foreach(MICRO OPT_FLAGS IN ZIP_LISTS PYSVS_MICROARCHS OPTIMIZATION_FLAGS)
    set(LIB_NAME "_pysvs_${MICRO}")

    pybind11_add_module(${LIB_NAME} MODULE ${CPP_FILES})
    target_link_libraries(${LIB_NAME} PUBLIC svs::svs)
    # Dependency "fmt::fmt" obtained from "svs"
    target_link_libraries(${LIB_NAME} PRIVATE svs::compile_options fmt::fmt)

    string(REPLACE "," ";" OPT_FLAGS ${OPT_FLAGS})
    message("OPT Flags: ${OPT_FLAGS}")
    target_compile_options(${LIB_NAME} PRIVATE ${OPT_FLAGS})

    # Comunicate to the C++ library the desired name of the library
    target_compile_options(${LIB_NAME} PRIVATE "-DPYSVS_MODULE_NAME=${LIB_NAME}")

    # If scikit build is running the compilation process,
    if(DEFINED SKBUILD)
        install(TARGETS ${LIB_NAME} DESTINATION .)
    endif()
endforeach()

# Install the manifest JSON file.
# This is kind of a hack to avoid the needing to explicitly move JSON file into the source
# folder of the python library.
if(DEFINED SKBUILD)
    install(FILES ${FLAGS_MANIFEST_JSON} DESTINATION .)
endif()


# # Copy the test dataset into a "test_data" directory in the python module source.
# file(COPY "../../data/test_dataset" DESTINATION "${CMAKE_CURRENT_LIST_DIR}/src/pysvs")

#####
##### Python installation logic.
#####

if(DEFINED SKBUILD)
    # Surpress Cmake `unused variable` warning.
    set(surpress_unused "${SKBUILD}")

    # TODO: If we ever become non-header-only, then we'll need to set up RPATHs correctly
    # to get binary compatibility.
    #
    # See: https://github.com/scikit-build/scikit-build-sample-projects/blob/master/projects/hello-cmake-package/CMakeLists.txt

    # # The extension module must load the hello library as a dependency when the
    # # extension module is loaded. The easiest way to locate the hello library is
    # # via RPATH. Absolute RPATHs are possible, but they make the resulting
    # # binaries not redistributable to other Python installations (conda is broke,
    # # wheel reuse is broke, and more!).
    # #
    # # Placing the hello library in the package and using relative RPATHs that
    # # doesn't point outside of the package means that the built package is
    # # relocatable. This allows for safe binary redistribution.
    # set_target_properties(_pysvs PROPERTIES INSTALL_RPATH
    #                                         "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
endif()
