# Copyright 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(svs_microarch_cmake_included)
    return()
endif()
set(svs_microarch_cmake_included true)

# N.B.: first microarch listed in targets file is treated as "base" microarch
# which is used to build base object files, shared libs and executables
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/microarch_targets_x86_64" SVS_SUPPORTED_MICROARCHS)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    if(APPLE)
        file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/microarch_targets_aarch64_darwin" SVS_SUPPORTED_MICROARCHS)
    else()
        file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/microarch_targets_aarch64" SVS_SUPPORTED_MICROARCHS)
    endif()
else()
    message(FATAL_ERROR "Unknown CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# List all known microarchs
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/microarch_list_x86_64" SVS_KNOWN_MICROARCHS)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    if(APPLE)
        file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/microarch_list_aarch64_darwin" SVS_KNOWN_MICROARCHS)
    else()
        file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/microarch_list_aarch64" SVS_KNOWN_MICROARCHS)
    endif()
else()
    message(FATAL_ERROR "Unknown CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

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
set(FLAGS_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/microarch.py")
set(FLAGS_TEXT_FILE "${CMAKE_CURRENT_BINARY_DIR}/optimization_flags.txt")

execute_process(
    COMMAND
        ${SVS_PYTHON_EXECUTABLE}
        ${FLAGS_SCRIPT}
        ${FLAGS_TEXT_FILE}
        --compiler ${CMAKE_CXX_COMPILER_ID}
        --compiler-version ${CMAKE_CXX_COMPILER_VERSION}
        --microarchitectures ${SVS_SUPPORTED_MICROARCHS}
    COMMAND_ERROR_IS_FATAL ANY
)
file(STRINGS "${FLAGS_TEXT_FILE}" OPTIMIZATION_FLAGS)

# Run the python script to generate a header with microarch-specific macros.
set(GENERATOR_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/microarch_generate_macros.py")
set(MICROARCH_MACROS_PROTOTYPE_HEADER "${CMAKE_CURRENT_LIST_DIR}/microarch_macros.h")
set(MICROARCH_MACROS_HEADER "${CMAKE_CURRENT_LIST_DIR}/../include/svs/lib/microarch_macros.h")

execute_process(
    COMMAND
        ${SVS_PYTHON_EXECUTABLE}
        ${GENERATOR_SCRIPT}
        --proto-header-file ${MICROARCH_MACROS_PROTOTYPE_HEADER}
        --output-header-file ${MICROARCH_MACROS_HEADER}
        --known-uarchs ${SVS_KNOWN_MICROARCHS}
        --supported-uarchs ${SVS_SUPPORTED_MICROARCHS}
    COMMAND_ERROR_IS_FATAL ANY
)

#####
##### Helper function to register specified static dimensions for dispatching by microarch.
#####

set(DIM_REGISTRY_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/microarch_register_dimensions.py")

function(svs_register_static_dimensions)
    execute_process(
        COMMAND
            ${SVS_PYTHON_EXECUTABLE}
            ${DIM_REGISTRY_SCRIPT}
            --header-file ${MICROARCH_MACROS_HEADER}
            --dimensions ${ARGN}
        COMMAND_ERROR_IS_FATAL ANY
    )
endfunction()

#####
##### Helper targets to support required microarchs and apply relevant compiler optimizations.
#####

# Set up "base" target to include opt. flags for base microarch
# and flags to enable support of other microarchs in dispatcher
add_library(svs_microarch_options_base INTERFACE)
add_library(svs::microarch_options_base ALIAS svs_microarch_options_base)

# Get opt. flags for base microarch
list(GET SVS_SUPPORTED_MICROARCHS 0 BASE_MICROARCH)
list(GET OPTIMIZATION_FLAGS 0 BASE_OPT_FLAGS)
string(REPLACE "," ";" BASE_OPT_FLAGS ${BASE_OPT_FLAGS})
message("Opt.flags[base=${BASE_MICROARCH}]: ${BASE_OPT_FLAGS}")

target_compile_options(svs_microarch_options_base INTERFACE ${BASE_OPT_FLAGS})
install(
    TARGETS svs_microarch_options_base
    EXPORT svs-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

foreach(MICROARCH OPT_FLAGS IN ZIP_LISTS SVS_SUPPORTED_MICROARCHS OPTIMIZATION_FLAGS)
    string(REPLACE "," ";" OPT_FLAGS ${OPT_FLAGS})
    message("Opt.flags[${MICROARCH}]: ${OPT_FLAGS}")

    # Create a new target for this microarch
    add_library(svs_microarch_options_${MICROARCH} INTERFACE)
    add_library(svs::microarch_options_${MICROARCH} ALIAS svs_microarch_options_${MICROARCH})
    target_compile_options(svs_microarch_options_${MICROARCH} INTERFACE ${OPT_FLAGS} -DSVS_MICROARCH_TARGET=${MICROARCH})
    install(
        TARGETS svs_microarch_options_${MICROARCH}
        EXPORT svs-targets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endforeach()

# function to create a set of object files with microarch instantiations
function(create_microarch_instantiations link_target)
    set(MICROARCH_OBJECT_FILES "")
    foreach(MICROARCH OPT_FLAGS IN ZIP_LISTS SVS_SUPPORTED_MICROARCHS OPTIMIZATION_FLAGS)
        set(OBJ_NAME "microarch_${MICROARCH}")
        add_library(${OBJ_NAME} OBJECT ${ARGN})

        target_link_libraries(${OBJ_NAME} PRIVATE ${SVS_LIB} svs::compile_options svs_microarch_options_${MICROARCH})
        target_sources(${link_target} INTERFACE $<TARGET_OBJECTS:${OBJ_NAME}>)

        list(APPEND MICROARCH_OBJECT_FILES $<TARGET_OBJECTS:${OBJ_NAME}>)
    endforeach()
    # Note: this specific way of setting the variable is required to make it available in all targeted scopes
    set(MICROARCH_OBJECT_FILES "${MICROARCH_OBJECT_FILES}" CACHE INTERNAL "Microarchitecture-specific object files")
endfunction()
