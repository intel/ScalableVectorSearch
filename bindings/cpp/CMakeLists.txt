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

cmake_minimum_required(VERSION 3.21)
project(svs_runtime VERSION 0.1.0 LANGUAGES CXX)
set(TARGET_NAME svs_runtime)

set(SVS_RUNTIME_HEADERS
    include/svs/runtime/version.h
    include/svs/runtime/api_defs.h
    include/svs/runtime/training.h
    include/svs/runtime/vamana_index.h
    include/svs/runtime/dynamic_vamana_index.h
    include/svs/runtime/flat_index.h
)

set(SVS_RUNTIME_SOURCES
    src/svs_runtime_utils.h
    src/dynamic_vamana_index_impl.h
    src/dynamic_vamana_index_leanvec_impl.h
    src/training_impl.h
    src/flat_index_impl.h
    src/api_defs.cpp
    src/training.cpp
    src/vamana_index.cpp
    src/dynamic_vamana_index.cpp
    src/flat_index.cpp
)

option(SVS_RUNTIME_ENABLE_LVQ_LEANVEC "Enable compilation of SVS runtime with LVQ and LeanVec support" ON)
if (SVS_RUNTIME_ENABLE_LVQ_LEANVEC)
    message(STATUS "SVS runtime will be built with LVQ support")
else()
    message(STATUS "SVS runtime will be built without LVQ or LeanVec support")
endif()

add_library(${TARGET_NAME} SHARED
    ${SVS_RUNTIME_HEADERS}
    ${SVS_RUNTIME_SOURCES}
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(OpenMP REQUIRED)
target_link_libraries(${TARGET_NAME} PUBLIC OpenMP::OpenMP_CXX)

target_compile_options(${TARGET_NAME} PRIVATE
    -DSVS_ENABLE_OMP=1
    -fvisibility=hidden
)

if(UNIX AND NOT APPLE)
    # Don't export 3rd-party symbols from the lib
    target_link_options(${TARGET_NAME} PRIVATE "SHELL:-Wl,--exclude-libs,ALL")
endif()

target_compile_features(${TARGET_NAME} INTERFACE cxx_std_20)
set_target_properties(${TARGET_NAME} PROPERTIES PUBLIC_HEADER "${SVS_RUNTIME_HEADERS}")
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 20)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(${TARGET_NAME} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR} )

if (SVS_RUNTIME_ENABLE_LVQ_LEANVEC)
    if(DEFINED SVS_LVQ_HEADER AND DEFINED SVS_LEANVEC_HEADER)
        # expected that pre-defined headers are implementation headers
        message(STATUS "Using pre-defined LVQ header: ${SVS_LVQ_HEADER}")
        message(STATUS "Using pre-defined LeanVec header: ${SVS_LEANVEC_HEADER}")
    else()
        set(SVS_LVQ_HEADER "svs/extensions/vamana/lvq.h")
        set(SVS_LEANVEC_HEADER "svs/extensions/vamana/leanvec.h")
    endif()

    if(RUNTIME_BINDINGS_PRIVATE_SOURCE_BUILD)
        message(STATUS "Building directly from private sources")
        target_link_libraries(${TARGET_NAME} PRIVATE
            svs::svs
        )
        link_mkl_static(${TARGET_NAME})
    elseif(TARGET svs::svs)
        message(FATAL_ERROR
            "Pre-built LVQ/LeanVec SVS library cannot be used in SVS main build. "
            "Please build SVS Runtime using bindings/cpp directory as CMake source root."
        )
    else()
        # Links to LTO-enabled static library, requires GCC/G++ 11.2
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "11.2" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.3")
            set(SVS_URL "https://github.com/intel/ScalableVectorSearch/releases/download/v0.1.0/svs-shared-library-0.1.0-lto.tar.gz"
                CACHE STRING "URL to download SVS shared library")
        else()
            message(WARNING
                "Pre-built LVQ/LeanVec SVS library requires GCC/G++ v.11.2 to apply LTO optimizations."
                "Current compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
            )
            set(SVS_URL "https://github.com/intel/ScalableVectorSearch/releases/download/v0.1.0/svs-shared-library-0.1.0.tar.gz"
                CACHE STRING "URL to download SVS shared library")
        endif()
        include(FetchContent)
        FetchContent_Declare(
            svs
            URL ${SVS_URL}
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(svs)
        list(APPEND CMAKE_PREFIX_PATH "${svs_SOURCE_DIR}")
        find_package(svs REQUIRED)
        target_link_libraries(${TARGET_NAME} PRIVATE
            svs::svs
            svs::svs_compile_options
            svs::svs_static_library
        )
    endif()
    target_compile_definitions(${TARGET_NAME} PRIVATE
        PUBLIC "SVS_LVQ_HEADER=\"${SVS_LVQ_HEADER}\""
        PUBLIC "SVS_LEANVEC_HEADER=\"${SVS_LEANVEC_HEADER}\""
    )
else()
    # Include the SVS library directly if needed.
    if (NOT TARGET svs::svs)
        add_subdirectory("../.." "${CMAKE_CURRENT_BINARY_DIR}/svs")
    endif()
    target_link_libraries(${TARGET_NAME} PRIVATE
       svs::svs
       svs_compile_options
       svs_x86_options_base
    )
endif()

# installing
include(GNUInstallDirs)

set(SVS_RUNTIME_EXPORT_NAME ${TARGET_NAME})
set(VERSION_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/${SVS_RUNTIME_EXPORT_NAME}ConfigVersion.cmake")
set(SVS_RUNTIME_CONFIG_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/svs_runtime)
set(SVS_RUNTIME_COMPONENT_NAME "Runtime")

install(TARGETS ${TARGET_NAME}
    EXPORT ${SVS_RUNTIME_EXPORT_NAME}
    COMPONENT ${SVS_RUNTIME_COMPONENT_NAME}
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/svs/runtime
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/svs/runtime
    COMPONENT ${SVS_RUNTIME_COMPONENT_NAME}
    DESTINATION include/svs
    FILES_MATCHING PATTERN "*.h"
)

install(EXPORT ${SVS_RUNTIME_EXPORT_NAME}
    COMPONENT ${SVS_RUNTIME_COMPONENT_NAME}
    NAMESPACE svs::
    DESTINATION ${SVS_RUNTIME_CONFIG_INSTALL_DIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/runtimeConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${SVS_RUNTIME_EXPORT_NAME}Config.cmake"
    INSTALL_DESTINATION "${SVS_RUNTIME_CONFIG_INSTALL_DIR}"
)

# Don't make compatibility guarantees until we reach a compatibility milestone.
write_basic_package_version_file(
    ${VERSION_CONFIG}
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY ExactVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${SVS_RUNTIME_EXPORT_NAME}Config.cmake"
    "${VERSION_CONFIG}"
    COMPONENT ${SVS_RUNTIME_COMPONENT_NAME}
    DESTINATION "${SVS_RUNTIME_CONFIG_INSTALL_DIR}"
)

# Build tests if requested
if(SVS_BUILD_RUNTIME_TESTS)
    add_subdirectory(tests)
endif()
