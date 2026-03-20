# Copyright 2026 Intel Corporation
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
project(svs_c_api VERSION 0.1.0 LANGUAGES CXX C)
set(TARGET_NAME svs_c_api)

set(SVS_C_API_HEADERS
    include/svs/c_api/svs_c_config.h
    include/svs/c_api/svs_c.h
)

set(SVS_C_API_SOURCES
    src/algorithm.hpp
    src/error.hpp
    src/index.hpp
    src/index_builder.hpp
    src/storage.hpp
    src/threadpool.hpp
    src/types_support.hpp

    src/error.cpp
    src/svs_c.cpp
    src/dispatcher_vamana.cpp
    src/dispatcher_dynamic_vamana.cpp
)

add_library(${TARGET_NAME} SHARED
    ${SVS_C_API_HEADERS}
    ${SVS_C_API_SOURCES}
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
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
set_target_properties(${TARGET_NAME} PROPERTIES PUBLIC_HEADER "${SVS_C_API_HEADERS}")
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 20)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(${TARGET_NAME} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR} )

target_link_libraries(${TARGET_NAME} PRIVATE
    svs::svs
)
if (SVS_EXPERIMENTAL_LINK_STATIC_MKL)
    link_mkl_static(${TARGET_NAME})
endif()

if (SVS_RUNTIME_ENABLE_LVQ_LEANVEC)
    message(STATUS "Enabling LVQ/LeanVec support in C API")
    target_compile_definitions(${TARGET_NAME} PRIVATE SVS_RUNTIME_ENABLE_LVQ_LEANVEC)
    if(SVS_LVQ_HEADER)
        target_compile_definitions(${TARGET_NAME} PRIVATE
            SVS_LVQ_HEADER="${SVS_LVQ_HEADER}"
        )
        # "SVS_LVQ_HEADER=\"${SVS_LVQ_HEADER}\""
    endif()
    if(SVS_LEANVEC_HEADER)
        target_compile_definitions(${TARGET_NAME} PRIVATE
            SVS_LEANVEC_HEADER="${SVS_LEANVEC_HEADER}"
        )
        # "SVS_LEANVEC_HEADER=\"${SVS_LEANVEC_HEADER}\""
    endif()

    if (RUNTIME_BINDINGS_PRIVATE_SOURCE_BUILD)
        message(STATUS "Building directly from private sources with IVF support")
        target_link_libraries(${TARGET_NAME} PRIVATE
            svs::svs
            svs_compile_options
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
            set(SVS_URL "https://github.com/intel/ScalableVectorSearch/releases/download/nightly/svs-shared-library-lto-nightly-2026-02-05-1017.tar.gz"
                CACHE STRING "URL to download SVS shared library")
        else()
            message(WARNING
                "Pre-built LVQ/LeanVec SVS library requires GCC/G++ v.11.2 to apply LTO optimizations."
                "Current compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
            )
            set(SVS_URL "https://github.com/intel/ScalableVectorSearch/releases/download/nightly/svs-shared-library-nightly-2026-02-05-1017.tar.gz"
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
else()
    message(STATUS "LVQ/LeanVec support is disabled in C API")
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

# Installing
include(GNUInstallDirs)

set(SVS_C_API_EXPORT_NAME ${TARGET_NAME})
set(VERSION_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/${SVS_C_API_EXPORT_NAME}ConfigVersion.cmake")
set(SVS_C_API_CONFIG_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/svs_c_api)
set(SVS_C_API_COMPONENT_NAME "C_API")

install(TARGETS ${TARGET_NAME}
    EXPORT ${SVS_C_API_EXPORT_NAME}
    COMPONENT ${SVS_C_API_COMPONENT_NAME}
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/svs/c_api
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/svs/c_api
    COMPONENT ${SVS_C_API_COMPONENT_NAME}
    DESTINATION include/svs
    FILES_MATCHING PATTERN "*.h"
)

install(EXPORT ${SVS_C_API_EXPORT_NAME}
    COMPONENT ${SVS_C_API_COMPONENT_NAME}
    NAMESPACE svs::
    DESTINATION ${SVS_C_API_CONFIG_INSTALL_DIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/c_apiConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${SVS_C_API_EXPORT_NAME}Config.cmake"
    INSTALL_DESTINATION "${SVS_C_API_CONFIG_INSTALL_DIR}"
)

# Don't make compatibility guarantees until we reach a compatibility milestone.
write_basic_package_version_file(
    ${VERSION_CONFIG}
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY ExactVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${SVS_C_API_EXPORT_NAME}Config.cmake"
    "${VERSION_CONFIG}"
    COMPONENT ${SVS_C_API_COMPONENT_NAME}
    DESTINATION "${SVS_C_API_CONFIG_INSTALL_DIR}"
)

# Build tests if requested
# if(SVS_BUILD_C_API_TESTS)
#     add_subdirectory(tests)
# endif()

add_subdirectory(samples)
