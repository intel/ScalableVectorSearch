# Copyright (C) 2023 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you ("License"). Unless the License provides otherwise,
# you may not use, modify, copy, publish, distribute, disclose or transmit
# this software or the related documents without Intel's prior written
# permission.
#
# This software and the related documents are provided as is, with no
# express or implied warranties, other than those that are expressly stated
# in the License.

cmake_minimum_required(VERSION 3.21)
project(svs_runtime VERSION 0.0.10 LANGUAGES CXX)
set(TARGET_NAME svs_runtime)

set(SVS_RUNTIME_HEADERS
    include/IndexSVSImplDefs.h
    include/IndexSVSFlatImpl.h
    include/IndexSVSVamanaImpl.h
)

set(SVS_RUNTIME_SOURCES
    src/IndexSVSImplUtils.h
    src/IndexSVSFlatImpl.cpp
    src/IndexSVSVamanaImpl.cpp
)

option(SVS_RUNTIME_ENABLE_LVQ "Enable compilation of SVS runtime with LVQ support" ON)
if (SVS_RUNTIME_ENABLE_LVQ)
    message(STATUS "SVS runtime will be built with LVQ support")
    list(APPEND SVS_RUNTIME_HEADERS
        include/IndexSVSVamanaLVQImpl.h
    )
    list(APPEND SVS_RUNTIME_SOURCES
        src/IndexSVSVamanaLVQImpl.cpp
    )
else()
    message(STATUS "SVS runtime will be built without LVQ support")
endif()

option(SVS_RUNTIME_ENABLE_LEANVEC "Enable compilation of SVS runtime with LeanVec support" ON)
if (SVS_RUNTIME_ENABLE_LEANVEC)
    message(STATUS "SVS runtime will be built with LeanVec support")
    list(APPEND SVS_RUNTIME_HEADERS
        include/IndexSVSVamanaLeanVecImpl.h
    )
    list(APPEND SVS_RUNTIME_SOURCES
        src/IndexSVSVamanaLeanVecImpl.cpp
    )
else()
    message(STATUS "SVS runtime will be built without LeanVec support")
endif()

find_package(OpenMP REQUIRED)

add_library(${TARGET_NAME} SHARED
    ${SVS_RUNTIME_HEADERS}
    ${SVS_RUNTIME_SOURCES}
)

target_link_libraries(${TARGET_NAME} PUBLIC
    OpenMP::OpenMP_CXX
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(${TARGET_NAME} PRIVATE
    -DSVS_ENABLE_OMP=1
    -fvisibility=hidden
)

target_compile_features(${TARGET_NAME} INTERFACE cxx_std_20)
set_target_properties(${TARGET_NAME} PROPERTIES PUBLIC_HEADER "${SVS_RUNTIME_HEADERS}")
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 20)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(${TARGET_NAME} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR} )

if ((SVS_RUNTIME_ENABLE_LVQ OR SVS_RUNTIME_ENABLE_LEANVEC))
    if(TARGET svs::svs_static_library)
        target_link_libraries(${TARGET_NAME} PRIVATE
            svs::svs
            svs::svs_static_library
            svs_compile_options
            svs_x86_options_base
        )
    else()
        set(SVS_URL "https://github.com/intel/ScalableVectorSearch/releases/download/v1.0.0-dev/svs-shared-library-1.0.0-NIGHTLY-20251017-faiss.tar.gz"
            CACHE STRING "URL to download SVS shared library")
        include(FetchContent)
        FetchContent_Declare(
            svs
            URL ${SVS_URL}
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

