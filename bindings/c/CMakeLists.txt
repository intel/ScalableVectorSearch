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
    src/storage.hpp
    src/index.hpp
    src/index_builder.hpp
    src/threadpool.hpp
    src/types_support.hpp
    src/svs_c.cpp
)

add_library(${TARGET_NAME} SHARED
    ${SVS_C_API_HEADERS}
    ${SVS_C_API_SOURCES}
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
set_target_properties(${TARGET_NAME} PROPERTIES PUBLIC_HEADER "${SVS_C_API_HEADERS}")
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 20)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(${TARGET_NAME} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR} )

target_link_libraries(${TARGET_NAME} PRIVATE
    svs::svs
)
link_mkl_static(${TARGET_NAME})
# target_compile_definitions(${TARGET_NAME} PRIVATE
#     PUBLIC "SVS_LVQ_HEADER=\"${SVS_LVQ_HEADER}\""
#     PUBLIC "SVS_LEANVEC_HEADER=\"${SVS_LEANVEC_HEADER}\""
# )

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
