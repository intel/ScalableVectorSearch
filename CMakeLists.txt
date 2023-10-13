cmake_minimum_required(VERSION 3.18)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

project(svs
    LANGUAGES CXX
    # The version is tested in the files:
    # - /tests/svs/lib/version.cpp
    # - /bindings/python/tests/test_common.py
    # Manually keep in-sync with:
    # - /bindings/python/setup.py
    VERSION 0.0.1
)

set(SVS_LIB svs_devel)

# Version variables.
set(SVS_VERSION ${PROJECT_VERSION})
set(SVS_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(SVS_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(SVS_VERSION_PATCH ${PROJECT_VERSION_PATCH})

add_library(${SVS_LIB} INTERFACE)
set_target_properties(${SVS_LIB} PROPERTIES EXPORT_NAME svs)
add_library(svs::svs ALIAS ${SVS_LIB})

# Runtime include directories are established in the installation logic.
target_include_directories(
    ${SVS_LIB}
    INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

# We use C++ 20 features in our exposed headers.
# Anyone using us as a depdency and including our headers will need to be C++20 compatible.
target_compile_features(${SVS_LIB} INTERFACE cxx_std_20)

# Populate with the version number macro.
target_compile_options(
    ${SVS_LIB}
    INTERFACE
    "-DSVS_VERSION_MAJOR=${SVS_VERSION_MAJOR}"
    "-DSVS_VERSION_MINOR=${SVS_VERSION_MINOR}"
    "-DSVS_VERSION_PATCH=${SVS_VERSION_PATCH}"
)

#####
##### Options and extra build steps
#####

include("cmake/options.cmake")

include("cmake/clang-tidy.cmake")
include("cmake/eve.cmake")
include("cmake/pthread.cmake")
include("cmake/numa.cmake")
include("cmake/robin-map.cmake")
include("cmake/fmt.cmake")
include("cmake/toml.cmake")

#####
##### Build Objects
#####

if(SVS_BUILD_BINARIES)
    add_subdirectory(utils)
endif()

if(SVS_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(SVS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(SVS_BUILD_DOCS)
    add_subdirectory(docs)
endif()

# The benchmark directory contains a sub-component that is used by both the benchmarking
# framework and the unit-tests.
#
# If only the unit tests are enabled, then the benchmark will be built as a minimal
# component to avoid excessive compilation time.
if(SVS_BUILD_BENCHMARK OR SVS_BUILD_TESTS)
    add_subdirectory(benchmark)
endif()

#####
##### Install Logic
#####

include(GNUInstallDirs)

# Location of auxiliary generated cmake files to help consumers of this package.
set(LIB_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/svs")

# Install headers and target information.
install(
    TARGETS svs_devel svs_compile_options svs_native_options
    EXPORT svs-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/svs"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.h"
)
install(
    EXPORT svs-targets
    NAMESPACE "svs::"
    DESTINATION "${LIB_CONFIG_INSTALL_DIR}"
)

#####
##### Config File
#####

set(VERSION_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/svsConfigVersion.cmake")

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/cmake/Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/svsConfig.cmake"
    INSTALL_DESTINATION "${LIB_CONFIG_INSTALL_DIR}"
)

# Don't make compatibility guarentees until we reach a compatibility milestone.
write_basic_package_version_file(
    ${VERSION_CONFIG}
    VERSION ${SVS_VERSION}
    COMPATIBILITY ExactVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/svsConfig.cmake"
    "${VERSION_CONFIG}"
    DESTINATION "${LIB_CONFIG_INSTALL_DIR}"
)

# Copy any "Find*" files that may be needed.
set(CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(SVS_CMAKE_FIND_FILES
    ${CMAKE_DIR}/FindNuma.cmake
)

install(FILES
    ${SVS_CMAKE_FIND_FILES}
    DESTINATION "${LIB_CONFIG_INSTALL_DIR}"
)

