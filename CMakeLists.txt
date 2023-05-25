cmake_minimum_required(VERSION 3.18)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

project(svs
    LANGUAGES CXX
    VERSION 0.0.1
)

set(SVS_LIB svs_devel)
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

#####
##### Options and extra build steps
#####

include("cmake/options.cmake")
include("cmake/clang-tidy.cmake")
include("cmake/pthread.cmake")
include("cmake/numa.cmake")
include("cmake/robin-map.cmake")
include("cmake/fmt.cmake")
include("cmake/toml.cmake")

# TODO: Right now this is always needed.
# Decoupling our dependence on EVE to disable quantization will be ... tricky ...
if(SVS_ENABLE_QUANTIZATION)
    include("cmake/eve.cmake")
endif()

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

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/cmake/Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/svsConfig.cmake"
    INSTALL_DESTINATION "${LIB_CONFIG_INSTALL_DIR}"
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/svsConfig.cmake"
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

