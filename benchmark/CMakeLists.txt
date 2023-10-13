# The svs-benchmark consists of two components:
#
# (1) A shared-library containing most of the implementation logic.
# (2) A thin executable that provides a command-line interface to the shared-library.
#
# This is because svs-benchmark provides utilities that can be reused for testing.
# This is done by linking the test executable to svs-benchmark shared library.

# As a compile-time optimization, if we're just building the tests, we disable all
# specializations compiled in the shared library.
if (${SVS_BUILD_BENCHMARK})
    set(SVS_BENCHMARK_MINIMAL OFF)
else()
    set(SVS_BENCHMARK_MINIMAL ON)
endif()

set(SHARED_LIBRARY_FILES
    src/benchmark.cpp
    src/vamana/build.cpp
)

add_library(svs_benchmark_library SHARED ${SHARED_LIBRARY_FILES})
target_include_directories(svs_benchmark_library PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

if (${SVS_BENCHMARK_MINIMAL})
    message("Compiling minimal benchmark library")
    target_compile_options(svs_benchmark_library PUBLIC -DSVS_BENCHMARK_MINIMAL)
endif()

# Link the library with the main SVS library.
target_link_libraries(
    svs_benchmark_library
    PUBLIC
        ${SVS_LIB}
        svs_compile_options
        svs_native_options
        fmt::fmt
)

# Build the shared library.
set(EXE_FILES src/main.cpp)
add_executable(svs_benchmark ${EXE_FILES})
target_link_libraries(svs_benchmark PRIVATE svs_benchmark_library)
