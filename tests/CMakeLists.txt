include(FetchContent)

# Do wide printing for the console logger for Catch2
# Reference: https://github.com/catchorg/Catch2/issues/1348
set(CATCH_CONFIG_CONSOLE_WIDTH "100" CACHE STRING "" FORCE)
set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(CATCH_CONFIG_ENABLE_BENCHMARKING OFF CACHE BOOL "" FORCE)
set(CATCH_CONFIG_FAST_COMPILE OFF CACHE BOOL "" FORCE)

# If we don't configure the prefix version of the Catch2 macros, then we get compiler
# errors about repeated definitions of the macro `INFO`
set(CATCH_CONFIG_PREFIX_ALL ON CACHE BOOL "" FORCE)

# Download the Catch2 unit testing suite.
# Reference: https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md
#
# Catch2 compiles using the compiler default (something like C++14).
# Unfortunately - this means we miss out on `std::string_view` for stringification.
# Since Catch2 does not provide a mechanism for building with a specific C++ version,
# we need to use this hack.
set(PRESET_CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD ${SVS_CXX_STANDARD})
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
)

FetchContent_MakeAvailable(Catch2)
set(CMAKE_CXX_STANDARD ${PRESET_CMAKE_CXX_STANRARD})

# Add top level test executable and link with the unit test library.
set(TEST_DIR ${CMAKE_CURRENT_LIST_DIR})
set(TEST_SOURCES
    # Testing Utilities
    ${TEST_DIR}/utils/utils.cpp
    ${TEST_DIR}/utils/benchmarks.cpp
    ${TEST_DIR}/utils/generators.cpp
    ${TEST_DIR}/utils/inverted_reference.cpp
    ${TEST_DIR}/utils/require_error.cpp
    ${TEST_DIR}/utils/schemas.cpp
    ${TEST_DIR}/utils/test_dataset.cpp
    ${TEST_DIR}/utils/vamana_reference.cpp
    # Lib
    ${TEST_DIR}/svs/lib/algorithms.cpp
    ${TEST_DIR}/svs/lib/array.cpp
    ${TEST_DIR}/svs/lib/datatype.cpp
    ${TEST_DIR}/svs/lib/dispatcher.cpp
    ${TEST_DIR}/svs/lib/exception.cpp
    ${TEST_DIR}/svs/lib/expected.cpp
    ${TEST_DIR}/svs/lib/invoke.cpp
    ${TEST_DIR}/svs/lib/file.cpp
    ${TEST_DIR}/svs/lib/file_iterator.cpp
    ${TEST_DIR}/svs/lib/float16.cpp
    ${TEST_DIR}/svs/lib/memory.cpp
    ${TEST_DIR}/svs/lib/meta.cpp
    ${TEST_DIR}/svs/lib/misc.cpp
    ${TEST_DIR}/svs/lib/narrow.cpp
    ${TEST_DIR}/svs/lib/neighbor.cpp
    # ${TEST_DIR}/svs/lib/numa.cpp -- requires SVS_EXPERIMENTAL_ENABLE_NUMA
    ${TEST_DIR}/svs/lib/prefetch.cpp
    ${TEST_DIR}/svs/lib/preprocessor.cpp
    ${TEST_DIR}/svs/lib/readwrite.cpp
    ${TEST_DIR}/svs/lib/saveload.cpp
    ${TEST_DIR}/svs/lib/saveload/load.cpp
    ${TEST_DIR}/svs/lib/spinlock.cpp
    ${TEST_DIR}/svs/lib/static.cpp
    ${TEST_DIR}/svs/lib/timing.cpp
    ${TEST_DIR}/svs/lib/tuples.cpp
    ${TEST_DIR}/svs/lib/threads/types.cpp
    ${TEST_DIR}/svs/lib/threads/thunks.cpp
    ${TEST_DIR}/svs/lib/threads/thread.cpp
    ${TEST_DIR}/svs/lib/threads/threadlocal.cpp
    ${TEST_DIR}/svs/lib/threads/threadpool.cpp
    ${TEST_DIR}/svs/lib/type_traits.cpp
    ${TEST_DIR}/svs/lib/version.cpp
    ${TEST_DIR}/svs/lib/uuid.cpp
    ${TEST_DIR}/svs/lib/concurrency/readwrite_protected.cpp
    # Third Party
    ${TEST_DIR}/svs/third-party/fmt.cpp
    ${TEST_DIR}/svs/third-party/toml.cpp
    # Benchmark
    ${TEST_DIR}/benchmark/benchmark.cpp
    # Concepts
    ${TEST_DIR}/svs/concepts/distance.cpp
    # Core
    ${TEST_DIR}/svs/core/allocator.cpp
    ${TEST_DIR}/svs/core/compact.cpp
    ${TEST_DIR}/svs/core/data.cpp
    ${TEST_DIR}/svs/core/data/block.cpp
    ${TEST_DIR}/svs/core/data/simple.cpp
    ${TEST_DIR}/svs/core/data/serialization.cpp
    ${TEST_DIR}/svs/core/data/view.cpp
    ${TEST_DIR}/svs/core/distance.cpp
    ${TEST_DIR}/svs/core/distances/simd_utils.cpp
    ${TEST_DIR}/svs/core/distances/distance_euclidean.cpp
    ${TEST_DIR}/svs/core/distances/inner_product.cpp
    ${TEST_DIR}/svs/core/distances/cosine.cpp
    ${TEST_DIR}/svs/core/graph.cpp
    ${TEST_DIR}/svs/core/io/vecs.cpp
    ${TEST_DIR}/svs/core/io/binary.cpp
    ${TEST_DIR}/svs/core/io/native.cpp
    ${TEST_DIR}/svs/core/io.cpp
    ${TEST_DIR}/svs/core/loading.cpp
    ${TEST_DIR}/svs/core/logging.cpp
    ${TEST_DIR}/svs/core/kmeans.cpp
    ${TEST_DIR}/svs/core/medioid.cpp
    ${TEST_DIR}/svs/core/recall.cpp
    ${TEST_DIR}/svs/core/translation.cpp
    # Index Specific Functionality
    ${TEST_DIR}/svs/index/index.cpp
    ${TEST_DIR}/svs/index/flat/inserters.cpp
    ${TEST_DIR}/svs/index/vamana/build_parameters.cpp
    ${TEST_DIR}/svs/index/vamana/consolidate.cpp
    ${TEST_DIR}/svs/index/vamana/filter.cpp
    ${TEST_DIR}/svs/index/vamana/index.cpp
    ${TEST_DIR}/svs/index/vamana/prune.cpp
    ${TEST_DIR}/svs/index/vamana/search_buffer.cpp
    ${TEST_DIR}/svs/index/vamana/search_parameters.cpp
    ${TEST_DIR}/svs/index/vamana/vamana_build.cpp
    ${TEST_DIR}/svs/index/vamana/vamana_build.cpp
    # Inverted
    ${TEST_DIR}/svs/index/inverted/clustering.cpp

    # # ${TEST_DIR}/svs/index/vamana/dynamic_index.cpp
)

#####
##### Integration Tests
#####

SET(INTEGRATION_TESTS
    ${TEST_DIR}/svs/index/vamana/dynamic_index_2.cpp
    # Higher level constructs
    ${TEST_DIR}/svs/orchestrators/vamana.cpp
    # Integration Tests
    ${TEST_DIR}/integration/exhaustive.cpp
    # ${TEST_DIR}/integration/numa_search.cpp -- requires SVS_EXPERIMENTAL_ENABLE_NUMA
)

if ((NOT CMAKE_BUILD_TYPE STREQUAL "Debug") OR SVS_FORCE_INTEGRATION_TESTS)
    message("Enabling Integration Tests!")
    list(APPEND TEST_SOURCES ${INTEGRATION_TESTS})
endif()

#####
##### Numa Tests
#####

set(NUMA_TESTS
    ${TEST_DIR}/svs/lib/numa.cpp
    ${TEST_DIR}/integration/numa_search.cpp
)

if (SVS_EXPERIMENTAL_ENABLE_NUMA)
    message("Enabling NUMA tests!")
    list(APPEND TEST_SOURCES ${NUMA_TESTS})
endif()

add_executable(tests ${TEST_SOURCES})

# Path to the test dataset.
set(DATA_DIRECTORY "${PROJECT_SOURCE_DIR}/data")

# Communicate to the test application where it can find test data sets.
target_compile_definitions(tests PRIVATE SVS_TEST_DATA_DIR="${DATA_DIRECTORY}")

target_link_libraries(tests PRIVATE ${SVS_LIB})
target_link_libraries(
    tests PRIVATE svs_compile_options svs_native_options svs_benchmark_library
)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

target_include_directories(tests PRIVATE ${PROJECT_SOURCE_DIR})

# Test enumeration
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(tests)

