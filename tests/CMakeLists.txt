include(FetchContent)

# Do wide printing for the console logger for Catch2
# Reference: https://github.com/catchorg/Catch2/issues/1348
set(CATCH_CONFIG_CONSOLE_WIDTH "100" CACHE STRING "" FORCE)
set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)

# If we don't configure the prefix version of the Catch2 macros, then we get compiler
# errors about repeated definitions of the macro `INFO`
set(CATCH_CONFIG_PREFIX_ALL ON CACHE BOOL "" FORCE)

# Download the Catch2 unit testing suite.
# Reference: https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.3.2
)

FetchContent_MakeAvailable(Catch2)

# Add top level test executable and link with the unit test library.
set(TEST_DIR ${CMAKE_CURRENT_LIST_DIR})
set(TEST_SOURCES
    # Testing Utilities
    ${TEST_DIR}/utils/utils.cpp
    ${TEST_DIR}/utils/benchmarks.cpp
    ${TEST_DIR}/utils/generators.cpp
    ${TEST_DIR}/utils/schemas.cpp
    # Lib
    ${TEST_DIR}/svs/lib/algorithms.cpp
    ${TEST_DIR}/svs/lib/array.cpp
    ${TEST_DIR}/svs/lib/datatype.cpp
    ${TEST_DIR}/svs/lib/dispatcher.cpp
    ${TEST_DIR}/svs/lib/exception.cpp
    ${TEST_DIR}/svs/lib/invoke.cpp
    ${TEST_DIR}/svs/lib/file.cpp
    ${TEST_DIR}/svs/lib/file_iterator.cpp
    ${TEST_DIR}/svs/lib/float16.cpp
    ${TEST_DIR}/svs/lib/meta.cpp
    ${TEST_DIR}/svs/lib/misc.cpp
    ${TEST_DIR}/svs/lib/narrow.cpp
    ${TEST_DIR}/svs/lib/neighbor.cpp
    # ${TEST_DIR}/svs/lib/numa.cpp -- requires SVS_EXPERIMENTAL_ENABLE_NUMA
    ${TEST_DIR}/svs/lib/readwrite.cpp
    ${TEST_DIR}/svs/lib/saveload.cpp
    ${TEST_DIR}/svs/lib/spinlock.cpp
    ${TEST_DIR}/svs/lib/static.cpp
    ${TEST_DIR}/svs/lib/timing.cpp
    ${TEST_DIR}/svs/lib/tuples.cpp
    ${TEST_DIR}/svs/lib/threads/types.cpp
    ${TEST_DIR}/svs/lib/threads/thunks.cpp
    ${TEST_DIR}/svs/lib/threads/thread.cpp
    ${TEST_DIR}/svs/lib/threads/threadlocal.cpp
    ${TEST_DIR}/svs/lib/threads/threadpool.cpp
    ${TEST_DIR}/svs/lib/version.cpp
    ${TEST_DIR}/svs/lib/uuid.cpp
    # Third Party
    ${TEST_DIR}/svs/third-party/fmt.cpp
    ${TEST_DIR}/svs/third-party/toml.cpp
    # Concepts
    ${TEST_DIR}/svs/concepts/distance.cpp
    # Core
    ${TEST_DIR}/svs/core/allocator.cpp
    ${TEST_DIR}/svs/core/compact.cpp
    ${TEST_DIR}/svs/core/data.cpp
    ${TEST_DIR}/svs/core/data/block.cpp
    ${TEST_DIR}/svs/core/data/simple.cpp
    ${TEST_DIR}/svs/core/distance.cpp
    ${TEST_DIR}/svs/core/distances/simd_utils.cpp
    ${TEST_DIR}/svs/core/distances/distance_euclidean.cpp
    ${TEST_DIR}/svs/core/distances/inner_product.cpp
    ${TEST_DIR}/svs/core/distances/cosine.cpp
    ${TEST_DIR}/svs/core/graph.cpp
    ${TEST_DIR}/svs/core/io/vecs.cpp
    ${TEST_DIR}/svs/core/io/native.cpp
    ${TEST_DIR}/svs/core/io.cpp
    ${TEST_DIR}/svs/core/loading.cpp
    ${TEST_DIR}/svs/core/kmeans.cpp
    ${TEST_DIR}/svs/core/medioid.cpp
    ${TEST_DIR}/svs/core/recall.cpp
    ${TEST_DIR}/svs/core/translation.cpp
    # Index Specific Functionality
    ${TEST_DIR}/svs/index/flat/inserters.cpp
    ${TEST_DIR}/svs/index/vamana/build_parameters.cpp
    ${TEST_DIR}/svs/index/vamana/consolidate.cpp
    ${TEST_DIR}/svs/index/vamana/index.cpp
    ${TEST_DIR}/svs/index/vamana/prune.cpp
    ${TEST_DIR}/svs/index/vamana/search_buffer.cpp
    ${TEST_DIR}/svs/index/vamana/vamana_build.cpp
    # # ${TEST_DIR}/svs/index/vamana/dynamic_index.cpp
    ${TEST_DIR}/svs/quantization/lvq/compressed.cpp
    ${TEST_DIR}/svs/quantization/lvq/datasets.cpp
    ${TEST_DIR}/svs/quantization/lvq/vectors.cpp
    ${TEST_DIR}/svs/quantization/lvq/global_bias.cpp
    ${TEST_DIR}/svs/quantization/lvq/vector_top.cpp
    ${TEST_DIR}/svs/quantization/lvq/lvq.cpp
)

#####
##### Integration Tests
#####

SET(INTEGRATION_TESTS
    ${TEST_DIR}/svs/index/vamana/dynamic_index_2.cpp
    # # Higher level constructs
    ${TEST_DIR}/svs/orchestrators/vamana.cpp
    # # Integration Tests
    ${TEST_DIR}/integration/exhaustive.cpp
    ${TEST_DIR}/integration/index_search.cpp
    ${TEST_DIR}/integration/index_build.cpp
    ${TEST_DIR}/integration/lvq_search.cpp
    # # ${TEST_DIR}/integration/numa_search.cpp -- requires SVS_EXPERIMENTAL_ENABLE_NUMA
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


#####
##### Test Executable
#####

add_executable(tests ${TEST_SOURCES})

# Path to the test dataset.
set(DATA_DIRECTORY "${PROJECT_SOURCE_DIR}/data")

# Communicate to the test application where it can find test data sets.
target_compile_definitions(tests PRIVATE SVS_TEST_DATA_DIR="${DATA_DIRECTORY}")

target_link_libraries(tests PRIVATE ${SVS_LIB})
target_link_libraries(tests PRIVATE svs_compile_options svs_native_options)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

target_include_directories(tests PRIVATE ${PROJECT_SOURCE_DIR})

# Test enumeration
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(tests)

