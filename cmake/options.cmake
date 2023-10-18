if(svs_options_cmake_included)
    return()
endif()
set(svs_options_cmake_included true)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#####
##### Official Options
#####

option(SVS_BUILD_BINARIES
    "Build the utility binaries"
    OFF # enabled by default
)

option(SVS_BUILD_TESTS
    "Build the unit test suite."
    OFF # disabled by default
)

option(SVS_BUILD_DOCS
    "Build the library documentation."
    OFF # disabled by default
)

option(SVS_BUILD_EXAMPLES
    "Build the examples. If combined with SVS_BUILD_TESTS, will also test the examples."
    OFF # disabled by default
)

option(SVS_BUILD_BENCHMARK
    "Build the benchmark executable."
    OFF # disabled by default
)

option(SVS_TEST_EXAMPLES
    "Enable ctest for examples without requiring building the test suite."
    OFF # disabled by default
)

option(SVS_NO_AVX512
    "Disable use of AVX512 intrinsics."
    OFF # disabled by default
)

option(SVS_FORCE_INTEGRATION_TESTS
    "Run integration tests in debug mode (slow)"
    OFF #disabled by default
)

#####
##### Experimental
#####

option(SVS_EXPERIMENTAL_CLANG_TIDY
    "Run the clang-tidy static analyzer on utility and binding code."
    OFF # disabled by default
)

option(SVS_EXPERIMENTAL_CHECK_BOUNDS
    "Enable bounds checking on many data accesses."
    OFF # diabled by default
)

option(SVS_EXPERIMENTAL_ENABLE_NUMA
    "Enable NUMA aware data structures. (Experimental)"
    OFF # disabled by default
)

#####
##### Configuration
#####

if (SVS_NO_AVX512)
    # AVX512F is the base for the AVX512 instruction set.
    # Adding the `-mno-avx512f` flag will disable all AVX512 dependent instructions.
    target_compile_options(${SVS_LIB} INTERFACE -mno-avx512f)
endif()

if (SVS_EXPERIMENTAL_CHECK_BOUNDS)
    target_compile_definitions(${SVS_LIB} INTERFACE -DSVS_CHECK_BOUNDS)
endif()

if (SVS_EXPERIMENTAL_ENABLE_NUMA)
    target_compile_options(${SVS_LIB} INTERFACE -DSVS_ENABLE_NUMA)
endif()

#####
##### Helper target to apply relevant compiler optimizations.
#####

add_library(svs_native_options INTERFACE)
add_library(svs::native_options ALIAS svs_native_options)
target_compile_options(svs_native_options INTERFACE -march=native -mtune=native)

# Use an internal INTERFACE target to apply the same build options to both the
# unit test and the compiled binaries.
add_library(svs_compile_options INTERFACE)
add_library(svs::compile_options ALIAS svs_compile_options)

target_compile_options(
    svs_compile_options
    INTERFACE
        -Werror
        -Wall
        -Wextra
        -Wpedantic
        -Wno-gnu-zero-variadic-macro-arguments
        -Wno-parentheses # GCC in CI has issues without it
)


if(CMAKE_BUILD_TYPE STREQUAL Release OR CMAKE_BUILD_TYPE STREQUAL RelWithDebugInfo)
    target_compile_options(svs_compile_options INTERFACE -O3)
endif()

#####
##### Compiler specific flags
#####

# Fix Clang complaining about the sized delete operator.
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${SVS_LIB} INTERFACE -fsized-deallocation)
endif()

# Provide better diagnostics for broken templates.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(
        svs_compile_options
        INTERFACE
        -fconcepts-diagnostics-depth=10
        -ftemplate-backtrace-limit=0
    )

    if (CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 12.0)
        # GCC-12 throws errors in its own intrinsics library with uninitialized variables.
        # See: https://sourceware.org/bugzilla/show_bug.cgi?id=19444
        #
        # Since we're largely header-only and can't separately compile distance related
        # functions, the best we can do is disable the offending checks.
        target_compile_options(svs_compile_options INTERFACE -Wno-uninitialized)
    endif()
endif()

