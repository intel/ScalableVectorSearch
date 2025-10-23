# Copilot Instructions for ScalableVectorSearch (SVS)

## Repository Overview

**Scalable Vector Search (SVS)** is a high-performance C++20 library for vector similarity search, optimized for Intel x86 architectures but portable to other platforms. The library implements state-of-the-art Vamana graph-based approximate nearest neighbor (ANN) search and supports billions of high-dimensional vectors with high accuracy and speed. SVS features:

- **Core language**: C++20 with modern concepts for optimal compiler optimizations
- **Can be used as**: Header-only library or with Python bindings
- **Runtime ISA dispatching**: Automatically uses best available instruction set (SSE, AVX2, AVX512)
- **Python bindings**: Require shape-specialized templates for different data dimensionalities
- **Key algorithms**: Vamana graph-based search, LVQ/LeanVec compression (proprietary, available via shared libraries)

**Repository size**: Medium (~10k LOC core library, extensive tests and examples)
**Build system**: CMake 3.21+ with C++20 compiler (GCC 11+, Clang 15+)
**Test framework**: Catch2 v3.4.0 (unit tests), ctest (integration tests)

## Critical Build Instructions

### Prerequisites
- CMake 3.21 or higher
- C++20 compiler: GCC 11+, GCC 12+, or Clang 15+
- Optional: Intel MKL (for IVF support with `-DSVS_EXPERIMENTAL_ENABLE_IVF=ON`)
- Python 3.9+ (for bindings)

### Standard Build Sequence (Always Follow This Order)

**ALWAYS use an out-of-source build directory. NEVER run cmake in the repository root.**

```bash
# 1. Create and enter build directory
mkdir -p build
cd build

# 2. Configure with CMake (use exact flags from CI for consistency)
cmake -DCMAKE_BUILD_TYPE=RelWithDebugInfo \
      -DSVS_BUILD_BINARIES=YES \
      -DSVS_BUILD_TESTS=YES \
      -DSVS_BUILD_EXAMPLES=YES \
      -DSVS_NO_AVX512=NO \
      -DSVS_EXPERIMENTAL_ENABLE_IVF=OFF \
      ..

# 3. Build (typically takes 5-10 minutes on 4 cores)
make -j$(nproc)

# 4. Run tests from build/tests directory
cd tests
ctest -C RelWithDebugInfo
# OR run the test executable directly with filters:
./tests "[integration][build]"
```

**Time expectations**:
- CMake configuration: ~18-20 seconds
- Full build (first time): ~5-10 minutes on 4 cores
- Test suite: ~1-2 minutes
- C++ examples: ~10 seconds

**Important**: If enabling IVF support (`-DSVS_EXPERIMENTAL_ENABLE_IVF=ON`), you MUST first install Intel MKL:
```bash
# On Ubuntu (requires Intel apt repository setup)
sudo apt install intel-oneapi-mkl intel-oneapi-mkl-devel
source /opt/intel/oneapi/setvars.sh
```

### Common Build Options (from cmake/options.cmake)

| Option | Default | Description |
|--------|---------|-------------|
| `SVS_BUILD_BINARIES` | OFF | Build utility binaries in utils/ |
| `SVS_BUILD_TESTS` | OFF | Build test suite (Catch2-based) |
| `SVS_BUILD_EXAMPLES` | OFF | Build C++ examples |
| `SVS_BUILD_BENCHMARK` | OFF | Build benchmark executable |
| `SVS_NO_AVX512` | OFF | Disable Intel AVX-512 intrinsics |
| `SVS_EXPERIMENTAL_ENABLE_IVF` | OFF | Enable IVF (requires MKL) |
| `CMAKE_BUILD_TYPE` | Release | Use `RelWithDebugInfo` for testing |

## Code Formatting and Linting

### Formatting (ALWAYS run before committing)

**Tool**: clang-format version 15.x (specified in `.pre-commit-config.yaml`)
- **DO NOT** use clang-format 16+ or 14 and below - version 15.x is required

```bash
# Format all code (run from repository root)
./tools/clang-format.sh clang-format

# Formatted directories: bindings/python/src, bindings/python/include, 
#                        include, benchmark, tests, utils, examples/cpp
```

### Pre-commit Hooks

The repository uses pre-commit for automated formatting checks:

```bash
# Install pre-commit (if not already installed)
pip install pre-commit

# Install hooks (one-time setup, takes 1-2 minutes)
pre-commit install-hooks

# Run manually (optional, CI will check)
pre-commit run --all-files
```

**CI check**: The `pre-commit.yml` workflow runs on all PRs and will fail if code is not formatted.

## Testing

### C++ Tests (Catch2)

Tests use Catch2 v3 with prefix macros (`CATCH_TEST_CASE`, `CATCH_REQUIRE`, etc.):

```bash
# From build/tests directory
cd build/tests

# Run all tests
ctest -C RelWithDebugInfo
# OR
./tests

# Run specific test tags
./tests "[integration][build]"
./tests "[core][distance]"

# List available tags
./tests --list-tags

# Run with verbose output
CTEST_OUTPUT_ON_FAILURE=1 ctest -C RelWithDebugInfo
```

**Test tags commonly used**: `[integration]`, `[build]`, `[core]`, `[distance]`, `[vamana]`, `[data]`

### C++ Examples

Examples are tested via ctest:

```bash
cd build/examples/cpp
ctest -C RelWithDebugInfo
# Runs 10 example tests (~9 seconds total)
```

### Python Tests

Python tests use pytest (location: `bindings/python/tests/`):

```bash
# Build Python bindings first (requires scikit-build)
cd bindings/python
pip install -e .

# Run tests
pytest tests/
```

## Project Structure

```
ScalableVectorSearch/
├── .github/
│   ├── workflows/           # CI/CD pipelines
│   │   ├── build-linux.yml  # Main build & test (Ubuntu 22.04, g++/clang)
│   │   ├── pre-commit.yml   # Format checking
│   │   ├── cibuildwheel.yml # Python wheel building
│   │   └── build-*.y{a}ml   # macOS, ARM builds
│   └── scripts/             # CI helper scripts
├── benchmark/               # Benchmarking framework
│   ├── include/             # Benchmark headers
│   └── src/                 # Benchmark implementations
├── bindings/python/         # Python API (pybind11-based)
│   ├── include/             # Python binding headers
│   ├── src/                 # Binding implementations
│   ├── tests/               # Python unit tests (pytest)
│   ├── setup.py             # Python package setup
│   └── pyproject.toml       # Build configuration
├── cmake/                   # CMake modules
│   ├── options.cmake        # ** BUILD OPTIONS (IMPORTANT) **
│   ├── multi-arch.cmake     # Multi-architecture support
│   └── *.cmake              # Dependency configs (eve, fmt, spdlog, etc.)
├── data/                    # Test data and schemas
│   ├── test_dataset/        # Small test datasets
│   └── schemas/             # TOML schemas for serialization
├── docker/                  # Docker build environments
├── examples/
│   ├── cpp/                 # C++ usage examples
│   │   ├── vamana.cpp       # Main search example
│   │   ├── types.cpp        # Supported types
│   │   ├── saveload.cpp     # Save/load patterns
│   │   ├── dispatcher.cpp   # Compile-time dispatch
│   │   └── shared/          # LVQ/LeanVec via shared library
│   └── python/              # Python examples
├── include/svs/             # ** CORE LIBRARY HEADERS **
│   ├── lib/                 # Foundation: arrays, threads, I/O, SIMD
│   ├── core/                # Core: distance, data structures, allocators
│   ├── index/               # Index implementations
│   │   ├── vamana/          # Vamana graph index
│   │   ├── flat/            # Flat (brute-force) index
│   │   └── inverted/        # Inverted index (IVF)
│   ├── orchestrators/       # High-level APIs with type-erasure for simple interfaces
│   ├── quantization/        # Vector quantization
│   └── extensions/          # svs_invoke overloads to hook into core SVS routines
├── tests/                   # ** C++ TEST SUITE **
│   ├── svs/                 # Unit tests (mirrors include/svs/)
│   ├── integration/         # Integration tests
│   ├── benchmark/           # Benchmark tests
│   └── utils/               # Test utilities
├── tools/
│   ├── clang-format.sh      # ** FORMATTING SCRIPT (USE THIS) **
│   └── benchmark_inputs/    # Benchmark configurations
├── utils/                   # Command-line utilities
│   ├── build_index.cpp      # Index building tool
│   ├── search_index.cpp     # Search tool
│   └── benchmarks/          # Benchmark runners
├── CMakeLists.txt           # Main CMake configuration
├── .pre-commit-config.yaml  # Pre-commit configuration
├── .clang-format            # Formatting rules
└── README.md                # Project documentation
```

## Key Files and Configurations

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Main build configuration, version (0.0.10) |
| `cmake/options.cmake` | **All build options and flags** |
| `.pre-commit-config.yaml` | Formatting tool versions (clang-format 15) |
| `.clang-format` | Code formatting rules |
| `tools/clang-format.sh` | **Script to format all code** |
| `.github/workflows/build-linux.yml` | **Reference CI configuration** |

## CI/CD Pipeline

Main checks that run on every PR:

1. **build-linux.yml**: Builds with multiple compilers (g++-11, g++-12, clang++-15) in `RelWithDebugInfo` mode, runs all C++ tests and examples
2. **pre-commit.yml**: Verifies code formatting with clang-format 15
3. **cibuildwheel.yml**: Builds Python wheels (uses custom manylinux2014 container)

**To replicate CI locally**: Use the exact cmake command from `build-linux.yml` configuration step.

## Common Issues and Workarounds

### Build Issues

1. **Problem**: Build fails with uninitialized variable warnings on GCC 12+
   - **Solution**: Already handled - GCC 12+ adds `-Wno-uninitialized` automatically in cmake/options.cmake

2. **Problem**: IVF tests fail or IVF won't build
   - **Solution**: IVF requires Intel MKL - either install MKL or use `-DSVS_EXPERIMENTAL_ENABLE_IVF=OFF`

3. **Problem**: Tests timeout or take very long
   - **Solution**: Integration tests can take 1-2 minutes; use specific test filters for faster iteration

### Formatting Issues

1. **Problem**: Pre-commit fails with wrong clang-format version
   - **Solution**: Ensure clang-format 15.x is installed (not 16+)

2. **Problem**: clang-format script fails
   - **Solution**: Run from repository root: `./tools/clang-format.sh clang-format`

## Quick Reference Commands

```bash
# Complete build from scratch
rm -rf build && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebugInfo -DSVS_BUILD_TESTS=YES -DSVS_BUILD_EXAMPLES=YES ..
make -j$(nproc)
cd tests && ./tests

# Format code before commit
./tools/clang-format.sh clang-format

# Run specific test subset
cd build/tests && ./tests "[integration]"

# Check available test tags
cd build/tests && ./tests --list-tags

# Clean and rebuild
rm -rf build && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebugInfo \
      -DSVS_BUILD_BINARIES=YES \
      -DSVS_BUILD_TESTS=YES \
      -DSVS_BUILD_EXAMPLES=YES \
      -DSVS_NO_AVX512=NO \
      -DSVS_EXPERIMENTAL_ENABLE_IVF=OFF \
      ..
make -j$(nproc)
```

## Important Notes for Coding Agents

1. **Trust these instructions first** - Only search the repository if information here is incomplete or incorrect
2. **Always build out-of-source** - Use a `build/` directory, never configure CMake in the repository root
3. **Follow the CI configuration** - Use the same cmake flags as `.github/workflows/build-linux.yml` for consistency
4. **Format before committing** - Run `./tools/clang-format.sh clang-format` to avoid CI failures. **IMPORTANT**: Only format files you modify; do not include formatting changes from other files in your PR
5. **Test early and often** - Build times are reasonable (~5-10 min), so test incrementally
6. **Tests are required** - New features and bugfixes must be accompanied by tests. For bugs, first reproduce the issue in a unit test, then fix it in the code
7. **Header-only library** - Most code is in `include/svs/`, changes don't require recompiling everything
8. **ISA dispatching** - Runtime dispatch means the same binary runs on different CPU architectures
9. **Type erasure** - Orchestrators use type-erasure to provide simple and consistent interfaces across different implementations
10. **Extensions system** - The `extensions/` directory provides `svs_invoke` overloads/specializations to hook into core SVS routines (similar to `std::invoke`)
11. **Test filters are your friend** - Use Catch2 tags to run subsets of tests during development
12. **Python bindings are specialized** - Changes to template parameters may require Python binding updates
13. **Version is synchronized** - Keep version in sync across `CMakeLists.txt`, `setup.py`, and test files

## Additional Resources

- **Documentation**: https://intel.github.io/ScalableVectorSearch
- **Main README**: See repository root `README.md` for algorithm details and performance benchmarks
- **C++ Examples**: See `examples/cpp/README.md` for usage patterns
- **Test Dataset**: Small test vectors are in `data/test_dataset/` for quick validation
