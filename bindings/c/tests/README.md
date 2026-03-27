# C API Tests

This directory contains comprehensive tests for the SVS C API using the Catch2 testing framework.

## Test Structure

The tests are organized into separate files by functionality:

- **c_api_error.cpp**: Tests for error handling functionality
- **c_api_algorithm.cpp**: Tests for algorithm creation and configuration (Vamana)
- **c_api_storage.cpp**: Tests for storage configurations (Simple, LeanVec, LVQ, SQ)
- **c_api_search_params.cpp**: Tests for search parameter creation and configuration
- **c_api_index_builder.cpp**: Tests for index builder creation and configuration
- **c_api_index.cpp**: Tests for index building, searching, and basic operations
- **c_api_dynamic_index.cpp**: Tests for dynamic index operations (add, delete, consolidate, compact)

Note: The main() function is provided by Catch2::Catch2WithMain automatically.

## Building the Tests

The tests are built as part of the C API build process. To build them:

```bash
# From the build directory
cmake -DSVS_BUILD_C_API_TESTS=ON ..
make svs_c_api_tests
```

To disable building tests:

```bash
cmake -DSVS_BUILD_C_API_TESTS=OFF ..
```

## Running the Tests

### Run all tests

```bash
./svs_c_api_tests
```

### Run specific test cases

```bash
# Run error handling tests only
./svs_c_api_tests "[c_api][error]"

# Run algorithm tests only
./svs_c_api_tests "[c_api][algorithm]"

# Run all index tests
./svs_c_api_tests "[c_api][index]"

# Run dynamic index tests
./svs_c_api_tests "[c_api][dynamic]"
```

### Run with verbose output

```bash
./svs_c_api_tests -s
```

### List all available tests

```bash
./svs_c_api_tests --list-tests
```

### Run with CTest

```bash
ctest -R svs_c_api_tests
```

## Test Coverage

The tests cover the following aspects of the C API:

### Error Handling

- Error handle creation and cleanup
- Error state checking
- Error codes and messages
- NULL error handle support

### Algorithm Configuration

- Vamana algorithm creation
- Parameter getters and setters (graph_degree, build_window_size, alpha, search_history)
- Invalid parameter handling

### Storage Configuration

- Simple storage (Float32, Float16, Int8, Uint8)
- LeanVec storage (various primary/secondary combinations)
- LVQ storage (with and without residual)
- Scalar Quantization storage

### Search Parameters

- Vamana search parameter creation
- Various window sizes

### Index Builder

- Index builder creation with different metrics (Euclidean, Cosine, Dot Product)
- Storage configuration
- Thread pool configuration (Native, OMP, Custom)

### Index Operations

- Index building from data
- Searching with queries
- Different K values
- Distance calculation
- Vector reconstruction
- Thread count management

### Dynamic Index Operations

- Dynamic index building with/without explicit IDs
- Adding points
- Deleting points
- ID existence checking
- Index consolidation
- Index compaction
- Search after modifications

## Test Patterns

The tests follow the patterns established in the SVS project:

1. Use `CATCH_TEST_CASE` for test case definitions
2. Use `CATCH_SECTION` for test subsections
3. Use `CATCH_REQUIRE` for assertions
4. Clean up all resources (free handles) after each test
5. Test both success and error paths
6. Test with and without NULL error handles

## Adding New Tests

When adding new tests:

1. Create a new `.cpp` file or add to an existing one
2. Follow the existing structure and naming conventions
3. Include proper copyright header
4. Use appropriate test tags: `[c_api][functionality]`
5. Add the new test file to `CMakeLists.txt` if needed
6. Clean up all allocated resources
7. Test both success and error conditions

## Dependencies

- Catch2 v3.x (automatically fetched if not found)
- SVS C API library
- C++17 or later compiler

## Notes

- Tests use a simple sequential thread pool for deterministic behavior
- Test data is generated programmatically for repeatability
- Some tests may be skipped if optional features are not enabled (e.g., LVQ/LeanVec)
