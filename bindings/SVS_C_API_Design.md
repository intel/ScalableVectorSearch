# SVS C API Design Proposal

## Overview

This document describes the design proposal for the Scalable Vector Search (SVS) C API. The API provides a C interface to SVS's vector similarity search capabilities, enabling integration with C applications and other languages that support C FFI (Foreign Function Interface).

### Design Goals

The SVS C API is designed with the following principles:

1. **Simplicity** - Provide a minimal, intuitive set of operations to create and use vector search indices
2. **Flexibility** - Allow fine-grained control over:
   - Index building parameters (graph degree, window sizes, etc.)
   - Memory allocation strategies (simple, hugepage, custom)
   - Thread pool configuration (native, OpenMP, custom)
   - Vector storage formats (simple, compressed, quantized)
   - Search parameters and filters
   - Logging system
3. **Safety** - Comprehensive error handling with detailed error messages
4. **Portability** - Standard C interface that works across platforms and languages

## Architecture Overview

The API is built around a builder pattern with the following core abstractions:

```
┌─────────────────┐
│ Index Builder   │  Configure index parameters
│  - Algorithm    │
│  - Storage      │
│  - Threadpool   │
└────────┬────────┘
         │ build()
         ↓
┌─────────────────┐
│ Index           │  Perform searches
│  - search()     │  with optional search params
└─────────────────┘
         │
         ├─ Search Params (optional)
         └─ Search Results
```

## Core Components

### 1. Index

The main search structure providing vector similarity search operations.

**Current Capabilities:**
- **TopK Search** - Find the k nearest neighbors for query vectors
- Configurable search parameters (window size, etc.)
- Multiple distance metrics (Euclidean, Cosine, Inner Product)

**Requirements:**
- Built from a non-empty dataset using Index Builder
- Immutable after creation

**Future Extensions:**
- Range search (all neighbors within distance threshold)
- Filtered search (predicate-based filtering)
- Dynamic updates (add/remove vectors)

### 2. Index Builder

Configures and creates index instances using the builder pattern.

**Required Parameters:**
- Algorithm configuration handle
- Vector dimensions  
- Distance metric (Euclidean, Cosine, Inner Product)

**Optional Configuration:**
- Storage format (default: Simple FP32)
- Thread pool kind and size (default: native with hardware concurrency)
- Custom thread pool interface (for advanced use cases)

### 3. Algorithm Configuration

Defines the search algorithm and its parameters.

**Current Support:**
- **Vamana** - Graph-based approximate nearest neighbor search
  - Graph degree (connectivity)
  - Build window size (construction search budget)
  - Default search window size
  - Alpha parameter (pruning threshold)
  - Search history mode

**Future Support:**
- **Flat** - Exhaustive brute-force search
- **IVF** - Inverted file with clustering

### 4. Storage Configuration

Defines how vectors are stored in memory, supporting various compression schemes.

| Storage Type | Configuration Options | Description |
|--------------|----------------------|-------------|
| **Simple** | FP32, FP16, INT8, UINT8, INT4, UINT4 | Uncompressed storage |
| **SQ** | INT8, UINT8 | Scalar quantization |
| **LVQ** | Primary: INT4/UINT4/INT8/UINT8<br>Residual: VOID/INT4/UINT4/INT8/UINT8 | Locally-adaptive vector quantization |
| **LeanVec** | Dimensions<br>Primary: data type<br>Secondary: data type | LeanVec dimensionality reduced storage |

**Example:**
```c
// Simple FP32 storage (default)
svs_storage_h storage = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT32, err);

// LVQ with 8-bit primary and 4-bit residual
svs_storage_h storage = svs_storage_create_lvq(
    SVS_DATA_TYPE_UINT8, SVS_DATA_TYPE_UINT4, err
);

// LeanVec with 128 dimensions
svs_storage_h storage = svs_storage_create_leanvec(
    128, SVS_DATA_TYPE_FLOAT16, SVS_DATA_TYPE_INT8, err
);

// Scalar quantization
svs_storage_h storage = svs_storage_create_sq(SVS_DATA_TYPE_INT8, err);
```

### 5. Thread Pool Configuration

Controls parallelization strategy for index operations.

| Type | Configuration | Use Case |
|------|---------------|----------|
| **Native** | Thread count | Default SVS thread pool (recommended) |
| **OpenMP** | Uses OMP_NUM_THREADS | Integration with OpenMP applications |
| **Single Thread** | No parallelization | Debugging or minimal overhead |
| **Custom** | User-defined interface | Custom scheduling/work-stealing |

**Custom Interface:**
```c
struct svs_threadpool_interface_ops {
    size_t (*size)(void* self);
    void (*parallel_for)(
        void* self,
        void (*func)(void* svs_param, size_t i),
        void* svs_param, // SVS state
        size_t n // Number of tasks
    );
};

struct svs_threadpool_interface {
    struct svs_threadpool_interface_ops ops;
    void* self;  // User-defined state
};
```

### 6. Search Parameters

Configures runtime search behavior (algorithm-specific).

**Vamana Search Parameters:**
- **Search window size** - Controls search accuracy vs. speed tradeoff
  - Larger values: more accurate but slower
  - Smaller values: faster but less accurate
  - Typically 50-200 for good recall

**Usage:**
```c
// Use custom search parameters
svs_search_params_h params = svs_search_params_create_vamana(100, err);
svs_search_results_t results = svs_index_search(
    index, queries, num_queries, k, params, err
);
svs_search_params_free(params);

// Or use defaults from algorithm configuration
svs_search_results_t results = svs_index_search(
    index, queries, num_queries, k, NULL, err
);
```

## Error Handling Strategy

The API uses a dual approach for error reporting: return codes and optional detailed error information.

### Return Values

- Functions returning handles return `NULL` on failure
- Functions returning booleans return `false` on failure
- All functions accept an optional `svs_error_h` parameter for detailed diagnostics

### Detailed Error Information

For comprehensive error diagnostics, create an error handle and pass it to API calls:

```c
// Create error handle
svs_error_h err = svs_error_create();

// Use in API calls (last parameter, can be NULL)
svs_algorithm_h algo = svs_algorithm_create_vamana(
    64,      // graph_degree
    128,     // build_window_size
    128,     // search_window_size
    err      // optional error handle (can be NULL)
);

if (algo == NULL) {
    // Check error status
    if (!svs_error_ok(err)) {
        // Query error details
        svs_error_code_t code = svs_error_get_code(err);
        const char* msg = svs_error_get_message(err);
        fprintf(stderr, "Error [%d]: %s\n", code, msg);
    }
}

// Error handle can be reused across multiple calls
svs_storage_h storage = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT32, err);

// Free error handle when done
svs_error_free(err);
```

### Error Codes

```c
enum svs_error_code {
    SVS_OK = 0,                       // Success
    SVS_ERROR_GENERIC = 1,            // Generic/unspecified error
    SVS_ERROR_INVALID_ARGUMENT = 2,   // Invalid function parameter
    SVS_ERROR_OUT_OF_MEMORY = 3,      // Memory allocation failed
    SVS_ERROR_INDEX_BUILD_FAILED = 4, // Index construction failed
    SVS_ERROR_NOT_IMPLEMENTED = 5,    // Feature not yet available
    SVS_ERROR_UNSUPPORTED_HW = 6,     // Hardware doesn't support required features
    SVS_ERROR_RUNTIME = 7,            // Runtime error during operation
    SVS_ERROR_UNKNOWN = 1000          // Unknown/unexpected error
};
```

### Best Practices

1. **Always check return values** - Test for `NULL` or `false` before using results
2. **Use error handles during development** - Provides detailed diagnostics and error messages
3. **Reuse error handles** - Single handle can be reused across multiple API calls
4. **Free all resources** - Always call appropriate `_free()` functions to prevent leaks
5. **Pass NULL for optional parameters** - Error handle and search params can be `NULL` if not needed
6. **Check `svs_error_ok()`** - Use this helper to check if operation succeeded

## Naming Conventions

Consistent naming improves API discoverability and reduces cognitive load.

### Prefixes

- `svs_` - All functions and types
- `SVS_` - Macros and constants

### Type Suffixes

| Suffix | Meaning | Example |
|--------|---------|----------|
| `_t` | Value type (enum, struct) | `svs_metric_t`, `svs_error_code_t` |
| `_h` | Handle (opaque pointer) | `svs_index_h`, `svs_algorithm_h` |
| `_i` | Interface structure | `svs_allocator_i`, `svs_threadpool_i` |

### Function Naming Pattern

```
svs_<object>[_<specialization>]_<operation>
```

**Examples:**

| Function | Breakdown | Description |
|----------|-----------|-------------|
| `svs_index_search()` | `svs` + `index` + `search` | Generic index search |
| `svs_algo_vamana_set_alpha()` | `svs` + `algo` + `vamana` + `set_alpha` | Set Vamana-specific parameter |
| `svs_storage_create_lvq()` | `svs` + `storage` + `create` + `lvq` | Create LVQ storage configuration |
| `svs_factory_set_threadpool()` | `svs` + `factory` + `set` + `threadpool` | Configure builder thread pool |

### Examples by Category

```c
// Handles (opaque pointers)
typedef struct svs_index* svs_index_h;
typedef struct svs_algorithm* svs_algorithm_h;
typedef struct svs_storage* svs_storage_h;

// Value types
typedef enum svs_metric svs_metric_t;
typedef enum svs_error_code svs_error_code_t;

// Interface structures
typedef struct svs_allocator_interface svs_allocator_i;
typedef struct svs_threadpool_interface svs_threadpool_i;
```

## API Reference

### Type Definitions

```c
// Opaque handles (suffix: _h)
typedef struct svs_error_desc* svs_error_h;
typedef struct svs_index* svs_index_h;
typedef struct svs_index_builder* svs_index_builder_h;
typedef struct svs_algorithm* svs_algorithm_h;
typedef struct svs_storage* svs_storage_h;
typedef struct svs_search_params* svs_search_params_h;

// Fully defined types (suffix: _t)
typedef enum svs_error_code svs_error_code_t;
typedef enum svs_distance_metric svs_distance_metric_t;
typedef enum svs_algorithm_type svs_algorithm_type_t;
typedef enum svs_data_type svs_data_type_t;
typedef enum svs_storage_kind svs_storage_kind_t;
typedef enum svs_threadpool_kind svs_threadpool_kind_t;

// Interface pointers
typedef struct svs_threadpool_interface* svs_threadpool_i;
typedef struct svs_search_results* svs_search_results_t;
```

### Error Handling API

```c
// Create and manage error handles
svs_error_h svs_error_create(void);
void svs_error_free(svs_error_h err);

// Query error information
bool svs_error_ok(svs_error_h err);
svs_error_code_t svs_error_get_code(svs_error_h err);
const char* svs_error_get_message(svs_error_h err);
```

### Algorithm API

Create and configure search algorithms.

```c
// Vamana graph-based approximate nearest neighbor search
svs_algorithm_h svs_algorithm_create_vamana(
    size_t graph_degree,        // Graph connectivity (e.g., 64)
    size_t build_window_size,   // Construction search window (e.g., 128)
    size_t search_window_size,  // Default query search window (e.g., 128)
    svs_error_h out_err        // optional, can be NULL
);

// Cleanup
void svs_algorithm_free(svs_algorithm_h algorithm);

// Get/Set Vamana parameters
bool svs_algorithm_vamana_get_alpha(
    svs_algorithm_h algorithm,
    float* out_alpha,
    svs_error_h out_err
);

bool svs_algorithm_vamana_set_alpha(
    svs_algorithm_h algorithm,
    float alpha,               // Pruning parameter (typically 1.0 - 1.4)
    svs_error_h out_err
);

bool svs_algorithm_vamana_get_graph_degree(
    svs_algorithm_h algorithm,
    size_t* out_graph_degree,
    svs_error_h out_err
);

bool svs_algorithm_vamana_set_graph_degree(
    svs_algorithm_h algorithm,
    size_t graph_degree,
    svs_error_h out_err
);

bool svs_algorithm_vamana_get_build_window_size(
    svs_algorithm_h algorithm,
    size_t* out_build_window_size,
    svs_error_h out_err
);

bool svs_algorithm_vamana_set_build_window_size(
    svs_algorithm_h algorithm,
    size_t build_window_size,
    svs_error_h out_err
);

bool svs_algorithm_vamana_get_use_search_history(
    svs_algorithm_h algorithm,
    bool* out_use_full_search_history,
    svs_error_h out_err
);

bool svs_algorithm_vamana_set_use_search_history(
    svs_algorithm_h algorithm,
    bool use_full_search_history,
    svs_error_h out_err
);
```

### Storage API

Configure vector storage format and compression.

```c
// Simple uncompressed storage
svs_storage_h svs_storage_create_simple(
    svs_data_type_t data_type, // SVS_DATA_TYPE_FLOAT32, FLOAT16, INT8, etc.
    svs_error_h out_err       // optional, can be NULL
);

// Scalar quantization
svs_storage_h svs_storage_create_sq(
    svs_data_type_t data_type, // SVS_DATA_TYPE_INT8, SVS_DATA_TYPE_UINT8
    svs_error_h out_err
);

// Locally-adaptive Vector Quantization (LVQ)
svs_storage_h svs_storage_create_lvq(
    svs_data_type_t primary,   // Primary quantization type
    svs_data_type_t residual,  // Residual type (or SVS_DATA_TYPE_VOID)
    svs_error_h out_err
);

// LeanVec two-level hierarchical storage
svs_storage_h svs_storage_create_leanvec(
    size_t leanvec_dims,       // Primary dimensions (usually much smaller)
    svs_data_type_t primary,   // Primary storage type
    svs_data_type_t secondary, // Secondary/residual storage type
    svs_error_h out_err
);

// Cleanup
void svs_storage_free(svs_storage_h storage);
```


### Search Parameters API

Configure runtime search behavior.

```c
// Create Vamana search parameters
svs_search_params_h svs_search_params_create_vamana(
    size_t search_window_size, // Search window size (e.g., 100)
    svs_error_h out_err       // optional, can be NULL
);

// Cleanup
void svs_search_params_free(svs_search_params_h params);
```

### Index Builder API

Configure and build index instances.

```c
// Create index builder with required parameters
svs_index_builder_h svs_index_builder_create(
    svs_distance_metric_t metric, // Distance metric
    size_t dimension,            // Vector dimensionality
    svs_algorithm_h algorithm,   // Algorithm configuration
    svs_error_h out_err         // optional, can be NULL
);

// Configure storage (optional, default: Simple FP32)
bool svs_index_builder_set_storage(
    svs_index_builder_h builder,
    svs_storage_h storage,       // Storage configuration
    svs_error_h out_err
);

// Configure thread pool (optional, default: native)
bool svs_index_builder_set_threadpool(
    svs_index_builder_h builder,
    svs_threadpool_kind_t kind,  // Thread pool type
    size_t num_threads,          // Number of threads (for native)
    svs_error_h out_err
);

// Configure custom thread pool (advanced)
bool svs_index_builder_set_threadpool_custom(
    svs_index_builder_h builder,
    svs_threadpool_interface_t pool, // Custom thread pool interface
    svs_error_h out_err
);

// Cleanup
void svs_index_builder_free(svs_index_builder_h builder);
```


### Index API

Build and query vector search indices.

```c
// Build index from vector data
svs_index_h svs_index_build(
    svs_index_builder_h builder,
    const float* data,         // Vector data [num_vectors × dimensions]
    size_t num_vectors,
    svs_error_h out_err       // optional, can be NULL
);

// Cleanup
void svs_index_free(svs_index_h index);
```

### Search Results

```c
// Search results structure
struct svs_search_results {
    size_t num_queries;        // Number of query vectors
    size_t* results_per_query; // Number of results per query
    size_t* indices;           // Indices of the nearest neighbors
    float* distances;          // Distances to the nearest neighbors
};

typedef struct svs_search_results* svs_search_results_t;

// Access pattern:
// For query i, neighbor j (where k is the number of neighbors):
//   index = results->indices[i * k + j]
//   distance = results->distances[i * k + j]
```

### Search Operations

```c
// Top-K nearest neighbor search
svs_search_results_t svs_index_search(
    svs_index_h index,
    const float* queries,      // Query vectors [num_queries × dimensions]
    size_t num_queries,
    size_t k,                  // Number of neighbors to return
    svs_search_params_h search_params, // optional, can be NULL for defaults
    svs_error_h out_err       // optional, can be NULL
);

// Cleanup search results
void svs_search_results_free(svs_search_results_t results);
```

## Complete Usage Example

```c
#include "svs/c_api/svs_c.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // 1. Create error handle for diagnostics
    svs_error_h err = svs_error_create();
    
    // 2. Create Vamana algorithm configuration
    svs_algorithm_h algo = svs_algorithm_create_vamana(
        64,   // graph_degree
        128,  // build_window_size
        128,  // default search_window_size
        err
    );
    if (!algo || !svs_error_ok(err)) {
        fprintf(stderr, "Algorithm creation failed: %s\n", 
                svs_error_get_message(err));
        svs_error_free(err);
        return 1;
    }
    
    // 3. Create index builder
    size_t dimensions = 128;
    svs_index_builder_h builder = svs_index_builder_create(
        SVS_DISTANCE_METRIC_EUCLIDEAN,
        dimensions,
        algo,
        err
    );
    
    // 4. Optional: Configure storage (default is FP32)
    svs_storage_h storage = svs_storage_create_simple(
        SVS_DATA_TYPE_FLOAT32, err
    );
    svs_index_builder_set_storage(builder, storage, err);
    
    // 5. Optional: Configure thread pool
    svs_index_builder_set_threadpool(
        builder,
        SVS_THREADPOOL_KIND_NATIVE,
        8,  // num_threads
        err
    );
    
    // 6. Prepare data
    size_t num_vectors = 10000;
    float* data = (float*)malloc(num_vectors * dimensions * sizeof(float));
    // ... fill data with vectors ...
    
    // 7. Build index
    svs_index_h index = svs_index_build(builder, data, num_vectors, err);
    if (!index || !svs_error_ok(err)) {
        fprintf(stderr, "Index build failed: %s\n", 
                svs_error_get_message(err));
        goto cleanup;
    }
    
    // 8. Prepare queries
    size_t num_queries = 10;
    float* queries = (float*)malloc(num_queries * dimensions * sizeof(float));
    // ... fill queries ...
    
    // 9. Perform search with default parameters
    size_t k = 5;
    svs_search_results_t results = svs_index_search(
        index, queries, num_queries, k, NULL, err
    );
    
    // Or with custom search parameters:
    // svs_search_params_h params = svs_search_params_create_vamana(100, err);
    // svs_search_results_t results = svs_index_search(
    //     index, queries, num_queries, k, params, err
    // );
    // svs_search_params_free(params);
    
    // 10. Process results
    if (results && svs_error_ok(err)) {
        for (size_t i = 0; i < results->num_queries; i++) {
            printf("Query %zu:\n", i);
            for (size_t j = 0; j < k; j++) {
                size_t idx = i * k + j;
                printf("  Index: %zu, Distance: %f\n",
                       results->indices[idx], results->distances[idx]);
            }
        }
        svs_search_results_free(results);
    }
    
    // 11. Cleanup
cleanup:
    if (index) svs_index_free(index);
    if (builder) svs_index_builder_free(builder);
    if (storage) svs_storage_free(storage);
    if (algo) svs_algorithm_free(algo);
    svs_error_free(err);
    
    free(data);
    free(queries);
    
    return 0;
}
```

## Next Steps

- See [ERROR_HANDLING.md](c/ERROR_HANDLING.md) for comprehensive error handling guide
- See [examples/c/](../examples/c/) for additional usage examples
- See [bindings/c/samples/](c/samples/) for complete sample applications

```
