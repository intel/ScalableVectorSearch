<!--
  ~ Copyright 2026 Intel Corporation
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~     http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
-->

# SVS C API Design

> Looking for build/consume instructions or a quick-start example? See
> [../README.md](../README.md). This document focuses on the design rationale,
> conventions, and the full API reference.

## Table of Contents

- [Overview](#overview)
  - [Design Goals](#design-goals)
- [Architecture Overview](#architecture-overview)
- [Error Handling Strategy](#error-handling-strategy)
  - [Return Values](#return-values)
  - [Detailed Error Information](#detailed-error-information)
  - [Error Codes](#error-codes)
  - [Best Practices](#best-practices)
- [Naming Conventions](#naming-conventions)
  - [Prefixes](#prefixes)
  - [Type Suffixes](#type-suffixes)
  - [Function Naming Pattern](#function-naming-pattern)
  - [Examples by Category](#examples-by-category)
- [Core Components](#core-components)
  - [1. Index](#1-index)
  - [2. Index Builder](#2-index-builder)
  - [3. Algorithm Configuration](#3-algorithm-configuration)
  - [4. Storage Configuration](#4-storage-configuration)
  - [5. Thread Pool Configuration](#5-thread-pool-configuration)
  - [6. Search Parameters](#6-search-parameters)
  - [7. ID Filter (optional)](#7-id-filter-optional)
- [API Reference](#api-reference)
  - [Public Headers](#public-headers)
  - [Type Definitions](#type-definitions)
  - [Version Information](#version-information)
  - [Error Handling API](#error-handling-api)
  - [Algorithm API](#algorithm-api)
  - [Storage API](#storage-api)
  - [Search Parameters API](#search-parameters-api)
  - [Index Builder API](#index-builder-api)
  - [Index API](#index-api)
  - [Dynamic Index Operations](#dynamic-index-operations)
  - [Index Introspection](#index-introspection)
  - [Search Results](#search-results)
  - [Search Operations](#search-operations)
- [Next Steps](#next-steps)

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
    SVS_ERROR_NOT_IMPLEMENTED = 5,    // Feature not yet available
    SVS_ERROR_UNSUPPORTED_HW = 6,     // Hardware doesn't support required features
    SVS_ERROR_RUNTIME = 7,            // Runtime error during operation
    SVS_ERROR_INVALID_OPERATION = 8,  // Operation not valid in the current state
    SVS_ERROR_UNKNOWN = 1000          // Unknown/unexpected error
};
```

User-provided callbacks (custom thread pool, ID filter) can report failures back to
the library by calling `svs_error_set()` on the `out_err` handle they receive.

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
| `_t` | Value type (enum, struct) | `svs_distance_metric_t`, `svs_error_code_t` |
| `_h` | Handle (opaque pointer) | `svs_index_h`, `svs_algorithm_h` |
| `_i` | Interface pointer type | `svs_threadpool_i`, `svs_id_filter_i` |

### Function Naming Pattern

```
svs_<object>[_<specialization>]_<operation>
```

**Examples:**

| Function | Breakdown | Description |
|----------|-----------|-------------|
| `svs_index_search_topk()` | `svs` + `index` + `search_topk` | TopK index search (with optional ID filter) |
| `svs_algorithm_vamana_set_alpha()` | `svs` + `algorithm` + `vamana` + `set_alpha` | Set Vamana-specific parameter |
| `svs_storage_create_lvq()` | `svs` + `storage` + `create` + `lvq` | Create LVQ storage configuration |
| `svs_index_builder_set_threadpool()` | `svs` + `index_builder` + `set_threadpool` | Configure builder thread pool |

### Examples by Category

```c
// Handles (opaque pointers)
typedef struct svs_index* svs_index_h;
typedef struct svs_algorithm* svs_algorithm_h;
typedef struct svs_storage* svs_storage_h;

// Value types
typedef enum svs_distance_metric svs_distance_metric_t;
typedef enum svs_error_code svs_error_code_t;

// Interface pointer types
typedef struct svs_threadpool_interface* svs_threadpool_i;
typedef struct svs_id_filter_interface*  svs_id_filter_i;
```

## Core Components

### 1. Index

The main search structure providing vector similarity search operations.

**Current Capabilities:**
- **TopK Search** - Find the k nearest neighbors for query vectors
- **Filtered TopK Search** - Optional caller-provided ID filter applied during topk search
- Configurable search parameters (window size, etc.)
- Multiple distance metrics (Euclidean, Cosine, Dot Product)
- **Persistence** - Save an index to disk and reload it later
- **Introspection** - Query total memory usage and per-component breakdown
- **Dynamic Updates** *(dynamic index only)* - Add / delete points, consolidate, compact

**Requirements:**
- Built from a non-empty dataset using Index Builder, or loaded from disk
- A static index is immutable after creation; a dynamic index additionally supports
  add/delete/consolidate/compact operations

**Future Extensions:**
- Range search (all neighbors within distance threshold)
- Additional algorithms (Flat, IVF)

### 2. Index Builder

Configures and creates index instances using the builder pattern.

**Required Parameters:**
- Algorithm configuration handle
- Vector dimensions  
- Distance metric (Euclidean, Cosine, Dot Product)

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

The custom thread pool interface is split into a versioned operations table and an
instance structure that carries an opaque `self` pointer. Ops tables must always be
initialised through the provided `SVS_INIT_THREADPOOL_OPS()` macro so that the
`version` and `struct_size` fields are populated correctly for forward compatibility.

```c
struct svs_threadpool_interface_ops {
    uint32_t version;      // Set by SVS_INIT_THREADPOOL_OPS
    size_t struct_size;    // Set by SVS_INIT_THREADPOOL_OPS
    size_t (*size)(void* self);
    bool (*parallel_for)(
        void* self,
        void (*func)(void* svs_param, size_t i),
        void* svs_param,       // SVS state
        size_t n,              // Number of tasks
        svs_error_h out_err    // Set via svs_error_set() to abort execution
    );
};

struct svs_threadpool_interface {
    struct svs_threadpool_interface_ops* ops;
    void* self;                // User-defined state
};

// Handy typedef used by the API surface
typedef struct svs_threadpool_interface* svs_threadpool_i;

// Initialisation macros
static svs_threadpool_ops_t my_ops =
    SVS_INIT_THREADPOOL_OPS(my_size_func, my_parallel_for_func);
static svs_threadpool_t my_pool = SVS_MAKE_INTERFACE(NULL, my_ops);
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
svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();

// Use custom search parameters
svs_search_params_h params = svs_search_params_create_vamana(100, err);
svs_index_search_topk(
    index, queries, num_queries, k, &results, params, /*id_filter=*/NULL, err
);
svs_search_params_free(params);

// Or use defaults from algorithm configuration (search_params = NULL)
svs_index_search_topk(
    index, queries, num_queries, k, &results, NULL, NULL, err
);

svs_search_results_free(&results);
```

### 7. ID Filter (optional)

A caller-supplied ID filter can be passed to `svs_index_search_topk()` to restrict
results to a subset of vector IDs. Like the thread pool, the filter is a versioned
ops table plus an opaque `self` pointer.

```c
struct svs_id_filter_interface_ops {
    uint32_t version;
    size_t struct_size;
    bool (*is_member)(void* self, size_t id);
    float (*filter_rate)(void* self); // Optional selectivity hint, or NULL / 0.0
};

struct svs_id_filter_interface {
    struct svs_id_filter_interface_ops* ops;
    void* self;
};
typedef struct svs_id_filter_interface* svs_id_filter_i;

static svs_id_filter_ops_t my_filter_ops =
    SVS_INIT_ID_FILTER_OPS(my_is_member, my_filter_rate);
static svs_id_filter_t my_filter = SVS_MAKE_INTERFACE(user_state, my_filter_ops);
```

Providing a non-zero `filter_rate` lets the search account for the expected
selectivity; if the observed hit rate ends up lower than the reported estimate the
function returns an empty result set for that query.

## API Reference

### Public Headers

```c
#include "svs/c/svs_c.h"           // Main C API
#include "svs/c/svs_c_config.h"    // API configuration macroses
#include "svs/c/svs_c_version.h"   // SVS_C_API_VERSION[_MAJOR|_MINOR|_PATCH|_STRING]
```

`svs_c_version.h` is generated by CMake from `svs_c_version.h.in` and defines both
the encoded `SVS_C_API_VERSION` integer (used by the `SVS_INIT_*` macros) and
human-readable component macros.

### Type Definitions

```c
// Opaque handles (suffix: _h)
typedef struct svs_error_desc*     svs_error_h;
typedef struct svs_index*          svs_index_h;
typedef struct svs_index_builder*  svs_index_builder_h;
typedef struct svs_algorithm*      svs_algorithm_h;
typedef struct svs_storage*        svs_storage_h;
typedef struct svs_search_params*  svs_search_params_h;

// Fully defined enum types (suffix: _t)
typedef enum svs_error_code        svs_error_code_t;
typedef enum svs_distance_metric   svs_distance_metric_t;
typedef enum svs_algorithm_type    svs_algorithm_type_t;
typedef enum svs_data_type         svs_data_type_t;
typedef enum svs_storage_kind      svs_storage_kind_t;
typedef enum svs_threadpool_kind   svs_threadpool_kind_t;

// Custom-interface value + pointer types
typedef struct svs_threadpool_interface_ops svs_threadpool_ops_t;
typedef struct svs_threadpool_interface     svs_threadpool_t;
typedef struct svs_threadpool_interface*    svs_threadpool_i;

typedef struct svs_id_filter_interface_ops  svs_id_filter_ops_t;
typedef struct svs_id_filter_interface      svs_id_filter_t;
typedef struct svs_id_filter_interface*     svs_id_filter_i;

// Result / introspection value types (defined in header)
typedef struct svs_search_results    svs_search_results_t;
typedef struct svs_memory_breakdown  svs_memory_breakdown_t;

// Distance metric enum values
//   SVS_DISTANCE_METRIC_EUCLIDEAN, SVS_DISTANCE_METRIC_COSINE,
//   SVS_DISTANCE_METRIC_DOT_PRODUCT
```

### Version Information

```c
uint32_t svs_get_version(void);          // (major<<16)|(minor<<8)|patch
const char* svs_get_version_string(void); // "major.minor.patch"
```

### Error Handling API

```c
svs_error_h svs_error_create(void);
void        svs_error_free(svs_error_h err);

// Set an error code + message. Intended for user callbacks (custom threadpool,
// ID filter) so they can propagate failures back to the library.
bool svs_error_set(svs_error_h err, svs_error_code_t code, const char* message);

bool             svs_error_ok(svs_error_h err);
svs_error_code_t svs_error_get_code(svs_error_h err);
const char*      svs_error_get_message(svs_error_h err);
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

// Introspection
bool svs_algorithm_get_type(
    svs_algorithm_h algorithm,
    svs_algorithm_type_t* out_type,
    svs_error_h out_err
);

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

// Introspection
bool svs_storage_get_kind(
    svs_storage_h storage,
    svs_storage_kind_t* out_kind,
    svs_error_h out_err
);

// Cleanup
void svs_storage_free(svs_storage_h storage);
```

> LVQ and LeanVec require a build that includes the compression backend and, in
> some cases, specific x86 ISA support. When they are unavailable the create
> functions return `NULL` and populate `out_err` with `SVS_ERROR_NOT_IMPLEMENTED`
> or `SVS_ERROR_UNSUPPORTED_HW`.

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
    svs_storage_h storage,
    svs_error_h out_err
);

// Configure thread pool (optional, default: native)
bool svs_index_builder_set_threadpool(
    svs_index_builder_h builder,
    svs_threadpool_kind_t kind,
    size_t num_threads,
    svs_error_h out_err
);

// Configure custom thread pool (advanced)
bool svs_index_builder_set_threadpool_custom(
    svs_index_builder_h builder,
    svs_threadpool_i pool,       // Pointer to svs_threadpool_interface
    svs_error_h out_err
);

// Cleanup
void svs_index_builder_free(svs_index_builder_h builder);
```

### Index API

Build, persist and manage vector search indices.

```c
// Build a static index from vector data
svs_index_h svs_index_build(
    svs_index_builder_h builder,
    const float* data,         // [num_vectors * dimensions]
    size_t num_vectors,
    svs_error_h out_err       // optional, can be NULL
);

// Build a dynamic index. Passing ids = NULL auto-generates IDs 0..num_vectors-1;
// blocksize_bytes = 0 selects an implementation-defined default.
svs_index_h svs_index_build_dynamic(
    svs_index_builder_h builder,
    const float* data,
    const size_t* ids /*=NULL*/,
    size_t num_vectors,
    size_t blocksize_bytes /*=0*/,
    svs_error_h out_err /*=NULL*/
);

// Load a previously saved static index. The builder supplies configuration
// (storage, threadpool, ...).
svs_index_h svs_index_load(
    svs_index_builder_h builder,
    const char* directory,
    svs_error_h out_err /*=NULL*/
);

// Load a previously saved dynamic index.
svs_index_h svs_index_load_dynamic(
    svs_index_builder_h builder,
    const char* directory,
    size_t blocksize_bytes /*=0*/,
    svs_error_h out_err /*=NULL*/
);

// Persist an index (static or dynamic) to disk.
bool svs_index_save(svs_index_h index, const char* directory, svs_error_h out_err);

// Cleanup
void svs_index_free(svs_index_h index);
```

### Dynamic Index Operations

Available only for indices created via `svs_index_build_dynamic()` /
`svs_index_load_dynamic()`.

```c
bool svs_index_dynamic_add_points(
    svs_index_h index,
    const float* new_points,
    const size_t* ids,
    size_t num_vectors,
    size_t* out_added_count /*=NULL*/,
    svs_error_h out_err /*=NULL*/
);

bool svs_index_dynamic_delete_points(
    svs_index_h index,
    const size_t* ids,
    size_t num_ids,
    size_t* out_deleted_count /*=NULL*/,
    svs_error_h out_err /*=NULL*/
);

bool svs_index_dynamic_has_id(
    svs_index_h index, size_t id, bool* out_has_id, svs_error_h out_err /*=NULL*/
);

// Reclaim space and consolidate deletes.
bool svs_index_dynamic_consolidate(svs_index_h index, svs_error_h out_err /*=NULL*/);
bool svs_index_dynamic_compact(
    svs_index_h index, size_t batchsize /*=0*/, svs_error_h out_err /*=NULL*/
);
```

### Index Introspection

```c
bool svs_index_get_num_threads(svs_index_h index, size_t* out_num_threads,
                               svs_error_h out_err);

// Only supported for indices built with SVS_THREADPOOL_KIND_NATIVE / _OMP.
// Returns false with SVS_ERROR_INVALID_OPERATION for _CUSTOM / _SINGLE_THREAD.
bool svs_index_set_num_threads(svs_index_h index, size_t num_threads,
                               svs_error_h out_err);

// Compute distance from a stored vector (by id) to a query.
bool svs_index_get_distance(svs_index_h index, size_t id, const float* query,
                            float* out_distance, svs_error_h out_err);

// Reconstruct stored vectors for a set of ids into a caller-provided buffer of
// size num_ids * data_dim.
bool svs_index_reconstruct(svs_index_h index, const size_t* ids, size_t num_ids,
                           float* out_data, size_t data_dim, svs_error_h out_err);

// Total memory (graph + data + metadata) used by the index.
bool svs_index_get_memory_usage(svs_index_h index, size_t* out_bytes,
                                svs_error_h out_err);

// Per-component memory breakdown.
struct svs_memory_breakdown {
    uint32_t version;
    size_t   struct_size;
    size_t   graph_bytes;
    size_t   data_bytes;
    size_t   metadata_bytes;
};

#define SVS_INIT_MEMORY_BREAKDOWN() /* zero-inits + version + struct_size */

bool svs_index_get_memory_breakdown(
    svs_index_h index,
    svs_memory_breakdown_t* out_breakdown,
    svs_error_h out_err
);
```

### Search Results

Search results are returned in a caller-provided CSR-layout struct. The struct is
versioned and forward-compatible: the library only writes fields covered by the
caller-supplied `struct_size`, so binaries built against an older header keep
working against newer libraries.

```c
struct svs_search_results {
    uint32_t version;
    size_t   struct_size;

    size_t   num_queries;      // Number of populated rows
    size_t   total_results;    // == offsets[num_queries]
    size_t*  offsets;          // Length num_queries + 1 (row starts)
    size_t*  indices;          // Length total_results
    float*   distances;        // Length total_results

    size_t   offsets_capacity; // Allocation size of offsets
    size_t   results_capacity; // Allocation size of indices / distances
    bool     owns_buffers;     // If true, the library allocated / will free them
};

// Row-oriented access (O(1)):
//   For query q neighbor j:
//     size_t begin = results.offsets[q];
//     size_t idx      = results.indices[begin + j];
//     float  distance = results.distances[begin + j];
//
// For fixed top-k searches offsets[q] == q * k, so results.indices[q*k + j] is
// equivalent.

// Convenience row accessor
static inline void svs_search_results_row(
    const struct svs_search_results* results,
    size_t q,
    const size_t** out_ids,
    const float**  out_distances,
    size_t*        out_count
);

// Initialisation macros
#define SVS_INIT_SEARCH_RESULTS()  /* library-owned buffers, allocated on demand */
#define SVS_INIT_SEARCH_RESULTS_WITH_BUFFERS(                                    \
    p_offsets, p_indices, p_distances, p_offsets_cap, p_results_cap)             \
    /* caller-owned buffers; call fails without reallocating if capacity too low */

// Release library-owned buffers and reset the struct. Safe on NULL,
// zero-initialised, and caller-owned-buffer instances (in which case only the
// descriptor is reset). Safe to call multiple times.
void svs_search_results_free(svs_search_results_t* results);
```

**Buffer reuse:** passing the same `svs_search_results_t` object to consecutive
searches lets the library reuse existing library-owned buffers whenever capacity
is sufficient; steady-state batches of the same shape allocate only on the first
call.

### Search Operations

```c
// TopK search with optional ID filter and search parameters.
// out_results should be initialised with SVS_INIT_SEARCH_RESULTS() (or
// SVS_INIT_SEARCH_RESULTS_WITH_BUFFERS()) before the first call.
bool svs_index_search_topk(
    svs_index_h index,
    const float* queries,       // [num_queries * dimensions]
    size_t num_queries,
    size_t k,
    svs_search_results_t* out_results,
    svs_search_params_h search_params /*=NULL*/,
    svs_id_filter_i id_filter         /*=NULL*/,
    svs_error_h out_err               /*=NULL*/
);

// Deprecated shim retained for source compatibility; equivalent to
// svs_index_search_topk() with id_filter = NULL.
SVS_DEPRECATED("Use svs_index_search_topk() instead")
static inline bool svs_index_search(
    svs_index_h index,
    const float* queries,
    size_t num_queries,
    size_t k,
    svs_search_results_t* out_results,
    svs_search_params_h search_params /*=NULL*/,
    svs_error_h out_err               /*=NULL*/
);
```

## Next Steps

- See the top-level [../README.md](../README.md) for a quick start, build/consume
  instructions, and a complete end-to-end usage example.
- See [../samples/](../samples/) for runnable sample applications:
  - `simple.c` – minimal static index build + search with a custom thread pool
  - `dynamic.c` – dynamic index with add / delete / consolidate
  - `save_load.c` – persisting and reloading indices from disk
- See [examples/c/](../../../examples/c/) for additional usage examples
