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
- [API Overview](#api-overview)
  - [Headers](#headers)
  - [Types](#types)
  - [Function groups](#function-groups)
  - [Conventions](#conventions)
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
svs_<object>[_<specialization>]_<operation>[_<specialization>]
```

The `<specialization>` qualifier sits next to the noun it modifies, so exactly
one of the two optional slots is used per name:

- **Leading form** — `svs_<object>_<specialization>_<operation>` when the
  operation acts on an object that is *already* of that specialization
  (e.g. `svs_algorithm_vamana_set_alpha` sets `alpha` on a Vamana algorithm;
  `svs_index_dynamic_add_points` adds points to a dynamic index).
- **Trailing form** — `svs_<object>_<operation>_<specialization>` when the
  operation *produces* or *targets* that specialization
  (e.g. `svs_algorithm_create_vamana` creates a Vamana algorithm;
  `svs_index_build_dynamic` builds a dynamic index).

**Examples:**

| Function | Breakdown | Description |
|----------|-----------|-------------|
| `svs_index_search_topk()` | `svs` + `index` + `search_topk` | TopK index search (with optional ID filter) |
| `svs_algorithm_create_vamana()` | `svs` + `algorithm` + `create` + `vamana` | Create a Vamana algorithm configuration |
| `svs_algorithm_vamana_set_alpha()` | `svs` + `algorithm` + `vamana` + `set_alpha` | Set `alpha` on a Vamana algorithm |
| `svs_storage_create_lvq()` | `svs` + `storage` + `create` + `lvq` | Create LVQ storage configuration |
| `svs_index_build_dynamic()` | `svs` + `index` + `build` + `dynamic` | Build a dynamic index |
| `svs_index_dynamic_add_points()` | `svs` + `index` + `dynamic` + `add_points` | Add points to a dynamic index |
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

## API Overview

A concise map of the public surface. See [svs/c/svs_c.h](../include/svs/c/svs_c.h)
for full signatures, parameters, and Doxygen documentation.

### Headers

- `svs/c/svs_c.h` — main C API (all functions and types below)
- `svs/c/svs_c_config.h` — configuration macros (`SVS_API`, `SVS_DEPRECATED`)
- `svs/c/svs_c_version.h` — generated; `SVS_C_API_VERSION[_MAJOR|_MINOR|_PATCH|_STRING]`

### Types

- **Opaque handles** (`_h`): `svs_error_h`, `svs_index_h`, `svs_index_builder_h`,
  `svs_algorithm_h`, `svs_storage_h`, `svs_search_params_h`
- **Enums** (`_t`): `svs_error_code_t`, `svs_distance_metric_t`,
  `svs_algorithm_type_t`, `svs_data_type_t`, `svs_storage_kind_t`,
  `svs_threadpool_kind_t`
- **Custom interfaces**: `svs_threadpool_i` and `svs_id_filter_i` (versioned
  ops-table + `self` pointer; build with `SVS_INIT_*_OPS()` / `SVS_MAKE_INTERFACE()`)
- **Value structs**: `svs_search_results_t` (CSR result buffer),
  `svs_memory_breakdown_t`

### Function groups

| Group | Key functions |
|-------|---------------|
| **Version** | `svs_get_version`, `svs_get_version_string` |
| **Error** | `svs_error_create` / `_free` / `_set` / `_ok` / `_get_code` / `_get_message` |
| **Algorithm** | `svs_algorithm_create_vamana`, `svs_algorithm_get_type`, `svs_algorithm_vamana_{get,set}_{alpha,graph_degree,build_window_size,use_search_history}`, `svs_algorithm_free` |
| **Storage** | `svs_storage_create_{simple,sq,lvq,leanvec}`, `svs_storage_get_kind`, `svs_storage_free` |
| **Search params** | `svs_search_params_create_vamana`, `svs_search_params_free` |
| **Builder** | `svs_index_builder_create`, `svs_index_builder_set_{storage,threadpool,threadpool_custom}`, `svs_index_builder_free` |
| **Index lifecycle** | `svs_index_build`, `svs_index_build_dynamic`, `svs_index_load`, `svs_index_load_dynamic`, `svs_index_save`, `svs_index_free` |
| **Dynamic ops** | `svs_index_dynamic_{add_points,delete_points,has_id,consolidate,compact}` |
| **Introspection** | `svs_index_get_num_threads` / `set_num_threads`, `svs_index_get_distance`, `svs_index_reconstruct`, `svs_index_get_memory_usage`, `svs_index_get_memory_breakdown` |
| **Search** | `svs_index_search_topk` (+ deprecated `svs_index_search`), `svs_search_results_free` |

### Conventions

- Every fallible call takes a trailing optional `svs_error_h out_err` (may be `NULL`).
- Constructors return an opaque handle or `NULL` on failure; other calls return
  `bool`. Out-values are written through `out_*` pointer parameters.
- LVQ/LeanVec require the compression backend and specific x86 ISA support; when
  unavailable, `svs_storage_create_{lvq,leanvec}` return `NULL` with
  `SVS_ERROR_NOT_IMPLEMENTED` or `SVS_ERROR_UNSUPPORTED_HW`.
- `svs_index_search_topk` writes results into a caller-provided
  `svs_search_results_t` (init with `SVS_INIT_SEARCH_RESULTS()` for library-owned
  buffers, or `SVS_INIT_SEARCH_RESULTS_WITH_BUFFERS()` for caller-owned). Rows use
  a CSR layout (`offsets`/`indices`/`distances`); read them with
  `svs_search_results_row()` and release with `svs_search_results_free()`.

## Next Steps

- See the top-level [../README.md](../README.md) for a quick start, build/consume
  instructions, and a complete end-to-end usage example.
- See [../samples/](../samples/) for runnable sample applications:
  - `simple.c` – minimal static index build + search with a custom thread pool
  - `dynamic.c` – dynamic index with add / delete / consolidate
  - `save_load.c` – persisting and reloading indices from disk
- See [examples/c/](../../../examples/c/) for additional usage examples
