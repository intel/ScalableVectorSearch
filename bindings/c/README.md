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

# SVS C API

A C ABI binding for [Scalable Vector Search](../..) that enables integration with
C applications and any language with C FFI support.

The API is built around a small set of opaque handles and a builder pattern:
configure an *algorithm*, optional *storage* and *thread pool*, hand them to an
*index builder*, then use the resulting *index* to run TopK searches (with
optional ID filtering), save/load the index, and — for dynamic indices — add or
delete points at runtime.

For the design rationale, naming conventions, and full API reference see
[docs/C_API_Design.md](docs/C_API_Design.md).

## Public Headers

```c
#include "svs/c_api/svs_c.h"           // Main C API
```

## Building and Consuming

The C API is built as a shared library target `svs_c_api` and installs a CMake
package `svs_c_api` with the imported target `svs::svs_c_api`.

### Build from source

Configure and build from the top of the ScalableVectorSearch tree; the C API is
picked up as a subdirectory under `bindings/c`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target svs_c_api
cmake --install build
```

### Consume from a downstream CMake project

```cmake
find_package(svs_c_api REQUIRED)
target_link_libraries(my_app PRIVATE svs::svs_c_api)
```

The library only exposes a C ABI, so downstream code can be plain C99+ (or
C++20+); the C++20 requirement is a private build-time detail of the library
itself.

## Language Requirements

- C consumers: **C99** or later
- C++ consumers: **C++20** or later

`svs_c.h` enforces this at include time via `#error` if the compiler standard
is below the required version.

## Quick Start

```c
#include "svs/c_api/svs_c.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    // 1. Create error handle for diagnostics
    svs_error_h err = svs_error_create();

    // 2. Create Vamana algorithm configuration
    svs_algorithm_h algo = svs_algorithm_create_vamana(
        64,   // graph_degree
        128,  // build_window_size
        128,  // default search_window_size
        err
    );
    if (!algo) {
        fprintf(stderr, "Algorithm creation failed: %s\n",
                svs_error_get_message(err));
        svs_error_free(err);
        return 1;
    }

    // 3. Create index builder
    size_t dimensions = 128;
    svs_index_builder_h builder = svs_index_builder_create(
        SVS_DISTANCE_METRIC_EUCLIDEAN, dimensions, algo, err
    );

    // 4. Optional: configure storage (default is Simple FP32)
    svs_storage_h storage = svs_storage_create_simple(SVS_DATA_TYPE_FLOAT32, err);
    svs_index_builder_set_storage(builder, storage, err);

    // 5. Optional: configure thread pool
    svs_index_builder_set_threadpool(
        builder, SVS_THREADPOOL_KIND_NATIVE, /*num_threads=*/8, err
    );

    // 6. Prepare data
    size_t num_vectors = 10000;
    float* data = (float*)malloc(num_vectors * dimensions * sizeof(float));
    // ... fill data with vectors ...

    // 7. Build index
    svs_index_h index = svs_index_build(builder, data, num_vectors, err);
    if (!index) {
        fprintf(stderr, "Index build failed: %s\n", svs_error_get_message(err));
        goto cleanup;
    }

    // 8. Prepare queries
    size_t num_queries = 10;
    float* queries = (float*)malloc(num_queries * dimensions * sizeof(float));
    // ... fill queries ...

    // 9. Perform search (library-owned result buffers; reused across calls)
    size_t k = 5;
    svs_search_results_t results = SVS_INIT_SEARCH_RESULTS();

    if (!svs_index_search_topk(
            index, queries, num_queries, k, &results,
            /*search_params=*/NULL,
            /*id_filter=*/NULL,
            err
        )) {
        fprintf(stderr, "Search failed: %s\n", svs_error_get_message(err));
        goto cleanup;
    }

    // 10. Process results
    for (size_t q = 0; q < results.num_queries; ++q) {
        const size_t* ids;
        const float*  dists;
        size_t count;
        svs_search_results_row(&results, q, &ids, &dists, &count);
        printf("Query %zu:\n", q);
        for (size_t j = 0; j < count; ++j) {
            printf("  Index: %zu, Distance: %f\n", ids[j], dists[j]);
        }
    }

    // 11. Optional: introspect memory usage
    svs_memory_breakdown_t breakdown = SVS_INIT_MEMORY_BREAKDOWN();
    if (svs_index_get_memory_breakdown(index, &breakdown, err)) {
        printf("Memory: graph=%zu data=%zu metadata=%zu bytes\n",
               breakdown.graph_bytes,
               breakdown.data_bytes,
               breakdown.metadata_bytes);
    }

    // 12. Optional: persist the index for later reuse
    svs_index_save(index, "/tmp/my_index", err);

cleanup:
    svs_search_results_free(&results);
    if (index)   svs_index_free(index);
    if (builder) svs_index_builder_free(builder);
    if (storage) svs_storage_free(storage);
    if (algo)    svs_algorithm_free(algo);
    svs_error_free(err);

    free(data);
    free(queries);

    return 0;
}
```

## Samples

Runnable sample applications live in [samples/](samples/):

- [`simple.c`](samples/simple.c) – minimal static index build + search with a
  custom thread pool
- [`dynamic.c`](samples/dynamic.c) – dynamic index with add / delete /
  consolidate
- [`save_load.c`](samples/save_load.c) – persisting and reloading indices from
  disk

Additional integration examples: [`examples/c/`](../../examples/c/).

## Further Reading

- [docs/C_API_Design.md](docs/C_API_Design.md) – design goals, architecture, core
  components, error-handling strategy, naming conventions, and complete API
  reference.
