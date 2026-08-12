/*
 * Copyright 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "svs/c/svs_c_config.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// SVS C API requires a C99 or later compiler, or a C++20 or later compiler. If the
// compiler does not meet these requirements, a compilation error will be generated with a
// clear message indicating the required standard version.
#if defined(__cplusplus)
#if __cplusplus < 202002L
#error \
    "svs_c.h requires C++20 or later (designated initializers in SVS_INIT_* / SVS_MAKE_INTERFACE)."
#endif
#elif defined(__STDC_VERSION__)
#if __STDC_VERSION__ < 199901L
#error \
    "svs_c.h requires C99 or later (designated initializers in SVS_INIT_* / SVS_MAKE_INTERFACE)."
#endif
#else
#error "svs_c.h requires C99 or later, or C++20 or later."
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum svs_error_code {
    SVS_OK = 0,
    SVS_ERROR_GENERIC = 1,
    SVS_ERROR_INVALID_ARGUMENT = 2,
    SVS_ERROR_OUT_OF_MEMORY = 3,
    SVS_ERROR_NOT_IMPLEMENTED = 5,
    SVS_ERROR_UNSUPPORTED_HW = 6,
    SVS_ERROR_RUNTIME = 7,
    SVS_ERROR_INVALID_OPERATION = 8,
    SVS_ERROR_UNKNOWN = 1000
};

typedef struct svs_error_desc* svs_error_h;

enum svs_distance_metric {
    SVS_DISTANCE_METRIC_EUCLIDEAN = 0,
    SVS_DISTANCE_METRIC_COSINE = 1,
    SVS_DISTANCE_METRIC_DOT_PRODUCT = 2
};

enum svs_algorithm_type {
    SVS_ALGORITHM_TYPE_VAMANA = 0,
    SVS_ALGORITHM_TYPE_FLAT = 1,
    SVS_ALGORITHM_TYPE_IVF = 2,
};

enum svs_data_type {
    SVS_DATA_TYPE_NONE = 0,
    SVS_DATA_TYPE_VOID = SVS_DATA_TYPE_NONE,
    SVS_DATA_TYPE_FLOAT64,
    SVS_DATA_TYPE_FLOAT32,
    SVS_DATA_TYPE_FLOAT16,
    SVS_DATA_TYPE_BFLOAT16,
    SVS_DATA_TYPE_INT64,
    SVS_DATA_TYPE_UINT64,
    SVS_DATA_TYPE_INT32,
    SVS_DATA_TYPE_UINT32,
    SVS_DATA_TYPE_INT16,
    SVS_DATA_TYPE_UINT16,
    SVS_DATA_TYPE_INT8,
    SVS_DATA_TYPE_UINT8,
    SVS_DATA_TYPE_INT4,
    SVS_DATA_TYPE_UINT4
};

enum svs_storage_kind {
    SVS_STORAGE_KIND_SIMPLE = 0,
    SVS_STORAGE_KIND_LEANVEC = 1,
    SVS_STORAGE_KIND_LVQ = 2,
    SVS_STORAGE_KIND_SQ = 3
};

enum svs_threadpool_kind {
    SVS_THREADPOOL_KIND_NATIVE = 0,
    SVS_THREADPOOL_KIND_OMP = 1,
    SVS_THREADPOOL_KIND_SINGLE_THREAD = 2,
    SVS_THREADPOOL_KIND_CUSTOM = 3
};

/// @brief Operations table for a custom thread pool interface
/// @remarks The user must ensure that the thread pool implementation is thread-safe and
/// that the provided function pointers remain valid for the lifetime of the thread pool
/// interface.
///
/// @var svs_threadpool_interface_ops::version
///   Version of the thread pool interface.
/// @var svs_threadpool_interface_ops::struct_size
///   Size of the structure, used for versioning and compatibility checks.
/// @var svs_threadpool_interface_ops::size
///   Function pointer to retrieve the number of threads in the thread pool.
///   @param self Pointer to the thread pool instance.
///   @return Number of threads in the thread pool.
///
/// @var svs_threadpool_interface_ops::parallel_for
///   Function pointer to execute a function in parallel across the thread pool.
///   The user is responsible for ensuring that @p func and @p svs_param remain valid
///   for the duration of the parallel execution. The implementation must call @p func
///   exactly once for each index in [0, n). If @p func signals failure via @p out_err,
///   the parallel execution should be aborted.
///   @param self Pointer to the thread pool instance.
///   @param func Function pointer to execute per iteration. Takes a user data pointer
///     (@p svs_param) and a zero-based iteration index (@p i).
///   @param svs_param Pointer to user-defined data passed to each @p func invocation.
///   @param n Number of iterations to execute in parallel.
///   @param out_err Handle to capture any error that occurs during execution. User code may
///   call svs_error_set() to set the error code and message if an error occurs.
///   @return @c true if all iterations completed successfully, @c false otherwise.
// clang-format off
struct svs_threadpool_interface_ops {
    uint32_t version;
    size_t struct_size;
    size_t (*size)(void* self);
    bool (*parallel_for)(
        void* self,
        void (*func)(void* svs_param, size_t i),
        void* svs_param,
        size_t n,
        svs_error_h out_err
    );
};
// clang-format on

/// @brief Macro to create a user-defined thread pool interface operations structure
/// @param size_func Function pointer to retrieve the number of threads in the thread pool
/// @param parallel_for_func Function pointer to execute a function in parallel across the
/// thread pool
#define SVS_INIT_THREADPOOL_OPS(size_func, parallel_for_func)                           \
    {                                                                                   \
        .version = SVS_C_API_VERSION,                                                   \
        .struct_size = sizeof(struct svs_threadpool_interface_ops), .size = &size_func, \
        .parallel_for = &parallel_for_func                                              \
    }

/// @brief Structure representing a custom thread pool interface
/// @var svs_threadpool_interface_ops::ops
///   Function pointers for the thread pool operations.
/// @var svs_threadpool_interface_ops::self
///   Pointer to the user-defined thread pool instance. This pointer is passed to the
///   function pointers in @p ops when they are called.
struct svs_threadpool_interface {
    struct svs_threadpool_interface_ops* ops;
    void* self;
};

/// @brief Operations table for a custom ID filter interface
/// @remarks The user must ensure that the ID filter implementation is thread-safe and
/// that the provided function pointers remain valid for the lifetime of the ID filter
/// interface.
/// @var svs_id_filter_interface_ops::version
///   Version of the ID filter interface.
/// @var svs_id_filter_interface_ops::struct_size
///   Size of the structure, used for versioning and compatibility checks.
/// @var svs_id_filter_interface_ops::is_member
///   Function pointer to check if a given ID is a member of the filter.
/// @var svs_id_filter_interface_ops::filter_rate
///   Optional function pointer to get the estimated selectivity of the filter, i.e., the
///   fraction of IDs that are expected to pass the filter. A value of 0.01 indicates that
///   1% of IDs are expected to pass, while a value of 1.0 indicates that all IDs are
///   expected to pass. If the filter does not provide an estimate, it should be set to NULL
///   or return 0.0.
struct svs_id_filter_interface_ops {
    uint32_t version;
    size_t struct_size;
    bool (*is_member)(void* self, size_t id);
    float (*filter_rate)(void* self);
};

/// @brief Macro to create a user-defined ID filter interface operations structure
/// @param is_member_func Function pointer to check if a given ID is a member of the filter
/// @param filter_rate_func Optional function pointer to get the estimated selectivity of
/// the filter
#define SVS_INIT_ID_FILTER_OPS(is_member_func, filter_rate_func)       \
    {                                                                  \
        .version = SVS_C_API_VERSION,                                  \
        .struct_size = sizeof(struct svs_id_filter_interface_ops),     \
        .is_member = &is_member_func, .filter_rate = &filter_rate_func \
    }

/// @brief Structure representing a custom ID filter interface
/// @var svs_id_filter_interface::ops
///   Function pointers for the ID filter operations.
/// @var svs_id_filter_interface::self
///  Pointer to the user-defined ID filter instance. This pointer is passed to the
///  function pointers in @p ops when they are called.
struct svs_id_filter_interface {
    struct svs_id_filter_interface_ops* ops;
    void* self;
};

/// @brief Macro to create a user-defined interface implementation structure
/// @param user_ptr Pointer to the user-defined object
/// @param vtable Function pointers for the interface operations
/// @return A fully initialized interface implementation structure
#define SVS_MAKE_INTERFACE(user_ptr, vtable) \
    { .ops = &vtable, .self = (void*)(user_ptr) }

/// @brief Structure to hold search results in a compressed sparse row (CSR) layout.
///
/// Row @p q (results for query @p q) occupies half-open range
/// [@p offsets[q], @p offsets[q+1]) in @p indices and @p distances. This supports
/// variadic per-query result counts (filtered search, range search) while keeping
/// the data flat and cache-friendly. For fixed top-k searches @p offsets[q] equals
/// @p q * k, so the classical @p indices[q*k+j] / @p distances[q*k+j] access pattern
/// remains valid.
///
/// Ownership: when @p owns_buffers is true the library allocated @p offsets,
/// @p indices, @p distances and svs_search_results_free() will release them.
/// When false, the caller is responsible for the storage; svs_search_results_free()
/// only resets the descriptor. On zero-initialized objects the free call is a no-op.
///
/// Buffer reuse: passing the same object to consecutive search calls lets the
/// library reuse existing library-owned buffers whenever capacity is sufficient;
/// steady-state batches of equal shape allocate only on the first call.
///
/// Forward-compatibility contract: On any write to this OUT struct, the library
/// only touches fields covered by the caller-supplied @p struct_size. A caller
/// compiled against an older header will never observe writes beyond its known
/// fields. Any optional field added in a future API version is written only when
/// the caller opts in by supplying a large-enough @p struct_size (e.g. via the
/// SVS_INIT_SEARCH_RESULTS() macro from the newer header).
///
/// @var svs_search_results::version
///   API version at which the struct was initialized (SVS_C_API_VERSION).
/// @var svs_search_results::struct_size
///   Size of this structure, used for versioning and forward compatibility.
/// @var svs_search_results::num_queries
///   Number of populated rows (queries) in this result set.
/// @var svs_search_results::total_results
///   Total number of populated results; equals offsets[num_queries].
/// @var svs_search_results::offsets
///   Row start offsets, length @p num_queries + 1. Monotonically non-decreasing.
/// @var svs_search_results::indices
///   Neighbor IDs, length @p total_results.
/// @var svs_search_results::distances
///   Neighbor distances, length @p total_results.
/// @var svs_search_results::offsets_capacity
///   Number of elements allocated in @p offsets.
/// @var svs_search_results::results_capacity
///   Number of elements allocated in @p indices and @p distances.
/// @var svs_search_results::owns_buffers
///   True if the library owns @p offsets, @p indices, and @p distances and must
///   free them; false if the buffers are caller-provided.
struct svs_search_results {
    uint32_t version;
    size_t struct_size;

    size_t num_queries;
    size_t total_results;
    size_t* offsets;
    size_t* indices;
    float* distances;

    size_t offsets_capacity;
    size_t results_capacity;
    bool owns_buffers;
};

/// @brief Macro to initialize a svs_search_results structure with default values
#define SVS_INIT_SEARCH_RESULTS()                                                       \
    {                                                                                   \
        .version = SVS_C_API_VERSION, .struct_size = sizeof(struct svs_search_results), \
        .num_queries = 0, .total_results = 0, .offsets = NULL, .indices = NULL,         \
        .distances = NULL, .offsets_capacity = 0, .results_capacity = 0,                \
        .owns_buffers = false                                                           \
    }

/// @brief Initialize a svs_search_results structure with caller-provided buffers.
///
/// Ownership stays with the caller (owns_buffers = false); svs_search_results_free()
/// will not release the buffers. The library reuses these buffers on search calls
/// as long as their capacities are sufficient; otherwise the call fails without
/// reallocating caller-owned storage.
///
/// @param p_offsets   Pointer to caller-owned offsets buffer (size_t[p_offsets_cap]).
///                    Must hold at least @p num_queries + 1 elements at call time.
/// @param p_indices   Pointer to caller-owned indices buffer (size_t[p_results_cap]).
/// @param p_distances Pointer to caller-owned distances buffer (float[p_results_cap]).
/// @param p_offsets_cap Number of elements allocated in @p p_offsets.
/// @param p_results_cap Number of elements allocated in @p p_indices and @p p_distances.
/// @example
/// size_t offsets[NQ + 1];
/// size_t indices[NQ * K];
/// float distances[NQ * K];
/// svs_search_results_t results = SVS_INIT_SEARCH_RESULTS_WITH_BUFFERS(
///     offsets, indices, distances, NQ + 1, NQ * K
/// );
#define SVS_INIT_SEARCH_RESULTS_WITH_BUFFERS(                                           \
    p_offsets, p_indices, p_distances, p_offsets_cap, p_results_cap                     \
)                                                                                       \
    {                                                                                   \
        .version = SVS_C_API_VERSION, .struct_size = sizeof(struct svs_search_results), \
        .num_queries = 0, .total_results = 0, .offsets = (p_offsets),                   \
        .indices = (p_indices), .distances = (p_distances),                             \
        .offsets_capacity = (p_offsets_cap), .results_capacity = (p_results_cap),       \
        .owns_buffers = false                                                           \
    }

/// @brief Convenience accessor for one query's row (O(1)).
/// @param results Pointer to a populated search results structure.
/// @param q Zero-based query index; must be < @p results->num_queries.
/// @param out_ids Optional out pointer to the first neighbor ID for query @p q.
/// @param out_distances Optional out pointer to the first neighbor distance for
///   query @p q.
/// @param out_count Optional out pointer to the number of results for query @p q.
static inline void svs_search_results_row(
    const struct svs_search_results* results,
    size_t q,
    const size_t** out_ids,
    const float** out_distances,
    size_t* out_count
) {
    if (q >= results->num_queries) {
        if (out_ids) {
            *out_ids = NULL;
        }
        if (out_distances) {
            *out_distances = NULL;
        }
        if (out_count) {
            *out_count = 0;
        }
        return;
    }
    size_t begin = results->offsets[q];
    size_t end = results->offsets[q + 1];
    if (out_ids) {
        *out_ids = results->indices + begin;
    }
    if (out_distances) {
        *out_distances = results->distances + begin;
    }
    if (out_count) {
        *out_count = end - begin;
    }
    return;
}

/// @brief Structure to hold memory breakdown for an index.
///
/// Forward-compatibility contract: On any write to this OUT struct, the library
/// only touches fields covered by the caller-supplied @p struct_size. A caller
/// compiled against an older header will never observe writes beyond its known
/// fields. Any optional field added in a future API version is written only when
/// the caller opts in by supplying a large-enough @p struct_size (e.g. via the
/// SVS_INIT_MEMORY_BREAKDOWN() macro from the newer header).
struct svs_memory_breakdown {
    uint32_t version; /// Version of the memory breakdown structure
    size_t
        struct_size; /// Size of the structure, used for versioning and compatibility checks
    size_t graph_bytes;    /// Allocated bytes for the graph structure
    size_t data_bytes;     /// Allocated bytes for the data vectors
    size_t metadata_bytes; /// Allocated bytes for metadata (entry points, status, etc.)
};

/// @brief Macro to initialize a svs_memory_breakdown structure with default values
#define SVS_INIT_MEMORY_BREAKDOWN()                                                       \
    {                                                                                     \
        .version = SVS_C_API_VERSION, .struct_size = sizeof(struct svs_memory_breakdown), \
        .graph_bytes = 0, .data_bytes = 0, .metadata_bytes = 0                            \
    }

// Handle typedefs; "_h" suffix indicates a handle to an opaque struct
typedef struct svs_index* svs_index_h;
typedef struct svs_index_builder* svs_index_builder_h;
typedef struct svs_algorithm* svs_algorithm_h;
typedef struct svs_storage* svs_storage_h;
typedef struct svs_search_params* svs_search_params_h;

// Fully defined types; "_t" suffix indicates a fully defined struct
typedef enum svs_error_code svs_error_code_t;
typedef enum svs_distance_metric svs_distance_metric_t;
typedef enum svs_algorithm_type svs_algorithm_type_t;
typedef enum svs_data_type svs_data_type_t;
typedef enum svs_storage_kind svs_storage_kind_t;
typedef enum svs_threadpool_kind svs_threadpool_kind_t;

typedef struct svs_threadpool_interface_ops svs_threadpool_ops_t;
typedef struct svs_threadpool_interface svs_threadpool_t;
typedef struct svs_threadpool_interface* svs_threadpool_i;

typedef struct svs_id_filter_interface_ops svs_id_filter_ops_t;
typedef struct svs_id_filter_interface svs_id_filter_t;
typedef struct svs_id_filter_interface* svs_id_filter_i;

typedef struct svs_search_results svs_search_results_t;
typedef struct svs_memory_breakdown svs_memory_breakdown_t;

/// @brief Get SVS version information
/// @return An integer representing the version of the SVS library, encoded as (major << 16)
/// | (minor << 8) | patch
SVS_API uint32_t svs_get_version();

/// @brief Get SVS version string
/// @return A string representing the version of the SVS library in "major.minor.patch"
/// format
SVS_API const char* svs_get_version_string();

/// @brief Create an error handle
/// @return A handle to the created error object or NULL if creation failed (e.g., due to
/// memory allocation failure)
SVS_API svs_error_h svs_error_create();

/// @brief Set an error code and message in the error handle
/// @param err The error handle to set
/// @param code The error code to set
/// @param message A string describing the error
/// @return true if the error was set successfully, false if failed (e.g., if the error
/// handle is NULL)
SVS_API bool svs_error_set(svs_error_h err, svs_error_code_t code, const char* message);

/// @brief Check if the error handle indicates success
/// @param err The error handle to check
/// @return true if no error occurred, false otherwise
SVS_API bool svs_error_ok(svs_error_h err);

/// @brief Get the error code from the error handle
/// @param err The error handle
/// @return The error code
SVS_API svs_error_code_t svs_error_get_code(svs_error_h err);

/// @brief Get the error message from the error handle
/// @param err The error handle
/// @return A string describing the error
/// @remarks The returned string is valid until the error handle is freed or modified.
SVS_API const char* svs_error_get_message(svs_error_h err);

/// @brief Free the error handle
/// @param err The error handle to free
SVS_API void svs_error_free(svs_error_h err);

/// @brief Create a Vamana algorithm configuration
/// @param graph_degree The graph degree parameter
/// @param build_window_size The build window size parameter
/// @param search_window_size Default search window size parameter
/// @param out_err An optional error handle to capture errors
/// @return A handle to the created Vamana algorithm
SVS_API svs_algorithm_h svs_algorithm_create_vamana(
    size_t graph_degree,
    size_t build_window_size,
    size_t search_window_size,
    svs_error_h out_err /*=NULL*/
);

/// @brief Get algorithm type from an algorithm handle
/// @param algorithm The algorithm handle
/// @param out_type Pointer to store the retrieved algorithm type
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_algorithm_get_type(
    svs_algorithm_h algorithm, svs_algorithm_type_t* out_type, svs_error_h out_err /*=NULL*/
);

/// @brief Free the algorithm configuration handle
/// @param algorithm The algorithm handle to free
SVS_API void svs_algorithm_free(svs_algorithm_h algorithm);

/// @brief Get the alpha parameter from a Vamana algorithm configuration
/// @param algorithm The algorithm handle
/// @param out_alpha Pointer to store the retrieved alpha parameter
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_algorithm_vamana_get_alpha(
    svs_algorithm_h algorithm, float* out_alpha, svs_error_h out_err /*=NULL*/
);

/// @brief Set the alpha parameter in a Vamana algorithm configuration
/// @param algorithm The algorithm handle
/// @param alpha The alpha parameter to set
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_algorithm_vamana_set_alpha(
    svs_algorithm_h algorithm, float alpha, svs_error_h out_err /*=NULL*/
);

/// @brief Get the graph degree parameter from a Vamana algorithm configuration
/// @param algorithm The algorithm handle
/// @param out_graph_degree Pointer to store the retrieved graph degree parameter
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_algorithm_vamana_get_graph_degree(
    svs_algorithm_h algorithm, size_t* out_graph_degree, svs_error_h out_err /*=NULL*/
);

/// @brief Set the graph degree parameter in a Vamana algorithm configuration
/// @param algorithm The algorithm handle
/// @param graph_degree The graph degree parameter to set
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_algorithm_vamana_set_graph_degree(
    svs_algorithm_h algorithm, size_t graph_degree, svs_error_h out_err /*=NULL*/
);

/// @brief Get the build window size parameter from a Vamana algorithm configuration
/// @param algorithm The algorithm handle
/// @param out_build_window_size Pointer to store the retrieved build window size parameter
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_algorithm_vamana_get_build_window_size(
    svs_algorithm_h algorithm, size_t* out_build_window_size, svs_error_h out_err /*=NULL*/
);

/// @brief Set the build window size parameter in a Vamana algorithm configuration
/// @param algorithm The algorithm handle
/// @param build_window_size The build window size parameter to set
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_algorithm_vamana_set_build_window_size(
    svs_algorithm_h algorithm, size_t build_window_size, svs_error_h out_err /*=NULL*/
);

/// @brief Get whether to use full search history in the Vamana algorithm
/// @param algorithm The algorithm handle
/// @param out_use_full_search_history Pointer to store whether full search history is used
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_algorithm_vamana_get_use_search_history(
    svs_algorithm_h algorithm,
    bool* out_use_full_search_history,
    svs_error_h out_err /*=NULL*/
);

/// @brief Set whether to use full search history in the Vamana algorithm
/// @param algorithm The algorithm handle
/// @param use_full_search_history Whether to use full search history
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_algorithm_vamana_set_use_search_history(
    svs_algorithm_h algorithm, bool use_full_search_history, svs_error_h out_err /*=NULL*/
);

/// @brief Create Vamana search parameters
/// @param search_window_size The search window size parameter
/// @param out_err An optional error handle to capture errors
/// @return A handle to the created Vamana search parameters
SVS_API svs_search_params_h svs_search_params_create_vamana(
    size_t search_window_size, svs_error_h out_err /*=NULL*/
);

/// @brief Free the search parameters handle
/// @param params The search parameters handle to free
SVS_API void svs_search_params_free(svs_search_params_h params);

/// @brief Create a simple storage configuration
/// @param data_type The data type of the vectors
/// @param out_err An optional error handle to capture errors
/// @return A handle to the created simple storage
SVS_API svs_storage_h
svs_storage_create_simple(svs_data_type_t data_type, svs_error_h out_err);

/// @brief Create a LeanVec storage configuration
/// @param leanvec_dims The number of LeanVec dimensions
/// @param primary The data type of the primary quantization
/// @param secondary The data type of the secondary quantization
/// @param out_err An optional error handle to capture errors
/// @return A handle to the created LeanVec storage
SVS_API svs_storage_h svs_storage_create_leanvec(
    size_t leanvec_dims,
    svs_data_type_t primary,
    svs_data_type_t secondary,
    svs_error_h out_err /*=NULL*/
);

/// @brief Create an LVQ storage configuration
/// @param primary The data type of the primary quantization
/// @param residual The data type of the residual quantization
/// @param out_err An optional error handle to capture errors
/// @return A handle to the created LVQ storage
SVS_API svs_storage_h svs_storage_create_lvq(
    svs_data_type_t primary, svs_data_type_t residual, svs_error_h out_err /*=NULL*/
);

/// @brief Create a Scalar Quantization storage configuration
/// @param data_type The data type of the quantized vectors
/// @param out_err An optional error handle to capture errors
/// @return A handle to the created Scalar Quantization storage
SVS_API svs_storage_h svs_storage_create_sq(
    svs_data_type_t data_type, svs_error_h out_err /*=NULL*/
);

/// @brief Get the kind of storage configuration
/// @param storage The storage handle
/// @param out_kind Pointer to store the retrieved storage kind
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_storage_get_kind(
    svs_storage_h storage, svs_storage_kind_t* out_kind, svs_error_h out_err /*=NULL*/
);

/// @brief Free the storage handle
/// @param storage The storage handle to free
SVS_API void svs_storage_free(svs_storage_h storage);

/// @brief Create an index builder configuration
/// @param metric The distance metric to use
/// @param dimension The dimensionality of the vectors
/// @param algorithm The algorithm configuration to use
/// @param out_err An optional error handle to capture errors
/// @return A handle to the created index builder
/// @remarks Default storage configuration is equivalent to
/// svs_storage_create_simple(SVS_DATA_TYPE_FLOAT32)
SVS_API svs_index_builder_h svs_index_builder_create(
    svs_distance_metric_t metric,
    size_t dimension,
    svs_algorithm_h algorithm,
    svs_error_h out_err /*=NULL*/
);

/// @brief Free the index builder handle
/// @param builder The index builder handle to free
SVS_API void svs_index_builder_free(svs_index_builder_h builder);

/// @brief Set the storage configuration for the index builder
/// @param builder The index builder handle
/// @param storage The storage configuration handle
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_index_builder_set_storage(
    svs_index_builder_h builder, svs_storage_h storage, svs_error_h out_err /*=NULL*/
);

/// @brief Set the thread pool configuration for the index builder
/// @param builder The index builder handle
/// @param kind The kind of thread pool to use
/// @param num_threads The number of threads to use (if applicable)
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_index_builder_set_threadpool(
    svs_index_builder_h builder,
    svs_threadpool_kind_t kind,
    size_t num_threads,
    svs_error_h out_err /*=NULL*/
);

/// @brief Set the custom thread pool for the index builder
/// @param builder The index builder handle
/// @param pool The custom thread pool interface
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_index_builder_set_threadpool_custom(
    svs_index_builder_h builder, svs_threadpool_i pool, svs_error_h out_err /*=NULL*/
);

/// @brief Build an index from the provided data
/// @param builder The index builder handle
/// @param data Pointer to the vector data (float array)
/// @param num_vectors The number of vectors in the data
/// @param out_err An optional error handle to capture errors
/// @return A handle to the built index
SVS_API svs_index_h svs_index_build(
    svs_index_builder_h builder,
    const float* data,
    size_t num_vectors,
    svs_error_h out_err /*=NULL*/
);

/// @brief Build a dynamic index from the provided data and IDs
/// @param builder The index builder handle
/// @param data Pointer to the vector data (float array)
/// @param ids Pointer to the vector IDs (size_t array). Can be NULL if IDs should be
/// auto-generated from 0 to num_vectors-1.
/// @param num_vectors The number of vectors in the data
/// @param blocksize_bytes The block size in bytes for dynamic index building (0 for
/// default)
/// @param out_err An optional error handle to capture errors
/// @return A handle to the built dynamic index
SVS_API svs_index_h svs_index_build_dynamic(
    svs_index_builder_h builder,
    const float* data,
    const size_t* ids /*=NULL*/,
    size_t num_vectors,
    size_t blocksize_bytes /*=0*/,
    svs_error_h out_err /*=NULL*/
);

/// @brief Load an index from disk
/// @param builder The index builder handle (used for configuration)
/// @param directory The directory path to load the index from
/// @param out_err An optional error handle to capture errors
/// @return A handle to the loaded index
SVS_API svs_index_h svs_index_load(
    svs_index_builder_h builder, const char* directory, svs_error_h out_err /*=NULL*/
);

/// @brief Load a dynamic index from disk
/// @param builder The index builder handle (used for configuration)
/// @param directory The directory path to load the index from
/// @param blocksize_bytes The block size in bytes for dynamic index loading (0 for default)
/// @param out_err An optional error handle to capture errors
/// @return A handle to the loaded dynamic index
SVS_API svs_index_h svs_index_load_dynamic(
    svs_index_builder_h builder,
    const char* directory,
    size_t blocksize_bytes /*=0*/,
    svs_error_h out_err /*=NULL*/
);

/// @brief Free the index handle
/// @param index The index handle to free
SVS_API void svs_index_free(svs_index_h index);

/// @brief TopK search the index with the provided queries and an optional ID filter
/// @details Performs a TopK search on the index with the provided queries and an optional
/// ID filter. The ID filter allows for filtering the search results based on specific IDs,
/// enabling more targeted searches. If the ID filter is NULL, the search will return the
/// top K results. If ID filter is provided, only the results that pass the filter will be
/// returned. Results are written into @p out_results in CSR layout (see
/// svs_search_results); library-owned buffers are reused across calls when their
/// capacity suffices. If ID filter is provided with `filter_rate > 0.0` then the
/// function will account for the actual filter hit rate during the search. If the
/// actual observed filter hit rate is less than the provided `filter_rate` value, the
/// function returns an empty result set.
/// @note After use, release library-owned buffers with svs_search_results_free().
/// @param index The index handle
/// @param queries Pointer to the query data (float array)
/// @param num_queries The number of query vectors
/// @param k The number of nearest neighbors to retrieve per query
/// @param out_results Pointer to a caller-provided results structure (typically
///   initialized with SVS_INIT_SEARCH_RESULTS()). See svs_search_results for
///   ownership and buffer-reuse semantics.
/// @param search_params The search parameters handle (can be NULL for defaults)
/// @param id_filter The ID filter interface (can be NULL for no filtering)
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_index_search_topk(
    svs_index_h index,
    const float* queries,
    size_t num_queries,
    size_t k,
    svs_search_results_t* out_results,
    svs_search_params_h search_params /*=NULL*/,
    svs_id_filter_i id_filter /*=NULL*/,
    svs_error_h out_err /*=NULL*/
);

/// @brief Release library-owned buffers held by a search results structure and
/// reset it to the SVS_INIT_SEARCH_RESULTS() state.
/// @param results Pointer to the results structure. Safe on NULL, on
/// zero-initialized objects, and on caller-owned buffers (in which case only the
/// descriptor is reset). Safe to call multiple times.
SVS_API void svs_search_results_free(svs_search_results_t* results);

/// @brief Search the index with the provided queries
/// @param index The index handle
/// @param queries Pointer to the query data (float array)
/// @param num_queries The number of query vectors
/// @param k The number of nearest neighbors to retrieve per query
/// @param out_results Pointer to a caller-provided results structure (typically
///   initialized with SVS_INIT_SEARCH_RESULTS()). See svs_search_results for
///   ownership and buffer-reuse semantics.
/// @param search_params The search parameters handle (can be NULL for defaults)
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
/// @deprecated Use svs_index_search_topk() instead, which additionally supports an
/// optional ID filter. This function is equivalent to calling svs_index_search_topk()
/// with a NULL id_filter.
SVS_DEPRECATED("Use svs_index_search_topk() instead")
static inline bool svs_index_search(
    svs_index_h index,
    const float* queries,
    size_t num_queries,
    size_t k,
    svs_search_results_t* out_results,
    svs_search_params_h search_params /*=NULL*/,
    svs_error_h out_err /*=NULL*/
) {
    return svs_index_search_topk(
        index, queries, num_queries, k, out_results, search_params, NULL, out_err
    );
}

/// @brief Save the index to disk
/// @param index The index handle
/// @param directory The directory path to save the index to
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool
svs_index_save(svs_index_h index, const char* directory, svs_error_h out_err /*=NULL*/);

/// @brief Add points to a dynamic index
/// @param index The dynamic index handle
/// @param new_points Pointer to the new vector data (float array)
/// @param ids Pointer to the new vector IDs (size_t array)
/// @param num_vectors The number of new vectors to add
/// @param out_added_count Optional pointer to store the number of successfully added
/// vectors
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_index_dynamic_add_points(
    svs_index_h index,
    const float* new_points,
    const size_t* ids,
    size_t num_vectors,
    size_t* out_added_count /*=NULL*/,
    svs_error_h out_err /*=NULL*/
);

/// @brief Delete points from a dynamic index
/// @param index The dynamic index handle
/// @param ids Pointer to the vector IDs to delete (size_t array)
/// @param num_ids The number of vector IDs to delete
/// @param out_deleted_count Optional pointer to store the number of successfully deleted
/// vectors
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_index_dynamic_delete_points(
    svs_index_h index,
    const size_t* ids,
    size_t num_ids,
    size_t* out_deleted_count /*=NULL*/,
    svs_error_h out_err /*=NULL*/
);

/// @brief Check if a dynamic index has a specific ID
/// @param index The dynamic index handle
/// @param id The vector ID to check for
/// @param out_has_id Pointer to store whether the ID exists in the index
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_index_dynamic_has_id(
    svs_index_h index, size_t id, bool* out_has_id, svs_error_h out_err /*=NULL*/
);

/// @brief Get the distance from a specific ID to a query vector in an index
/// @param index The index handle
/// @param id The vector ID to get the distance for
/// @param query Pointer to the query vector data (float array)
/// @param out_distance Pointer to store the retrieved distance
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_index_get_distance(
    svs_index_h index,
    size_t id,
    const float* query,
    float* out_distance,
    svs_error_h out_err /*=NULL*/
);

/// @brief Reconstruct the vectors for specific IDs in an index
/// @param index The index handle
/// @param ids Pointer to the vector IDs to reconstruct (size_t array)
/// @param num_ids The number of vector IDs to reconstruct
/// @param out_data Pointer to store the reconstructed vector data (float array with size
/// num_ids * data_dim)
/// @param data_dim The dimensionality of the vectors
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_index_reconstruct(
    svs_index_h index,
    const size_t* ids,
    size_t num_ids,
    float* out_data,
    size_t data_dim,
    svs_error_h out_err /*=NULL*/
);

/// @brief Consolidate a dynamic index to optimize storage and search performance
/// @param index The dynamic index handle
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool
svs_index_dynamic_consolidate(svs_index_h index, svs_error_h out_err /*=NULL*/);

/// @brief Compact a dynamic index to remove deleted entries and optimize storage
/// @param index The dynamic index handle
/// @param batchsize The batch size for compaction (0 for default)
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_index_dynamic_compact(
    svs_index_h index, size_t batchsize /*=0*/, svs_error_h out_err /*=NULL*/
);

/// @brief Get number of threads used for search in the index's thread pool
/// @param index The index handle
/// @param out_num_threads Pointer to store the retrieved number of threads
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
SVS_API bool svs_index_get_num_threads(
    svs_index_h index, size_t* out_num_threads, svs_error_h out_err /*=NULL*/
);

/// @brief Set number of threads for search in the index's thread pool
/// @param index The index handle
/// @param num_threads The number of threads to set
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
/// @remarks This function is only supported for indices built with threadpool kinds
/// SVS_THREADPOOL_KIND_NATIVE or SVS_THREADPOOL_KIND_OMP. Attempting to call this
/// function on indices built with SVS_THREADPOOL_KIND_CUSTOM or
/// SVS_THREADPOOL_KIND_SINGLE_THREAD will fail and return false.
/// @error On failure, if out_err is provided, it will contain:
/// - SVS_ERROR_INVALID_OPERATION if the index's threadpool kind is unresizable
/// - SVS_ERROR_INVALID_ARGUMENT if num_threads is invalid or zero
/// - SVS_ERROR_RUNTIME for other runtime failures
SVS_API bool svs_index_set_num_threads(
    svs_index_h index, size_t num_threads, svs_error_h out_err /*=NULL*/
);

/// @brief Get the total memory usage of the index in bytes
/// @param index The index handle
/// @param out_bytes Pointer to store the total memory usage in bytes
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
/// @remarks This returns the sum of graph_bytes + data_bytes + metadata_bytes
SVS_API bool svs_index_get_memory_usage(
    svs_index_h index, size_t* out_bytes, svs_error_h out_err /*=NULL*/
);

/// @brief Get the memory breakdown for the index
/// @param index The index handle
/// @param out_breakdown Pointer to store the memory breakdown structure
/// @param out_err An optional error handle to capture errors
/// @return true on success, false on failure
/// @remarks The breakdown reports allocated memory for graph, data, and metadata
/// components. Uses capacity-based accounting for datasets that support it, reflecting
/// the true memory footprint including over-allocation.
SVS_API bool svs_index_get_memory_breakdown(
    svs_index_h index, svs_memory_breakdown_t* out_breakdown, svs_error_h out_err /*=NULL*/
);

#ifdef __cplusplus
}
#endif
