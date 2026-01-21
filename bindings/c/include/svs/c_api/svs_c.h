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

#include "svs_c_config.h"

#ifdef __cplusplus
extern "C" {
#endif
#include <stdbool.h>
#include <stddef.h>

enum svs_error_code {
    SVS_OK = 0,
    SVS_ERROR_GENERIC = 1,
    SVS_ERROR_INVALID_ARGUMENT = 2,
    SVS_ERROR_OUT_OF_MEMORY = 3,
    SVS_ERROR_INDEX_BUILD_FAILED = 4,
    SVS_ERROR_NOT_IMPLEMENTED = 5
};

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
    SVS_DATA_TYPE_VOID = 0,
    SVS_DATA_TYPE_FLOAT32 = 32,
    SVS_DATA_TYPE_FLOAT16 = 16,
    SVS_DATA_TYPE_INT8 = 8,
    SVS_DATA_TYPE_UINT8 = SVS_DATA_TYPE_INT8 - 1,
    SVS_DATA_TYPE_INT4 = 4,
    SVS_DATA_TYPE_UINT4 = SVS_DATA_TYPE_INT4 - 1
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

// clang-format off
struct svs_threadpool_interface_ops {
    size_t (*size)(void* self);
    void (*parallel_for)(
        void* self,
        void (*func)(void* svs_param, size_t n),
        void* svs_param,
        size_t n
    );
};
// clang-format on

struct svs_threadpool_interface {
    struct svs_threadpool_interface_ops ops;
    void* self;
};

/// @brief Structure to hold search results
struct svs_search_results {
    size_t num_queries;        /// Number of query vectors
    size_t* results_per_query; /// Number of results per query
    size_t* indices;           /// Indices of the nearest neighbors
    float* distances;          /// Distances to the nearest neighbors
};

// Handle typedefs; "_h" suffix indicates a handle to an opaque struct
typedef struct svs_error_desc* svs_error_h;
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
typedef enum svs_threadpool_kind svs_threadpool_kind_t;

typedef struct svs_threadpool_interface* svs_threadpool_interface_t;
typedef struct svs_search_results* svs_search_results_t;

/// @brief Create an error handle
/// @return A handle to the created error object
SVS_API svs_error_h svs_error_init();

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

/// @brief Free the algorithm configuration handle
/// @param algorithm The algorithm handle to free
SVS_API void svs_algorithm_free(svs_algorithm_h algorithm);

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
/// @param lenavec_dims The number of LeanVec dimensions
/// @param primary The data type of the primary quantization
/// @param secondary The data type of the secondary quantization
/// @param out_err An optional error handle to capture errors
/// @return A handle to the created LeanVec storage
SVS_API svs_storage_h svs_storage_create_leanvec(
    size_t lenavec_dims,
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
    svs_index_builder_h builder,
    svs_threadpool_interface_t pool,
    svs_error_h out_err /*=NULL*/
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

/// @brief Free the index handle
/// @param index The index handle to free
SVS_API void svs_index_free(svs_index_h index);

/// @brief Search the index with the provided queries
/// @param index The index handle
/// @param queries Pointer to the query data (float array)
/// @param num_queries The number of query vectors
/// @param k The number of nearest neighbors to retrieve per query
/// @param search_params The search parameters handle (can be NULL for defaults)
/// @param out_err An optional error handle to capture errors
/// @return A pointer to the search results structure
SVS_API svs_search_results_t svs_index_search(
    svs_index_h index,
    const float* queries,
    size_t num_queries,
    size_t k,
    svs_search_params_h search_params /*=NULL*/,
    svs_error_h out_err /*=NULL*/
);

/// @brief Free the search results structure
/// @param results The search results structure to release
SVS_API void svs_search_results_free(svs_search_results_t results);

#ifdef __cplusplus
}
#endif
