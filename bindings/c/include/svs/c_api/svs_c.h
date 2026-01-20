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
    SVS_DATA_TYPE_INT8 = 9,
    SVS_DATA_TYPE_UINT8 = 8,
    SVS_DATA_TYPE_INT4 = 5,
    SVS_DATA_TYPE_UINT4 = 4
};

enum svs_storage_kind {
    SVS_STORAGE_KIND_SIMPLE = 0,
    SVS_STORAGE_KIND_LEANVEC = 1,
    SVS_STORAGE_KIND_LVQ = 2,
    SVS_STORAGE_KIND_SQ = 3
};

enum svs_thread_pool_kind {
    SVS_THREAD_POOL_KIND_NATIVE = 0,
    SVS_THREAD_POOL_KIND_OMP = 1,
    SVS_THREAD_POOL_KIND_SINGLE_THREAD = 2,
    SVS_THREAD_POOL_KIND_MANUAL = 3
};

struct svs_search_results {
    size_t num_queries;
    size_t* results_per_query;
    size_t* indices;
    float* distances;
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
typedef enum svs_thread_pool_kind svs_thread_pool_kind_t;

typedef struct svs_search_results* svs_search_results_t;

SVS_API svs_error_h svs_error_init();
SVS_API bool svs_error_ok(svs_error_h err);
SVS_API svs_error_code_t svs_error_get_code(svs_error_h err);
SVS_API const char* svs_error_get_message(svs_error_h err);
SVS_API void svs_error_free(svs_error_h err);

SVS_API svs_algorithm_h svs_algorithm_create_vamana(
    size_t graph_degree,
    size_t build_window_size,
    size_t search_window_size,
    svs_error_h out_err /*=NULL*/
);
SVS_API void svs_algorithm_free(svs_algorithm_h algorithm);

SVS_API svs_search_params_h svs_search_params_create_vamana(
    size_t search_window_size, svs_error_h out_err /*=NULL*/
);
SVS_API void svs_search_params_free(svs_search_params_h params);

SVS_API svs_storage_h
svs_storage_create_simple(svs_data_type_t data_type, svs_error_h out_err);
SVS_API svs_storage_h svs_storage_create_leanvec(
    size_t lenavec_dims,
    svs_data_type_t primary,
    svs_data_type_t secondary,
    svs_error_h out_err /*=NULL*/
);
SVS_API svs_storage_h svs_storage_create_lvq(
    svs_data_type_t primary, svs_data_type_t residual, svs_error_h out_err /*=NULL*/
);
SVS_API svs_storage_h svs_storage_create_sq(
    svs_data_type_t data_type, svs_error_h out_err /*=NULL*/
);
SVS_API void svs_storage_free(svs_storage_h storage);

SVS_API svs_index_builder_h svs_index_builder_create(
    svs_distance_metric_t metric,
    size_t dimension,
    svs_algorithm_h algorithm,
    svs_error_h out_err /*=NULL*/
);
SVS_API void svs_index_builder_free(svs_index_builder_h builder);

SVS_API bool svs_index_builder_set_storage(
    svs_index_builder_h builder, svs_storage_h storage, svs_error_h out_err /*=NULL*/
);

SVS_API bool svs_index_builder_set_thread_pool(
    svs_index_builder_h builder,
    svs_thread_pool_kind_t kind,
    size_t num_threads,
    svs_error_h out_err /*=NULL*/
);

SVS_API svs_index_h svs_index_build(
    svs_index_builder_h builder,
    const float* data,
    size_t num_vectors,
    svs_error_h out_err /*=NULL*/
);
SVS_API void svs_index_free(svs_index_h index);

SVS_API svs_search_results_t svs_index_search(
    svs_index_h index,
    const float* queries,
    size_t num_queries,
    size_t k,
    svs_search_params_h search_params /*=NULL*/,
    svs_error_h out_err /*=NULL*/
);
SVS_API void svs_search_results_free(svs_search_results_t results);

#ifdef __cplusplus
}
#endif
