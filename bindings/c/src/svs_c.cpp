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

#include "svs/c_api/svs_c.h"

#include "algorithm.hpp"
#include "error.hpp"
#include "index.hpp"
#include "index_builder.hpp"
#include "storage.hpp"
#include "threadpool.hpp"
#include "types_support.hpp"

#include <svs/core/data.h>
#include <svs/core/query_result.h>
#include <svs/orchestrators/vamana.h>

// C API implementation
struct svs_index {
    std::shared_ptr<svs::c_runtime::Index> impl;
};

struct svs_index_builder {
    std::shared_ptr<svs::c_runtime::IndexBuilder> impl;
};

struct svs_algorithm {
    std::shared_ptr<svs::c_runtime::Algorithm> impl;
};

struct svs_search_params {
    std::shared_ptr<svs::c_runtime::Algorithm::SearchParams> impl;
};

struct svs_storage {
    std::shared_ptr<svs::c_runtime::Storage> impl;
};

extern "C" svs_algorithm_h svs_algorithm_create_vamana(
    size_t graph_degree,
    size_t build_window_size,
    size_t search_window_size,
    svs_error_h out_err
) {
    using namespace svs::c_runtime;
    return wrap_exceptions(
        [&]() {
            EXPECT_ARG_GT_THAN(graph_degree, 0);
            EXPECT_ARG_GT_THAN(build_window_size, 0);
            EXPECT_ARG_GT_THAN(search_window_size, 0);
            auto algorithm = std::make_shared<AlgorithmVamana>(
                graph_degree, build_window_size, search_window_size
            );
            auto result = new svs_algorithm;
            result->impl = algorithm;
            return result;
        },
        out_err
    );
}

extern "C" void svs_algorithm_free(svs_algorithm_h algorithm) { delete algorithm; }

#define EXPECT_VAMANA(algorithm)                              \
    EXPECT_ARG_NOT_NULL(algorithm);                           \
    INVALID_ARGUMENT_IF(                                      \
        (algorithm->impl->type != SVS_ALGORITHM_TYPE_VAMANA), \
        "Algorithm type does not support this operation"      \
    )

extern "C" bool svs_algorithm_vamana_get_alpha(
    svs_algorithm_h algorithm, float* out_alpha, svs_error_h out_err
) {
    using namespace svs::c_runtime;
    return wrap_exceptions(
        [&]() {
            EXPECT_VAMANA(algorithm);
            EXPECT_ARG_NOT_NULL(out_alpha);
            auto vamana_algorithm =
                std::static_pointer_cast<AlgorithmVamana>(algorithm->impl);
            *out_alpha = vamana_algorithm->build_parameters().alpha;
            return true;
        },
        out_err
    );
}

extern "C" bool svs_algorithm_vamana_set_alpha(
    svs_algorithm_h algorithm, float alpha, svs_error_h out_err
) {
    using namespace svs::c_runtime;
    return wrap_exceptions(
        [&]() {
            EXPECT_VAMANA(algorithm);
            EXPECT_ARG_GT_THAN(alpha, 0.0f);
            auto vamana_algorithm =
                std::static_pointer_cast<AlgorithmVamana>(algorithm->impl);
            vamana_algorithm->build_parameters().alpha = alpha;
            return true;
        },
        out_err
    );
}

extern "C" bool svs_algorithm_vamana_get_graph_degree(
    svs_algorithm_h algorithm, size_t* out_graph_degree, svs_error_h out_err
) {
    using namespace svs::c_runtime;
    return wrap_exceptions(
        [&]() {
            EXPECT_VAMANA(algorithm);
            EXPECT_ARG_NOT_NULL(out_graph_degree);
            auto vamana_algorithm =
                std::static_pointer_cast<AlgorithmVamana>(algorithm->impl);
            *out_graph_degree = vamana_algorithm->build_parameters().graph_max_degree;
            return true;
        },
        out_err
    );
}

extern "C" bool svs_algorithm_vamana_set_graph_degree(
    svs_algorithm_h algorithm, size_t graph_degree, svs_error_h out_err
) {
    using namespace svs::c_runtime;
    return wrap_exceptions(
        [&]() {
            EXPECT_VAMANA(algorithm);
            EXPECT_ARG_GT_THAN(graph_degree, 0);
            auto vamana_algorithm =
                std::static_pointer_cast<AlgorithmVamana>(algorithm->impl);
            vamana_algorithm->build_parameters().graph_max_degree = graph_degree;
            return true;
        },
        out_err
    );
}

extern "C" bool svs_algorithm_vamana_get_build_window_size(
    svs_algorithm_h algorithm, size_t* out_build_window_size, svs_error_h out_err
) {
    using namespace svs::c_runtime;
    return wrap_exceptions(
        [&]() {
            EXPECT_VAMANA(algorithm);
            EXPECT_ARG_NOT_NULL(out_build_window_size);
            auto vamana_algorithm =
                std::static_pointer_cast<AlgorithmVamana>(algorithm->impl);
            *out_build_window_size = vamana_algorithm->build_parameters().window_size;
            return true;
        },
        out_err
    );
}

extern "C" bool svs_algorithm_vamana_set_build_window_size(
    svs_algorithm_h algorithm, size_t build_window_size, svs_error_h out_err
) {
    using namespace svs::c_runtime;
    return wrap_exceptions(
        [&]() {
            EXPECT_VAMANA(algorithm);
            EXPECT_ARG_GT_THAN(build_window_size, 0);
            auto vamana_algorithm =
                std::static_pointer_cast<AlgorithmVamana>(algorithm->impl);
            vamana_algorithm->build_parameters().window_size = build_window_size;
            return true;
        },
        out_err
    );
}

extern "C" bool svs_algorithm_vamana_get_use_search_history(
    svs_algorithm_h algorithm, bool* out_use_full_search_history, svs_error_h out_err
) {
    using namespace svs::c_runtime;
    return wrap_exceptions(
        [&]() {
            EXPECT_VAMANA(algorithm);
            EXPECT_ARG_NOT_NULL(out_use_full_search_history);
            auto vamana_algorithm =
                std::static_pointer_cast<AlgorithmVamana>(algorithm->impl);
            *out_use_full_search_history =
                vamana_algorithm->build_parameters().use_full_search_history;
            return true;
        },
        out_err
    );
}

extern "C" bool svs_algorithm_vamana_set_use_search_history(
    svs_algorithm_h algorithm, bool use_full_search_history, svs_error_h out_err
) {
    using namespace svs::c_runtime;
    return wrap_exceptions(
        [&]() {
            EXPECT_VAMANA(algorithm);
            auto vamana_algorithm =
                std::static_pointer_cast<AlgorithmVamana>(algorithm->impl);
            vamana_algorithm->build_parameters().use_full_search_history =
                use_full_search_history;
            return true;
        },
        out_err
    );
}

extern "C" svs_search_params_h
svs_search_params_create_vamana(size_t search_window_size, svs_error_h out_err) {
    using namespace svs::c_runtime;
    return wrap_exceptions(
        [&]() {
            EXPECT_ARG_GT_THAN(search_window_size, 0);
            auto params =
                std::make_shared<AlgorithmVamana::SearchParams>(search_window_size);
            auto result = new svs_search_params;
            result->impl = params;
            return result;
        },
        out_err
    );
}

extern "C" void svs_search_params_free(svs_search_params_h params) { delete params; }

extern "C" svs_storage_h
svs_storage_create_simple(svs_data_type_t data_type, svs_error_h out_err) {
    using namespace svs::c_runtime;
    return wrap_exceptions(
        [&]() {
            INVALID_ARGUMENT_IF(
                data_type != SVS_DATA_TYPE_FLOAT32 && data_type != SVS_DATA_TYPE_FLOAT16,
                "Simple storage only supports float32 and float16 data types"
            );
            auto storage = std::make_shared<StorageSimple>(data_type);
            auto result = new svs_storage;
            result->impl = storage;
            return result;
        },
        out_err
    );
}

extern "C" svs_storage_h svs_storage_create_leanvec(
    size_t leanvec_dims,
    svs_data_type_t primary,
    svs_data_type_t secondary,
    svs_error_h out_err
) {
    using namespace svs::c_runtime;
    return wrap_exceptions(
        [&]() {
            EXPECT_ARG_GT_THAN(leanvec_dims, 0);
            NOT_IMPLEMENTED_IF(
                (primary == SVS_DATA_TYPE_FLOAT32 || primary == SVS_DATA_TYPE_FLOAT16 ||
                 secondary == SVS_DATA_TYPE_FLOAT32 || secondary == SVS_DATA_TYPE_FLOAT16),
                "Unsupported simple data types for LeanVec primary and secondary"
            );
            INVALID_ARGUMENT_IF(
                (primary != SVS_DATA_TYPE_INT4 && primary != SVS_DATA_TYPE_UINT4 &&
                 primary != SVS_DATA_TYPE_INT8 && primary != SVS_DATA_TYPE_UINT8),
                "Unsupported data type for LeanVec primary storage"
            );
            INVALID_ARGUMENT_IF(
                (secondary != SVS_DATA_TYPE_INT4 && secondary != SVS_DATA_TYPE_UINT4 &&
                 secondary != SVS_DATA_TYPE_INT8 && secondary != SVS_DATA_TYPE_UINT8),
                "Unsupported data type for LeanVec secondary storage"
            );

            auto storage =
                std::make_shared<StorageLeanVec>(leanvec_dims, primary, secondary);
            auto result = new svs_storage;
            result->impl = storage;
            return result;
        },
        out_err
    );
}

extern "C" svs_storage_h svs_storage_create_lvq(
    svs_data_type_t primary, svs_data_type_t residual, svs_error_h out_err
) {
    using namespace svs::c_runtime;
    return wrap_exceptions(
        [&]() {
            INVALID_ARGUMENT_IF(
                (primary != SVS_DATA_TYPE_INT4 && primary != SVS_DATA_TYPE_UINT4 &&
                 primary != SVS_DATA_TYPE_INT8 && primary != SVS_DATA_TYPE_UINT8),
                "Unsupported data type for LeanVec primary storage"
            );
            INVALID_ARGUMENT_IF(
                (residual != SVS_DATA_TYPE_INT4 && residual != SVS_DATA_TYPE_UINT4 &&
                 residual != SVS_DATA_TYPE_INT8 && residual != SVS_DATA_TYPE_UINT8 &&
                 residual != SVS_DATA_TYPE_VOID),
                "Unsupported data type for LeanVec secondary storage"
            );
            auto storage = std::make_shared<StorageLVQ>(primary, residual);
            auto result = new svs_storage;
            result->impl = storage;
            return result;
        },
        out_err
    );
}

extern "C" svs_storage_h
svs_storage_create_sq(svs_data_type_t data_type, svs_error_h out_err) {
    using namespace svs::c_runtime;
    return wrap_exceptions(
        [&]() {
            INVALID_ARGUMENT_IF(
                data_type != SVS_DATA_TYPE_UINT8 && data_type != SVS_DATA_TYPE_INT8,
                "Scalar quantization only supports 8-bit data types"
            );
            auto storage = std::make_shared<StorageSQ>(data_type);
            auto result = new svs_storage;
            result->impl = storage;
            return result;
        },
        out_err
    );
}

extern "C" void svs_storage_free(svs_storage_h storage) { delete storage; }

extern "C" svs_index_builder_h svs_index_builder_create(
    svs_distance_metric_t metric,
    size_t dimension,
    svs_algorithm_h algorithm,
    svs_error_h out_err
) {
    using namespace svs::c_runtime;
    return wrap_exceptions(
        [&]() {
            INVALID_ARGUMENT_IF(
                metric != SVS_DISTANCE_METRIC_EUCLIDEAN &&
                    metric != SVS_DISTANCE_METRIC_COSINE &&
                    metric != SVS_DISTANCE_METRIC_DOT_PRODUCT,
                "Unsupported distance metric"
            );
            EXPECT_ARG_GT_THAN(dimension, 0);
            EXPECT_ARG_NOT_NULL(algorithm);
            auto builder =
                std::make_shared<IndexBuilder>(metric, dimension, algorithm->impl);
            auto result = new svs_index_builder;
            result->impl = builder;
            return result;
        },
        out_err
    );
}

extern "C" void svs_index_builder_free(svs_index_builder_h builder) { delete builder; }

extern "C" bool svs_index_builder_set_storage(
    svs_index_builder_h builder, svs_storage_h storage, svs_error_h out_err
) {
    using namespace svs::c_runtime;
    return wrap_exceptions(
        [&]() {
            EXPECT_ARG_NOT_NULL(builder);
            EXPECT_ARG_NOT_NULL(storage);
            builder->impl->set_storage(storage->impl);
            return true;
        },
        out_err
    );
}

extern "C" bool svs_index_builder_set_threadpool(
    svs_index_builder_h builder,
    svs_threadpool_kind_t kind,
    size_t num_threads,
    svs_error_h out_err
) {
    if (builder == nullptr) {
        SET_ERROR(out_err, SVS_ERROR_INVALID_ARGUMENT, "Invalid argument");
        return false;
    }
    using namespace svs::c_runtime;
    return wrap_exceptions(
        [&]() {
            EXPECT_ARG_NOT_NULL(builder);
            EXPECT_ARG_GT_THAN(num_threads, 0);
            builder->impl->set_threadpool_builder({kind, num_threads});
            return true;
        },
        out_err
    );
}

extern "C" bool svs_index_builder_set_threadpool_custom(
    svs_index_builder_h builder,
    svs_threadpool_interface_t pool,
    svs_error_h out_err /*=NULL*/
) {
    using namespace svs::c_runtime;
    return wrap_exceptions(
        [&]() {
            EXPECT_ARG_NOT_NULL(builder);
            EXPECT_ARG_NOT_NULL(pool);
            builder->impl->set_threadpool_builder({pool});
            return true;
        },
        out_err
    );
}

extern "C" svs_index_h svs_index_build(
    svs_index_builder_h builder, const float* data, size_t num_vectors, svs_error_h out_err
) {
    using namespace svs::c_runtime;
    return wrap_exceptions(
        [&]() {
            EXPECT_ARG_NOT_NULL(builder);
            EXPECT_ARG_GT_THAN(num_vectors, 0);
            EXPECT_ARG_NOT_NULL(data);
            NOT_IMPLEMENTED_IF(
                (builder->impl->algorithm->type != SVS_ALGORITHM_TYPE_VAMANA),
                "Only Vamana algorithm is currently supported for index building"
            );
            auto src_data = svs::data::ConstSimpleDataView<float>(
                data, num_vectors, builder->impl->dimension
            );

            auto index = builder->impl->build(src_data);
            if (index == nullptr) {
                SET_ERROR(out_err, SVS_ERROR_INDEX_BUILD_FAILED, "Index build failed");
                return svs_index_h{nullptr};
            }

            auto result = new svs_index;
            result->impl = index;
            return result;
        },
        out_err
    );
}

extern "C" void svs_index_free(svs_index_h index) { delete index; }

extern "C" svs_search_results_t svs_index_search(
    svs_index_h index,
    const float* queries,
    size_t num_queries,
    size_t k,
    svs_search_params_h search_params,
    svs_error_h out_err
) {
    if (index == nullptr || queries == nullptr || num_queries == 0 || k == 0) {
        SET_ERROR(out_err, SVS_ERROR_INVALID_ARGUMENT, "Invalid argument");
        return nullptr;
    }
    if (index->impl->algorithm != SVS_ALGORITHM_TYPE_VAMANA) {
        SET_ERROR(out_err, SVS_ERROR_NOT_IMPLEMENTED, "Not implemented");
        return nullptr;
    }

    using namespace svs::c_runtime;
    return wrap_exceptions(
        [&]() {
            EXPECT_ARG_NOT_NULL(index);
            EXPECT_ARG_NOT_NULL(queries);
            EXPECT_ARG_GT_THAN(num_queries, 0);
            EXPECT_ARG_GT_THAN(k, 0);
            auto& vamana_index = static_cast<IndexVamana&>(*index->impl).index;

            auto queries_view = svs::data::ConstSimpleDataView<float>(
                queries, num_queries, vamana_index.dimensions()
            );

            auto search_results = index->impl->search(
                queries_view, k, search_params == nullptr ? nullptr : search_params->impl
            );

            svs_search_results_t results =
                new svs_search_results{0, nullptr, nullptr, nullptr};

            results->num_queries = num_queries;
            results->results_per_query = new size_t[num_queries];
            results->indices = new size_t[num_queries * k];
            results->distances = new float[num_queries * k];

            for (size_t i = 0; i < num_queries; ++i) {
                results->results_per_query[i] = k;
                for (size_t j = 0; j < k; ++j) {
                    results->indices[i * k + j] = search_results.index(i, j);
                    results->distances[i * k + j] = search_results.distance(i, j);
                }
            }

            return results;
        },
        out_err
    );
}

extern "C" void svs_search_results_free(svs_search_results_t results) {
    if (results == nullptr) {
        return;
    }
    delete[] results->results_per_query;
    delete[] results->indices;
    delete[] results->distances;
    delete results;
}
