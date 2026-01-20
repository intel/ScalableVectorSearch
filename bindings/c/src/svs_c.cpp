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
#include "index.hpp"
#include "index_builder.hpp"
#include "storage.hpp"
#include "thread_pool.hpp"
#include "types_support.hpp"

#include <svs/core/data.h>
#include <svs/core/query_result.h>
#include <svs/orchestrators/vamana.h>

// C API implementation
struct svs_error_desc {
    svs_error_code_t code;
    std::string message;
};

#define SET_ERROR(err, c, msg)      \
    do {                            \
        if (err) {                  \
            (err)->code = (c);      \
            (err)->message = (msg); \
        }                           \
    } while (0)

template <typename Callable, typename Result = std::invoke_result_t<Callable>>
inline Result
runtime_error_wrapper(Callable&& func, svs_error_h err, Result err_res = {}) noexcept {
    try {
        SET_ERROR(err, SVS_OK, "Success");
        return func();
    } catch (const std::invalid_argument& ex) {
        SET_ERROR(err, SVS_ERROR_INVALID_ARGUMENT, ex.what());
        return err_res;
    } catch (const std::exception& ex) {
        SET_ERROR(err, SVS_ERROR_GENERIC, ex.what());
        return err_res;
    } catch (...) {
        SET_ERROR(err, SVS_ERROR_GENERIC, "An unknown error has occurred.");
        return err_res;
    }
}

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

extern "C" svs_error_h svs_error_init() { return new svs_error_desc{SVS_OK, "Success"}; }
extern "C" bool svs_error_ok(svs_error_h err) { return err->code == SVS_OK; }
extern "C" svs_error_code_t svs_error_get_code(svs_error_h err) { return err->code; }
extern "C" const char* svs_error_get_message(svs_error_h err) {
    return err->message.c_str();
}
extern "C" void svs_error_free(svs_error_h err) { delete err; }

extern "C" svs_algorithm_h svs_algorithm_create_vamana(
    size_t graph_degree,
    size_t build_window_size,
    size_t search_window_size,
    svs_error_h out_err
) {
    return runtime_error_wrapper(
        [&]() {
            using namespace svs::c_runtime;
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

extern "C" svs_search_params_h svs_search_params_create_vamana(
    size_t search_window_size,
    svs_error_h out_err
) {
    return runtime_error_wrapper(
        [&]() {
            using namespace svs::c_runtime;
            auto params = std::make_shared<AlgorithmVamana::SearchParams>(
                search_window_size
            );
            auto result = new svs_search_params;
            result->impl = params;
            return result;
        },
        out_err
    );
}

extern "C" void svs_search_params_free(svs_search_params_h params) {
    delete params;
}

extern "C" svs_storage_h
svs_storage_create_simple(svs_data_type_t data_type, svs_error_h out_err) {
    return runtime_error_wrapper(
        [&]() {
            using namespace svs::c_runtime;
            auto storage = std::make_shared<StorageSimple>(data_type);
            auto result = new svs_storage;
            result->impl = storage;
            return result;
        },
        out_err
    );
}

extern "C" svs_storage_h svs_storage_create_leanvec(
    size_t lenavec_dims,
    svs_data_type_t primary,
    svs_data_type_t secondary,
    svs_error_h out_err
) {
    return runtime_error_wrapper(
        [&]() {
            using namespace svs::c_runtime;
            auto storage =
                std::make_shared<StorageLeanVec>(lenavec_dims, primary, secondary);
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
    return runtime_error_wrapper(
        [&]() {
            using namespace svs::c_runtime;
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
    if (builder == nullptr || storage == nullptr) {
        SET_ERROR(out_err, SVS_ERROR_INVALID_ARGUMENT, "Invalid argument");
        return false;
    }
    return runtime_error_wrapper(
        [&]() {
            builder->impl->set_storage(storage->impl);
            return true;
        },
        out_err
    );
}

extern "C" bool svs_index_builder_set_thread_pool(
    svs_index_builder_h builder,
    svs_thread_pool_kind_t kind,
    size_t num_threads,
    svs_error_h out_err
) {
    if (builder == nullptr) {
        SET_ERROR(out_err, SVS_ERROR_INVALID_ARGUMENT, "Invalid argument");
        return false;
    }
    return runtime_error_wrapper(
        [&]() {
            builder->impl->set_thread_pool({kind, num_threads});
            return true;
        },
        out_err
    );
}

extern "C" svs_index_h svs_index_build(
    svs_index_builder_h builder, const float* data, size_t num_vectors, svs_error_h out_err
) {
    if (builder == nullptr || num_vectors == 0 || data == nullptr) {
        SET_ERROR(out_err, SVS_ERROR_INVALID_ARGUMENT, "Invalid argument");
        return nullptr;
    }
    if (builder->impl->algorithm->type != SVS_ALGORITHM_TYPE_VAMANA) {
        SET_ERROR(out_err, SVS_ERROR_NOT_IMPLEMENTED, "Not implemented");
        return nullptr;
    }
    if (builder->impl->storage->kind != SVS_STORAGE_KIND_SIMPLE &&
        builder->impl->storage->kind != SVS_STORAGE_KIND_LEANVEC) {
        SET_ERROR(out_err, SVS_ERROR_NOT_IMPLEMENTED, "Not implemented");
        return nullptr;
    }

    return runtime_error_wrapper(
        [&]() {
            using namespace svs::c_runtime;
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

    return runtime_error_wrapper(
        [&]() {
            using namespace svs::c_runtime;

            auto& vamana_index = static_cast<IndexVamana&>(*index->impl).index;

            auto queries_view = svs::data::ConstSimpleDataView<float>(
                queries, num_queries, vamana_index.dimensions()
            );

            auto search_results = index->impl->search(queries_view, k, search_params->impl);

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
