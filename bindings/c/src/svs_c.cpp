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

#include <svs/concepts/data.h>
#include <svs/core/distance.h>
#include <svs/core/query_result.h>
#include <svs/index/vamana/build_params.h>
#include <svs/orchestrators/vamana.h>

struct SVS_Algorithm {
    svs_algorithm_type type;
    SVS_Algorithm(svs_algorithm_type type)
        : type(type) {}
    virtual ~SVS_Algorithm() = default;
};

struct SVS_AlgorithmVamana : public SVS_Algorithm {
    size_t graph_degree;
    size_t build_window_size;
    size_t search_window_size;

    SVS_AlgorithmVamana(
        size_t graph_degree, size_t build_window_size, size_t search_window_size
    )
        : SVS_Algorithm{SVS_ALGORITHM_TYPE_VAMANA}
        , graph_degree(graph_degree)
        , build_window_size(build_window_size)
        , search_window_size(search_window_size) {}

    svs::index::vamana::VamanaBuildParameters get_build_parameters() const {
        svs::index::vamana::VamanaBuildParameters params;
        params.graph_max_degree = graph_degree;
        params.window_size = build_window_size;
        return params;
    }
};

struct SVS_Storage {
    svs_storage_kind kind;
    SVS_Storage(svs_storage_kind kind)
        : kind(kind) {}
    virtual ~SVS_Storage() = default;
};

struct SVS_StorageSimple : public SVS_Storage {
    svs_data_type_t data_type;

    SVS_StorageSimple(svs_data_type_t data_type)
        : SVS_Storage{SVS_STORAGE_KIND_SIMPLE}
        , data_type(data_type) {}
};

struct SVS_StorageLeanVec : public SVS_Storage {
    size_t lenavec_dims;
    svs_data_type_t primary_type;
    svs_data_type_t secondary_type;

    SVS_StorageLeanVec(
        size_t lenavec_dims, svs_data_type_t primary, svs_data_type_t secondary
    )
        : SVS_Storage{SVS_STORAGE_KIND_LEANVEC}
        , lenavec_dims(lenavec_dims)
        , primary_type(primary)
        , secondary_type(secondary) {}
};

struct SVS_IndexBuilder {
    svs_distance_metric_t distance_metric;
    size_t dimension;
    std::shared_ptr<SVS_Algorithm> algorithm;
    std::shared_ptr<SVS_Storage> storage;

    SVS_IndexBuilder(
        svs_distance_metric_t distance_metric,
        size_t dimension,
        std::shared_ptr<SVS_Algorithm> algorithm
    )
        : distance_metric(distance_metric)
        , dimension(dimension)
        , algorithm(std::move(algorithm))
        , storage(std::make_shared<SVS_StorageSimple>(SVS_DATA_TYPE_FLOAT32)) {}

    ~SVS_IndexBuilder() {}

    void set_storage(std::shared_ptr<SVS_Storage> storage) {
        this->storage = std::move(storage);
    }
};

struct SVS_Index {
    svs_algorithm_type algorithm;
    SVS_Index(svs_algorithm_type algorithm)
        : algorithm(algorithm) {}
    virtual ~SVS_Index() = default;
};

struct SVS_IndexVamana : public SVS_Index {
    svs::Vamana index;
    SVS_IndexVamana(svs::Vamana&& index)
        : SVS_Index{SVS_ALGORITHM_TYPE_VAMANA}
        , index(std::move(index)) {}
    ~SVS_IndexVamana() {}
};

// C API function implementations
struct svs_index {
    std::shared_ptr<SVS_Index> impl;
};

struct svs_index_builder {
    std::shared_ptr<SVS_IndexBuilder> impl;
};

struct svs_algorithm {
    std::shared_ptr<SVS_Algorithm> impl;
};

struct svs_storage {
    std::shared_ptr<SVS_Storage> impl;
};

extern "C" svs_algorithm_t svs_algorithm_create_vamana(
    size_t graph_degree,
    size_t build_window_size,
    size_t search_window_size,
    svs_error_code_t* out_code
) {
    auto algorithm = std::make_shared<SVS_AlgorithmVamana>(
        graph_degree, build_window_size, search_window_size
    );
    if (out_code) {
        *out_code = SVS_OK;
    }
    auto result = new svs_algorithm;
    result->impl = algorithm;
    return result;
}

extern "C" void svs_algorithm_free(svs_algorithm_t algorithm) { delete algorithm; }

extern "C" svs_storage_t
svs_storage_create_simple(svs_data_type_t data_type, svs_error_code_t* out_code) {
    auto storage = std::make_shared<SVS_StorageSimple>(data_type);
    if (out_code) {
        *out_code = SVS_OK;
    }
    auto result = new svs_storage;
    result->impl = storage;
    return result;
}

extern "C" svs_storage_t svs_storage_create_leanvec(
    size_t lenavec_dims,
    svs_data_type_t primary,
    svs_data_type_t secondary,
    svs_error_code_t* out_code
) {
    auto storage = std::make_shared<SVS_StorageLeanVec>(lenavec_dims, primary, secondary);
    if (out_code) {
        *out_code = SVS_OK;
    }
    auto result = new svs_storage;
    result->impl = storage;
    return result;
}

extern "C" void svs_storage_free(svs_storage_t storage) { delete storage; }

extern "C" svs_index_builder_t svs_index_builder_create(
    svs_distance_metric_t metric,
    size_t dimension,
    svs_algorithm_t algorithm,
    svs_error_code_t* out_code
) {
    auto builder = std::make_shared<SVS_IndexBuilder>(metric, dimension, algorithm->impl);
    if (out_code) {
        *out_code = SVS_OK;
    }
    auto result = new svs_index_builder;
    result->impl = builder;
    return result;
}

extern "C" void svs_index_builder_free(svs_index_builder_t builder) { delete builder; }

extern "C" void svs_index_builder_set_storage(
    svs_index_builder_t builder, svs_storage_t storage, svs_error_code_t* out_code
) {
    builder->impl->set_storage(storage->impl);
    if (out_code) {
        *out_code = SVS_OK;
    }
    return;
}

extern "C" svs_index_t svs_index_build(
    svs_index_builder_t builder,
    const float* data,
    size_t num_vectors,
    svs_error_code_t* out_code
) {
    if (builder == nullptr || num_vectors == 0 || data == nullptr) {
        if (out_code) {
            *out_code = SVS_ERROR_INVALID_ARGUMENT;
        }
        return nullptr;
    }
    if (builder->impl->algorithm->type != SVS_ALGORITHM_TYPE_VAMANA) {
        if (out_code) {
            *out_code = SVS_ERROR_NOT_IMPLEMENTED;
        }
        return nullptr;
    }
    if (builder->impl->storage->kind != SVS_STORAGE_KIND_SIMPLE) {
        if (out_code) {
            *out_code = SVS_ERROR_NOT_IMPLEMENTED;
        }
        return nullptr;
    }

    svs::DistanceType distance_type;
    switch (builder->impl->distance_metric) {
        case SVS_DISTANCE_METRIC_EUCLIDEAN:
            distance_type = svs::DistanceType::L2;
            break;
        case SVS_DISTANCE_METRIC_DOT_PRODUCT:
            distance_type = svs::DistanceType::MIP;
            break;
        case SVS_DISTANCE_METRIC_COSINE:
            distance_type = svs::DistanceType::Cosine;
            break;
        default:
            if (out_code) {
                *out_code = SVS_ERROR_INVALID_ARGUMENT;
            }
            return nullptr;
    }

    auto src_data =
        svs::data::ConstSimpleDataView<float>(data, num_vectors, builder->impl->dimension);

    auto simple_data = svs::data::SimpleData<float>(num_vectors, builder->impl->dimension);

    svs::data::copy(src_data, simple_data);

    auto vamana_algorithm =
        std::static_pointer_cast<SVS_AlgorithmVamana>(builder->impl->algorithm);
    auto index = std::make_shared<SVS_IndexVamana>(svs::Vamana::build<float>(
        vamana_algorithm->get_build_parameters(), std::move(simple_data), distance_type
    ));
    if (out_code) {
        *out_code = SVS_OK;
    }
    auto result = new svs_index;
    result->impl = index;
    return result;
}

extern "C" void svs_index_free(svs_index_t index) { delete index; }

extern "C" svs_search_results_t
svs_index_search(svs_index_t index, const float* queries, size_t num_queries, size_t k) {
    if (index == nullptr || queries == nullptr || num_queries == 0 || k == 0) {
        return nullptr;
    }
    if (index->impl->algorithm != SVS_ALGORITHM_TYPE_VAMANA) {
        return nullptr;
    }
    auto& vamana_index = static_cast<SVS_IndexVamana&>(*index->impl).index;

    auto queries_view = svs::data::ConstSimpleDataView<float>(
        queries, num_queries, vamana_index.dimensions()
    );

    auto& vamana_idx = static_cast<SVS_IndexVamana&>(*index->impl).index;
    auto vamana_results = svs::QueryResult<size_t>(num_queries, k);

    vamana_idx.search(
        vamana_results.view(), queries_view, vamana_idx.get_search_parameters()
    );

    svs_search_results_t results = new svs_search_results{0, nullptr, nullptr, nullptr};

    results->num_queries = num_queries;
    results->results_per_query = new size_t[num_queries];
    results->indices = new size_t[num_queries * k];
    results->distances = new float[num_queries * k];

    for (size_t i = 0; i < num_queries; ++i) {
        results->results_per_query[i] = k;
        for (size_t j = 0; j < k; ++j) {
            results->indices[i * k + j] = vamana_results.index(i, j);
            results->distances[i * k + j] = vamana_results.distance(i, j);
        }
    }

    return results;
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
