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

#include "svs/c_api/svs_c.h"

// #include <svs/concepts/data.h>
// #include <svs/core/distance.h>
// #include <svs/core/query_result.h>
#include <svs/index/vamana/build_params.h>
#include <svs/index/vamana/search_params.h>
#include <svs/orchestrators/vamana.h>

namespace svs::c_runtime {

struct Algorithm {
    struct SearchParams {
        svs_algorithm_type type;
        SearchParams(svs_algorithm_type type)
            : type(type) {}
        virtual ~SearchParams() = default;
    };

    svs_algorithm_type type;
    Algorithm(svs_algorithm_type type)
        : type(type) {}
    virtual ~Algorithm() = default;

    virtual std::shared_ptr<SearchParams> get_default_search_params() const = 0;
};

struct AlgorithmVamana : public Algorithm {
    struct SearchParams : public Algorithm::SearchParams {
        size_t search_window_size;
        SearchParams(size_t search_window_size)
            : Algorithm::SearchParams{SVS_ALGORITHM_TYPE_VAMANA}
            , search_window_size(search_window_size) {}

        svs::index::vamana::VamanaSearchParameters get_search_parameters() const {
            svs::index::vamana::VamanaSearchParameters params;
            params.buffer_config_ =
                svs::index::vamana::SearchBufferConfig{search_window_size};
            return params;
        }
    };

    size_t graph_degree;
    size_t build_window_size;
    SearchParams default_search_params;

    AlgorithmVamana(
        size_t graph_degree, size_t build_window_size, size_t search_window_size
    )
        : Algorithm{SVS_ALGORITHM_TYPE_VAMANA}
        , graph_degree(graph_degree)
        , build_window_size(build_window_size)
        , default_search_params(search_window_size) {}

    svs::index::vamana::VamanaBuildParameters get_build_parameters() const {
        svs::index::vamana::VamanaBuildParameters params;
        params.graph_max_degree = graph_degree;
        params.window_size = build_window_size;
        return params;
    }

    std::shared_ptr<Algorithm::SearchParams> get_default_search_params() const override {
        return std::make_shared<SearchParams>(default_search_params);
    }
};

} // namespace svs::c_runtime