/*
 * Copyright 2025 Intel Corporation
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

#include "svs/orchestrators/ivf.h"
#include "svsmain.h"

// stl
#include <functional>
#include <map>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace svs::index::ivf;

template <typename BuildType, typename Params, typename Data, typename Dist>
void build_ivf_clustering(
    Params params,
    Data data,
    Dist dist,
    size_t n_threads,
    const std::string& clustering_directory
) {
    auto clustering = svs::IVF::build_clustering<BuildType>(
        std::move(params), std::move(data), std::move(dist), n_threads
    );
    svs::lib::save_to_disk(clustering, clustering_directory);
}

const std::string HELP =
    R"(
The required arguments are as follows:

(1) Data Element Type (string). Options: (int8, uint8, float, float16, bfloat16)
(2) Path to vector dataset (.vecs format) (string).
(3) Number of clusters to be built
(4) Number of threads to use for index construction (integer).
(5) Should use hierarchical Kmeans? (0/1)
(6) Clustering directory for saving.
(7) Distance type (string - distance type)
)";

int svs_main(std::vector<std::string> args) {
    if (args.size() != 8) {
        std::cout << "Expected 7 arguments. Instead, got " << args.size() - 1 << ". "
                  << "The required positional arguments are given below." << std::endl
                  << std::endl
                  << HELP << std::endl;
        return 1;
    }

    size_t i = 1;
    const auto& data_type(args[i++]);
    const auto& vecs_filename(args[i++]);
    const size_t n_clusters = std::stoull(args[i++]);
    const size_t n_threads = std::stoull(args[i++]);
    const size_t is_hierarchical = std::stoull(args[i++]);
    const std::string& clustering_directory(args[i++]);
    const auto& distance_type = args[i++];

    const size_t D = svs::Dynamic;
    using Alloc = svs::HugepageAllocator<float>;

    auto data = svs::VectorDataLoader<float, D, Alloc>(vecs_filename);

    auto dist_disp = [&]<typename dist_type>(dist_type dist) {
        auto params = svs::index::ivf::IVFBuildParameters(n_clusters, 10000, 10, false, .1);
        if (is_hierarchical) {
            params.is_hierarchical_ = true;
        }
        if (data_type == "float") {
            build_ivf_clustering<float>(
                std::move(params),
                std::move(data),
                std::move(dist),
                n_threads,
                clustering_directory
            );
        } else if (data_type == "float16") {
            build_ivf_clustering<svs::Float16>(
                std::move(params),
                std::move(data),
                std::move(dist),
                n_threads,
                clustering_directory
            );
        } else if (data_type == "bfloat16") {
            build_ivf_clustering<svs::BFloat16>(
                std::move(params),
                std::move(data),
                std::move(dist),
                n_threads,
                clustering_directory
            );
        } else {
            throw ANNEXCEPTION("Unsupported data type: ", data_type, '.');
        }
    };

    if (distance_type == std::string("L2")) {
        dist_disp(svs::distance::DistanceL2{});
    } else if (distance_type == std::string("MIP")) {
        dist_disp(svs::distance::DistanceIP{});
    } else {
        throw ANNEXCEPTION(
            "Unsupported distance type. Valid values: L2/MIP. Received: ",
            distance_type,
            '!'
        );
    }

    return 0;
}

SVS_DEFINE_MAIN();
