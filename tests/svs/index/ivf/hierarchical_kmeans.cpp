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

// header under test
#include "svs/index/ivf/hierarchical_kmeans.h"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

// catch
#include "catch2/catch_test_macros.hpp"

// stl
#include <unordered_map>

namespace {

template <typename BuildType, svs::data::ImmutableMemoryDataset Data, typename Distance>
void test_hierarchical_kmeans_clustering(const Data& data, Distance distance) {
    namespace ivf = svs::index::ivf;

    for (size_t n_centroids : {155}) {
        for (size_t minibatch : {25}) {
            for (size_t iters : {3}) {
                for (float training_fraction : {0.55}) {
                    for (size_t l1_clusters : {0, 9}) {
                        auto params = ivf::IVFBuildParameters()
                            .num_centroids(n_centroids)
                            .minibatch_size(minibatch)
                            .num_iterations(iters)
                            .is_hierarchical(true)
                            .training_fraction(training_fraction)
                            .hierarchical_level1_clusters(l1_clusters);

                        auto threadpool = svs::threads::as_threadpool(10);
                        auto [centroids, clusters] =
                            hierarchical_kmeans_clustering<BuildType>(
                                params, data, distance, threadpool
                            );

                        CATCH_REQUIRE(centroids.size() == n_centroids);
                        CATCH_REQUIRE(centroids.dimensions() == data.dimensions());
                        CATCH_REQUIRE(clusters.size() == n_centroids);
                    }
                }
            }
        }
    }
}

} // namespace

CATCH_TEST_CASE("Hierarchical Kmeans Param Check", "[ivf][hierarchial_parameter_check]") {
    CATCH_SECTION("Uncompressed Data") {
        auto data = svs::data::SimpleData<float>::load(test_dataset::data_svs_file());
        test_hierarchical_kmeans_clustering<float>(data, svs::DistanceIP());
        test_hierarchical_kmeans_clustering<svs::Float16>(data, svs::DistanceIP());
        test_hierarchical_kmeans_clustering<svs::BFloat16>(data, svs::DistanceIP());
        test_hierarchical_kmeans_clustering<svs::BFloat16>(data, svs::DistanceL2());
    }
}
