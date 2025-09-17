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
#include "svs/index/ivf/common.h"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

// catch
#include "catch2/catch_test_macros.hpp"

CATCH_TEST_CASE("Kmeans Clustering", "[ivf][parameters]") {
    namespace ivf = svs::index::ivf;
    CATCH_SECTION("IVF Build Parameters") {
        auto p = ivf::IVFBuildParameters();

        // Test the setter methods.
#define XX(name, v)                        \
    CATCH_REQUIRE(p.name##_ != v);         \
    CATCH_REQUIRE(p.name(v).name##_ == v); \
    CATCH_REQUIRE(p.name##_ == v);

        XX(num_centroids, 10);
        XX(minibatch_size, 100);
        XX(num_iterations, 1000);
        XX(is_hierarchical, false);
        XX(training_fraction, 0.05F);
        XX(hierarchical_level1_clusters, 10);
        XX(seed, 0x1234);
#undef XX

        // Saving and loading
        svs_test::prepare_temp_directory();
        auto dir = svs_test::temp_directory();
        CATCH_REQUIRE(svs::lib::test_self_save_load(p, dir));
    }

    CATCH_SECTION("IVF Search Parameters") {
        auto p = ivf::IVFSearchParameters();

        // Test the setter methods.
#define XX(name, v)                        \
    CATCH_REQUIRE(p.name##_ != v);         \
    CATCH_REQUIRE(p.name(v).name##_ == v); \
    CATCH_REQUIRE(p.name##_ == v);

        XX(n_probes, 10);
        XX(k_reorder, 100);
#undef XX

        // Saving and loading
        svs_test::prepare_temp_directory();
        auto dir = svs_test::temp_directory();
        CATCH_REQUIRE(svs::lib::test_self_save_load(p, dir));
    }
}
