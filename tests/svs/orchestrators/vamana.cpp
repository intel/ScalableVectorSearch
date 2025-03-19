/*
 * Copyright 2023 Intel Corporation
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

// SVS
#include "svs/orchestrators/vamana.h"

// Catch2
#include "catch2/catch_test_macros.hpp"
#include <catch2/catch_approx.hpp>

// svsbenchmark
#include "svs-benchmark/benchmark.h"

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"
#include "tests/utils/vamana_reference.h"

CATCH_TEST_CASE("Vamana Index", "[managers][vamana]") {
    // Todo?
}
CATCH_TEST_CASE("Vamana Index Default Parameters", "[managers][vamana]") {
    using Catch::Approx;
    std::filesystem::path data_path = test_dataset::data_svs_file();

    CATCH_SECTION("L2 Distance Defaults") {
        auto expected_result = test_dataset::vamana::expected_build_results(
            svs::L2, svsbenchmark::Uncompressed(svs::DataType::float32)
        );
        auto build_params = expected_result.build_parameters_.value();
        auto data_loader = svs::data::SimpleData<float>::load(data_path);
        svs::Vamana index = svs::Vamana::build<float>(build_params, data_loader, svs::L2);
        CATCH_REQUIRE(index.get_alpha() == Approx(1.2f));
    }

    CATCH_SECTION("MIP Distance Defaults") {
        auto expected_result = test_dataset::vamana::expected_build_results(
            svs::MIP, svsbenchmark::Uncompressed(svs::DataType::float32)
        );
        auto build_params = expected_result.build_parameters_.value();
        auto data_loader = svs::data::SimpleData<float>::load(data_path);
        svs::Vamana index = svs::Vamana::build<float>(build_params, data_loader, svs::MIP);
        CATCH_REQUIRE(index.get_alpha() == Approx(0.95f));
    }

    CATCH_SECTION("Invalid Alpha for L2") {
        auto expected_result = test_dataset::vamana::expected_build_results(
            svs::L2, svsbenchmark::Uncompressed(svs::DataType::float32)
        );
        auto build_params = expected_result.build_parameters_.value();
        build_params.alpha = 0.8f;
        auto data_loader = svs::data::SimpleData<float>::load(data_path);
        CATCH_REQUIRE_THROWS_WITH(
            svs::Vamana::build<float>(build_params, data_loader, svs::L2),
            "For L2 distance, alpha must be >= 1.0"
        );
    }

    CATCH_SECTION("Invalid Alpha for MIP") {
        auto expected_result = test_dataset::vamana::expected_build_results(
            svs::MIP, svsbenchmark::Uncompressed(svs::DataType::float32)
        );
        auto build_params = expected_result.build_parameters_.value();
        build_params.alpha = 1.2f;
        auto data_loader = svs::data::SimpleData<float>::load(data_path);
        CATCH_REQUIRE_THROWS_WITH(
            svs::Vamana::build<float>(build_params, data_loader, svs::MIP),
            "For MIP/Cosine distance, alpha must be <= 1.0"
        );
    }
}