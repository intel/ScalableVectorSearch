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

#include "svs/runtime/api_defs.h"
#include "svs/runtime/dynamic_ivf_index.h"
#include "svs/runtime/ivf_index.h"

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

#include "utils.h"

namespace {

// Generate test data
std::vector<float> create_test_data(size_t n, size_t d, unsigned int seed = 123) {
    std::vector<float> data(n * d);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dis(gen);
    }
    return data;
}

constexpr size_t test_d = 64;
constexpr size_t test_n = 100;

// Global test data - generated once and reused across all tests
inline const std::vector<float>& get_test_data() {
    static const std::vector<float> test_data = create_test_data(test_n, test_d, 123);
    return test_data;
}

// Compute recall@k: fraction of ground-truth neighbors found in the result set,
// averaged over all queries.
double compute_recall(
    const std::vector<size_t>& result_labels,
    const std::vector<size_t>& gt_labels,
    size_t nq,
    size_t k
) {
    size_t total_found = 0;
    for (size_t q = 0; q < nq; ++q) {
        for (size_t i = 0; i < k; ++i) {
            size_t gt_id = gt_labels[q * k + i];
            for (size_t j = 0; j < k; ++j) {
                if (result_labels[q * k + j] == gt_id) {
                    ++total_found;
                    break;
                }
            }
        }
    }
    return static_cast<double>(total_found) / static_cast<double>(nq * k);
}

} // namespace

CATCH_TEST_CASE("IVFIndexBuildAndSearch", "[runtime][ivf]") {
    std::cout << "[IVF] Running IVFIndexBuildAndSearch..." << std::endl;
    const auto& test_data = get_test_data();

    // Build static IVF index
    svs::runtime::v0::IVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10; // Small number for test data
    build_params.num_iterations = 5;

    svs::runtime::v0::Status status = svs::runtime::v0::IVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        test_n,
        test_data.data(),
        build_params
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Search
    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 10;

    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    status = index->search(nq, xq, k, distances.data(), result_labels.data());
    CATCH_REQUIRE(status.ok());

    // Verify results are reasonable (at least some results found)
    bool found_valid = false;
    for (int i = 0; i < nq * k; ++i) {
        if (result_labels[i] < test_n) {
            found_valid = true;
            break;
        }
    }
    CATCH_REQUIRE(found_valid);

    svs::runtime::v0::IVFIndex::destroy(index);
}

CATCH_TEST_CASE("IVFIndexWriteAndRead", "[runtime][ivf]") {
    std::cout << "[IVF] Running IVFIndexWriteAndRead..." << std::endl;
    const auto& test_data = get_test_data();

    // Build static IVF index
    svs::runtime::v0::IVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    svs::runtime::v0::Status status = svs::runtime::v0::IVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        test_n,
        test_data.data(),
        build_params
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    svs_test::prepare_temp_directory();
    auto temp_dir = svs_test::temp_directory();
    auto filename = temp_dir / "static_ivf_test.bin";

    // Serialize
    std::ofstream out(filename, std::ios::binary);
    CATCH_REQUIRE(out.is_open());
    status = index->save(out);
    CATCH_REQUIRE(status.ok());
    out.close();

    // Deserialize
    svs::runtime::v0::IVFIndex* loaded = nullptr;
    std::ifstream in(filename, std::ios::binary);
    CATCH_REQUIRE(in.is_open());
    status = svs::runtime::v0::IVFIndex::load(
        &loaded, in, svs::runtime::v0::MetricType::L2, svs::runtime::v0::StorageKind::FP32
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(loaded != nullptr);
    in.close();

    // Test search on loaded index
    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 10;

    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    status = loaded->search(nq, xq, k, distances.data(), result_labels.data());
    CATCH_REQUIRE(status.ok());

    // Clean up
    svs::runtime::v0::IVFIndex::destroy(index);
    svs::runtime::v0::IVFIndex::destroy(loaded);
}

CATCH_TEST_CASE("DynamicIVFIndexBuildAndSearch", "[runtime][ivf]") {
    std::cout << "[IVF] Running DynamicIVFIndexBuildAndSearch..." << std::endl;
    const auto& test_data = get_test_data();

    // Build dynamic IVF index with initial data
    svs::runtime::v0::DynamicIVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    svs::runtime::v0::IVFIndex::SearchParams search_params;
    search_params.n_probes = 3;

    std::vector<size_t> labels(test_n);
    std::iota(labels.begin(), labels.end(), 0);

    svs::runtime::v0::Status status = svs::runtime::v0::DynamicIVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        test_n,
        test_data.data(),
        labels.data(),
        build_params,
        search_params
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Search
    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 10;

    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    status = index->search(nq, xq, k, distances.data(), result_labels.data());
    CATCH_REQUIRE(status.ok());

    // Verify that increasing n_probes improves recall for dynamic IVF
    // Ground truth: search with all centroids probed
    svs::runtime::v0::IVFIndex::SearchParams exhaustive_params;
    exhaustive_params.n_probes = 10;
    std::vector<float> gt_distances(nq * k);
    std::vector<size_t> gt_labels(nq * k);
    status =
        index->search(nq, xq, k, gt_distances.data(), gt_labels.data(), &exhaustive_params);
    CATCH_REQUIRE(status.ok());

    // Low n_probes
    svs::runtime::v0::IVFIndex::SearchParams low_params;
    low_params.n_probes = 1;
    std::vector<float> dist_low(nq * k);
    std::vector<size_t> labels_low(nq * k);
    status = index->search(nq, xq, k, dist_low.data(), labels_low.data(), &low_params);
    CATCH_REQUIRE(status.ok());

    // High n_probes
    svs::runtime::v0::IVFIndex::SearchParams high_params;
    high_params.n_probes = 5;
    std::vector<float> dist_high(nq * k);
    std::vector<size_t> labels_high(nq * k);
    status = index->search(nq, xq, k, dist_high.data(), labels_high.data(), &high_params);
    CATCH_REQUIRE(status.ok());

    double recall_low = compute_recall(labels_low, gt_labels, nq, k);
    double recall_high = compute_recall(labels_high, gt_labels, nq, k);

    std::cout << "  [Dynamic] recall@" << k << " with n_probes=1: " << recall_low
              << std::endl;
    std::cout << "  [Dynamic] recall@" << k << " with n_probes=5: " << recall_high
              << std::endl;

    CATCH_REQUIRE(recall_high >= recall_low);
    CATCH_REQUIRE(recall_high > 0.0);

    svs::runtime::v0::DynamicIVFIndex::destroy(index);
}

CATCH_TEST_CASE("DynamicIVFIndexAddAndRemove", "[runtime][ivf]") {
    std::cout << "[IVF] Running DynamicIVFIndexAddAndRemove..." << std::endl;
    const auto& test_data = get_test_data();

    // Build empty dynamic IVF index first, then add data
    svs::runtime::v0::DynamicIVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    // Create with initial data (needed for clustering)
    std::vector<size_t> labels(test_n);
    std::iota(labels.begin(), labels.end(), 0);

    svs::runtime::v0::Status status = svs::runtime::v0::DynamicIVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        test_n,
        test_data.data(),
        labels.data(),
        build_params
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Check has_id for existing IDs
    bool exists = false;
    status = index->has_id(&exists, 0);
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(exists);

    status = index->has_id(&exists, test_n - 1);
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(exists);

    // Check has_id for non-existing ID
    status = index->has_id(&exists, test_n + 100);
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(!exists);

    // Remove some IDs
    std::vector<size_t> ids_to_remove = {0, 1, 2};
    status = index->remove(ids_to_remove.size(), ids_to_remove.data());
    CATCH_REQUIRE(status.ok());

    // Verify removed IDs no longer exist
    status = index->has_id(&exists, 0);
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(!exists);

    // Consolidate and compact
    status = index->consolidate();
    CATCH_REQUIRE(status.ok());

    status = index->compact();
    CATCH_REQUIRE(status.ok());

    // Search should still work
    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 10;

    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    status = index->search(nq, xq, k, distances.data(), result_labels.data());
    CATCH_REQUIRE(status.ok());

    svs::runtime::v0::DynamicIVFIndex::destroy(index);
}

CATCH_TEST_CASE("DynamicIVFIndexWriteAndRead", "[runtime][ivf]") {
    std::cout << "[IVF] Running DynamicIVFIndexWriteAndRead..." << std::endl;
    const auto& test_data = get_test_data();

    // Build dynamic IVF index
    svs::runtime::v0::DynamicIVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    std::vector<size_t> labels(test_n);
    std::iota(labels.begin(), labels.end(), 0);

    svs::runtime::v0::Status status = svs::runtime::v0::DynamicIVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        test_n,
        test_data.data(),
        labels.data(),
        build_params
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    svs_test::prepare_temp_directory();
    auto temp_dir = svs_test::temp_directory();
    auto filename = temp_dir / "dynamic_ivf_test.bin";

    // Serialize
    std::ofstream out(filename, std::ios::binary);
    CATCH_REQUIRE(out.is_open());
    status = index->save(out);
    CATCH_REQUIRE(status.ok());
    out.close();

    // Deserialize
    svs::runtime::v0::DynamicIVFIndex* loaded = nullptr;
    std::ifstream in(filename, std::ios::binary);
    CATCH_REQUIRE(in.is_open());
    status = svs::runtime::v0::DynamicIVFIndex::load(
        &loaded, in, svs::runtime::v0::MetricType::L2, svs::runtime::v0::StorageKind::FP32
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(loaded != nullptr);
    in.close();

    // Test search on loaded index
    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 10;

    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    status = loaded->search(nq, xq, k, distances.data(), result_labels.data());
    CATCH_REQUIRE(status.ok());

    // Clean up
    svs::runtime::v0::DynamicIVFIndex::destroy(index);
    svs::runtime::v0::DynamicIVFIndex::destroy(loaded);
}

CATCH_TEST_CASE("DynamicIVFIndexRemoveSelected", "[runtime][ivf]") {
    std::cout << "[IVF] Running DynamicIVFIndexRemoveSelected..." << std::endl;
    const auto& test_data = get_test_data();

    // Build dynamic IVF index
    svs::runtime::v0::DynamicIVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    std::vector<size_t> labels(test_n);
    std::iota(labels.begin(), labels.end(), 0);

    svs::runtime::v0::Status status = svs::runtime::v0::DynamicIVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        test_n,
        test_data.data(),
        labels.data(),
        build_params
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Remove IDs in range [0, 20) using selector
    size_t min_id = 0;
    size_t max_id = 20;
    test_utils::IDFilterRange selector(min_id, max_id);

    size_t num_removed = 0;
    status = index->remove_selected(&num_removed, selector);
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(num_removed == max_id - min_id);

    // Verify removed IDs no longer exist
    bool exists = false;
    for (size_t i = min_id; i < max_id; ++i) {
        status = index->has_id(&exists, i);
        CATCH_REQUIRE(status.ok());
        CATCH_REQUIRE(!exists);
    }

    // Verify IDs outside range still exist
    status = index->has_id(&exists, max_id);
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(exists);

    svs::runtime::v0::DynamicIVFIndex::destroy(index);
}

CATCH_TEST_CASE("IVFIndexSearchWithParams", "[runtime][ivf]") {
    std::cout << "[IVF] Running IVFIndexSearchWithParams..." << std::endl;
    const auto& test_data = get_test_data();

    // Build static IVF index
    svs::runtime::v0::IVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    svs::runtime::v0::IVFIndex::SearchParams default_search_params;
    default_search_params.n_probes = 2;

    svs::runtime::v0::Status status = svs::runtime::v0::IVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        test_n,
        test_data.data(),
        build_params,
        default_search_params
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 10;

    // Step 1: Get ground-truth by searching with all centroids probed
    svs::runtime::v0::IVFIndex::SearchParams exhaustive_params;
    exhaustive_params.n_probes = 10; // all centroids

    std::vector<float> gt_distances(nq * k);
    std::vector<size_t> gt_labels(nq * k);
    status =
        index->search(nq, xq, k, gt_distances.data(), gt_labels.data(), &exhaustive_params);
    CATCH_REQUIRE(status.ok());

    // Step 2: Search with low n_probes
    svs::runtime::v0::IVFIndex::SearchParams low_params;
    low_params.n_probes = 1;

    std::vector<float> distances_low(nq * k);
    std::vector<size_t> labels_low(nq * k);
    status = index->search(nq, xq, k, distances_low.data(), labels_low.data(), &low_params);
    CATCH_REQUIRE(status.ok());

    // Step 3: Search with high n_probes
    svs::runtime::v0::IVFIndex::SearchParams high_params;
    high_params.n_probes = 5;

    std::vector<float> distances_high(nq * k);
    std::vector<size_t> labels_high(nq * k);
    status =
        index->search(nq, xq, k, distances_high.data(), labels_high.data(), &high_params);
    CATCH_REQUIRE(status.ok());

    // Step 4: Compute recall for both and verify higher n_probes gives >= recall
    double recall_low = compute_recall(labels_low, gt_labels, nq, k);
    double recall_high = compute_recall(labels_high, gt_labels, nq, k);

    std::cout << "  recall@" << k << " with n_probes=1: " << recall_low << std::endl;
    std::cout << "  recall@" << k << " with n_probes=5: " << recall_high << std::endl;

    CATCH_REQUIRE(recall_high >= recall_low);
    // With 5 out of 10 centroids probed, recall should be reasonably high
    CATCH_REQUIRE(recall_high > 0.0);

    svs::runtime::v0::IVFIndex::destroy(index);
}

CATCH_TEST_CASE("IVFIndexCheckStorageKind", "[runtime][ivf]") {
    std::cout << "[IVF] Running IVFIndexCheckStorageKind..." << std::endl;
    // FP32 should always be supported
    CATCH_REQUIRE(
        svs::runtime::v0::IVFIndex::check_storage_kind(svs::runtime::v0::StorageKind::FP32)
            .ok()
    );
    CATCH_REQUIRE(svs::runtime::v0::DynamicIVFIndex::check_storage_kind(
                      svs::runtime::v0::StorageKind::FP32
    )
                      .ok());

    // FP16 should always be supported
    CATCH_REQUIRE(
        svs::runtime::v0::IVFIndex::check_storage_kind(svs::runtime::v0::StorageKind::FP16)
            .ok()
    );
    CATCH_REQUIRE(svs::runtime::v0::DynamicIVFIndex::check_storage_kind(
                      svs::runtime::v0::StorageKind::FP16
    )
                      .ok());

    // SQI8 should always be supported
    CATCH_REQUIRE(
        svs::runtime::v0::IVFIndex::check_storage_kind(svs::runtime::v0::StorageKind::SQI8)
            .ok()
    );
    CATCH_REQUIRE(svs::runtime::v0::DynamicIVFIndex::check_storage_kind(
                      svs::runtime::v0::StorageKind::SQI8
    )
                      .ok());

    // LVQ and LeanVec support depends on build configuration
    // check_storage_kind will return ok when built with LVQ/LeanVec support
    auto lvq_status =
        svs::runtime::v0::IVFIndex::check_storage_kind(svs::runtime::v0::StorageKind::LVQ4x4
        );
    auto leanvec_status = svs::runtime::v0::IVFIndex::check_storage_kind(
        svs::runtime::v0::StorageKind::LeanVec4x4
    );
    // Just verify the calls don't crash - actual support depends on build flags
    (void)lvq_status;
    (void)leanvec_status;
}

CATCH_TEST_CASE("IVFIndexBuildAndSearchLVQ", "[runtime][ivf]") {
    std::cout << "[IVF] Running IVFIndexBuildAndSearchLVQ..." << std::endl;
    const auto& test_data = get_test_data();

    svs::runtime::v0::IVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    svs::runtime::v0::Status status = svs::runtime::v0::IVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::LVQ4x4,
        test_n,
        test_data.data(),
        build_params
    );

    if (!svs::runtime::v0::IVFIndex::check_storage_kind(
             svs::runtime::v0::StorageKind::LVQ4x4
        )
             .ok()) {
        CATCH_REQUIRE(!status.ok());
        CATCH_SKIP("LVQ storage kind is not supported in this build configuration.");
    }
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Search
    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 10;

    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    status = index->search(nq, xq, k, distances.data(), result_labels.data());
    CATCH_REQUIRE(status.ok());

    svs::runtime::v0::IVFIndex::destroy(index);
}

CATCH_TEST_CASE("IVFIndexWriteAndReadLVQ", "[runtime][ivf]") {
    std::cout << "[IVF] Running IVFIndexWriteAndReadLVQ..." << std::endl;
    const auto& test_data = get_test_data();

    svs::runtime::v0::IVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    svs::runtime::v0::Status status = svs::runtime::v0::IVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::LVQ4x4,
        test_n,
        test_data.data(),
        build_params
    );

    if (!svs::runtime::v0::IVFIndex::check_storage_kind(
             svs::runtime::v0::StorageKind::LVQ4x4
        )
             .ok()) {
        CATCH_REQUIRE(!status.ok());
        CATCH_SKIP("LVQ storage kind is not supported in this build configuration.");
    }
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    svs_test::prepare_temp_directory();
    auto temp_dir = svs_test::temp_directory();
    auto filename = temp_dir / "ivf_lvq_test.bin";

    // Serialize
    std::ofstream out(filename, std::ios::binary);
    CATCH_REQUIRE(out.is_open());
    status = index->save(out);
    CATCH_REQUIRE(status.ok());
    out.close();

    // Deserialize
    svs::runtime::v0::IVFIndex* loaded = nullptr;
    std::ifstream in(filename, std::ios::binary);
    CATCH_REQUIRE(in.is_open());
    status = svs::runtime::v0::IVFIndex::load(
        &loaded, in, svs::runtime::v0::MetricType::L2, svs::runtime::v0::StorageKind::LVQ4x4
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(loaded != nullptr);
    in.close();

    // Search loaded index
    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 10;

    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    status = loaded->search(nq, xq, k, distances.data(), result_labels.data());
    CATCH_REQUIRE(status.ok());

    svs::runtime::v0::IVFIndex::destroy(index);
    svs::runtime::v0::IVFIndex::destroy(loaded);
}

CATCH_TEST_CASE("DynamicIVFIndexBuildAndSearchLVQ", "[runtime][ivf]") {
    std::cout << "[IVF] Running DynamicIVFIndexBuildAndSearchLVQ..." << std::endl;
    const auto& test_data = get_test_data();

    svs::runtime::v0::DynamicIVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    std::vector<size_t> labels(test_n);
    std::iota(labels.begin(), labels.end(), 0);

    svs::runtime::v0::Status status = svs::runtime::v0::DynamicIVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::LVQ4x4,
        test_n,
        test_data.data(),
        labels.data(),
        build_params
    );

    if (!svs::runtime::v0::DynamicIVFIndex::check_storage_kind(
             svs::runtime::v0::StorageKind::LVQ4x4
        )
             .ok()) {
        CATCH_REQUIRE(!status.ok());
        CATCH_SKIP("LVQ storage kind is not supported in this build configuration.");
    }
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Search
    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 10;

    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    status = index->search(nq, xq, k, distances.data(), result_labels.data());
    CATCH_REQUIRE(status.ok());

    svs::runtime::v0::DynamicIVFIndex::destroy(index);
}

CATCH_TEST_CASE("IVFIndexBuildAndSearchLeanVec", "[runtime][ivf]") {
    std::cout << "[IVF] Running IVFIndexBuildAndSearchLeanVec..." << std::endl;
    const auto& test_data = get_test_data();

    svs::runtime::v0::IVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    svs::runtime::v0::Status status = svs::runtime::v0::IVFIndexLeanVec::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::LeanVec4x4,
        test_n,
        test_data.data(),
        32, // leanvec_dims
        build_params
    );

    if (!svs::runtime::v0::IVFIndex::check_storage_kind(
             svs::runtime::v0::StorageKind::LeanVec4x4
        )
             .ok()) {
        CATCH_REQUIRE(!status.ok());
        CATCH_SKIP("LeanVec storage kind is not supported in this build configuration.");
    }
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Search
    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 10;

    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    status = index->search(nq, xq, k, distances.data(), result_labels.data());
    CATCH_REQUIRE(status.ok());

    svs::runtime::v0::IVFIndex::destroy(index);
}

CATCH_TEST_CASE("IVFIndexWriteAndReadLeanVec", "[runtime][ivf]") {
    std::cout << "[IVF] Running IVFIndexWriteAndReadLeanVec..." << std::endl;
    const auto& test_data = get_test_data();

    svs::runtime::v0::IVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    svs::runtime::v0::Status status = svs::runtime::v0::IVFIndexLeanVec::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::LeanVec4x4,
        test_n,
        test_data.data(),
        32, // leanvec_dims
        build_params
    );

    if (!svs::runtime::v0::IVFIndex::check_storage_kind(
             svs::runtime::v0::StorageKind::LeanVec4x4
        )
             .ok()) {
        CATCH_REQUIRE(!status.ok());
        CATCH_SKIP("LeanVec storage kind is not supported in this build configuration.");
    }
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    svs_test::prepare_temp_directory();
    auto temp_dir = svs_test::temp_directory();
    auto filename = temp_dir / "ivf_leanvec_test.bin";

    // Serialize
    std::ofstream out(filename, std::ios::binary);
    CATCH_REQUIRE(out.is_open());
    status = index->save(out);
    CATCH_REQUIRE(status.ok());
    out.close();

    // Deserialize
    svs::runtime::v0::IVFIndex* loaded = nullptr;
    std::ifstream in(filename, std::ios::binary);
    CATCH_REQUIRE(in.is_open());
    status = svs::runtime::v0::IVFIndex::load(
        &loaded,
        in,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::LeanVec4x4
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(loaded != nullptr);
    in.close();

    // Search loaded index
    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 10;

    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    status = loaded->search(nq, xq, k, distances.data(), result_labels.data());
    CATCH_REQUIRE(status.ok());

    svs::runtime::v0::IVFIndex::destroy(index);
    svs::runtime::v0::IVFIndex::destroy(loaded);
}

CATCH_TEST_CASE("DynamicIVFIndexBuildAndSearchLeanVec", "[runtime][ivf]") {
    std::cout << "[IVF] Running DynamicIVFIndexBuildAndSearchLeanVec..." << std::endl;
    const auto& test_data = get_test_data();

    svs::runtime::v0::DynamicIVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    std::vector<size_t> labels(test_n);
    std::iota(labels.begin(), labels.end(), 0);

    svs::runtime::v0::Status status = svs::runtime::v0::DynamicIVFIndexLeanVec::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::LeanVec4x4,
        test_n,
        test_data.data(),
        labels.data(),
        32, // leanvec_dims
        build_params
    );

    if (!svs::runtime::v0::DynamicIVFIndex::check_storage_kind(
             svs::runtime::v0::StorageKind::LeanVec4x4
        )
             .ok()) {
        CATCH_REQUIRE(!status.ok());
        CATCH_SKIP("LeanVec storage kind is not supported in this build configuration.");
    }
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Search
    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 10;

    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    status = index->search(nq, xq, k, distances.data(), result_labels.data());
    CATCH_REQUIRE(status.ok());

    svs::runtime::v0::DynamicIVFIndex::destroy(index);
}

CATCH_TEST_CASE("IVFIndexInnerProduct", "[runtime][ivf]") {
    std::cout << "[IVF] Running IVFIndexInnerProduct..." << std::endl;
    const auto& test_data = get_test_data();

    // Build static IVF index with inner product metric
    svs::runtime::v0::IVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    svs::runtime::v0::Status status = svs::runtime::v0::IVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::INNER_PRODUCT,
        svs::runtime::v0::StorageKind::FP32,
        test_n,
        test_data.data(),
        build_params
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Search
    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 10;

    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    status = index->search(nq, xq, k, distances.data(), result_labels.data());
    CATCH_REQUIRE(status.ok());

    svs::runtime::v0::IVFIndex::destroy(index);
}

CATCH_TEST_CASE("IVFIndexSetIntraQueryThreads", "[runtime][ivf]") {
    std::cout << "[IVF] Running IVFIndexSetIntraQueryThreads..." << std::endl;
    const auto& test_data = get_test_data();

    svs::runtime::v0::IVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    svs::runtime::v0::Status status = svs::runtime::v0::IVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        test_n,
        test_data.data(),
        build_params,
        /*num_threads=*/2,
        /*intra_query_threads=*/1
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    size_t got = 0;
    status = index->get_intra_query_threads(&got);
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(got == 1);

    status = index->set_intra_query_threads(3);
    CATCH_REQUIRE(status.ok());

    status = index->get_intra_query_threads(&got);
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(got == 3);

    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 10;
    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);
    status = index->search(nq, xq, k, distances.data(), result_labels.data());
    CATCH_REQUIRE(status.ok());

    status = index->set_intra_query_threads(0);
    CATCH_REQUIRE(!status.ok());

    svs::runtime::v0::IVFIndex::destroy(index);
}

CATCH_TEST_CASE("IVFIndexFilteredSearchWithoutBatchEstimate", "[runtime][ivf]") {
    std::cout << "[IVF] Running IVFIndexFilteredSearchWithoutBatchEstimate..." << std::endl;
    const auto& test_data = get_test_data();

    svs::runtime::v0::IVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    svs::runtime::v0::Status status = svs::runtime::v0::IVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        test_n,
        test_data.data(),
        build_params
    );
    CATCH_REQUIRE(status.ok());

    const size_t min_id = 10;
    const size_t max_id = 50;
    test_utils::IDFilterRange filter(min_id, max_id);

    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 5;
    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    // Test with filter_estimate_batch=false
    svs::runtime::v0::IVFIndex::SearchParams sp;
    sp.n_probes = 10;
    sp.filter_estimate_batch = false;
    status = index->search(nq, xq, k, distances.data(), result_labels.data(), &sp, &filter);
    CATCH_REQUIRE(status.ok());

    // Verify all returned IDs are within filter range
    for (int q = 0; q < nq; ++q) {
        for (int j = 0; j < k; ++j) {
            size_t id = result_labels[q * k + j];
            if (id < test_n) {
                CATCH_REQUIRE(id >= min_id);
                CATCH_REQUIRE(id < max_id);
            }
        }
    }

    svs::runtime::v0::IVFIndex::destroy(index);
}

CATCH_TEST_CASE("IVFIndexIntraQueryThreadsConsistency", "[runtime][ivf]") {
    std::cout << "[IVF] Running IVFIndexIntraQueryThreadsConsistency..." << std::endl;
    const auto& test_data = get_test_data();

    svs::runtime::v0::IVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    svs::runtime::v0::Status status = svs::runtime::v0::IVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        test_n,
        test_data.data(),
        build_params,
        /*num_threads=*/2,
        /*intra_query_threads=*/1
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 10;

    // Run search with intra_query_threads=1
    std::vector<float> distances1(nq * k);
    std::vector<size_t> labels1(nq * k);
    status = index->search(nq, xq, k, distances1.data(), labels1.data());
    CATCH_REQUIRE(status.ok());

    // Change to intra_query_threads=2 and search again
    status = index->set_intra_query_threads(2);
    CATCH_REQUIRE(status.ok());
    size_t got = 0;
    status = index->get_intra_query_threads(&got);
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(got == 2);

    std::vector<float> distances2(nq * k);
    std::vector<size_t> labels2(nq * k);
    status = index->search(nq, xq, k, distances2.data(), labels2.data());
    CATCH_REQUIRE(status.ok());

    // Change to intra_query_threads=4 and search again
    status = index->set_intra_query_threads(4);
    CATCH_REQUIRE(status.ok());
    status = index->get_intra_query_threads(&got);
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(got == 4);

    std::vector<float> distances4(nq * k);
    std::vector<size_t> labels4(nq * k);
    status = index->search(nq, xq, k, distances4.data(), labels4.data());
    CATCH_REQUIRE(status.ok());

    // Results should be identical regardless of thread count
    // (same neighbors found, possibly different distance rounding)
    for (int q = 0; q < nq; ++q) {
        for (int j = 0; j < k; ++j) {
            size_t idx = q * k + j;
            CATCH_REQUIRE(labels1[idx] == labels2[idx]);
            CATCH_REQUIRE(labels1[idx] == labels4[idx]);
            // Distances might have slight floating-point differences
            CATCH_REQUIRE(std::abs(distances1[idx] - distances2[idx]) < 1e-4);
            CATCH_REQUIRE(std::abs(distances1[idx] - distances4[idx]) < 1e-4);
        }
    }

    svs::runtime::v0::IVFIndex::destroy(index);
}

CATCH_TEST_CASE("DynamicIVFIndexSetIntraQueryThreads", "[runtime][ivf]") {
    std::cout << "[IVF] Running DynamicIVFIndexSetIntraQueryThreads..." << std::endl;
    const auto& test_data = get_test_data();

    svs::runtime::v0::DynamicIVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    std::vector<size_t> labels(test_n);
    std::iota(labels.begin(), labels.end(), 0);

    svs::runtime::v0::Status status = svs::runtime::v0::DynamicIVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        test_n,
        test_data.data(),
        labels.data(),
        build_params,
        /*default_search_params=*/{},
        /*num_threads=*/2,
        /*intra_query_threads=*/1
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    size_t got = 0;
    status = index->get_intra_query_threads(&got);
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(got == 1);

    status = index->set_intra_query_threads(2);
    CATCH_REQUIRE(status.ok());

    status = index->get_intra_query_threads(&got);
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(got == 2);

    svs::runtime::v0::DynamicIVFIndex::destroy(index);
}

CATCH_TEST_CASE("IVFIndexSearchWithIDFilter", "[runtime][ivf]") {
    std::cout << "[IVF] Running IVFIndexSearchWithIDFilter..." << std::endl;
    const auto& test_data = get_test_data();

    svs::runtime::v0::IVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    svs::runtime::v0::Status status = svs::runtime::v0::IVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        test_n,
        test_data.data(),
        build_params
    );
    CATCH_REQUIRE(status.ok());

    const size_t min_id = 10;
    const size_t max_id = 50;
    test_utils::IDFilterRange filter(min_id, max_id);

    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 5;
    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    svs::runtime::v0::IVFIndex::SearchParams sp;
    sp.n_probes = 10;
    status = index->search(nq, xq, k, distances.data(), result_labels.data(), &sp, &filter);
    CATCH_REQUIRE(status.ok());

    for (int q = 0; q < nq; ++q) {
        for (int j = 0; j < k; ++j) {
            size_t id = result_labels[q * k + j];
            if (id < test_n) {
                CATCH_REQUIRE(id >= min_id);
                CATCH_REQUIRE(id < max_id);
            }
        }
    }

    svs::runtime::v0::IVFIndex::destroy(index);
}

CATCH_TEST_CASE("DynamicIVFIndexSearchWithIDFilter", "[runtime][ivf]") {
    std::cout << "[IVF] Running DynamicIVFIndexSearchWithIDFilter..." << std::endl;
    const auto& test_data = get_test_data();

    svs::runtime::v0::DynamicIVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    std::vector<size_t> labels(test_n);
    std::iota(labels.begin(), labels.end(), 0);

    svs::runtime::v0::Status status = svs::runtime::v0::DynamicIVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        test_n,
        test_data.data(),
        labels.data(),
        build_params
    );
    CATCH_REQUIRE(status.ok());

    const size_t min_id = 20;
    const size_t max_id = 60;
    test_utils::IDFilterRange filter(min_id, max_id);

    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 5;
    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    svs::runtime::v0::IVFIndex::SearchParams sp;
    sp.n_probes = 10;
    status = index->search(nq, xq, k, distances.data(), result_labels.data(), &sp, &filter);
    CATCH_REQUIRE(status.ok());

    for (int q = 0; q < nq; ++q) {
        for (int j = 0; j < k; ++j) {
            size_t id = result_labels[q * k + j];
            if (id < test_n) {
                CATCH_REQUIRE(id >= min_id);
                CATCH_REQUIRE(id < max_id);
            }
        }
    }

    svs::runtime::v0::DynamicIVFIndex::destroy(index);
}

CATCH_TEST_CASE("IVFIndexSearchWithRestrictiveFilter", "[runtime][ivf][filtered_search]") {
    std::cout << "[IVF] Running IVFIndexSearchWithRestrictiveFilter..." << std::endl;
    const auto& test_data = get_test_data();

    svs::runtime::v0::IVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    svs::runtime::v0::Status status = svs::runtime::v0::IVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        test_n,
        test_data.data(),
        build_params
    );
    CATCH_REQUIRE(status.ok());

    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 5;

    // 10% selectivity: accept only IDs 0-9 out of 100
    const size_t min_id = 0;
    const size_t max_id = test_n / 10;
    test_utils::IDFilterRange filter(min_id, max_id);

    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    svs::runtime::v0::IVFIndex::SearchParams sp;
    sp.n_probes = 10;
    status = index->search(nq, xq, k, distances.data(), result_labels.data(), &sp, &filter);
    CATCH_REQUIRE(status.ok());

    for (int i = 0; i < nq * k; ++i) {
        if (svs::runtime::v0::is_specified(result_labels[i])) {
            CATCH_REQUIRE(result_labels[i] >= min_id);
            CATCH_REQUIRE(result_labels[i] < max_id);
        }
    }

    svs::runtime::v0::IVFIndex::destroy(index);
}

CATCH_TEST_CASE("IVFIndexFilterStopEarlyExit", "[runtime][ivf][filtered_search]") {
    std::cout << "[IVF] Running IVFIndexFilterStopEarlyExit..." << std::endl;
    const auto& test_data = get_test_data();

    svs::runtime::v0::IVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    svs::runtime::v0::Status status = svs::runtime::v0::IVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        test_n,
        test_data.data(),
        build_params
    );
    CATCH_REQUIRE(status.ok());

    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 5;

    // 10% selectivity: accept only IDs 0-9 out of 100
    const size_t min_id = 0;
    const size_t max_id = test_n / 10;
    test_utils::IDFilterRange filter(min_id, max_id);

    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    // Set filter_stop = 0.5 (50%). With ~10% hit rate, search should give up
    // and return unspecified results.
    svs::runtime::v0::IVFIndex::SearchParams sp;
    sp.n_probes = 10;
    sp.filter_stop = 0.5f;

    status = index->search(nq, xq, k, distances.data(), result_labels.data(), &sp, &filter);
    CATCH_REQUIRE(status.ok());

    for (int i = 0; i < nq * k; ++i) {
        CATCH_REQUIRE(!svs::runtime::v0::is_specified(result_labels[i]));
    }

    // Now search without filter_stop — should find valid results in the filter range.
    std::vector<float> distances2(nq * k);
    std::vector<size_t> result_labels2(nq * k);

    svs::runtime::v0::IVFIndex::SearchParams sp_no_stop;
    sp_no_stop.n_probes = 10;
    status = index->search(
        nq, xq, k, distances2.data(), result_labels2.data(), &sp_no_stop, &filter
    );
    CATCH_REQUIRE(status.ok());

    for (int i = 0; i < nq * k; ++i) {
        if (svs::runtime::v0::is_specified(result_labels2[i])) {
            CATCH_REQUIRE(result_labels2[i] >= min_id);
            CATCH_REQUIRE(result_labels2[i] < max_id);
        }
    }

    svs::runtime::v0::IVFIndex::destroy(index);
}

CATCH_TEST_CASE(
    "DynamicIVFIndexSearchWithRestrictiveFilter", "[runtime][ivf][filtered_search]"
) {
    std::cout << "[IVF] Running DynamicIVFIndexSearchWithRestrictiveFilter..." << std::endl;
    const auto& test_data = get_test_data();

    svs::runtime::v0::DynamicIVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    std::vector<size_t> labels(test_n);
    std::iota(labels.begin(), labels.end(), 0);

    svs::runtime::v0::Status status = svs::runtime::v0::DynamicIVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        test_n,
        test_data.data(),
        labels.data(),
        build_params
    );
    CATCH_REQUIRE(status.ok());

    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 5;

    // 10% selectivity: accept only IDs 0-9 out of 100
    const size_t min_id = 0;
    const size_t max_id = test_n / 10;
    test_utils::IDFilterRange filter(min_id, max_id);

    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    svs::runtime::v0::IVFIndex::SearchParams sp;
    sp.n_probes = 10;
    status = index->search(nq, xq, k, distances.data(), result_labels.data(), &sp, &filter);
    CATCH_REQUIRE(status.ok());

    for (int i = 0; i < nq * k; ++i) {
        if (svs::runtime::v0::is_specified(result_labels[i])) {
            CATCH_REQUIRE(result_labels[i] >= min_id);
            CATCH_REQUIRE(result_labels[i] < max_id);
        }
    }

    svs::runtime::v0::DynamicIVFIndex::destroy(index);
}

CATCH_TEST_CASE("DynamicIVFIndexFilterStopEarlyExit", "[runtime][ivf][filtered_search]") {
    std::cout << "[IVF] Running DynamicIVFIndexFilterStopEarlyExit..." << std::endl;
    const auto& test_data = get_test_data();

    svs::runtime::v0::DynamicIVFIndex* index = nullptr;
    svs::runtime::v0::IVFIndex::BuildParams build_params;
    build_params.num_centroids = 10;
    build_params.num_iterations = 5;

    std::vector<size_t> labels(test_n);
    std::iota(labels.begin(), labels.end(), 0);

    svs::runtime::v0::Status status = svs::runtime::v0::DynamicIVFIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        test_n,
        test_data.data(),
        labels.data(),
        build_params
    );
    CATCH_REQUIRE(status.ok());

    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 5;

    // 10% selectivity: accept only IDs 0-9 out of 100
    const size_t min_id = 0;
    const size_t max_id = test_n / 10;
    test_utils::IDFilterRange filter(min_id, max_id);

    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    // Set filter_stop = 0.5 (50%). With ~10% hit rate, search should give up
    // and return unspecified results.
    svs::runtime::v0::IVFIndex::SearchParams sp;
    sp.n_probes = 10;
    sp.filter_stop = 0.5f;

    status = index->search(nq, xq, k, distances.data(), result_labels.data(), &sp, &filter);
    CATCH_REQUIRE(status.ok());

    for (int i = 0; i < nq * k; ++i) {
        CATCH_REQUIRE(!svs::runtime::v0::is_specified(result_labels[i]));
    }

    // Now search without filter_stop — should find valid results in the filter range.
    std::vector<float> distances2(nq * k);
    std::vector<size_t> result_labels2(nq * k);

    svs::runtime::v0::IVFIndex::SearchParams sp_no_stop;
    sp_no_stop.n_probes = 10;
    status = index->search(
        nq, xq, k, distances2.data(), result_labels2.data(), &sp_no_stop, &filter
    );
    CATCH_REQUIRE(status.ok());

    for (int i = 0; i < nq * k; ++i) {
        if (svs::runtime::v0::is_specified(result_labels2[i])) {
            CATCH_REQUIRE(result_labels2[i] >= min_id);
            CATCH_REQUIRE(result_labels2[i] < max_id);
        }
    }

    svs::runtime::v0::DynamicIVFIndex::destroy(index);
}
