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

#include "svs/runtime/api_defs.h"
#include "svs/runtime/dynamic_vamana_index.h"
#include "svs/runtime/flat_index.h"
#include "svs/runtime/training.h"
#include "svs/runtime/vamana_index.h"

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

// For memory tracking
#include <fstream>
#include <sys/resource.h>
#include <unistd.h>

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

size_t get_current_rss() {
    std::ifstream statm("/proc/self/statm");
    if (!statm.is_open()) {
        return 0;
    }
    size_t vsize, rss;
    statm >> vsize >> rss;
    size_t page_size = sysconf(_SC_PAGESIZE);
    return rss * page_size;
}

struct UsageInfo {
    size_t file_size;
    size_t rss_increase;
};

} // namespace

// Template function to write and read an index
template <typename BuildFunc>
void write_and_read_index(
    BuildFunc build_func,
    const std::vector<float>& xb,
    size_t n,
    size_t d,
    svs::runtime::v0::StorageKind storage_kind,
    svs::runtime::v0::MetricType metric = svs::runtime::v0::MetricType::L2
) {
    // Build index
    svs::runtime::v0::DynamicVamanaIndex* index = nullptr;
    svs::runtime::v0::Status status = build_func(&index);
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Add data to index
    std::vector<size_t> labels(n);
    std::iota(labels.begin(), labels.end(), 0);

    status = index->add(n, labels.data(), xb.data());
    CATCH_REQUIRE(status.ok());

    svs_test::prepare_temp_directory();
    auto temp_dir = svs_test::temp_directory();
    auto filename = temp_dir / "index_test.bin";

    // Serialize
    std::ofstream out(filename, std::ios::binary);
    CATCH_REQUIRE(out.is_open());
    status = index->save(out);
    CATCH_REQUIRE(status.ok());
    out.close();

    // Deserialize
    svs::runtime::v0::DynamicVamanaIndex* loaded = nullptr;
    std::ifstream in(filename, std::ios::binary);
    CATCH_REQUIRE(in.is_open());

    status = svs::runtime::v0::DynamicVamanaIndex::load(&loaded, in, metric, storage_kind);
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(loaded != nullptr);
    in.close();

    // Test basic functionality of loaded index
    const int nq = 5;
    const float* xq = xb.data();
    const int k = 10;

    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    status = loaded->search(nq, xq, k, distances.data(), result_labels.data());
    CATCH_REQUIRE(status.ok());

    // Clean up
    svs::runtime::v0::DynamicVamanaIndex::destroy(index);
    svs::runtime::v0::DynamicVamanaIndex::destroy(loaded);
}

// Helper that writes and reads and index of requested size
// Reports memory usage
UsageInfo run_save_and_load_test(const size_t target_mibytes) {
    // Generate requested MiB of test data
    constexpr size_t mem_test_d = 128;
    const size_t target_bytes = target_mibytes * 1024 * 1024;
    const size_t mem_test_n = target_bytes / (mem_test_d * sizeof(float));

    svs_test::prepare_temp_directory();
    auto temp_dir = svs_test::temp_directory();
    auto filename = temp_dir / "memory_test_index.bin";

    {
        // Build Vamana FP32 index, scoped for memory cleanup
        auto large_test_data = create_test_data(mem_test_n, mem_test_d, 456);

        // Add data to index
        std::vector<size_t> labels(mem_test_n);
        std::iota(labels.begin(), labels.end(), 0);

        size_t mem_before = get_current_rss();
        svs::runtime::v0::DynamicVamanaIndex* index = nullptr;
        svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
        svs::runtime::v0::Status status = svs::runtime::v0::DynamicVamanaIndex::build(
            &index,
            mem_test_d,
            svs::runtime::v0::MetricType::L2,
            svs::runtime::v0::StorageKind::FP32,
            build_params
        );
        CATCH_REQUIRE(status.ok());
        CATCH_REQUIRE(index != nullptr);
        status = index->add(mem_test_n, labels.data(), large_test_data.data());
        CATCH_REQUIRE(status.ok());

        std::ofstream out(filename, std::ios::binary);
        CATCH_REQUIRE(out.is_open());
        status = index->save(out);
        CATCH_REQUIRE(status.ok());
        out.close();

        svs::runtime::v0::DynamicVamanaIndex::destroy(index);
        index = nullptr;
    }

    // Investigate the file size on disk
    size_t file_size = std::filesystem::file_size(filename);

    // Load the index from disk
    std::ifstream in(filename, std::ios::binary);
    CATCH_REQUIRE(in.is_open());

    // Monitor RSS increase
    size_t rss_before = get_current_rss();

    svs::runtime::v0::DynamicVamanaIndex* loaded = nullptr;
    auto status = svs::runtime::v0::DynamicVamanaIndex::load(
        &loaded, in, svs::runtime::v0::MetricType::L2, svs::runtime::v0::StorageKind::FP32
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(loaded != nullptr);
    in.close();

    size_t rss_delta = get_current_rss() - rss_before;

    // Clean up
    svs::runtime::v0::DynamicVamanaIndex::destroy(loaded);
    loaded = nullptr;

    return {file_size, rss_delta};
}

CATCH_TEST_CASE("WriteAndReadIndexSVS", "[runtime]") {
    const auto& test_data = get_test_data();
    auto build_func = [](svs::runtime::v0::DynamicVamanaIndex** index) {
        svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
        return svs::runtime::v0::DynamicVamanaIndex::build(
            index,
            test_d,
            svs::runtime::v0::MetricType::L2,
            svs::runtime::v0::StorageKind::FP32,
            build_params
        );
    };
    write_and_read_index(
        build_func, test_data, test_n, test_d, svs::runtime::v0::StorageKind::FP32
    );
}

CATCH_TEST_CASE("WriteAndReadIndexSVSFP16", "[runtime]") {
    const auto& test_data = get_test_data();
    auto build_func = [](svs::runtime::v0::DynamicVamanaIndex** index) {
        svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
        return svs::runtime::v0::DynamicVamanaIndex::build(
            index,
            test_d,
            svs::runtime::v0::MetricType::L2,
            svs::runtime::v0::StorageKind::FP16,
            build_params
        );
    };
    write_and_read_index(
        build_func, test_data, test_n, test_d, svs::runtime::v0::StorageKind::FP16
    );
}

CATCH_TEST_CASE("WriteAndReadIndexSVSSQI8", "[runtime]") {
    const auto& test_data = get_test_data();
    auto build_func = [](svs::runtime::v0::DynamicVamanaIndex** index) {
        svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
        return svs::runtime::v0::DynamicVamanaIndex::build(
            index,
            test_d,
            svs::runtime::v0::MetricType::L2,
            svs::runtime::v0::StorageKind::SQI8,
            build_params
        );
    };
    write_and_read_index(
        build_func, test_data, test_n, test_d, svs::runtime::v0::StorageKind::SQI8
    );
}

CATCH_TEST_CASE("WriteAndReadIndexSVSLVQ4x4", "[runtime]") {
    const auto& test_data = get_test_data();
    auto build_func = [](svs::runtime::v0::DynamicVamanaIndex** index) {
        svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
        return svs::runtime::v0::DynamicVamanaIndex::build(
            index,
            test_d,
            svs::runtime::v0::MetricType::L2,
            svs::runtime::v0::StorageKind::LVQ4x4,
            build_params
        );
    };
    write_and_read_index(
        build_func, test_data, test_n, test_d, svs::runtime::v0::StorageKind::LVQ4x4
    );
}

CATCH_TEST_CASE("WriteAndReadIndexSVSVamanaLeanVec4x4", "[runtime]") {
    const auto& test_data = get_test_data();
    auto build_func = [](svs::runtime::v0::DynamicVamanaIndex** index) {
        svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
        return svs::runtime::v0::DynamicVamanaIndexLeanVec::build(
            index,
            test_d,
            svs::runtime::v0::MetricType::L2,
            svs::runtime::v0::StorageKind::LeanVec4x4,
            32,
            build_params
        );
    };
    write_and_read_index(
        build_func, test_data, test_n, test_d, svs::runtime::v0::StorageKind::LeanVec4x4
    );
}

CATCH_TEST_CASE("LeanVecWithTrainingData", "[runtime]") {
    const auto& test_data = get_test_data();
    // Build LeanVec index with explicit training
    svs::runtime::v0::DynamicVamanaIndex* index = nullptr;
    svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
    svs::runtime::v0::Status status = svs::runtime::v0::DynamicVamanaIndexLeanVec::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::LeanVec4x4,
        32,
        build_params
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Add data - should work with provided leanvec dims
    std::vector<size_t> labels(test_n);
    std::iota(labels.begin(), labels.end(), 0);

    status = index->add(test_n, labels.data(), test_data.data());
    CATCH_REQUIRE(status.ok());

    svs::runtime::v0::DynamicVamanaIndex::destroy(index);
}

CATCH_TEST_CASE("FlatIndexWriteAndRead", "[runtime]") {
    const auto& test_data = get_test_data();
    svs::runtime::v0::FlatIndex* index = nullptr;
    svs::runtime::v0::Status status = svs::runtime::v0::FlatIndex::build(
        &index, test_d, svs::runtime::v0::MetricType::L2
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Add data
    status = index->add(test_n, test_data.data());
    CATCH_REQUIRE(status.ok());

    svs_test::prepare_temp_directory();
    auto temp_dir = svs_test::temp_directory();
    auto filename = temp_dir / "flat_index_test.bin";

    // Serialize
    std::ofstream out(filename, std::ios::binary);
    CATCH_REQUIRE(out.is_open());
    status = index->save(out);
    CATCH_REQUIRE(status.ok());
    out.close();

    // Deserialize
    svs::runtime::v0::FlatIndex* loaded = nullptr;
    std::ifstream in(filename, std::ios::binary);
    CATCH_REQUIRE(in.is_open());

    status =
        svs::runtime::v0::FlatIndex::load(&loaded, in, svs::runtime::v0::MetricType::L2);
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(loaded != nullptr);
    in.close();

    // Test search
    const int nq = 5;
    const float* xq = test_data.data();
    const int k = 10;

    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    status = loaded->search(nq, xq, k, distances.data(), result_labels.data());
    CATCH_REQUIRE(status.ok());

    // Clean up
    svs::runtime::v0::FlatIndex::destroy(index);
    svs::runtime::v0::FlatIndex::destroy(loaded);
}

CATCH_TEST_CASE("SearchWithIDFilter", "[runtime]") {
    const auto& test_data = get_test_data();
    // Build index
    svs::runtime::v0::DynamicVamanaIndex* index = nullptr;
    svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
    svs::runtime::v0::Status status = svs::runtime::v0::DynamicVamanaIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        build_params
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Add data
    std::vector<size_t> labels(test_n);
    std::iota(labels.begin(), labels.end(), 0);
    status = index->add(test_n, labels.data(), test_data.data());
    CATCH_REQUIRE(status.ok());

    const int nq = 8;
    const float* xq = test_data.data();
    const int k = 10;

    size_t min_id = test_n / 5;
    size_t max_id = test_n * 4 / 5;
    test_utils::IDFilterRange selector(min_id, max_id);

    std::vector<float> distances(nq * k);
    std::vector<size_t> result_labels(nq * k);

    status = index->search(
        nq, xq, k, distances.data(), result_labels.data(), nullptr, &selector
    );
    CATCH_REQUIRE(status.ok());

    // All returned labels must fall inside the selected range
    for (int i = 0; i < nq * k; ++i) {
        CATCH_REQUIRE(result_labels[i] >= min_id);
        CATCH_REQUIRE(result_labels[i] < max_id);
    }

    svs::runtime::v0::DynamicVamanaIndex::destroy(index);
}

CATCH_TEST_CASE("RangeSearchFunctional", "[runtime]") {
    const auto& test_data = get_test_data();
    // Build index
    svs::runtime::v0::DynamicVamanaIndex* index = nullptr;
    svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
    svs::runtime::v0::Status status = svs::runtime::v0::DynamicVamanaIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        build_params
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Add data
    std::vector<size_t> labels(test_n);
    std::iota(labels.begin(), labels.end(), 0);
    status = index->add(test_n, labels.data(), test_data.data());
    CATCH_REQUIRE(status.ok());

    const int nq = 5;
    const float* xq = test_data.data();

    // Small radius search
    test_utils::TestResultsAllocator allocator_small;
    status = index->range_search(nq, xq, 0.05f, allocator_small);
    CATCH_REQUIRE(status.ok());

    // Larger radius to exercise loop continuation
    test_utils::TestResultsAllocator allocator_big;
    status = index->range_search(nq, xq, 5.0f, allocator_big);
    CATCH_REQUIRE(status.ok());

    svs::runtime::v0::DynamicVamanaIndex::destroy(index);
}

CATCH_TEST_CASE("MemoryUsageOnLoad", "[runtime][memory]") {
    CATCH_SECTION("SmallIndex") {
        auto stats = run_save_and_load_test(10);
        CATCH_REQUIRE(stats.file_size < 20 * 1024 * 1024);
        CATCH_REQUIRE(stats.rss_increase < 1.2 * stats.file_size);
    }

    CATCH_SECTION("MediumIndex") {
        auto stats = run_save_and_load_test(50);
        CATCH_REQUIRE(stats.file_size < 100 * 1024 * 1024);
        CATCH_REQUIRE(stats.rss_increase < 1.2 * stats.file_size);
    }

    CATCH_SECTION("LargeIndex") {
        auto stats = run_save_and_load_test(200);
        CATCH_REQUIRE(stats.file_size < 400 * 1024 * 1024);
        CATCH_REQUIRE(stats.rss_increase < 1.2 * stats.file_size);
    }
}
