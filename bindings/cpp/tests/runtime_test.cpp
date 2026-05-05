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
#include <optional>
#include <sys/resource.h>
#include <type_traits>
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
template <typename Index, typename BuildFunc>
void write_and_read_index(
    BuildFunc build_func,
    const std::vector<float>& xb,
    size_t n,
    size_t d,
    std::optional<svs::runtime::v0::StorageKind> storage_kind = std::nullopt,
    svs::runtime::v0::MetricType metric = svs::runtime::v0::MetricType::L2
) {
    // Build index
    Index* index = nullptr;
    svs::runtime::v0::Status status = build_func(&index);

    // Stop here if storage kind is not supported on this platform
    if constexpr (std::is_base_of_v<svs::runtime::v0::VamanaIndex, Index>) {
        if (storage_kind.has_value()) {
            if (!Index::check_storage_kind(*storage_kind).ok()) {
                CATCH_REQUIRE(!status.ok());
                return;
            }
        }
    }
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Add data to index
    if constexpr (std::is_same_v<Index, svs::runtime::v0::FlatIndex> || std::is_same_v<Index, svs::runtime::v0::VamanaIndex>) {
        status = index->add(n, xb.data());
    } else {
        std::vector<size_t> labels(n);
        std::iota(labels.begin(), labels.end(), 0);
        status = index->add(n, labels.data(), xb.data());
    }
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
    Index* loaded = nullptr;
    std::ifstream in(filename, std::ios::binary);
    CATCH_REQUIRE(in.is_open());

    if constexpr (std::is_same_v<Index, svs::runtime::v0::FlatIndex>) {
        status = Index::load(&loaded, in, metric);
    } else {
        CATCH_REQUIRE(storage_kind.has_value());
        status = Index::load(&loaded, in, metric, *storage_kind);
    }
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
    Index::destroy(index);
    Index::destroy(loaded);
}

// Helper that writes and reads and index of requested size
// Reports memory usage
UsageInfo run_save_and_load_test(
    const size_t target_mibytes, size_t d, size_t graph_max_degree, size_t blocksize_exp
) {
    // Generate requested MiB of test data
    const size_t target_bytes = target_mibytes * 1024 * 1024;
    const size_t mem_test_n = target_bytes / (d * sizeof(float));

    svs_test::prepare_temp_directory();
    auto temp_dir = svs_test::temp_directory();
    auto filename = temp_dir / "memory_test_index.bin";

    {
        // Build Vamana FP32 index, scoped for memory cleanup
        auto large_test_data = create_test_data(mem_test_n, d, 456);

        // Add data to index
        std::vector<size_t> labels(mem_test_n);
        std::iota(labels.begin(), labels.end(), 0);

        size_t mem_before = get_current_rss();
        svs::runtime::v0::DynamicVamanaIndex* index = nullptr;
        svs::runtime::v0::VamanaIndex::BuildParams build_params{graph_max_degree};
        svs::runtime::v0::VamanaIndex::DynamicIndexParams dynamic_index_params{
            blocksize_exp};
        svs::runtime::v0::Status status = svs::runtime::v0::DynamicVamanaIndex::build(
            &index,
            d,
            svs::runtime::v0::MetricType::L2,
            svs::runtime::v0::StorageKind::FP32,
            build_params,
            {},
            dynamic_index_params
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
        svs::runtime::v0::VamanaIndex::DynamicIndexParams dynamic_index_params{15};
        return svs::runtime::v0::DynamicVamanaIndex::build(
            index,
            test_d,
            svs::runtime::v0::MetricType::L2,
            svs::runtime::v0::StorageKind::FP32,
            build_params,
            {},
            dynamic_index_params
        );
    };
    write_and_read_index<svs::runtime::v0::DynamicVamanaIndex>(
        build_func, test_data, test_n, test_d, svs::runtime::v0::StorageKind::FP32
    );
}

CATCH_TEST_CASE("WriteAndReadIndexSVSFP16", "[runtime]") {
    const auto& test_data = get_test_data();
    auto build_func = [](svs::runtime::v0::DynamicVamanaIndex** index) {
        svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
        svs::runtime::v0::VamanaIndex::DynamicIndexParams dynamic_index_params{16};
        return svs::runtime::v0::DynamicVamanaIndex::build(
            index,
            test_d,
            svs::runtime::v0::MetricType::L2,
            svs::runtime::v0::StorageKind::FP16,
            build_params,
            {},
            dynamic_index_params
        );
    };
    write_and_read_index<svs::runtime::v0::DynamicVamanaIndex>(
        build_func, test_data, test_n, test_d, svs::runtime::v0::StorageKind::FP16
    );
}

CATCH_TEST_CASE("WriteAndReadIndexSVSSQI8", "[runtime]") {
    const auto& test_data = get_test_data();
    auto build_func = [](svs::runtime::v0::DynamicVamanaIndex** index) {
        svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
        svs::runtime::v0::VamanaIndex::DynamicIndexParams dynamic_index_params{17};
        return svs::runtime::v0::DynamicVamanaIndex::build(
            index,
            test_d,
            svs::runtime::v0::MetricType::L2,
            svs::runtime::v0::StorageKind::SQI8,
            build_params,
            {},
            dynamic_index_params
        );
    };
    write_and_read_index<svs::runtime::v0::DynamicVamanaIndex>(
        build_func, test_data, test_n, test_d, svs::runtime::v0::StorageKind::SQI8
    );
}

CATCH_TEST_CASE("WriteAndReadIndexSVSLVQ4x4", "[runtime]") {
    const auto& test_data = get_test_data();
    auto build_func = [](svs::runtime::v0::DynamicVamanaIndex** index) {
        svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
        svs::runtime::v0::VamanaIndex::DynamicIndexParams dynamic_index_params{18};
        return svs::runtime::v0::DynamicVamanaIndex::build(
            index,
            test_d,
            svs::runtime::v0::MetricType::L2,
            svs::runtime::v0::StorageKind::LVQ4x4,
            build_params,
            {},
            dynamic_index_params
        );
    };
    write_and_read_index<svs::runtime::v0::DynamicVamanaIndex>(
        build_func, test_data, test_n, test_d, svs::runtime::v0::StorageKind::LVQ4x4
    );
}

CATCH_TEST_CASE("WriteAndReadIndexSVSVamanaLeanVec4x4", "[runtime]") {
    const auto& test_data = get_test_data();
    auto build_func = [](svs::runtime::v0::DynamicVamanaIndex** index) {
        svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
        svs::runtime::v0::VamanaIndex::DynamicIndexParams dynamic_index_params{19};
        return svs::runtime::v0::DynamicVamanaIndexLeanVec::build(
            index,
            test_d,
            svs::runtime::v0::MetricType::L2,
            svs::runtime::v0::StorageKind::LeanVec4x4,
            32,
            build_params,
            {},
            dynamic_index_params
        );
    };
    write_and_read_index<svs::runtime::v0::DynamicVamanaIndex>(
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
    if (!svs::runtime::v0::DynamicVamanaIndex::check_storage_kind(
             svs::runtime::v0::StorageKind::LeanVec4x4
        )
             .ok()) {
        CATCH_REQUIRE(!status.ok());
        CATCH_SKIP("Storage kind is not supported, skipping test.");
    }
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Add data - should work with provided leanvec dims
    std::vector<size_t> labels(test_n);
    std::iota(labels.begin(), labels.end(), 0);

    status = index->add(test_n, labels.data(), test_data.data());
    CATCH_REQUIRE(status.ok());

    svs::runtime::v0::DynamicVamanaIndex::destroy(index);
}

CATCH_TEST_CASE("LeanVecWithTrainingDataCustomBlockSize", "[runtime]") {
    const auto& test_data = get_test_data();
    size_t block_size_exp = 17; // block_size_bytes = 2^block_size_exp
    // Build LeanVec index with explicit training
    svs::runtime::v0::DynamicVamanaIndex* index = nullptr;
    svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
    svs::runtime::v0::VamanaIndex::DynamicIndexParams dynamic_index_params{block_size_exp};
    svs::runtime::v0::Status status = svs::runtime::v0::DynamicVamanaIndexLeanVec::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::LeanVec4x4,
        32,
        build_params,
        {},
        dynamic_index_params
    );
    if (!svs::runtime::v0::DynamicVamanaIndex::check_storage_kind(
             svs::runtime::v0::StorageKind::LeanVec4x4
        )
             .ok()) {
        CATCH_REQUIRE(!status.ok());
        CATCH_SKIP("Storage kind is not supported, skipping test.");
    }
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Add data - should work with provided leanvec dims
    std::vector<size_t> labels(test_n);
    std::iota(labels.begin(), labels.end(), 0);

    status = index->add(test_n, labels.data(), test_data.data());
    CATCH_REQUIRE(status.ok());

    CATCH_REQUIRE(index->blocksize_bytes() == 1u << block_size_exp);

    svs::runtime::v0::DynamicVamanaIndex::destroy(index);
}

CATCH_TEST_CASE("TrainingDataCustomBlockSize", "[runtime]") {
    const auto& test_data = get_test_data();
    size_t block_size_exp = 17; // block_size_bytes = 2^block_size_exp
    // Build LeanVec index with explicit training
    svs::runtime::v0::DynamicVamanaIndex* index = nullptr;
    svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
    svs::runtime::v0::VamanaIndex::DynamicIndexParams dynamic_index_params{block_size_exp};
    svs::runtime::v0::Status status = svs::runtime::v0::DynamicVamanaIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        build_params,
        {},
        dynamic_index_params
    );
    if (!svs::runtime::v0::DynamicVamanaIndex::check_storage_kind(
             svs::runtime::v0::StorageKind::FP32
        )
             .ok()) {
        CATCH_REQUIRE(!status.ok());
        CATCH_SKIP("Storage kind is not supported, skipping test.");
    }
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Add data - should work with provided leanvec dims
    std::vector<size_t> labels(test_n);
    std::iota(labels.begin(), labels.end(), 0);

    status = index->add(test_n, labels.data(), test_data.data());
    CATCH_REQUIRE(status.ok());

    CATCH_REQUIRE(index->blocksize_bytes() == 1u << block_size_exp);

    svs::runtime::v0::DynamicVamanaIndex::destroy(index);
}

CATCH_TEST_CASE("FlatIndexWriteAndRead", "[runtime]") {
    const auto& test_data = get_test_data();
    auto build_func = [](svs::runtime::v0::FlatIndex** index) {
        return svs::runtime::v0::FlatIndex::build(
            index, test_d, svs::runtime::v0::MetricType::L2
        );
    };
    write_and_read_index<svs::runtime::v0::FlatIndex>(
        build_func, test_data, test_n, test_d
    );
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
    constexpr auto file_threshold =
        [](size_t generated_data_bytes, size_t dim, size_t graph_max_degree) {
            // The index consists of the vectors (d * float) plus the graph (i.e., neighbor
            // indices; R * size_t). With d=128 and R=64, the graph overhead is 50% of the
            // vector size. So the total size is ~1.5x the vector size. Using 1.5x should be
            // safe, as R is the max degree, and thus the average degree is usually lower.
            size_t num_vectors = generated_data_bytes / (dim * sizeof(float));
            size_t graph_size = num_vectors * graph_max_degree * sizeof(size_t);
            return generated_data_bytes + graph_size;
        };

    constexpr auto rss_threshold = [](size_t generated_data_bytes,
                                      size_t allocator_block_size) {
        // Alias long names
        const size_t g = generated_data_bytes;
        const size_t a = allocator_block_size;
        // On load, the allocator allocates blocks of block_size
        // We allow for size_on_disk / block_size + 1 for the graph
        size_t per_entity = size_t((g + a - 1) / a * a);
        // Graph and Data can both be loaded with blocked allocators, therefore
        // we allow two times the aligned size calculated from the data size
        return per_entity + per_entity;
    };

    constexpr size_t MiB = 1024 * 1024;

    CATCH_SECTION("SmallIndex") {
        auto stats = run_save_and_load_test(10, 128, 64, 30);
        CATCH_REQUIRE(stats.file_size < file_threshold(10 * MiB, 128, 64));
        CATCH_REQUIRE(stats.rss_increase < rss_threshold(10 * MiB, 1024 * MiB));
    }

    CATCH_SECTION("MediumIndex") {
        auto stats = run_save_and_load_test(50, 128, 64, 30);
        CATCH_REQUIRE(stats.file_size < file_threshold(50 * MiB, 128, 64));
        CATCH_REQUIRE(stats.rss_increase < rss_threshold(50 * MiB, 1024 * MiB));
    }

    CATCH_SECTION("LargeIndex") {
        auto stats = run_save_and_load_test(200, 128, 64, 30);
        CATCH_REQUIRE(stats.file_size < file_threshold(200 * MiB, 128, 64));
        CATCH_REQUIRE(stats.rss_increase < rss_threshold(200 * MiB, 1024 * MiB));
    }
}

CATCH_TEST_CASE("SetIfSpecifiedUtility", "[runtime]") {
    using svs::runtime::set_if_specified;
    using svs::runtime::v0::is_specified;
    using svs::runtime::v0::OptionalBool;
    using svs::runtime::v0::Unspecify;

    CATCH_SECTION("OptionalBool") {
        OptionalBool undef;
        OptionalBool t(true);
        OptionalBool f(false);

        CATCH_REQUIRE(!is_specified(undef));
        CATCH_REQUIRE(is_specified(t));
        CATCH_REQUIRE(is_specified(f));

        bool target = true;
        set_if_specified(target, undef);
        CATCH_REQUIRE(target);
        set_if_specified(target, f);
        CATCH_REQUIRE(!target);
        set_if_specified(target, undef);
        CATCH_REQUIRE(!target);
        set_if_specified(target, t);
        CATCH_REQUIRE(target);
    }

    CATCH_SECTION("size_t") {
        size_t undef = Unspecify<size_t>();
        size_t val = 42;

        CATCH_REQUIRE(!is_specified(undef));
        CATCH_REQUIRE(is_specified(val));

        size_t target = 100;
        set_if_specified(target, undef);
        CATCH_REQUIRE(target == 100);
        set_if_specified(target, val);
        CATCH_REQUIRE(target == 42);
        set_if_specified(target, size_t{0});
        CATCH_REQUIRE(target == 0);
    }

    CATCH_SECTION("float") {
        float undef = Unspecify<float>();
        float val = 3.14f;

        CATCH_REQUIRE(!is_specified(undef));
        CATCH_REQUIRE(is_specified(val));

        float target = 1.0f;
        set_if_specified(target, undef);
        CATCH_REQUIRE(target == 1.0f);
        set_if_specified(target, val);
        CATCH_REQUIRE(target == 3.14f);
        set_if_specified(target, 0.0f);
        CATCH_REQUIRE(target == 0.0f);
    }

    CATCH_SECTION("int") {
        int undef = Unspecify<int>();
        int val = -7;

        CATCH_REQUIRE(!is_specified(undef));
        CATCH_REQUIRE(is_specified(val));

        int target = 10;
        set_if_specified(target, undef);
        CATCH_REQUIRE(target == 10);
        set_if_specified(target, val);
        CATCH_REQUIRE(target == -7);
        set_if_specified(target, 0);
        CATCH_REQUIRE(target == 0);
    }

    CATCH_SECTION("bool") {
        auto undef = Unspecify<bool>();
        auto val = true;

        CATCH_REQUIRE(!is_specified(undef));
        CATCH_REQUIRE(is_specified(val));

        bool target = false;
        set_if_specified(target, undef);
        CATCH_REQUIRE(target == false);
        set_if_specified(target, val);
        CATCH_REQUIRE(target == true);
        set_if_specified(target, false);
        CATCH_REQUIRE(target == false);
    }
}

CATCH_TEST_CASE("WriteAndReadStaticIndexSVS", "[runtime][static_vamana]") {
    const auto& test_data = get_test_data();
    auto build_func = [](svs::runtime::v0::VamanaIndex** index) {
        svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
        return svs::runtime::v0::VamanaIndex::build(
            index,
            test_d,
            svs::runtime::v0::MetricType::L2,
            svs::runtime::v0::StorageKind::FP32,
            build_params
        );
    };
    write_and_read_index<svs::runtime::v0::VamanaIndex>(
        build_func, test_data, test_n, test_d, svs::runtime::v0::StorageKind::FP32
    );
}

CATCH_TEST_CASE("WriteAndReadStaticIndexSVSFP16", "[runtime][static_vamana]") {
    const auto& test_data = get_test_data();
    auto build_func = [](svs::runtime::v0::VamanaIndex** index) {
        svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
        return svs::runtime::v0::VamanaIndex::build(
            index,
            test_d,
            svs::runtime::v0::MetricType::L2,
            svs::runtime::v0::StorageKind::FP16,
            build_params
        );
    };
    write_and_read_index<svs::runtime::v0::VamanaIndex>(
        build_func, test_data, test_n, test_d, svs::runtime::v0::StorageKind::FP16
    );
}

CATCH_TEST_CASE("WriteAndReadStaticIndexSVSSQI8", "[runtime][static_vamana]") {
    const auto& test_data = get_test_data();
    auto build_func = [](svs::runtime::v0::VamanaIndex** index) {
        svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
        return svs::runtime::v0::VamanaIndex::build(
            index,
            test_d,
            svs::runtime::v0::MetricType::L2,
            svs::runtime::v0::StorageKind::SQI8,
            build_params
        );
    };
    write_and_read_index<svs::runtime::v0::VamanaIndex>(
        build_func, test_data, test_n, test_d, svs::runtime::v0::StorageKind::SQI8
    );
}

CATCH_TEST_CASE("WriteAndReadStaticIndexSVSLVQ4x4", "[runtime][static_vamana]") {
    const auto& test_data = get_test_data();
    auto build_func = [](svs::runtime::v0::VamanaIndex** index) {
        svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
        return svs::runtime::v0::VamanaIndex::build(
            index,
            test_d,
            svs::runtime::v0::MetricType::L2,
            svs::runtime::v0::StorageKind::LVQ4x4,
            build_params
        );
    };
    write_and_read_index<svs::runtime::v0::VamanaIndex>(
        build_func, test_data, test_n, test_d, svs::runtime::v0::StorageKind::LVQ4x4
    );
}

CATCH_TEST_CASE("WriteAndReadStaticIndexSVSVamanaLeanVec4x4", "[runtime][static_vamana]") {
    const auto& test_data = get_test_data();
    auto build_func = [](svs::runtime::v0::VamanaIndex** index) {
        svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
        return svs::runtime::v0::VamanaIndexLeanVec::build(
            index,
            test_d,
            svs::runtime::v0::MetricType::L2,
            svs::runtime::v0::StorageKind::LeanVec4x4,
            32,
            build_params
        );
    };
    write_and_read_index<svs::runtime::v0::VamanaIndex>(
        build_func, test_data, test_n, test_d, svs::runtime::v0::StorageKind::LeanVec4x4
    );
}

CATCH_TEST_CASE("StaticIndexLeanVecWithTrainingData", "[runtime][static_vamana]") {
    const auto& test_data = get_test_data();
    const size_t leanvec_dims = 32;
    // Build LeanVec index with explicit training
    svs::runtime::v0::VamanaIndex* index = nullptr;
    svs::runtime::v0::VamanaIndex::BuildParams build_params{64};

    // Prepare training data
    svs::runtime::v0::LeanVecTrainingData* training_data = nullptr;
    svs::runtime::v0::Status status = svs::runtime::v0::LeanVecTrainingData::build(
        &training_data, test_d, test_n, test_data.data(), leanvec_dims
    );
    if (!svs::runtime::v0::VamanaIndexLeanVec::check_storage_kind(
             svs::runtime::v0::StorageKind::LeanVec4x4
        )
             .ok()) {
        CATCH_REQUIRE(!status.ok());
        CATCH_SKIP("Storage kind is not supported, skipping test.");
    }

    status = svs::runtime::v0::VamanaIndexLeanVec::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::LeanVec4x4,
        training_data,
        build_params
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    status = index->add(test_n, test_data.data());
    CATCH_REQUIRE(status.ok());

    svs::runtime::v0::VamanaIndex::destroy(index);
    svs::runtime::v0::LeanVecTrainingData::destroy(training_data);
}

CATCH_TEST_CASE("SearchWithIDFilterStatic", "[runtime][static_vamana]") {
    const auto& test_data = get_test_data();
    // Build index
    svs::runtime::v0::VamanaIndex* index = nullptr;
    svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
    svs::runtime::v0::Status status = svs::runtime::v0::VamanaIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        build_params
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Add data
    status = index->add(test_n, test_data.data());
    CATCH_REQUIRE(status.ok());

    // Second attempt to add data should fail on static index
    status = index->add(test_n, test_data.data());
    CATCH_REQUIRE(!status.ok());

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

    svs::runtime::v0::VamanaIndex::destroy(index);
}

CATCH_TEST_CASE("RangeSearchFunctionalStatic", "[runtime][static_vamana]") {
    const auto& test_data = get_test_data();
    // Build index
    svs::runtime::v0::VamanaIndex* index = nullptr;
    svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
    svs::runtime::v0::Status status = svs::runtime::v0::VamanaIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        build_params
    );
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Add data
    status = index->add(test_n, test_data.data());
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

    svs::runtime::v0::VamanaIndex::destroy(index);
}

// =====================================================================
// Deferred-compression tests
//
// These exercise the runtime orchestrator that watches the live insert count
// and triggers the swap from the initial uncompressed storage backend
// (FP32 / FP16) to the configured trained target (LVQ / LeanVec / SQ) once
// the threshold is reached, while reusing the existing graph.
// =====================================================================
namespace {

// Add `flat_data` in two chunks straddling `threshold`. Returns the storage
// kind reported between the two `add()` calls and after the second.
struct DelayedSwapObservation {
    svs::runtime::v0::StorageKind kind_before_swap;
    svs::runtime::v0::StorageKind kind_after_swap;
};

DelayedSwapObservation add_in_two_halves(
    svs::runtime::v0::DynamicVamanaIndex& index,
    const std::vector<float>& flat_data,
    size_t n_total,
    size_t d,
    size_t first_chunk
) {
    CATCH_REQUIRE(first_chunk < n_total);
    std::vector<size_t> ids(n_total);
    std::iota(ids.begin(), ids.end(), 0);

    auto status = index.add(first_chunk, ids.data(), flat_data.data());
    CATCH_REQUIRE(status.ok());
    auto kind_before = index.get_current_storage_kind();

    status = index.add(
        n_total - first_chunk,
        ids.data() + first_chunk,
        flat_data.data() + first_chunk * d
    );
    CATCH_REQUIRE(status.ok());
    auto kind_after = index.get_current_storage_kind();

    return {kind_before, kind_after};
}

} // namespace

CATCH_TEST_CASE(
    "Deferred compression FP32 -> LVQ4x8 via runtime",
    "[runtime][deferred_compression]"
) {
    const auto& test_data = get_test_data();
    const size_t threshold = test_n / 2;
    const size_t first_chunk = threshold / 2; // stays under threshold

    svs::runtime::v0::DynamicVamanaIndex* index = nullptr;
    svs::runtime::v0::VamanaIndex::BuildParams build_params{/*graph_max_degree=*/64};
    svs::runtime::v0::VamanaIndex::DynamicIndexParams dyn_params{};
    dyn_params.deferred_compression_threshold = threshold;
    dyn_params.initial_storage_kind = svs::runtime::v0::StorageKind::FP32;

    auto status = svs::runtime::v0::DynamicVamanaIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::LVQ4x8,
        build_params,
        {},
        dyn_params
    );
    if (!svs::runtime::v0::DynamicVamanaIndex::check_storage_kind(
             svs::runtime::v0::StorageKind::LVQ4x8)
             .ok()) {
        CATCH_REQUIRE(!status.ok());
        return;
    }
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Before any add: nothing built yet, current kind is the initial kind.
    CATCH_REQUIRE(
        index->get_current_storage_kind() == svs::runtime::v0::StorageKind::FP32
    );

    auto obs = add_in_two_halves(*index, test_data, test_n, test_d, first_chunk);
    CATCH_REQUIRE(obs.kind_before_swap == svs::runtime::v0::StorageKind::FP32);
    CATCH_REQUIRE(obs.kind_after_swap == svs::runtime::v0::StorageKind::LVQ4x8);

    // Sanity-check search after the swap (the graph should survive).
    const int nq = 5;
    const int k = 5;
    std::vector<float> distances(nq * k);
    std::vector<size_t> labels(nq * k);
    status =
        index->search(nq, test_data.data(), k, distances.data(), labels.data());
    CATCH_REQUIRE(status.ok());

    svs::runtime::v0::DynamicVamanaIndex::destroy(index);
}

CATCH_TEST_CASE(
    "Deferred compression FP32 -> LeanVec4x8 via runtime",
    "[runtime][deferred_compression]"
) {
    const auto& test_data = get_test_data();
    const size_t threshold = test_n / 2;
    const size_t first_chunk = threshold / 2;

    svs::runtime::v0::DynamicVamanaIndex* index = nullptr;
    svs::runtime::v0::VamanaIndex::BuildParams build_params{/*graph_max_degree=*/64};
    svs::runtime::v0::VamanaIndex::DynamicIndexParams dyn_params{};
    dyn_params.deferred_compression_threshold = threshold;
    dyn_params.initial_storage_kind = svs::runtime::v0::StorageKind::FP32;

    auto status = svs::runtime::v0::DynamicVamanaIndexLeanVec::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::LeanVec4x8,
        /*leanvec_dims=*/test_d / 2,
        build_params,
        {},
        dyn_params
    );
    if (!svs::runtime::v0::DynamicVamanaIndex::check_storage_kind(
             svs::runtime::v0::StorageKind::LeanVec4x8)
             .ok()) {
        CATCH_REQUIRE(!status.ok());
        return;
    }
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    CATCH_REQUIRE(
        index->get_current_storage_kind() == svs::runtime::v0::StorageKind::FP32
    );

    auto obs = add_in_two_halves(*index, test_data, test_n, test_d, first_chunk);
    CATCH_REQUIRE(obs.kind_before_swap == svs::runtime::v0::StorageKind::FP32);
    CATCH_REQUIRE(obs.kind_after_swap == svs::runtime::v0::StorageKind::LeanVec4x8);

    const int nq = 5;
    const int k = 5;
    std::vector<float> distances(nq * k);
    std::vector<size_t> labels(nq * k);
    status =
        index->search(nq, test_data.data(), k, distances.data(), labels.data());
    CATCH_REQUIRE(status.ok());

    svs::runtime::v0::DynamicVamanaIndex::destroy(index);
}

CATCH_TEST_CASE(
    "Deferred compression disabled by default keeps eager behavior via runtime",
    "[runtime][deferred_compression]"
) {
    const auto& test_data = get_test_data();

    svs::runtime::v0::DynamicVamanaIndex* index = nullptr;
    svs::runtime::v0::VamanaIndex::BuildParams build_params{/*graph_max_degree=*/64};
    // Default ctor: threshold == 0 -> eager.
    svs::runtime::v0::VamanaIndex::DynamicIndexParams dyn_params{};

    auto status = svs::runtime::v0::DynamicVamanaIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::LVQ4x8,
        build_params,
        {},
        dyn_params
    );
    if (!svs::runtime::v0::DynamicVamanaIndex::check_storage_kind(
             svs::runtime::v0::StorageKind::LVQ4x8)
             .ok()) {
        CATCH_REQUIRE(!status.ok());
        return;
    }
    CATCH_REQUIRE(status.ok());

    // Eager mode: target kind is in effect from the start, before and after add.
    CATCH_REQUIRE(
        index->get_current_storage_kind() == svs::runtime::v0::StorageKind::LVQ4x8
    );

    std::vector<size_t> ids(test_n);
    std::iota(ids.begin(), ids.end(), 0);
    status = index->add(test_n, ids.data(), test_data.data());
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(
        index->get_current_storage_kind() == svs::runtime::v0::StorageKind::LVQ4x8
    );

    svs::runtime::v0::DynamicVamanaIndex::destroy(index);
}

CATCH_TEST_CASE(
    "Deferred compression rejects non-FP initial storage kinds via runtime",
    "[runtime][deferred_compression]"
) {
    svs::runtime::v0::DynamicVamanaIndex* index = nullptr;
    svs::runtime::v0::VamanaIndex::BuildParams build_params{/*graph_max_degree=*/64};
    svs::runtime::v0::VamanaIndex::DynamicIndexParams dyn_params{};
    dyn_params.deferred_compression_threshold = 1024;
    dyn_params.initial_storage_kind = svs::runtime::v0::StorageKind::LVQ4x4;

    auto status = svs::runtime::v0::DynamicVamanaIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::LVQ4x8,
        build_params,
        {},
        dyn_params
    );
    CATCH_REQUIRE(!status.ok());
    CATCH_REQUIRE(index == nullptr);
}

CATCH_TEST_CASE(
    "Deferred compression first-add at-threshold builds target directly via runtime",
    "[runtime][deferred_compression]"
) {
    // When the very first add already meets the deferred-compression threshold,
    // the runtime should skip the uncompressed staging build and construct the
    // target compressed backend directly. Externally this is observable as the
    // `current_storage_kind` jumping straight from the initial kind (pre-build)
    // to the target kind after the first add, without ever materializing the
    // uncompressed backend.
    const auto& test_data = get_test_data();
    const size_t threshold = test_n / 2; // first add will be `test_n` >= threshold

    svs::runtime::v0::DynamicVamanaIndex* index = nullptr;
    svs::runtime::v0::VamanaIndex::BuildParams build_params{/*graph_max_degree=*/64};
    svs::runtime::v0::VamanaIndex::DynamicIndexParams dyn_params{};
    dyn_params.deferred_compression_threshold = threshold;
    dyn_params.initial_storage_kind = svs::runtime::v0::StorageKind::FP32;

    auto status = svs::runtime::v0::DynamicVamanaIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::LVQ4x8,
        build_params,
        {},
        dyn_params
    );
    if (!svs::runtime::v0::DynamicVamanaIndex::check_storage_kind(
             svs::runtime::v0::StorageKind::LVQ4x8)
             .ok()) {
        CATCH_REQUIRE(!status.ok());
        return;
    }
    CATCH_REQUIRE(status.ok());

    // Pre-add: index reports initial kind.
    CATCH_REQUIRE(
        index->get_current_storage_kind() == svs::runtime::v0::StorageKind::FP32
    );

    std::vector<size_t> ids(test_n);
    std::iota(ids.begin(), ids.end(), 0);
    status = index->add(test_n, ids.data(), test_data.data());
    CATCH_REQUIRE(status.ok());

    // Post-first-add: target kind is already in effect.
    CATCH_REQUIRE(
        index->get_current_storage_kind() == svs::runtime::v0::StorageKind::LVQ4x8
    );

    const int nq = 5;
    const int k = 5;
    std::vector<float> distances(nq * k);
    std::vector<size_t> labels(nq * k);
    status =
        index->search(nq, test_data.data(), k, distances.data(), labels.data());
    CATCH_REQUIRE(status.ok());

    svs::runtime::v0::DynamicVamanaIndex::destroy(index);
}
