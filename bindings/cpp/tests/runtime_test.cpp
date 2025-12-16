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
    svs::runtime::v0::IndexBlockSize blocksize = svs::runtime::v0::IndexBlockSize(30),
    std::optional<svs::runtime::v0::StorageKind> storage_kind = std::nullopt,
    svs::runtime::v0::MetricType metric = svs::runtime::v0::MetricType::L2
) {
    // Build index
    Index* index = nullptr;
    svs::runtime::v0::Status status = build_func(&index);

    // Stop here if storage kind is not supported on this platform
    if constexpr (std::is_same_v<Index, svs::runtime::v0::DynamicVamanaIndex>) {
        if (storage_kind.has_value()) {
            if (!svs::runtime::v0::DynamicVamanaIndex::check_storage_kind(*storage_kind)
                     .ok()) {
                CATCH_REQUIRE(!status.ok());
                return;
            }
        }
    }
    CATCH_REQUIRE(status.ok());
    CATCH_REQUIRE(index != nullptr);

    // Add data to index
    if constexpr (std::is_same_v<Index, svs::runtime::v0::FlatIndex>) {
        status = index->add(n, xb.data());
    } else {
        std::vector<size_t> labels(n);
        std::iota(labels.begin(), labels.end(), 0);
        status = index->add(n, labels.data(), xb.data(), blocksize);
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

    if constexpr (!std::is_same_v<Index, svs::runtime::v0::FlatIndex>) {
        CATCH_REQUIRE(index->blocksize_bytes() == blocksize.BlockSizeBytes());
    }
    // Clean up
    Index::destroy(index);
    Index::destroy(loaded);
}

// Helper that writes and reads and index of requested size
// Reports memory usage
UsageInfo run_save_and_load_test(
    const size_t target_mibytes,
    size_t d,
    size_t graph_max_degree,
    svs::runtime::v0::IndexBlockSize blocksize
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
        svs::runtime::v0::Status status = svs::runtime::v0::DynamicVamanaIndex::build(
            &index,
            d,
            svs::runtime::v0::MetricType::L2,
            svs::runtime::v0::StorageKind::FP32,
            build_params
        );
        CATCH_REQUIRE(status.ok());
        CATCH_REQUIRE(index != nullptr);
        status = index->add(mem_test_n, labels.data(), large_test_data.data(), blocksize);
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
    write_and_read_index<svs::runtime::v0::DynamicVamanaIndex>(
        build_func,
        test_data,
        test_n,
        test_d,
        svs::runtime::v0::IndexBlockSize(15),
        svs::runtime::v0::StorageKind::FP32
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
    write_and_read_index<svs::runtime::v0::DynamicVamanaIndex>(
        build_func,
        test_data,
        test_n,
        test_d,
        svs::runtime::v0::IndexBlockSize(16),
        svs::runtime::v0::StorageKind::FP16
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
    write_and_read_index<svs::runtime::v0::DynamicVamanaIndex>(
        build_func,
        test_data,
        test_n,
        test_d,
        svs::runtime::v0::IndexBlockSize(17),
        svs::runtime::v0::StorageKind::SQI8
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
    write_and_read_index<svs::runtime::v0::DynamicVamanaIndex>(
        build_func,
        test_data,
        test_n,
        test_d,
        svs::runtime::v0::IndexBlockSize(18),
        svs::runtime::v0::StorageKind::LVQ4x4
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
    write_and_read_index<svs::runtime::v0::DynamicVamanaIndex>(
        build_func,
        test_data,
        test_n,
        test_d,
        svs::runtime::v0::IndexBlockSize(19),
        svs::runtime::v0::StorageKind::LeanVec4x4
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

    int block_size_exp = 17; // block_size_bytes = 2^block_size_exp
    status = index->add(
        test_n,
        labels.data(),
        test_data.data(),
        svs::runtime::v0::IndexBlockSize(block_size_exp)
    );
    CATCH_REQUIRE(status.ok());

    CATCH_REQUIRE(index->blocksize_bytes().raw() == block_size_exp);

    svs::runtime::v0::DynamicVamanaIndex::destroy(index);
}

CATCH_TEST_CASE("TrainingDataCustomBlockSize", "[runtime]") {
    const auto& test_data = get_test_data();
    // Build LeanVec index with explicit training
    svs::runtime::v0::DynamicVamanaIndex* index = nullptr;
    svs::runtime::v0::VamanaIndex::BuildParams build_params{64};
    svs::runtime::v0::Status status = svs::runtime::v0::DynamicVamanaIndex::build(
        &index,
        test_d,
        svs::runtime::v0::MetricType::L2,
        svs::runtime::v0::StorageKind::FP32,
        build_params
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

    int block_size_exp = 17; // block_size_bytes = 2^block_size_exp
    status = index->add(
        test_n,
        labels.data(),
        test_data.data(),
        svs::runtime::v0::IndexBlockSize(block_size_exp)
    );
    CATCH_REQUIRE(status.ok());

    CATCH_REQUIRE(index->blocksize_bytes().raw() == block_size_exp);

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
    status = index->add(
        test_n, labels.data(), test_data.data(), svs::runtime::v0::IndexBlockSize(30)
    );
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
    status = index->add(
        test_n, labels.data(), test_data.data(), svs::runtime::v0::IndexBlockSize(30)
    );
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
        auto stats =
            run_save_and_load_test(10, 128, 64, svs::runtime::v0::IndexBlockSize(30));
        CATCH_REQUIRE(stats.file_size < file_threshold(10 * MiB, 128, 64));
        CATCH_REQUIRE(stats.rss_increase < rss_threshold(10 * MiB, 1024 * MiB));
    }

    CATCH_SECTION("MediumIndex") {
        auto stats =
            run_save_and_load_test(50, 128, 64, svs::runtime::v0::IndexBlockSize(30));
        CATCH_REQUIRE(stats.file_size < file_threshold(50 * MiB, 128, 64));
        CATCH_REQUIRE(stats.rss_increase < rss_threshold(50 * MiB, 1024 * MiB));
    }

    CATCH_SECTION("LargeIndex") {
        auto stats =
            run_save_and_load_test(200, 128, 64, svs::runtime::v0::IndexBlockSize(30));
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
