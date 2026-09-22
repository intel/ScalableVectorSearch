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

// header under test.
#include "svs/index/vamana/dynamic_index.h"
#include "svs/index/vamana/consolidate.h"

// stl
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

// svs
#include "svs/core/recall.h"
#include "svs/lib/timing.h"
#include "svs/orchestrators/dynamic_vamana.h"

// catch2
#include "catch2/catch_test_macros.hpp"
#include <catch2/catch_approx.hpp>

// tests
#include "tests/utils/test_dataset.h"
#include "tests/utils/utils.h"

// The MutableVamanaIndex "Soft Deletion" test uses outdated API.
#if 0
namespace {
template <typename T> auto copy_dataset(const T& data) {
    auto copy = svs::data::SimplePolymorphicData<typename T::element_type, T::extent>{
        data.size(), data.dimensions()};
    for (size_t i = 0; i < data.size(); ++i) {
        copy.set_datum(i, data.get_datum(i));
    }
    return copy;
}

template <typename T, typename U> void check_results(const T& results, const U& deleted) {
    for (size_t i = 0; i < svs::getsize<0>(results); ++i) {
        for (size_t j = 0; j < svs::getsize<1>(results); ++j) {
            CATCH_REQUIRE(!deleted.contains(results.at(i, j)));
        }
    }
}

template <typename T, typename U>
void check_deleted(const T& index, const U& deleted, size_t imax) {
    for (size_t i = 0; i < imax; ++i) {
        if (deleted.contains(i)) {
            CATCH_REQUIRE(index.is_deleted(i));
        } else {
            CATCH_REQUIRE(!index.is_deleted(i));
        }
    }
}

template <typename Left, typename Right>
void check_equal(const Left& left, const Right& right) {
    CATCH_REQUIRE(left.size() == right.size());
    CATCH_REQUIRE(left.dimensions() == right.dimensions());

    for (size_t i = 0, imax = left.size(); i < imax; ++i) {
        const auto& datum_left = left.get_datum(i);
        const auto& datum_right = right.get_datum(i);
        CATCH_REQUIRE(std::equal(datum_left.begin(), datum_left.end(), datum_right.begin())
        );
    }
}

} // namespace

#if defined(NDEBUG)
const double DELETE_PERCENT = 0.3;
#else
const double DELETE_PERCENT = 0.05;
#endif

CATCH_TEST_CASE("MutableVamanaIndex", "[graph_index]") {
    const size_t num_threads = 2;
    const size_t num_neighbors = 10;

    const auto base_data = test_dataset::data_blocked_f32();
    // const auto base_data = test_dataset::data_f32();
    const auto queries = test_dataset::queries();
    const auto groundtruth = test_dataset::groundtruth_euclidean();

    CATCH_SECTION("Soft Deletion") {
        // In this section, we test soft deletion.
        // The idea is as follows:
        //
        // (1) Load the test index.
        // (2) Run a round of queries to ensure that everything loading correctly.
        // (3) Set a target deletion percentage where all the neighbors returned by
        //     all results returned by the previous query plus a random collection of extras
        //     are deleted.
        //
        // (4) Rerun queries, make sure accuracy is still high and that no deleted indices
        //     are present in the results.
        auto entry_point = svs::index::load_entry_point(test_dataset::metadata_file());

        auto index = svs::index::MutableVamanaIndex{
            test_dataset::graph_blocked(),
            base_data.copy(),
            entry_point,
            svs::distance::DistanceL2(),
            svs::threads::UnitRange<size_t>(0, base_data.size()),
            num_threads};

        check_equal(base_data, index);
        index.debug_check_graph_consistency(false);

        auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        index.set_search_window_size(num_neighbors);

        auto tic = svs::lib::now();
        index.search(queries.view(), num_neighbors, results.view());
        auto original_time = svs::lib::time_difference(svs::lib::now(), tic);
        auto original_recall = svs::k_recall_at_n(groundtruth, results);
        CATCH_REQUIRE(index.entry_point() == entry_point);

        std::unordered_set<uint32_t> ids_to_delete{};
        double delete_percent = DELETE_PERCENT;
        for (size_t i = 0; i < groundtruth.size(); ++i) {
            auto slice = groundtruth.get_datum(i);
            for (size_t j = 0; j < num_neighbors; ++j) {
                auto id = slice[j];

                // For now - don't delete the entry point.
                if (id != entry_point) {
                    ids_to_delete.insert(slice[j]);
                }
            }

            if (ids_to_delete.size() > delete_percent * base_data.size()) {
                break;
            }
        }

        index.set_threadpool(threads::CppAsyncThreadPool(num_threads));

        std::cout << "Deleting " << ids_to_delete.size() << " entries!" << std::endl;
        index.delete_entries(ids_to_delete);
        check_deleted(index, ids_to_delete, base_data.size());
        index.debug_check_graph_consistency(true);
        CATCH_REQUIRE_THROWS_AS(
            index.debug_check_graph_consistency(false), svs::ANNException
        );
        CATCH_REQUIRE(index.entry_point() == entry_point);
        // Make sure the correct points were deleted.
        tic = svs::lib::now();
        index.search(queries.view(), num_neighbors, results.view());
        auto new_time = svs::lib::time_difference(tic);

        // Make sure none of the returned results are in the deleted list.
        check_results(results.indices(), ids_to_delete);

        index.set_threadpool(threads::QueueThreadPoolWrapper(num_threads));

        auto results_reference = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        index.exhaustive_search(queries.view(), num_neighbors, results_reference.view());
        auto new_recall = svs::k_recall_at_n(results_reference.indices(), results);

        // Perform graph consolidation and see how the results are effected.
        index.set_alpha(1.2);
        index.consolidate();
        index.debug_check_graph_consistency(false);
        tic = svs::lib::now();
        index.search(queries.view(), num_neighbors, results.view());
        auto post_consolidate_time = svs::lib::time_difference(tic);
        auto post_consolidate_recall =
            svs::k_recall_at_n(results_reference.indices(), results);

        // Check deletion again.
        check_deleted(index, ids_to_delete, base_data.size());
        CATCH_REQUIRE(index.entry_point() == entry_point);

        std::cout << "Original recall: " << original_recall
                  << ", New Recall: " << new_recall
                  << ", Post Recall: " << post_consolidate_recall << std::endl;
        std::cout << "Original Time: " << original_time << " (s), New Time: " << new_time
                  << " (s) Post Time: " << post_consolidate_time << std::endl;
        CATCH_REQUIRE(new_recall > original_recall);
        check_results(results.indices(), ids_to_delete);

        // Now - delete the entry point and consolidate.
        ids_to_delete.insert(entry_point);
        std::vector<size_t> entry_point_vector{};
        entry_point_vector.push_back(entry_point);
        index.delete_entries(entry_point_vector);
        index.set_alpha(1.2);
        index.consolidate();
        index.debug_check_graph_consistency(false);

        auto& threadpool =
            index.get_threadpool_handle().get<threads::CppAsyncThreadPool>.get();
        threadpool.resize(3);
        CATCH_REQUIRE(index.get_num_threads() == 3);
        threadpool.resize(num_threads);
        CATCH_REQUIRE(index.get_num_threads() == num_threads);

        CATCH_REQUIRE(index.entry_point() != entry_point);
        index.search(queries.view(), num_neighbors, results.view());
        auto post_entrypoint_recall =
            svs::k_recall_at_n(results_reference.indices(), results);
        std::cout << "Post entry-point deletion recall: " << post_entrypoint_recall
                  << std::endl;

        // Add the deleted points back in.
        auto points = svs::data::SimpleData<float, svs::Dynamic>(
            ids_to_delete.size(), base_data.dimensions()
        );

        size_t i = 0;
        for (const auto& j : ids_to_delete) {
            points.set_datum(i, base_data.get_datum(j));
            ++i;
        }

        index.set_threadpool(threads::DefaultThreadPool(num_threads));
        tic = svs::lib::now();
        index.add_points(points, ids_to_delete);
        auto insert_time = svs::lib::time_difference(tic);
        std::cout << "Insertion took: " << insert_time << " seconds!" << std::endl;

        // Check that the stored dataset and the original dataset are equal.
        check_equal(base_data, index);
        index.debug_check_graph_consistency(false);

        tic = svs::lib::now();
        index.search(queries.view(), num_neighbors, results.view());
        auto post_add_time = svs::lib::time_difference(tic);
        auto post_reinsertion_recall = svs::k_recall_at_n(groundtruth, results);
        std::cout << "Post reinsertion recall: " << post_reinsertion_recall << " in "
                  << post_add_time << " seconds." << std::endl;
    }
}
#endif

CATCH_TEST_CASE(
    "MutableVamana Index Save and Load", "[graph_index][dynamic_index][saveload]"
) {
    const size_t num_threads = 2;
    using Distance = svs::distance::DistanceL2;

    auto data = test_dataset::data_blocked_f32();
    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);

    svs::index::vamana::VamanaBuildParameters parameters{1.2, 64, 10, 20, 10, true};
    auto index = svs::index::vamana::MutableVamanaIndex(
        parameters, std::move(data), indices, Distance(), num_threads
    );

    const size_t num_neighbors = 10;
    auto queries = test_dataset::queries();
    auto search_params = svs::index::vamana::VamanaSearchParameters{};
    search_params.buffer_config_ = svs::index::vamana::SearchBufferConfig{num_neighbors};
    auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
    index.search(results.view(), queries.cview(), search_params);

    CATCH_SECTION("Load MutableVamana Index being serialized natively to stream") {
        std::stringstream stream;
        index.save(stream);
        {
            using Data_t = svs::data::BlockedData<float>;

            auto loaded = svs::DynamicVamana::assemble<float, Data_t>(
                stream, Distance(), num_threads
            );

            CATCH_REQUIRE(loaded.size() == index.size());
            CATCH_REQUIRE(loaded.dimensions() == index.dimensions());
            CATCH_REQUIRE(loaded.get_alpha() == index.get_alpha());
            CATCH_REQUIRE(loaded.get_graph_max_degree() == index.get_graph_max_degree());
            CATCH_REQUIRE(loaded.get_max_candidates() == index.get_max_candidates());
            CATCH_REQUIRE(
                loaded.get_construction_window_size() ==
                index.get_construction_window_size()
            );
            CATCH_REQUIRE(loaded.get_prune_to() == index.get_prune_to());
            CATCH_REQUIRE(
                loaded.get_full_search_history() == index.get_full_search_history()
            );
            index.on_ids([&](size_t e) { CATCH_REQUIRE(loaded.has_id(e)); });

            auto loaded_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
            loaded.search(loaded_results.view(), queries.cview(), search_params);
            for (size_t q = 0; q < queries.size(); ++q) {
                for (size_t i = 0; i < num_neighbors; ++i) {
                    CATCH_REQUIRE(loaded_results.index(q, i) == results.index(q, i));
                    CATCH_REQUIRE(
                        loaded_results.distance(q, i) ==
                        Catch::Approx(results.distance(q, i)).epsilon(1e-5)
                    );
                }
            }
        }
    }

    CATCH_SECTION("Load MutableVamana Index being serialized with intermediate files") {
        std::stringstream stream;
        {
            svs::lib::UniqueTempDirectory tempdir{"svs_dynvamana_save"};
            const auto config_dir = tempdir.get() / "config";
            const auto graph_dir = tempdir.get() / "graph";
            const auto data_dir = tempdir.get() / "data";
            std::filesystem::create_directories(config_dir);
            std::filesystem::create_directories(graph_dir);
            std::filesystem::create_directories(data_dir);
            index.save(config_dir, graph_dir, data_dir);
            svs::lib::DirectoryArchiver::pack(tempdir, stream);
        }
        {
            using Data_t = svs::data::BlockedData<float>;

            auto loaded = svs::DynamicVamana::assemble<float, Data_t>(
                stream, Distance(), num_threads
            );

            CATCH_REQUIRE(loaded.size() == index.size());
            CATCH_REQUIRE(loaded.dimensions() == index.dimensions());
            CATCH_REQUIRE(loaded.get_alpha() == index.get_alpha());
            CATCH_REQUIRE(loaded.get_graph_max_degree() == index.get_graph_max_degree());
            CATCH_REQUIRE(loaded.get_max_candidates() == index.get_max_candidates());
            CATCH_REQUIRE(
                loaded.get_construction_window_size() ==
                index.get_construction_window_size()
            );
            CATCH_REQUIRE(loaded.get_prune_to() == index.get_prune_to());
            CATCH_REQUIRE(
                loaded.get_full_search_history() == index.get_full_search_history()
            );
            index.on_ids([&](size_t e) { CATCH_REQUIRE(loaded.has_id(e)); });

            auto loaded_results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
            loaded.search(loaded_results.view(), queries.cview(), search_params);
            for (size_t q = 0; q < queries.size(); ++q) {
                for (size_t i = 0; i < num_neighbors; ++i) {
                    CATCH_REQUIRE(loaded_results.index(q, i) == results.index(q, i));
                    CATCH_REQUIRE(
                        loaded_results.distance(q, i) ==
                        Catch::Approx(results.distance(q, i)).epsilon(1e-5)
                    );
                }
            }
        }
    }
}

CATCH_TEST_CASE("MutableVamana Index Relabel", "[graph_index][dynamic_index]") {
    const size_t num_threads = 2;
    using Distance = svs::distance::DistanceL2;

    auto data = test_dataset::data_blocked_f32();
    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);

    svs::index::vamana::VamanaBuildParameters parameters{1.2, 64, 10, 20, 10, true};
    auto index = svs::index::vamana::MutableVamanaIndex(
        parameters, std::move(data), indices, Distance(), num_threads
    );

    const size_t old_id = indices.front();
    const size_t new_id = 999'999;
    CATCH_REQUIRE(index.has_id(old_id));
    CATCH_REQUIRE(!index.has_id(new_id));

    auto internal_id_before = index.translate_external_id(old_id);
    auto datum_before = index.get_datum(old_id);
    auto datum_copy = std::vector<float>(datum_before.begin(), datum_before.end());
    auto size_before = index.size();

    index.replace_external_id(old_id, new_id);

    // The old label is gone, the new one exists, and nothing moved: same internal id,
    // same stored data, same count -- this is a pure rename, not a delete + re-add.
    CATCH_REQUIRE(!index.has_id(old_id));
    CATCH_REQUIRE(index.has_id(new_id));
    CATCH_REQUIRE(index.size() == size_before);
    CATCH_REQUIRE(index.translate_external_id(new_id) == internal_id_before);
    auto datum_after = index.get_datum(new_id);
    CATCH_REQUIRE(std::equal(datum_after.begin(), datum_after.end(), datum_copy.begin()));

    // Every other id is untouched.
    for (size_t i = 1; i < indices.size(); ++i) {
        CATCH_REQUIRE(index.has_id(indices[i]));
    }

    CATCH_SECTION("Renaming a non-existent ID throws") {
        // `old_id` was already renamed away above, so it no longer exists either.
        CATCH_REQUIRE_THROWS_AS(
            index.replace_external_id(old_id, new_id + 1), svs::ANNException
        );
        CATCH_REQUIRE(!index.has_id(old_id));
        CATCH_REQUIRE(!index.has_id(new_id + 1));
    }

    CATCH_SECTION("Renaming onto an existing ID throws") {
        auto other_id = indices[1];
        CATCH_REQUIRE_THROWS_AS(
            index.replace_external_id(new_id, other_id), svs::ANNException
        );
        // State unchanged: the relabel performed above still holds.
        CATCH_REQUIRE(index.has_id(new_id));
        CATCH_REQUIRE(index.has_id(other_id));
    }
}

CATCH_TEST_CASE("MutableVamana Index Memory Usage", "[graph_index][dynamic_index]") {
    const size_t num_threads = 2;
    using Distance = svs::distance::DistanceL2;

    auto data = test_dataset::data_blocked_f32();
    const size_t data_size = data.size();
    // Expected data bytes are capacity-based; capture them before the dataset is moved
    // into the index so the test can pin the exact value.
    const size_t expected_data_bytes = data.capacity() * data.element_size();
    std::vector<size_t> indices(data_size);
    std::iota(indices.begin(), indices.end(), 0);

    svs::index::vamana::VamanaBuildParameters parameters{1.2, 64, 10, 20, 10, true};
    auto index = svs::index::vamana::MutableVamanaIndex(
        parameters, std::move(data), indices, Distance(), num_threads
    );

    const size_t expected_graph_bytes = index.view_graph().get_data().capacity() *
                                        index.view_graph().get_data().element_size();
    using Index = decltype(index);
    const size_t expected_metadata_bytes =
        data_size * sizeof(svs::index::vamana::SlotMetadata) +
        sizeof(typename Index::internal_id_type) +
        2 * indices.size() *
            (sizeof(typename Index::external_id_type) +
             sizeof(typename Index::internal_id_type));
    const size_t expected_total_bytes =
        expected_data_bytes + expected_graph_bytes + expected_metadata_bytes;

    // Dynamic get_memory_usage() should exactly match the capacity-based graph and data
    // bytes plus the deterministic metadata implied by the input ids.
    const auto breakdown = index.get_memory_breakdown();
    CATCH_REQUIRE(breakdown.graph_bytes == expected_graph_bytes);
    CATCH_REQUIRE(breakdown.data_bytes == expected_data_bytes);
    CATCH_REQUIRE(breakdown.metadata_bytes == expected_metadata_bytes);
    CATCH_REQUIRE(breakdown.total() == expected_total_bytes);
    const size_t usage = index.get_memory_breakdown().total();
    CATCH_REQUIRE(usage == expected_total_bytes);
}

CATCH_TEST_CASE(
    "MutableVamana Index Compact Tolerates Stale Edges",
    "[graph_index][dynamic_index][compact]"
) {
    const size_t num_threads = 2;
    const size_t num_neighbors = 10;
    using Distance = svs::distance::DistanceL2;

    auto reference_data = test_dataset::data_blocked_f32();
    auto data = test_dataset::data_blocked_f32();
    const size_t data_size = data.size();
    std::vector<size_t> indices(data_size);
    std::iota(indices.begin(), indices.end(), 0);

    svs::index::vamana::VamanaBuildParameters parameters{1.2, 64, 10, 20, 10, true};
    auto index = svs::index::vamana::MutableVamanaIndex(
        parameters, std::move(data), indices, Distance(), num_threads
    );
    const size_t entry_point = index.entry_point();

    // Delete a fifth of the ids (skipping the entry point, which is a separate,
    // unfixed defect at line 988) so surviving nodes are left with stale edges
    // once no consolidate() runs before compact().
    std::vector<size_t> ids_to_delete{};
    for (size_t i = 0; i < data_size; i += 5) {
        if (i != entry_point) {
            ids_to_delete.push_back(i);
        }
    }
    index.delete_entries(ids_to_delete);
    std::unordered_set<size_t> deleted_set(ids_to_delete.begin(), ids_to_delete.end());

    // Confirm the pathological precondition actually holds instead of assuming it:
    // at least one surviving node's adjacency list still references a deleted slot.
    auto has_stale_edge = [&]() {
        for (size_t i = 0; i < data_size; ++i) {
            if (index.is_deleted(i)) {
                continue;
            }
            for (auto j : index.view_graph().get_node(i)) {
                if (index.is_deleted(j)) {
                    return true;
                }
            }
        }
        return false;
    };
    CATCH_REQUIRE(has_stale_edge());

    auto queries = test_dataset::queries();
    auto search_params = svs::index::vamana::VamanaSearchParameters{};
    search_params.buffer_config_ = svs::index::vamana::SearchBufferConfig{num_neighbors};

    CATCH_SECTION("Compact without consolidate tolerates stale edges") {
        CATCH_REQUIRE_NOTHROW(index.compact());

        // Every surviving node's adjacency must now reference only valid slots.
        index.debug_check_graph_consistency(false);
        CATCH_REQUIRE(index.size() == data_size - ids_to_delete.size());

        for (size_t id = 0; id < data_size; ++id) {
            CATCH_REQUIRE(index.has_id(id) == !deleted_set.contains(id));
        }
        // Surviving ids must still translate to their original vector contents.
        for (size_t id = 0; id < data_size; id += 7) {
            if (deleted_set.contains(id)) {
                continue;
            }
            auto datum = index.get_datum(id);
            auto ref = reference_data.get_datum(id);
            CATCH_REQUIRE(std::equal(datum.begin(), datum.end(), ref.begin()));
        }

        auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        CATCH_REQUIRE_NOTHROW(index.search(results.view(), queries.cview(), search_params));
        for (size_t q = 0; q < queries.size(); ++q) {
            for (size_t i = 0; i < num_neighbors; ++i) {
                CATCH_REQUIRE(!deleted_set.contains(results.index(q, i)));
            }
        }
    }

    CATCH_SECTION("Consolidate before compact is unaffected by the stale-edge fix") {
        index.consolidate();
        index.debug_check_graph_consistency(false);

        // Capture pre-compact adjacency lengths (post-consolidate, so no stale
        // edges remain) to prove the new filtering drops nothing when it is a
        // no-op: every remapped list must keep its pre-compact length.
        auto surviving = index.nonmissing_indices();
        std::unordered_map<size_t, size_t> pre_sizes;
        for (auto old_id : surviving) {
            pre_sizes[old_id] = index.view_graph().get_node(old_id).size();
        }

        CATCH_REQUIRE_NOTHROW(index.compact());
        index.debug_check_graph_consistency(false);
        CATCH_REQUIRE(index.size() == surviving.size());

        size_t new_id = 0;
        for (auto old_id : surviving) {
            CATCH_REQUIRE(
                index.view_graph().get_node(new_id).size() == pre_sizes.at(old_id)
            );
            ++new_id;
        }

        for (auto id : ids_to_delete) {
            CATCH_REQUIRE(!index.has_id(id));
        }

        auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        CATCH_REQUIRE_NOTHROW(index.search(results.view(), queries.cview(), search_params));
        for (size_t q = 0; q < queries.size(); ++q) {
            for (size_t i = 0; i < num_neighbors; ++i) {
                CATCH_REQUIRE(!deleted_set.contains(results.index(q, i)));
            }
        }
    }
}

namespace {
// Build a fresh index over the standard test dataset. External ids are assigned
// `0..data_size` in order, so a slot's external and internal id coincide until the
// index is mutated.
template <typename Distance>
auto build_dynamic_index(const Distance& distance, size_t num_threads) {
    auto data = test_dataset::data_blocked_f32();
    const size_t data_size = data.size();
    std::vector<size_t> indices(data_size);
    std::iota(indices.begin(), indices.end(), 0);

    svs::index::vamana::VamanaBuildParameters parameters{1.2, 64, 10, 20, 10, true};
    return svs::index::vamana::MutableVamanaIndex(
        parameters, std::move(data), indices, distance, num_threads
    );
}

// Every fifth id, skipping the entry point so deletion never has to special-case it.
template <typename Index> std::vector<size_t> pick_ids_to_delete(const Index& index) {
    const size_t entry_point = index.entry_point();
    const size_t data_size = index.view_graph().n_nodes();
    std::vector<size_t> ids;
    for (size_t i = 0; i < data_size; i += 5) {
        if (i != entry_point) {
            ids.push_back(i);
        }
    }
    return ids;
}
} // namespace

CATCH_TEST_CASE(
    "MutableVamana Index Rolling Consolidation", "[graph_index][dynamic_index][consolidate]"
) {
    const size_t num_threads = 2;
    using Distance = svs::distance::DistanceL2;
    const Distance distance{};

    CATCH_SECTION("Slices sweeping the full range match one full consolidate()") {
        // Single-threaded builds so the two graphs start out bit-identical; the
        // initial build's edge set is not conserved across parallel runs.
        auto index_a = build_dynamic_index(distance, 1);
        auto index_b = build_dynamic_index(distance, 1);
        CATCH_REQUIRE(index_a.entry_point() == index_b.entry_point());

        auto ids_to_delete = pick_ids_to_delete(index_a);
        index_a.delete_entries(ids_to_delete);
        index_b.delete_entries(ids_to_delete);

        index_a.consolidate();

        // A batch size that does not evenly divide the node count, so the last slice
        // is a short one and the sweep still must land exactly on the end.
        const size_t batch_size = 37;
        const size_t num_nodes = index_b.view_graph().n_nodes();
        size_t swept = 0;
        while (swept < num_nodes) {
            index_b.consolidate_slice(batch_size);
            swept += batch_size;
        }

        CATCH_REQUIRE_NOTHROW(index_a.debug_check_graph_consistency(false));
        CATCH_REQUIRE_NOTHROW(index_b.debug_check_graph_consistency(false));

        CATCH_REQUIRE(index_a.view_graph().n_nodes() == index_b.view_graph().n_nodes());
        size_t mismatched_nodes = 0;
        for (size_t i = 0; i < num_nodes; ++i) {
            auto list_a = index_a.view_graph().get_node(i);
            auto list_b = index_b.view_graph().get_node(i);
            std::vector<uint32_t> sorted_a(list_a.begin(), list_a.end());
            std::vector<uint32_t> sorted_b(list_b.begin(), list_b.end());
            std::sort(sorted_a.begin(), sorted_a.end());
            std::sort(sorted_b.begin(), sorted_b.end());
            if (sorted_a != sorted_b) {
                ++mismatched_nodes;
            }
        }
        // Greedy pruning breaks exact distance ties by candidate-set iteration
        // order, which depends on the scratch set's capacity history; splitting a
        // sweep into many freshly-scoped calls can thus disagree with one big call
        // on the rare exactly-tied candidate. Almost all nodes must still match.
        CATCH_REQUIRE(mismatched_nodes < num_nodes / 100);
    }

    CATCH_SECTION("A slot is reused only after the reclamation delay") {
        auto index = build_dynamic_index(distance, num_threads);
        auto ids_to_delete = pick_ids_to_delete(index);
        const size_t victim = ids_to_delete.front();
        const size_t victim_slot = index.translate_external_id(victim);
        index.delete_entries(ids_to_delete);

        const size_t batch_size = 41;
        auto sweep_one_revolution = [&]() {
            const size_t total = index.view_graph().n_nodes();
            size_t swept = 0;
            while (swept < total) {
                index.consolidate_slice(batch_size);
                swept += batch_size;
            }
        };

        auto add_one_point = [&](size_t external_id) {
            auto point = svs::data::SimpleData<float, svs::Dynamic>(1, index.dimensions());
            point.set_datum(0, std::vector<float>(index.dimensions(), 0.0f));
            std::vector<size_t> ids{external_id};
            return index.add_points(point, ids, true);
        };

        // One revolution: the victim's own deletion state is still current, so it is
        // not the one freed; the reclamation rule requires a second revolution.
        sweep_one_revolution();
        auto first_slots = add_one_point(1'000'000);
        CATCH_REQUIRE(first_slots.front() != victim_slot);

        sweep_one_revolution();
        auto second_slots = add_one_point(1'000'001);
        CATCH_REQUIRE(second_slots.front() == victim_slot);
    }

    CATCH_SECTION("compact() after a partial set of slices does not throw") {
        auto index = build_dynamic_index(distance, num_threads);
        auto ids_to_delete = pick_ids_to_delete(index);
        index.delete_entries(ids_to_delete);

        // Sweep less than one full revolution, leaving the cursor mid-way.
        index.consolidate_slice(index.view_graph().n_nodes() / 3);

        CATCH_REQUIRE_NOTHROW(index.compact());
        CATCH_REQUIRE_NOTHROW(index.debug_check_graph_consistency(false));

        const size_t num_neighbors = 10;
        auto queries = test_dataset::queries();
        auto search_params = svs::index::vamana::VamanaSearchParameters{};
        search_params.buffer_config_ =
            svs::index::vamana::SearchBufferConfig{num_neighbors};
        auto results = svs::QueryResult<size_t>(queries.size(), num_neighbors);
        CATCH_REQUIRE_NOTHROW(index.search(results.view(), queries.cview(), search_params));

        std::unordered_set<size_t> deleted_set(ids_to_delete.begin(), ids_to_delete.end());
        for (size_t q = 0; q < queries.size(); ++q) {
            for (size_t i = 0; i < num_neighbors; ++i) {
                CATCH_REQUIRE(!deleted_set.contains(results.index(q, i)));
            }
        }
    }

    CATCH_SECTION("compact() repairs an edge deleted after its source was swept") {
        auto index = build_dynamic_index(distance, num_threads);
        const size_t entry_point = index.entry_point();

        // A node whose entire neighbor list can be deleted without touching the
        // entry point, so every one of its edges becomes a stale edge below.
        size_t witness = 0;
        bool found = false;
        for (size_t i = 0; i < index.view_graph().n_nodes(); ++i) {
            if (i == entry_point) {
                continue;
            }
            auto neighbors = index.view_graph().get_node(i);
            bool has_entry_point =
                std::find(neighbors.begin(), neighbors.end(), entry_point) !=
                neighbors.end();
            if (!neighbors.empty() && !has_entry_point) {
                witness = i;
                found = true;
                break;
            }
        }
        CATCH_REQUIRE(found);
        CATCH_REQUIRE(witness + 1 < index.view_graph().n_nodes());

        auto neighbors = index.view_graph().get_node(witness);
        std::vector<size_t> victims(neighbors.begin(), neighbors.end());

        // Sweep past `witness` while every one of its neighbors is still `Valid`, so
        // this revolution will not revisit it: this is the witness from the AR-9
        // follow-up, where a slot deleted after its source was already swept escapes
        // a drain that only covers `[cursor, n_nodes)`.
        index.consolidate_slice(witness + 1);
        index.delete_entries(victims);

        index.compact();
        CATCH_REQUIRE_NOTHROW(index.debug_check_graph_consistency(false));

        // Every original neighbor was deleted, so a repaired witness must have been
        // given a fresh, live replacement list; a dropped-not-repaired witness has
        // none, since compact()'s stale-edge filter removes rather than replaces.
        size_t new_witness = index.translate_external_id(witness);
        CATCH_REQUIRE(!index.view_graph().get_node(new_witness).empty());
    }
}
