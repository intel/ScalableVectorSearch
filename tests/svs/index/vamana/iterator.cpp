/*
 * Copyright 2024 Intel Corporation
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
#include "svs/index/vamana/iterator.h"
#include "svs/index/vamana/multi.h"

// svstest
#include "tests/utils/test_dataset.h"
#include "tests/utils/vamana_reference.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

// A static countdown to throwing an exception.
// This enables testing the restart functionality of the iterator.
size_t EXCEPTION_COUNTDOWN = 0;

// A wrapper around the L2 distance that throws when the countdown reaches zero.
struct ThrowingL2 {
    using compare = std::less<>;
    static float compute(std::span<const float> left, std::span<const float> right) {
        // If the exception countdown is active (non-zero), decrement the countdown.
        // If this action caused the countdown to hit zero, throw an exception.
        if (EXCEPTION_COUNTDOWN != 0) {
            --EXCEPTION_COUNTDOWN;
            if (EXCEPTION_COUNTDOWN == 0) {
                throw ANNEXCEPTION("Exception countdown triggered!");
            }
        }
        auto real = svs::DistanceL2{};
        return svs::distance::compute(real, left, right);
    }
};

} // namespace

template <> struct svs::index::vamana::PruneStrategy<ThrowingL2> {
    using type = ProgressivePruneStrategy;
};

namespace {

const size_t QUERIES_TO_CHECK = 10;

// Common test routines for the static and dynamic indexes.
template <typename Index, typename IDChecker = svs::lib::Returns<svs::lib::Const<true>>>
void check(
    Index& index,
    svs::data::ConstSimpleDataView<float> queries,
    svs::data::ConstSimpleDataView<uint32_t> groundtruth,
    IDChecker& checker
) {
    const size_t num_neighbors = 100;
    // Through an exception during search every `throw_exception_every` batches.
    const size_t throw_exception_every = 3;
    const auto batchsizes = std::vector<size_t>{{10, 20, 25, 50, 100}};

    CATCH_REQUIRE(index.size() > num_neighbors);
    auto p = svs::index::vamana::VamanaSearchParameters{
        {num_neighbors, num_neighbors}, false, 0, 0
    };

    auto scratch = index.scratchspace(p);

    auto id_to_distance = std::unordered_map<size_t, float>();
    auto id_buffer = std::vector<size_t>();

    CATCH_REQUIRE(checker(id_to_distance));

    auto from_iterator = std::unordered_set<size_t>();
    for (size_t query_index = 0; query_index < QUERIES_TO_CHECK; ++query_index) {
        auto query = queries.get_datum(query_index);

        // Perform a single, full-precision search to obtain reference results.
        index.search(query, scratch);
        const auto& buffer = scratch.buffer;

        id_to_distance.clear();
        id_buffer.clear();
        for (const auto& neighbor : buffer) {
            size_t id = [&]() -> size_t {
                if constexpr (Index::needs_id_translation) {
                    return index.translate_internal_id(neighbor.id());
                } else {
                    return neighbor.id();
                }
            }();
            id_to_distance.insert({id, neighbor.distance()});
            id_buffer.push_back(id);
        }

        // Ensure we have reasonable recall between.
        CATCH_REQUIRE(
            svs::lib::count_intersect(id_buffer, groundtruth.get_datum(query_index)) >=
            0.9 * num_neighbors
        );

        // Begin performing batch searches.
        for (auto batchsize : batchsizes) {
            CATCH_REQUIRE(num_neighbors % batchsize == 0);
            size_t num_batches = num_neighbors / batchsize;

            auto iterator = index.make_batch_iterator(query);
            CATCH_REQUIRE(iterator.size() == 0);
            iterator.next(batchsize);

            from_iterator.clear();
            size_t similar_count = 0;

            // IDs returned from the most recent batch.
            // Keep track of this because we want to ensure that if an exception is thrown
            // during search, the state of the iterator is unchanged.
            auto ids_returned_this_batch = std::vector<size_t>();
            for (size_t batch = 0; batch < num_batches; ++batch) {
                // Make sure the batch number is the same.
                CATCH_REQUIRE(iterator.batch_number() == batch + 1);
                ids_returned_this_batch.clear();
                for (auto i : iterator) {
                    auto id = i.id();
                    // Make sure that this ID has not been returned yet.
                    CATCH_REQUIRE(!from_iterator.contains(id));
                    auto itr = id_to_distance.find(id);
                    if (itr != id_to_distance.end()) {
                        // Make sure the returned distances match.
                        CATCH_REQUIRE(itr->second == i.distance());
                        ++similar_count;
                    }

                    // Insert the ID into the `from_iterator` container to detect for
                    // duplicates from future calls.
                    from_iterator.insert(id);
                    ids_returned_this_batch.push_back(id);
                }

                // The number of IDs returned should equal the number of IDs reported
                // by the iterator.
                CATCH_REQUIRE(ids_returned_this_batch.size() == iterator.size());
                CATCH_REQUIRE(ids_returned_this_batch.size() == batchsize);

                // Now we've extracted the neighbors, decide if we are going to try again
                // but throw an exception.
                //
                // If so, we want to ensure that the buffer is left in a sane state.
                // Furthermore, on the next iteration, we want to make sure we can resume
                // search without incident.
                if (batch % throw_exception_every == 0) {
                    EXCEPTION_COUNTDOWN = 50;
                    CATCH_REQUIRE_THROWS_AS(iterator.next(batchsize), svs::ANNException);
                    // The batch reported by the iterator must be unchanged.
                    CATCH_REQUIRE(iterator.batch_number() == batch + 1);
                    // The contents of the iterator should be unchanged.
                    CATCH_REQUIRE(iterator.size() == ids_returned_this_batch.size());
                    CATCH_REQUIRE(std::equal(
                        iterator.begin(),
                        iterator.end(),
                        ids_returned_this_batch.begin(),
                        [](svs::NeighborLike auto left, size_t right) {
                            return left.id() == right;
                        }
                    ));
                }

                iterator.next(batchsize);
            }

            // Make sure the expected number of neighbors has been obtained.
            CATCH_REQUIRE(from_iterator.size() == num_neighbors);

            // Ensure that the results returned by the iterator are "substantively similar"
            // to those returned from the full search.
            CATCH_REQUIRE(similar_count >= 0.98 * num_neighbors);
        }

        // Invoke the checker on the IDs returned from the iterator.
        CATCH_REQUIRE(checker(from_iterator));
    }
}

template <typename Index>
void check(
    Index& index,
    svs::data::ConstSimpleDataView<float> queries,
    svs::data::ConstSimpleDataView<uint32_t> groundtruth
) {
    auto checker = svs::lib::Returns<svs::lib::Const<true>>();
    check(index, queries, groundtruth, checker);
}

struct DynamicChecker {
    DynamicChecker(const std::unordered_set<size_t>& valid_ids)
        : valid_ids_{valid_ids} {}

    // Check whether `id` is valid or not.
    bool check(size_t id) {
        seen_.insert(id);
        return valid_ids_.contains(id);
    }

    template <std::integral I> bool operator()(const std::unordered_map<I, float>& ids) {
        for (const auto& itr : ids) {
            if (!check(itr.first)) {
                return false;
            }
        }
        return true;
    }

    template <std::integral I> bool operator()(const std::unordered_set<I>& ids) {
        for (auto itr : ids) {
            if (!check(itr)) {
                return false;
            }
        }
        return true;
    }

    void clear() { seen_.clear(); }

    // Valid IDs
    const std::unordered_set<size_t>& valid_ids_;
    std::unordered_set<size_t> seen_;
};

} // namespace

CATCH_TEST_CASE("Vamana Iterator", "[index][vamana][iterator]") {
    // This tests the general behavior of the iterator for correctness.
    // It is not concerned with whether the returned neighbors are accurate.
    //
    // That responsibility is delegated to the integration tests.
    auto queries = test_dataset::queries();
    auto gt = test_dataset::groundtruth_euclidean();
    CATCH_SECTION("Static Index") {
        auto index = test_dataset::vamana::load_test_index(ThrowingL2());
        check(index, queries.cview(), gt.cview());
    }

    // For the dynamic index, iterated search should honor the internal deleted state of
    // IDs.
    CATCH_SECTION("Dynamic Index") {
        auto index = test_dataset::vamana::load_dynamic_test_index(ThrowingL2());
        auto original = test_dataset::data_f32();

        // Increase the number of threads to help a little with run time.
        index.set_threadpool(svs::threads::DefaultThreadPool(2));
        auto itr = svs::threads::UnitRange{0, index.size()};
        auto valid_ids = std::unordered_set<size_t>{itr.begin(), itr.end()};
        auto checker = DynamicChecker{valid_ids};
        check(index, queries.cview(), gt.cview(), checker);

        // Delete the best candidate for each of the test queries.
        auto ids_to_delete = std::vector<size_t>();
        for (size_t i = 0; i < QUERIES_TO_CHECK; ++i) {
            auto nearest_neighbor = gt.get_datum(i).front();
            auto itr =
                std::find(ids_to_delete.begin(), ids_to_delete.end(), nearest_neighbor);
            if (itr == ids_to_delete.end()) {
                ids_to_delete.push_back(nearest_neighbor);
                CATCH_REQUIRE(valid_ids.erase(nearest_neighbor) == 1);
                CATCH_REQUIRE(checker.seen_.contains(nearest_neighbor));
            }
        }

        fmt::print("Deleting\n");
        index.delete_entries(ids_to_delete);
        checker.clear();
        check(index, queries.cview(), gt.cview(), checker);

        for (auto id : ids_to_delete) {
            CATCH_REQUIRE(!checker.seen_.contains(id));
        }

        // Compact and consolidate.
        index.consolidate();
        index.compact();

        fmt::print("Compacting\n");
        checker.clear();
        check(index, queries.cview(), gt.cview(), checker);
        for (auto id : ids_to_delete) {
            CATCH_REQUIRE(!checker.seen_.contains(id));
        }

        // Add back the points we deleted and try again.
        fmt::print("Adding\n");
        auto slots = index.add_points(
            svs::data::make_const_view(original, ids_to_delete), ids_to_delete
        );

        checker.clear();
        for (auto id : ids_to_delete) {
            auto [_, inserted] = valid_ids.insert(id);
            CATCH_REQUIRE(inserted);
        }

        check(index, queries.cview(), gt.cview(), checker);
        for (auto id : ids_to_delete) {
            CATCH_REQUIRE(checker.seen_.contains(id));
        }
    }

    // Multi-vector batch iterator should also pass non-multi-vector tests
    CATCH_SECTION("Multi batch iterator") {
        auto index = svs::index::vamana::auto_multi_dynamic_assemble(
            test_dataset::vamana_config_file(),
            test_dataset::graph(),
            test_dataset::data_f32(),
            ThrowingL2(),
            1,
            svs::index::vamana::MultiMutableVamanaLoad::FROM_STATIC
        );
        auto original = test_dataset::data_f32();
        index.set_threadpool(svs::threads::DefaultThreadPool(2));
        auto itr = svs::threads::UnitRange{0, index.size()};
        auto valid_ids = std::unordered_set<size_t>{itr.begin(), itr.end()};
        auto checker = DynamicChecker{valid_ids};
        check(index, queries.cview(), gt.cview(), checker);

        // Delete the best candidate for each of the test queries.
        auto ids_to_delete = std::vector<size_t>();
        for (size_t i = 0; i < QUERIES_TO_CHECK; ++i) {
            auto nearest_neighbor = gt.get_datum(i).front();
            auto itr =
                std::find(ids_to_delete.begin(), ids_to_delete.end(), nearest_neighbor);
            if (itr == ids_to_delete.end()) {
                ids_to_delete.push_back(nearest_neighbor);
                CATCH_REQUIRE(valid_ids.erase(nearest_neighbor) == 1);
                CATCH_REQUIRE(checker.seen_.contains(nearest_neighbor));
            }
        }

        fmt::print("Deleting\n");
        index.delete_entries(ids_to_delete);
        checker.clear();
        check(index, queries.cview(), gt.cview(), checker);

        for (auto id : ids_to_delete) {
            CATCH_REQUIRE(!checker.seen_.contains(id));
        }

        // Compact and consolidate.
        index.consolidate();
        index.compact();

        fmt::print("Compacting\n");
        checker.clear();
        check(index, queries.cview(), gt.cview(), checker);
        for (auto id : ids_to_delete) {
            CATCH_REQUIRE(!checker.seen_.contains(id));
        }

        // Add back the points we deleted and try again.
        fmt::print("Adding\n");
        auto slots = index.add_points(
            svs::data::make_const_view(original, ids_to_delete), ids_to_delete
        );

        checker.clear();
        for (auto id : ids_to_delete) {
            auto [_, inserted] = valid_ids.insert(id);
            CATCH_REQUIRE(inserted);
        }

        check(index, queries.cview(), gt.cview(), checker);
        for (auto id : ids_to_delete) {
            CATCH_REQUIRE(checker.seen_.contains(id));
        }
    }
}
