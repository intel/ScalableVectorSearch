/**
 *    Copyright (C) 2025 Intel Corporation
 *
 *    This software and the related documents are Intel copyrighted materials,
 *    and your use of them is governed by the express license under which they
 *    were provided to you ("License"). Unless the License provides otherwise,
 *    you may not use, modify, copy, publish, distribute, disclose or transmit
 *    this software or the related documents without Intel's prior written
 *    permission.
 *
 *    This software and the related documents are provided as is, with no
 *    express or implied warranties, other than those that are expressly stated
 *    in the License.
 */

// svs
#include "svs/index/vamana/iterator.h"
#include "svs/quantization/scalar/scalar.h"

// svstest
#include "tests/utils/test_dataset.h"
#include "tests/utils/vamana_reference.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace scalar = svs::quantization::scalar;

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
    const auto batchsizes = std::vector<size_t>{{10, 20, 25, 50, 100}};

    CATCH_REQUIRE(index.size() > num_neighbors);
    auto p = svs::index::vamana::VamanaSearchParameters{
        {num_neighbors, num_neighbors}, false, 0, 0};

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

        size_t num_matches =
            svs::lib::count_intersect(id_buffer, groundtruth.get_datum(query_index));
        // Ensure we have reasonable recall between.
        CATCH_REQUIRE(num_matches >= 0.80 * num_neighbors);

        // Begin performing batch searches.
        for (auto batchsize : batchsizes) {
            CATCH_REQUIRE(num_neighbors % batchsize == 0);
            size_t num_batches = num_neighbors / batchsize;

            // Initialize the base search parameters with more than the configured batch
            // size. This will check that the internal limiting mechanisms only return
            // at most `batchsize` elements.
            auto sp = svs::index::vamana::VamanaSearchParameters{
                {batchsize + 10, batchsize + 10}, false, 0, 0};

            auto iterator = svs::index::vamana::BatchIterator{
                index, query, svs::index::vamana::DefaultSchedule{sp, batchsize}};

            // TODO: how do we communicate if something goes wrong on the on the iterator
            // end and we cannot return `batch_size()` neighbors?
            CATCH_REQUIRE(iterator.size() == batchsize);

            from_iterator.clear();
            size_t similar_count = 0;

            // IDs returned from the most recent batch.
            // Keep track of this because we want to ensure that if an exception is thrown
            // during search, the state of the iterator is unchanged.
            auto ids_returned_this_batch = std::vector<size_t>();
            for (size_t batch = 0; batch < num_batches; ++batch) {
                // Make sure the batch number is the same.
                CATCH_REQUIRE(iterator.batch() == batch);
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

                iterator.next();
            }

            // Make sure the expected number of neighbors has been obtained.
            CATCH_REQUIRE(from_iterator.size() == num_neighbors);

            // Ensure that the results returned by the iterator are "substantively similar"
            // to those returned from the full search.
            CATCH_REQUIRE(similar_count >= 0.90 * num_neighbors);
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

template <typename Distance, typename Data>
void static_index_with_iterator(const Distance& distance, Data data) {
    auto queries = test_dataset::queries();
    auto gt = test_dataset::groundtruth_euclidean();

    auto index = svs::index::vamana::auto_assemble(
        test_dataset::vamana_config_file(),
        test_dataset::graph(),
        std::move(data),
        distance,
        1
    );
    check(index, queries.cview(), gt.cview());
}

template <typename Distance, typename Data>
void dynamic_index_with_iterator(const Distance& distance, Data data) {
    auto queries = test_dataset::queries();
    auto gt = test_dataset::groundtruth_euclidean();
    auto original = test_dataset::data_f32();

    auto index = svs::index::vamana::auto_dynamic_assemble(
        test_dataset::vamana_config_file(),
        test_dataset::graph(),
        std::move(data),
        distance,
        1,
        true // debug_load_from_static
    );

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
        auto itr = std::find(ids_to_delete.begin(), ids_to_delete.end(), nearest_neighbor);
        if (itr == ids_to_delete.end()) {
            ids_to_delete.push_back(nearest_neighbor);
            CATCH_REQUIRE(valid_ids.erase(nearest_neighbor) == 1);
            CATCH_REQUIRE(checker.seen_.contains(nearest_neighbor));
        }
    }

    index.delete_entries(ids_to_delete);
    checker.clear();
    check(index, queries.cview(), gt.cview(), checker);

    for (auto id : ids_to_delete) {
        CATCH_REQUIRE(!checker.seen_.contains(id));
    }

    // Compact and consolidate.
    index.consolidate();
    index.compact();

    checker.clear();
    check(index, queries.cview(), gt.cview(), checker);
    for (auto id : ids_to_delete) {
        CATCH_REQUIRE(!checker.seen_.contains(id));
    }

    // Add back the points we deleted and try again.
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

CATCH_TEST_CASE("LVQ/Leanvec Vamana Iterator", "[integration][vamana][iterator][scalar]") {
    auto dist = svs::distance::DistanceL2();
    auto original = test_dataset::data_f32();
    constexpr size_t E = 128;

    CATCH_SECTION("Static Index") {
        static_index_with_iterator(
            dist, scalar::SQDataset<std::int8_t, E>::compress(original)
        );
    }

    CATCH_SECTION("Dynamic Index") {
        using A = svs::lib::Allocator<std::int8_t>;
        using blocked_type = svs::data::Blocked<A>;

        dynamic_index_with_iterator(
            dist, scalar::SQDataset<std::int8_t, E, blocked_type>::compress(original)
        );
    }
}