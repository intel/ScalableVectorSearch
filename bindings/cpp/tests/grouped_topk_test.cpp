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

#include "grouped_topk.h"
#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <utility>
#include <vector>

namespace {

struct TestNeighbor {
    size_t id_;
    float distance_;
    size_t id() const { return id_; }
    float distance() const { return distance_; }
};

class ScriptedIterator {
  public:
    explicit ScriptedIterator(std::vector<std::vector<TestNeighbor>> batches)
        : batches_(std::move(batches)) {}

    void next(size_t /*batch_size*/) {
        if (current_ < batches_.size()) {
            results_ = batches_[current_++];
        } else {
            results_.clear();
        }
    }
    size_t size() const { return results_.size(); }
    const std::vector<TestNeighbor>& results() const { return results_; }
    bool done() const { return current_ == batches_.size(); }
    size_t batches_pulled() const { return current_; }

  private:
    std::vector<std::vector<TestNeighbor>> batches_;
    std::vector<TestNeighbor> results_;
    size_t current_ = 0;
};

const auto closer = [](float a, float b) { return a < b; };

} // namespace

CATCH_TEST_CASE("GroupedTopkConsumesWholeBatch", "[runtime][grouped_search]") {
    test_utils::IDGrouperTable grouper({0, 0, 1, 0});
    ScriptedIterator iterator({{{0, 10.0f}, {1, 11.0f}}, {{2, 1.0f}, {3, 2.0f}}});

    auto best =
        svs::runtime::grouped_topk(iterator, 2, nullptr, 0.0f, grouper, closer, 2, 100);

    CATCH_REQUIRE(best.size() == 2);
    CATCH_REQUIRE(best[0].id() == 2);
    CATCH_REQUIRE(best[1].id() == 3);
    CATCH_REQUIRE(best[1].distance() == 2.0f);
    CATCH_REQUIRE(iterator.batches_pulled() == 2);
}

CATCH_TEST_CASE("GroupedTopkNoGroupIdsAreTheirOwnGroups", "[runtime][grouped_search]") {
    test_utils::IDGrouperNone grouper;
    ScriptedIterator iterator({{{0, 1.0f}, {1, 2.0f}}, {{2, 3.0f}, {3, 4.0f}}});

    auto best =
        svs::runtime::grouped_topk(iterator, 3, nullptr, 0.0f, grouper, closer, 2, 100);

    CATCH_REQUIRE(best.size() == 3);
    CATCH_REQUIRE(best[0].id() == 0);
    CATCH_REQUIRE(best[1].id() == 1);
    CATCH_REQUIRE(best[2].id() == 2);
}

CATCH_TEST_CASE("GroupedTopkKeepsClosestGroups", "[runtime][grouped_search]") {
    test_utils::IDGrouperTable grouper({0, 1, 2});
    ScriptedIterator iterator({{{0, 10.0f}}, {{1, 1.0f}, {2, 2.0f}}});

    auto best =
        svs::runtime::grouped_topk(iterator, 2, nullptr, 0.0f, grouper, closer, 1, 100);

    CATCH_REQUIRE(best.size() == 2);
    CATCH_REQUIRE(best[0].id() == 1);
    CATCH_REQUIRE(best[1].id() == 2);
}

CATCH_TEST_CASE("GroupedTopkFilterStopGivesUp", "[runtime][grouped_search]") {
    test_utils::IDGrouperTable grouper({0, 1, 2, 3});
    test_utils::IDFilterRange filter(0, 1);
    ScriptedIterator iterator({{{0, 1.0f}, {1, 2.0f}, {2, 3.0f}, {3, 4.0f}}, {{4, 5.0f}}});

    auto best =
        svs::runtime::grouped_topk(iterator, 2, &filter, 0.5f, grouper, closer, 4, 100);

    CATCH_REQUIRE(best.empty());
    CATCH_REQUIRE(iterator.batches_pulled() == 1);
}

CATCH_TEST_CASE("GroupedBestKeepsBestPerGroup", "[runtime][grouped_search]") {
    svs::runtime::GroupedBest<TestNeighbor> best(4);
    best.offer(TestNeighbor{0, 5.0f}, 0, closer);
    best.offer(TestNeighbor{1, 6.0f}, 1, closer);
    best.offer(TestNeighbor{2, 7.0f}, svs::runtime::IDGrouper::NO_GROUP, closer);
    best.offer(TestNeighbor{3, 1.0f}, 0, closer);
    best.offer(TestNeighbor{4, 8.0f}, 1, closer);
    best.offer(TestNeighbor{5, 2.0f}, svs::runtime::IDGrouper::NO_GROUP, closer);
    CATCH_REQUIRE(best.size() == 4);

    auto kept = best.take();
    CATCH_REQUIRE(kept.size() == 4);
    CATCH_REQUIRE(kept[0].id() == 3);
    CATCH_REQUIRE(kept[1].id() == 1);
    CATCH_REQUIRE(kept[2].id() == 2);
    CATCH_REQUIRE(kept[3].id() == 5);
    CATCH_REQUIRE(best.size() == 0);
}
