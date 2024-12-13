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

// svs
#include "svs/orchestrators/vamana.h" // bulk of the dependencies required.

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace {

svs::data::SimpleData<float> initialize_example_data() {
    // Create a dataset with 7 elements, each with 4 dimensions.
    auto data = svs::data::SimpleData<float>(7, 4);

    // Fill data `i` with all `i`s.
    auto buffer = std::vector<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        auto fi = svs::lib::narrow<float>(i);
        buffer = {fi, fi, fi, fi};
        data.set_datum(i, buffer);
    }
    return data;
}

svs::Vamana make_example_index() {
    // Build the index.
    auto build_parameters =
        svs::index::vamana::VamanaBuildParameters{1.2, 16, 32, 16, 16, true};
    return svs::Vamana::build<float>(
        build_parameters, initialize_example_data(), svs::DistanceL2{}
    );
}

void test_default_schedule() {
    auto index = make_example_index();

    // Base search parameters for the iterator schedule.
    auto base_parameters = svs::index::vamana::VamanaSearchParameters{}.buffer_config({4});

    // The default schedule take base parameters and a batch size.
    // On each iteration,
    auto schedule = svs::index::vamana::DefaultSchedule{base_parameters, 3};

    // Create a batch iterator over the index for the query.
    // After the constructor returns, the contents of the first batch will be available.
    auto itr = [&]() {
        // Construct a query a query in a scoped block to demonstrate that the iterator
        // maintains an internal copy.
        auto query = std::vector<float>{3.25, 3.25, 3.25, 3.25};
        return index.batch_iterator(svs::lib::as_const_span(query), schedule);
    }();

    // The iterator was configured to yield three neighbors on each invocation.
    // This information is available through the `size()` method.
    CATCH_REQUIRE(itr.size() == 3);
    // There are more neighbors to return.
    CATCH_REQUIRE(!itr.done());
    // The current batch of neighbors is for batch 0.
    CATCH_REQUIRE(itr.batch() == 0);

    // Obtain a view of the current list candidates.
    std::span<const svs::Neighbor<size_t>> results = itr.results();
    CATCH_REQUIRE(results.size() == 3);

    // We constructed the dataset in such a way that we know what the results should be.
    CATCH_REQUIRE(results[0].id() == 3);
    CATCH_REQUIRE(results[1].id() == 4);
    CATCH_REQUIRE(results[2].id() == 2);

    // Once we've finished with the current batch of neighbors, we can step the iterator
    // to the next batch.
    //
    // Using the `DefaultSchedule`, we will retrieve at most 3 new candidates.
    itr.next();
    CATCH_REQUIRE(itr.size() == 3);
    CATCH_REQUIRE(!itr.done());
    CATCH_REQUIRE(itr.batch() == 1);

    results = itr.results();
    CATCH_REQUIRE(results[0].id() == 5);
    CATCH_REQUIRE(results[1].id() == 1);
    CATCH_REQUIRE(results[2].id() == 6);

    // So far, the iterator has yielded 6 of the 7 vectors in the dataset.
    // This call to `next()` should only yield a single neighbor - the last on in the index.
    itr.next();
    CATCH_REQUIRE(itr.size() == 1);
    CATCH_REQUIRE(itr.done());
    CATCH_REQUIRE(itr.batch() == 2);
    results = itr.results();
    CATCH_REQUIRE(results[0].id() == 0);

    // Calling `next()` again should yield no more candidates.
    itr.next();
    CATCH_REQUIRE(itr.size() == 0);
    CATCH_REQUIRE(itr.batch() == 2);
    CATCH_REQUIRE(itr.done());

    itr.next();
    CATCH_REQUIRE(itr.size() == 0);
    CATCH_REQUIRE(itr.batch() == 2);
    CATCH_REQUIRE(itr.done());

    // Update with a new schedule.
    {
        auto newquery = std::vector<float>{2.25, 2.25, 2.25, 2.25};
        auto schedule = svs::index::vamana::DefaultSchedule{base_parameters, 4};
        itr.update(svs::lib::as_const_span(newquery), schedule);
    }
    CATCH_REQUIRE(itr.batch() == 0);
    CATCH_REQUIRE(itr.size() == 4);
    CATCH_REQUIRE(!itr.done());
    results = itr.results();
    CATCH_REQUIRE(results[0].id() == 2);
    CATCH_REQUIRE(results[1].id() == 3);
    CATCH_REQUIRE(results[2].id() == 1);
    CATCH_REQUIRE(results[3].id() == 4);

    itr.next();
    CATCH_REQUIRE(itr.batch() == 1);
    CATCH_REQUIRE(itr.size() == 3);
    CATCH_REQUIRE(itr.done());

    results = itr.results();
    CATCH_REQUIRE(results[0].id() == 0);
    CATCH_REQUIRE(results[1].id() == 5);
    CATCH_REQUIRE(results[2].id() == 6);

    itr.next();
    CATCH_REQUIRE(itr.size() == 0);
    CATCH_REQUIRE(itr.batch() == 1);
    CATCH_REQUIRE(itr.done());

    itr.next();
    CATCH_REQUIRE(itr.size() == 0);
    CATCH_REQUIRE(itr.batch() == 1);
    CATCH_REQUIRE(itr.done());
}
} // namespace

CATCH_TEST_CASE("Vamana Iterator Example", "[index][vamana][iterator]") {
    test_default_schedule();
}
