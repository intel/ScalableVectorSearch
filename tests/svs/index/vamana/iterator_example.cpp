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

void test_iterator() {
    auto index = make_example_index();

    // Set batch size to 3.
    size_t batchsize = 3;

    // Create a batch iterator over the index for the query.
    auto itr = [&]() {
        // Construct a query a query in a scoped block to demonstrate that the iterator
        // maintains an internal copy.
        auto query = std::vector<float>{3.25, 3.25, 3.25, 3.25};
        return index.batch_iterator(svs::lib::as_const_span(query));
    }();

    // Ensure the iterator is initialized correctly. No search happens at this point.
    CATCH_REQUIRE(itr.size() == 0);
    CATCH_REQUIRE(itr.batch_number() == 0);
    auto buffer_config = itr.parameters_for_current_iteration().buffer_config_;
    CATCH_REQUIRE(buffer_config.get_search_window_size() == 0);
    CATCH_REQUIRE(
        buffer_config.get_total_capacity() == SVS_ITERATOR_EXTRA_BUFFER_CAPACITY_DEFAULT
    );

    itr.next(batchsize);

    // The iterator was configured to yield three neighbors on each invocation.
    // This information is available through the `size()` method.
    CATCH_REQUIRE(itr.size() == 3);
    // There are more neighbors to return.
    CATCH_REQUIRE(!itr.done());
    // The current batch of neighbors is for batch 1.
    CATCH_REQUIRE(itr.batch_number() == 1);
    buffer_config = itr.parameters_for_current_iteration().buffer_config_;
    CATCH_REQUIRE(buffer_config.get_search_window_size() == 3);
    CATCH_REQUIRE(
        buffer_config.get_total_capacity() == SVS_ITERATOR_EXTRA_BUFFER_CAPACITY_DEFAULT + 3
    );

    // Obtain a view of the current list candidates.
    std::span<const svs::Neighbor<size_t>> results = itr.results();
    CATCH_REQUIRE(results.size() == 3);

    // We constructed the dataset in such a way that we know what the results should be.
    CATCH_REQUIRE(results[0].id() == 3);
    CATCH_REQUIRE(results[1].id() == 4);
    CATCH_REQUIRE(results[2].id() == 2);

    // This will yield the next batch of neighbors (only 2 in this case).
    itr.next(batchsize - 1);
    CATCH_REQUIRE(itr.size() == batchsize - 1);
    CATCH_REQUIRE(!itr.done());
    CATCH_REQUIRE(itr.batch_number() == 2);

    results = itr.results();
    CATCH_REQUIRE(results[0].id() == 5);
    CATCH_REQUIRE(results[1].id() == 1);

    // So far, the iterator has yielded 5 of the 7 vectors in the dataset.
    // This call to `next()` should only yield two neighbors.
    itr.next(batchsize);
    CATCH_REQUIRE(itr.size() == 2);
    CATCH_REQUIRE(itr.done());
    CATCH_REQUIRE(itr.batch_number() == 3);
    results = itr.results();
    CATCH_REQUIRE(results[0].id() == 6);
    CATCH_REQUIRE(results[1].id() == 0);

    // Calling `next()` again should yield no more candidates.
    itr.next(batchsize);
    CATCH_REQUIRE(itr.size() == 0);
    CATCH_REQUIRE(itr.batch_number() == 3);
    CATCH_REQUIRE(itr.done());

    itr.next(batchsize);
    CATCH_REQUIRE(itr.size() == 0);
    CATCH_REQUIRE(itr.batch_number() == 3);
    CATCH_REQUIRE(itr.done());

    // Update with a new query and increase the batch size.
    {
        auto newquery = std::vector<float>{2.25, 2.25, 2.25, 2.25};
        itr.update(svs::lib::as_const_span(newquery));
    }
    batchsize = 4;
    itr.next(batchsize);

    CATCH_REQUIRE(itr.batch_number() == 1);
    CATCH_REQUIRE(itr.size() == 4);
    CATCH_REQUIRE(!itr.done());
    results = itr.results();
    CATCH_REQUIRE(results[0].id() == 2);
    CATCH_REQUIRE(results[1].id() == 3);
    CATCH_REQUIRE(results[2].id() == 1);
    CATCH_REQUIRE(results[3].id() == 4);

    itr.next(batchsize);
    CATCH_REQUIRE(itr.batch_number() == 2);
    CATCH_REQUIRE(itr.size() == 3);
    CATCH_REQUIRE(itr.done());

    results = itr.results();
    CATCH_REQUIRE(results[0].id() == 0);
    CATCH_REQUIRE(results[1].id() == 5);
    CATCH_REQUIRE(results[2].id() == 6);

    itr.next(batchsize);
    CATCH_REQUIRE(itr.size() == 0);
    CATCH_REQUIRE(itr.batch_number() == 2);
    CATCH_REQUIRE(itr.done());

    itr.next(batchsize + 1);
    CATCH_REQUIRE(itr.size() == 0);
    CATCH_REQUIRE(itr.batch_number() == 2);
    CATCH_REQUIRE(itr.done());

    // Create another instance of batch iterator to test the setting of extra search buffer
    // size than the default value
    size_t extra_buffer_size = 25;
    itr = [&]() {
        auto query = std::vector<float>{3.25, 3.25, 3.25, 3.25};
        return index.batch_iterator(svs::lib::as_const_span(query), extra_buffer_size);
    }();

    // Ensure the iterator is initialized correctly. No search happens at this point.
    CATCH_REQUIRE(itr.size() == 0);
    CATCH_REQUIRE(itr.batch_number() == 0);
    buffer_config = itr.parameters_for_current_iteration().buffer_config_;
    CATCH_REQUIRE(buffer_config.get_search_window_size() == 0);
    CATCH_REQUIRE(buffer_config.get_total_capacity() == extra_buffer_size);

    itr.next(4);
    CATCH_REQUIRE(itr.size() == 4);
    CATCH_REQUIRE(!itr.done());
    CATCH_REQUIRE(itr.batch_number() == 1);
    buffer_config = itr.parameters_for_current_iteration().buffer_config_;
    CATCH_REQUIRE(buffer_config.get_search_window_size() == 4);
    CATCH_REQUIRE(buffer_config.get_total_capacity() == extra_buffer_size + 4);

    // Obtain a view of the current list candidates.
    results = itr.results();
    CATCH_REQUIRE(results.size() == 4);

    // We constructed the dataset in such a way that we know what the results should be.
    CATCH_REQUIRE(results[0].id() == 3);
    CATCH_REQUIRE(results[1].id() == 4);
    CATCH_REQUIRE(results[2].id() == 2);
    CATCH_REQUIRE(results[3].id() == 5);
}
} // namespace

CATCH_TEST_CASE("Vamana Iterator Example", "[index][vamana][iterator]") { test_iterator(); }
