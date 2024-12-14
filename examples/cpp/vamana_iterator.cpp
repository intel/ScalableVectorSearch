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

//! [Example All]

//! [Includes]
// SVS Dependencies
#include "svs/orchestrators/vamana.h" // bulk of the dependencies required.

// Alternative main definition
#include "svsmain.h"

// stl
#include <string>
#include <string_view>
#include <vector>
//! [Includes]

void check(bool value, std::string_view expr, svs::lib::detail::LineInfo linfo) {
    if (!value) {
        throw svs::lib::ANNException(
            "expression \"{}\" evaluated to false in {}", expr, linfo
        );
    }
}

#define CHECK(expr) check(expr, #expr, SVS_LINEINFO);

//! [Example Index Construction]
[[nodiscard]] svs::Vamana make_example_index() {
    // Build the index.
    auto build_parameters = svs::index::vamana::VamanaBuildParameters{
        1.2, // alpha
        16,  // graph_max_degree
        32,  // window_size
        16,  // max_candidate_pool_size
        16,  // prune_to
        true // use_full_search_history
    };

    // Create a dataset with 7 elements, each with 4 dimensions.
    // The contents of the dataset are
    // {0, 0, 0, 0}
    // {1, 1, 1, 1}
    // ...
    // {6, 6, 6, 6}
    auto data = svs::data::SimpleData<float>(7, 4);

    // Fill data `i` with all `i`s.
    auto buffer = std::vector<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        auto fi = svs::lib::narrow<float>(i);
        buffer = {fi, fi, fi, fi};
        data.set_datum(i, buffer);
    }

    // Build the index.
    return svs::Vamana::build<float>(build_parameters, std::move(data), svs::DistanceL2{});
}
//! [Example Index Construction]

void demonstrate_default_schedule() {
    //! [Setup]
    auto index = make_example_index();

    // Base search parameters for the iterator schedule.
    // This uses a search window size/capacity of 4.
    auto base_parameters = svs::index::vamana::VamanaSearchParameters{}.buffer_config({4});

    // The default schedule take base parameters and a batch size.
    // Each iteration will yield 3 elements that have not been yielded previously.
    size_t batchsize = 3;
    auto schedule = svs::index::vamana::DefaultSchedule{base_parameters, batchsize};

    // Create a batch iterator over the index for the query.
    // After the constructor returns, the contents of the first batch will be available.
    auto itr = [&]() {
        // Construct a query a query in a scoped block to demonstrate that the iterator
        // maintains an internal copy.
        auto query = std::vector<float>{3.25, 3.25, 3.25, 3.25};

        // Make a batch iterator for the query using the provided schedule.
        return index.batch_iterator(svs::lib::as_const_span(query), schedule);
    }();
    //! [Setup]

    //! [Initial Checks]
    // The iterator was configured to yield three neighbors on each invocation.
    // This information is available through the `size()` method.
    CHECK(itr.size() == 3);

    // The contents of the iterator are for batch 0.
    CHECK(itr.batch() == 0);

    // Obtain a view of the current list candidates.
    std::span<const svs::Neighbor<size_t>> results = itr.results();
    CHECK(results.size() == 3);

    // We constructed the dataset in such a way that we know what the results should be.
    fmt::print("Neighbor 0 = {}\n", results[0].id());
    fmt::print("Neighbor 1 = {}\n", results[1].id());
    fmt::print("Neighbor 2 = {}\n", results[2].id());
    CHECK(!itr.done());
    CHECK(results[0].id() == 3);
    CHECK(results[1].id() == 4);
    CHECK(results[2].id() == 2);
    //! [Initial Checks]

    //! [Next Iteration]
    // Once we've finished with the current batch of neighbors, we can step the iterator
    // to the next batch.
    //
    // Using the `DefaultSchedule`, we will retrieve at most 3 new candidates.
    itr.next();
    CHECK(itr.size() == 3);

    // The contents of the iterator are for batch 1.
    CHECK(itr.batch() == 1);

    // Update and inspect the results.
    results = itr.results();
    fmt::print("Neighbor 3 = {}\n", results[0].id());
    fmt::print("Neighbor 4 = {}\n", results[1].id());
    fmt::print("Neighbor 5 = {}\n", results[2].id());
    CHECK(!itr.done());
    CHECK(results[0].id() == 5);
    CHECK(results[1].id() == 1);
    CHECK(results[2].id() == 6);
    //! [Next Iteration]

    //! [Final Iteration]
    // So far, the iterator has yielded 6 of the 7 vectors in the dataset.
    // This call to `next()` should only yield a single neighbor - the last on in the index.
    itr.next();
    CHECK(itr.size() == 1);
    CHECK(itr.done());
    results = itr.results();
    fmt::print("Neighbor 6 = {}\n", results[0].id());
    CHECK(results[0].id() == 0);
    //! [Final Iteration]

    //! [Beyond Final Iteration]
    // Calling `next()` again should yield no more candidates.
    itr.next();
    CHECK(itr.size() == 0);
    CHECK(itr.done());
    //! [Beyond Final Iteration]
}

// Alternative main definition
int svs_main(std::vector<std::string> SVS_UNUSED(args)) {
    demonstrate_default_schedule();
    return 0;
}

SVS_DEFINE_MAIN();
//! [Example All]
