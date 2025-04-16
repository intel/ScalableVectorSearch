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

void demonstrate_iterator() {
    //! [Setup]
    auto index = make_example_index();

    // Each iteration will yield 3 elements that have not been yielded previously.
    size_t batchsize = 3;

    // Create a batch iterator over the index for the query.
    auto itr = [&]() {
        // Construct a query a query in a scoped block to demonstrate that the iterator
        // maintains an internal copy.
        auto query = std::vector<float>{3.25, 3.25, 3.25, 3.25};

        // Make a batch iterator for the query.
        return index.batch_iterator(svs::lib::as_const_span(query));
    }();
    //! [Setup]

    //! [First Iteration]
    itr.next(batchsize);
    CHECK(itr.size() == 3);
    CHECK(itr.batch_number() == 1);

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
    //! [First Iteration]

    //! [Next Iteration]
    // This will yield the next batch of neighbors.
    itr.next(batchsize);
    CHECK(itr.size() == 3);

    // The contents of the iterator are for batch 2.
    CHECK(itr.batch_number() == 2);

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
    itr.next(batchsize);
    CHECK(itr.size() == 1);
    CHECK(itr.done());
    results = itr.results();
    fmt::print("Neighbor 6 = {}\n", results[0].id());
    CHECK(results[0].id() == 0);
    //! [Final Iteration]

    //! [Beyond Final Iteration]
    // Calling `next()` again should yield no more candidates.
    itr.next(batchsize);
    CHECK(itr.size() == 0);
    CHECK(itr.done());
    //! [Beyond Final Iteration]
}

// Alternative main definition
int svs_main(std::vector<std::string> SVS_UNUSED(args)) {
    demonstrate_iterator();
    return 0;
}

SVS_DEFINE_MAIN();
//! [Example All]
