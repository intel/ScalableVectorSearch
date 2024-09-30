/*
 * Copyright (C) 2023 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials,
 * and your use of them is governed by the express license under which they
 * were provided to you ("License"). Unless the License provides otherwise,
 * you may not use, modify, copy, publish, distribute, disclose or transmit
 * this software or the related documents without Intel's prior written
 * permission.
 *
 * This software and the related documents are provided as is, with no
 * express or implied warranties, other than those that are expressly stated
 * in the License.
 */
// header under test
#include "svs/lib/algorithms.h"

// test utils
#include "tests/utils/generators.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <random>
#include <vector>

CATCH_TEST_CASE("Algorithms", "[lib][lib-algorithms]") {
    CATCH_SECTION("All unique") {
        auto x = std::vector<int>{0, 2, 5, 100, 4, 99};
        CATCH_REQUIRE(svs::lib::all_unique(x.begin(), x.end()));
        x.push_back(2);
        CATCH_REQUIRE(!svs::lib::all_unique(x.begin(), x.end()));
        x.clear();
        CATCH_REQUIRE(svs::lib::all_unique(x.begin(), x.end()));
    }

    CATCH_SECTION("Bounded Merge") {
        auto sizes = std::vector<size_t>{0, 1, 2, 5, 10};
        auto numtests = 100;
        auto generator = svs_test::make_generator<int32_t>(-100, 100);

        auto source1 = std::vector<int32_t>();
        auto source2 = std::vector<int32_t>();
        auto bounded_dest = std::vector<int32_t>();
        auto reference = std::vector<int32_t>();

        for (size_t s1 : sizes) {
            source1.resize(s1);
            for (size_t s2 : sizes) {
                source2.resize(s2);
                for (int x = 0; x < numtests; ++x) {
                    svs_test::populate(source1, generator);
                    svs_test::populate(source2, generator);
                    std::sort(source1.begin(), source1.end());
                    std::sort(source2.begin(), source2.end());

                    reference.resize(s1 + s2);
                    std::merge(
                        source1.begin(),
                        source1.end(),
                        source2.begin(),
                        source2.end(),
                        reference.begin(),
                        std::less<>()
                    );

                    auto bounded_sizes =
                        std::array<size_t, 5>{{1, (s1 + s2) / 2, s1, s2, s1 + s2}};

                    for (auto s : bounded_sizes) {
                        if (s1 == 0 && s2 == 0 && s > 0) {
                            continue;
                        }
                        bounded_dest.resize(s);
                        svs::lib::ranges::bounded_merge(
                            source1, source2, bounded_dest, std::less<>()
                        );

                        auto eq = std::equal(
                            bounded_dest.begin(), bounded_dest.end(), reference.begin()
                        );
                        if (!eq) {
                            fmt::print("s = {}\n", s);
                            fmt::print("source1 = {}\n", fmt::join(source1, ", "));
                            fmt::print("source2 = {}\n", fmt::join(source2, ", "));
                            fmt::print("Reference: {}\n", fmt::join(reference, ", "));
                            fmt::print("Generated: {}\n", fmt::join(bounded_dest, ", "));
                        }
                        CATCH_REQUIRE(eq);
                    }
                }
            }
        }
    }
}
