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
#include "svs/lib/concurrency/readwrite_protected.h"
#include "svs/lib/threads.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <array>
#include <chrono>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

// A collection of long vectors that take a non-negligible amount of time to copy.
std::array<std::vector<std::string>, 3>
create_heavy_objects(size_t elements_per_vector = 20, size_t chars_per_string = 1000) {
    constexpr std::string_view chars = "0123456789abcdefghijklmnopqrstuvwxyz";
    auto ret = std::array<std::vector<std::string>, 3>{};

    auto rng = std::default_random_engine();
    auto dist = std::uniform_int_distribution<size_t>(0, chars.size() - 1);

    for (size_t i = 0; i < ret.size(); ++i) {
        auto v = std::vector<std::string>();
        for (size_t j = 0; j < elements_per_vector; ++j) {
            auto str = std::string();
            for (size_t k = 0; k < chars_per_string; ++k) {
                str.push_back(chars[dist(rng)]);
            }
            v.push_back(str);
        }
        ret[i] = std::move(v);
    }
    return ret;
}

template <typename T, size_t N>
bool test_in(const T& needle, const std::array<T, N>& haystack) {
    for (size_t i = 0; i < N; ++i) {
        if (needle == haystack[i]) {
            return true;
        }
    }
    return false;
}

void stress_test() {
    // Test setup
    size_t num_writers = 2; // Number of writer threads
    size_t num_readers = 2; // Number of reader threads

    size_t completed_writes = 10; // Number of required successful writes per thread
    size_t completed_reads = 30;  // Number of required successful reads per thread

    // Results from reader threads.
    // Acquire the lock before appending.
    auto results = std::vector<std::vector<std::string>>();
    auto result_mutex = std::mutex();

    const auto source = create_heavy_objects();

    // The device under test.
    auto dut = svs::lib::ReadWriteProtected<std::vector<std::string>>(source[0]);

    // The writer job.
    auto writer = [&dut, &source, completed_writes]() {
        auto rng = std::default_random_engine();
        auto sleep_dist = std::uniform_int_distribution<size_t>(1, 5);
        auto source_dist = std::uniform_int_distribution<size_t>(0, source.size() - 1);

        for (size_t i = 0; i < completed_writes; ++i) {
            // Switch between copy construction and move construction.
            const auto& s = source[source_dist(rng)];
            if (i % 2 == 0) {
                dut.set(s);
            } else {
                auto copy = s;
                dut.set(std::move(copy));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_dist(rng)));
        }
    };

    auto reader = [&dut, &results, &result_mutex, completed_reads]() {
        auto rng = std::default_random_engine();
        auto sleep_dist = std::uniform_int_distribution<size_t>(1, 2);

        for (size_t i = 0; i < completed_reads; ++i) {
            auto result = dut.get();
            {
                std::lock_guard lock{result_mutex};
                results.push_back(std::move(result));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_dist(rng)));
        }
    };

    auto job = [&reader, &writer, num_writers](size_t tid) {
        if (tid < num_writers) {
            writer();
        } else {
            reader();
        }
    };

    // Run all jobs.
    auto threadpool = svs::threads::NativeThreadPool(num_writers + num_readers);
    threadpool.run(svs::threads::FunctionRef{job});

    // Make sure the final results make sense.
    // * Final count is correct.
    // * All observed reads are correct and untorn.
    CATCH_REQUIRE(results.size() == num_readers * completed_reads);
    for (size_t i = 0; i < results.size(); ++i) {
        CATCH_REQUIRE(test_in(results[i], source));
    }

    // Make sure all source object made it into the results
    for (const auto& src : source) {
        CATCH_REQUIRE(std::find(results.begin(), results.end(), src) != results.end());
    }
}

} // namespace

CATCH_TEST_CASE("ReadWriteProtected", "[lib][concurrency]") { stress_test(); }
