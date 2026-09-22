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

// Header under test.
#include "svs-benchmark/latency.h"

// svs
#include "svs/lib/exception.h"

// catch2
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

// stdlib
#include <cmath>
#include <vector>

CATCH_TEST_CASE("LatencyStats", "[benchmark][latency]") {
    CATCH_SECTION("Known-answer: five samples") {
        auto latencies = std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0};
        auto stats = svsbenchmark::compute_latency_stats(latencies);

        CATCH_REQUIRE(stats.count == 5);
        CATCH_REQUIRE(stats.min_time == 1.0);
        CATCH_REQUIRE(stats.max_time == 5.0);
        CATCH_REQUIRE(stats.mean_time == Catch::Approx(3.0));
        // Population variance of {1..5} about mean 3 is (4+1+0+1+4)/5 = 2.
        CATCH_REQUIRE(stats.std_time == Catch::Approx(std::sqrt(2.0)));
        // Nearest-rank: p50 -> ceil(0.5*5)=3 -> sorted[2] = 3.0.
        CATCH_REQUIRE(stats.p50_time == 3.0);
        // p95 -> ceil(0.95*5)=5 -> sorted[4] = 5.0; p99 same rank.
        CATCH_REQUIRE(stats.p95_time == 5.0);
        CATCH_REQUIRE(stats.p99_time == 5.0);
    }

    CATCH_SECTION("Known-answer: ten samples") {
        auto latencies = std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        auto stats = svsbenchmark::compute_latency_stats(latencies);

        CATCH_REQUIRE(stats.count == 10);
        CATCH_REQUIRE(stats.min_time == 1.0);
        CATCH_REQUIRE(stats.max_time == 10.0);
        CATCH_REQUIRE(stats.mean_time == Catch::Approx(5.5));
        // Population variance of {1..10} about mean 5.5 is 8.25.
        CATCH_REQUIRE(stats.std_time == Catch::Approx(std::sqrt(8.25)));
        // p50 -> ceil(0.5*10)=5 -> sorted[4] = 5.0.
        CATCH_REQUIRE(stats.p50_time == 5.0);
        // p95 -> ceil(0.95*10)=10 -> sorted[9] = 10.0; p99 same rank.
        CATCH_REQUIRE(stats.p95_time == 10.0);
        CATCH_REQUIRE(stats.p99_time == 10.0);
    }

    CATCH_SECTION("Histogram recovers percentiles within tolerance") {
        auto latencies = std::vector<double>();
        latencies.reserve(1000);
        for (size_t i = 1; i <= 1000; ++i) {
            latencies.push_back(0.001 * static_cast<double>(i));
        }
        auto stats = svsbenchmark::compute_latency_stats(latencies);

        // 1% relative tolerance comfortably covers the histogram's ~0.2% worst-case bin
        // width at 256 subdivisions per octave.
        CATCH_REQUIRE(
            stats.latency_hdr.percentile(50.0) ==
            Catch::Approx(stats.p50_time).epsilon(0.01)
        );
        CATCH_REQUIRE(
            stats.latency_hdr.percentile(95.0) ==
            Catch::Approx(stats.p95_time).epsilon(0.01)
        );
        CATCH_REQUIRE(
            stats.latency_hdr.percentile(99.0) ==
            Catch::Approx(stats.p99_time).epsilon(0.01)
        );
    }

    CATCH_SECTION("Merging per-thread histograms matches a single-threaded histogram") {
        auto all = std::vector<double>();
        auto first_half = std::vector<double>();
        auto second_half = std::vector<double>();
        all.reserve(2000);
        for (size_t i = 1; i <= 2000; ++i) {
            double v = 0.0005 * static_cast<double>(i);
            all.push_back(v);
            (i <= 1000 ? first_half : second_half).push_back(v);
        }

        auto combined_stats = svsbenchmark::compute_latency_stats(all);

        auto merged = svsbenchmark::LatencyHistogram();
        for (double v : first_half) {
            merged.record(v);
        }
        auto other = svsbenchmark::LatencyHistogram();
        for (double v : second_half) {
            other.record(v);
        }
        merged.merge(other);

        CATCH_REQUIRE(merged.count() == 2000);
        CATCH_REQUIRE(
            merged.percentile(50.0) == Catch::Approx(combined_stats.p50_time).epsilon(0.01)
        );
        CATCH_REQUIRE(
            merged.percentile(99.0) == Catch::Approx(combined_stats.p99_time).epsilon(0.01)
        );

        // Merging histograms with different bucket layouts would silently misattribute
        // counts, so it is rejected instead.
        auto mismatched = svsbenchmark::LatencyHistogram(1e-6, 10.0);
        CATCH_REQUIRE_THROWS_AS(merged.merge(mismatched), svs::ANNException);
    }

    CATCH_SECTION("Edge case: empty input") {
        auto latencies = std::vector<double>();
        auto stats = svsbenchmark::compute_latency_stats(latencies);

        // Defined behavior for empty input: zero count, all time fields zero, and an
        // empty (but usable) histogram rather than a thrown exception.
        CATCH_REQUIRE(stats.count == 0);
        CATCH_REQUIRE(stats.mean_time == 0.0);
        CATCH_REQUIRE(stats.std_time == 0.0);
        CATCH_REQUIRE(stats.min_time == 0.0);
        CATCH_REQUIRE(stats.max_time == 0.0);
        CATCH_REQUIRE(stats.p50_time == 0.0);
        CATCH_REQUIRE(stats.p95_time == 0.0);
        CATCH_REQUIRE(stats.p99_time == 0.0);
        CATCH_REQUIRE(stats.latency_hdr.count() == 0);
        CATCH_REQUIRE(stats.latency_hdr.percentile(50.0) == 0.0);
    }

    CATCH_SECTION("Edge case: single sample") {
        auto latencies = std::vector<double>{0.042};
        auto stats = svsbenchmark::compute_latency_stats(latencies);

        CATCH_REQUIRE(stats.count == 1);
        CATCH_REQUIRE(stats.mean_time == 0.042);
        CATCH_REQUIRE(stats.std_time == 0.0);
        CATCH_REQUIRE(stats.min_time == 0.042);
        CATCH_REQUIRE(stats.max_time == 0.042);
        CATCH_REQUIRE(stats.p50_time == 0.042);
        CATCH_REQUIRE(stats.p95_time == 0.042);
        CATCH_REQUIRE(stats.p99_time == 0.042);
        CATCH_REQUIRE(
            stats.latency_hdr.percentile(50.0) == Catch::Approx(0.042).epsilon(0.01)
        );
    }

    CATCH_SECTION("Edge case: two samples") {
        auto latencies = std::vector<double>{1.0, 3.0};
        auto stats = svsbenchmark::compute_latency_stats(latencies);

        CATCH_REQUIRE(stats.count == 2);
        CATCH_REQUIRE(stats.mean_time == Catch::Approx(2.0));
        CATCH_REQUIRE(stats.std_time == Catch::Approx(1.0));
        CATCH_REQUIRE(stats.min_time == 1.0);
        CATCH_REQUIRE(stats.max_time == 3.0);
        // Nearest-rank with an even sample count is asymmetric: p50 -> ceil(1.0)=1 ->
        // sorted[0]; p95/p99 -> ceil(1.9 or 1.98)=2 -> sorted[1].
        CATCH_REQUIRE(stats.p50_time == 1.0);
        CATCH_REQUIRE(stats.p95_time == 3.0);
        CATCH_REQUIRE(stats.p99_time == 3.0);
    }

    CATCH_SECTION("Edge case: all-identical samples") {
        auto latencies = std::vector<double>(100, 7.5);
        auto stats = svsbenchmark::compute_latency_stats(latencies);

        CATCH_REQUIRE(stats.count == 100);
        CATCH_REQUIRE(stats.mean_time == 7.5);
        CATCH_REQUIRE(stats.std_time == 0.0);
        CATCH_REQUIRE(stats.min_time == 7.5);
        CATCH_REQUIRE(stats.max_time == 7.5);
        CATCH_REQUIRE(stats.p50_time == 7.5);
        CATCH_REQUIRE(stats.p95_time == 7.5);
        CATCH_REQUIRE(stats.p99_time == 7.5);
        CATCH_REQUIRE(
            stats.latency_hdr.percentile(50.0) == Catch::Approx(7.5).epsilon(0.01)
        );
    }
}

CATCH_TEST_CASE("LatencyHistogram construction", "[benchmark][latency]") {
    CATCH_REQUIRE_THROWS_AS(svsbenchmark::LatencyHistogram(-1.0, 10.0), svs::ANNException);
    CATCH_REQUIRE_THROWS_AS(svsbenchmark::LatencyHistogram(10.0, 1.0), svs::ANNException);
}
