/**
 *    Copyright (C) 2023-present, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

// stdlib
#include <atomic>
#include <chrono>
#include <memory>
// #include <omp.h>
#include <random>
#include <thread>
#include <tuple>

// local includes
#include "svs/lib/exception.h"
#include "svs/lib/misc.h"
#include "svs/lib/threads/thread.h"
#include "svs/lib/threads/threadpool.h"

// catch macros
#include "catch2/catch_test_macros.hpp"

CATCH_TEST_CASE("Thread Pool", "[core][threads][threadpool]") {
    using namespace std::chrono_literals;
    CATCH_SECTION("Exceptions") {
        auto v = std::vector<uint64_t>{};
        auto v_mutex = std::mutex{};

        auto pool = svs::threads::NativeThreadPool(4);
        CATCH_SECTION("Just One Thread Crashed") {
            for (size_t i = 0; i < pool.size(); ++i) {
                v.clear();
                CATCH_REQUIRE(v.empty());
                try {
                    svs::threads::run(pool, [&v, &v_mutex, i](uint64_t tid) {
                        if (tid == i) {
                            throw std::runtime_error("This is a test");
                        } else {
                            std::lock_guard lock{v_mutex};
                            v.push_back(tid);
                        }
                    });
                } catch (const svs::threads::ThreadingException& error) {
                    std::string what{error.what()};
                    std::string expected =
                        "Thread " + std::to_string(i) + ": This is a test";
                    auto pos = what.find(expected);
                    CATCH_REQUIRE(pos != std::string::npos);
                }
                // All other items should have been added to the vector.
                for (size_t j = 0; j < pool.size(); ++j) {
                    if (j == i) {
                        continue;
                    }
                    CATCH_REQUIRE(std::find(v.begin(), v.end(), j) != v.end());
                }
            }
        }

        CATCH_SECTION("All Threads Crash") {
            v.clear();
            try {
                svs::threads::run(pool, [](uint64_t tid) {
                    throw std::runtime_error("I crashed " + std::to_string(tid));
                });
            } catch (const svs::threads::ThreadingException& error) {
                std::string what{error.what()};
                for (size_t i = 0; i < pool.size(); ++i) {
                    std::string expected =
                        "Thread " + std::to_string(i) + ": I crashed " + std::to_string(i);
                    auto pos = what.find(expected);
                    CATCH_REQUIRE(pos != std::string::npos);
                }
            }

            // Now try again - all threads should be restarted.
            svs::threads::run(pool, [&v, &v_mutex](uint64_t tid) {
                std::lock_guard lock{v_mutex};
                v.push_back(tid);
            });

            for (size_t j = 0; j < pool.size(); ++j) {
                CATCH_REQUIRE(std::find(v.begin(), v.end(), j) != v.end());
            }
        }
    }

    CATCH_SECTION("Static Partition") {
        namespace threads = svs::threads;
        auto pool = svs::threads::NativeThreadPool(4);
        std::mutex mutex;

        std::vector<uint64_t> seen_threads{};
        std::vector<svs::threads::UnitRange<uint64_t>> ranges{};

        // Make sure that if the number of threads exceeds the number of available work
        // partitions that:
        //
        // 1. Everything remains within bounds.
        // 2. Threads that have no work are never launched.
        CATCH_SECTION("No Oversubscription") {
            threads::run(
                pool,
                threads::StaticPartition(size_t{3}),
                [&](const auto& range, uint64_t tid) {
                    std::lock_guard lock{mutex};
                    seen_threads.push_back(tid);
                    ranges.push_back(threads::UnitRange<size_t>{
                        *(range.begin()), *(range.end())});
                }
            );
        }

        CATCH_REQUIRE(seen_threads.size() == 3);
        CATCH_REQUIRE(ranges.size() == 3);

        std::sort(seen_threads.begin(), seen_threads.end());
        std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
            return a.start() < b.start();
        });

        CATCH_REQUIRE(seen_threads[0] == 0);
        CATCH_REQUIRE(seen_threads[1] == 1);
        CATCH_REQUIRE(seen_threads[2] == 2);

        CATCH_REQUIRE(ranges[0] == threads::UnitRange<uint64_t>(0, 1));
        CATCH_REQUIRE(ranges[1] == threads::UnitRange<uint64_t>(1, 2));
        CATCH_REQUIRE(ranges[2] == threads::UnitRange<uint64_t>(2, 3));
    }

    CATCH_SECTION("Parallel versus Sequential") {
        auto v = std::vector<uint64_t>(100'000);
        constexpr size_t num_threads = 2;

        /////
        ///// Sequential
        /////
        auto start_time = std::chrono::steady_clock::now();
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = 1;
        }
        auto stop_time = std::chrono::steady_clock::now();
        auto time_seconds = std::chrono::duration<float>(stop_time - start_time).count();
        std::cout << "Sequential Loop: " << time_seconds << " seconds" << std::endl;

        start_time = std::chrono::steady_clock::now();
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = 1;
        }
        stop_time = std::chrono::steady_clock::now();
        time_seconds = std::chrono::duration<float>(stop_time - start_time).count();
        std::cout << "Sequential Loop: " << time_seconds << " seconds" << std::endl;
        CATCH_REQUIRE(std::all_of(v.begin(), v.end(), [](const uint64_t& v) {
            return v == 1;
        }));

        /////
        ///// Sequential ThreadPool
        /////

        auto f = [&v](const auto& is, size_t /*unused*/) {
            for (auto i : is) {
                v[i] = 2;
            }
        };

        auto sequential_pool = svs::threads::SequentialThreadPool{};
        start_time = std::chrono::steady_clock::now();
        svs::threads::run(sequential_pool, svs::threads::StaticPartition{v.size()}, f);
        stop_time = std::chrono::steady_clock::now();
        time_seconds = std::chrono::duration<float>(stop_time - start_time).count();
        std::cout << "Sequential Pool: " << time_seconds << " seconds" << std::endl;

        start_time = std::chrono::steady_clock::now();
        svs::threads::run(sequential_pool, svs::threads::StaticPartition{v.size()}, f);
        stop_time = std::chrono::steady_clock::now();
        time_seconds = std::chrono::duration<float>(stop_time - start_time).count();
        std::cout << "Sequential Pool: " << time_seconds << " seconds" << std::endl;

        /////
        ///// Parallal
        /////

        auto pool = svs::threads::NativeThreadPool(num_threads);
        start_time = std::chrono::steady_clock::now();
        svs::threads::run(pool, svs::threads::StaticPartition{v.size()}, f);
        stop_time = std::chrono::steady_clock::now();
        time_seconds = std::chrono::duration<float>(stop_time - start_time).count();
        std::cout << "Parallel: " << time_seconds << " seconds" << std::endl;

        start_time = std::chrono::steady_clock::now();
        svs::threads::run(pool, svs::threads::StaticPartition{v.size()}, f);
        stop_time = std::chrono::steady_clock::now();
        time_seconds = std::chrono::duration<float>(stop_time - start_time).count();
        std::cout << "Parallel: " << time_seconds << " seconds" << std::endl;

        CATCH_REQUIRE(std::all_of(v.begin(), v.end(), [](const uint64_t& v) {
            return v == 2;
        }));
    }
}
