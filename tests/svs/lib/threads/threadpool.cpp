/*
 * Copyright 2023 Intel Corporation
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

        auto pool = svs::threads::DefaultThreadPool(4);
        CATCH_SECTION("Just One Thread Crashed") {
            for (size_t i = 0; i < pool.size(); ++i) {
                v.clear();
                CATCH_REQUIRE(v.empty());
                try {
                    svs::threads::parallel_for(pool, [&v, &v_mutex, i](uint64_t tid) {
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
                svs::threads::parallel_for(pool, [](uint64_t tid) {
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
            svs::threads::parallel_for(pool, [&v, &v_mutex](uint64_t tid) {
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
        auto pool = svs::threads::DefaultThreadPool(4);
        std::mutex mutex;

        std::vector<uint64_t> seen_threads{};
        std::vector<svs::threads::UnitRange<uint64_t>> ranges{};

        // Make sure that if the number of threads exceeds the number of available work
        // partitions that:
        //
        // 1. Everything remains within bounds.
        // 2. Threads that have no work are never launched.
        CATCH_SECTION("No Oversubscription") {
            threads::parallel_for(
                pool,
                threads::StaticPartition(size_t{3}),
                [&](const auto& range, uint64_t tid) {
                    std::lock_guard lock{mutex};
                    seen_threads.push_back(tid);
                    ranges.push_back(threads::UnitRange<uint64_t>{
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

    CATCH_SECTION("Sequential and Parallel For") {
        auto v = std::vector<uint64_t>(100'000);
        constexpr size_t num_threads = 2;
        auto start_time = std::chrono::steady_clock::now();
        auto stop_time = std::chrono::steady_clock::now();
        float time_seconds;

        auto f = [&v](const auto& is, size_t /*unused*/) {
            for (auto i : is) {
                v[i] = 2;
            }
        };

        /////
        ///// Sequential
        /////
        CATCH_SECTION("Sequential") {
            start_time = std::chrono::steady_clock::now();
            for (size_t i = 0; i < v.size(); ++i) {
                v[i] = 1;
            }
            stop_time = std::chrono::steady_clock::now();
            time_seconds = std::chrono::duration<float>(stop_time - start_time).count();
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
        }

        /////
        ///// Sequential ThreadPool
        /////

        CATCH_SECTION("SequentialThreadPool") {
            auto sequential_pool = svs::threads::SequentialThreadPool{};
            start_time = std::chrono::steady_clock::now();
            svs::threads::parallel_for(
                sequential_pool, svs::threads::StaticPartition{v.size()}, f
            );
            stop_time = std::chrono::steady_clock::now();
            time_seconds = std::chrono::duration<float>(stop_time - start_time).count();
            std::cout << "Sequential Pool: " << time_seconds << " seconds" << std::endl;

            start_time = std::chrono::steady_clock::now();
            svs::threads::parallel_for(
                sequential_pool, svs::threads::StaticPartition{v.size()}, f
            );
            stop_time = std::chrono::steady_clock::now();
            time_seconds = std::chrono::duration<float>(stop_time - start_time).count();
            std::cout << "SequentialThreadPool: " << time_seconds << " seconds"
                      << std::endl;

            CATCH_REQUIRE(std::all_of(v.begin(), v.end(), [](const uint64_t& v) {
                return v == 2;
            }));
        }

        /////
        ///// NativeThreadPool
        /////

        CATCH_SECTION("NativeThreadPool") {
            auto pool = svs::threads::NativeThreadPool(num_threads);
            start_time = std::chrono::steady_clock::now();
            svs::threads::parallel_for(pool, svs::threads::StaticPartition{v.size()}, f);
            stop_time = std::chrono::steady_clock::now();
            time_seconds = std::chrono::duration<float>(stop_time - start_time).count();
            std::cout << "NativeThreadPool: " << time_seconds << " seconds" << std::endl;

            start_time = std::chrono::steady_clock::now();
            svs::threads::parallel_for(pool, svs::threads::StaticPartition{v.size()}, f);
            stop_time = std::chrono::steady_clock::now();
            time_seconds = std::chrono::duration<float>(stop_time - start_time).count();
            std::cout << "NativeThreadPool: " << time_seconds << " seconds" << std::endl;

            CATCH_REQUIRE(std::all_of(v.begin(), v.end(), [](const uint64_t& v) {
                return v == 2;
            }));
        }

        /////
        ///// CppAsyncThreadPool
        /////
        CATCH_SECTION("CppAsyncThreadPool") {
            auto pool = svs::threads::CppAsyncThreadPool(num_threads);
            start_time = std::chrono::steady_clock::now();
            svs::threads::parallel_for(pool, svs::threads::StaticPartition{v.size()}, f);
            stop_time = std::chrono::steady_clock::now();
            time_seconds = std::chrono::duration<float>(stop_time - start_time).count();
            std::cout << "CppAsyncThreadPool: " << time_seconds << " seconds" << std::endl;

            start_time = std::chrono::steady_clock::now();
            svs::threads::parallel_for(pool, svs::threads::StaticPartition{v.size()}, f);
            stop_time = std::chrono::steady_clock::now();
            time_seconds = std::chrono::duration<float>(stop_time - start_time).count();
            std::cout << "CppAsyncThreadPool: " << time_seconds << " seconds" << std::endl;

            CATCH_REQUIRE(std::all_of(v.begin(), v.end(), [](const uint64_t& v) {
                return v == 2;
            }));
        }

        /////
        ///// QueueThreadPool
        /////
        CATCH_SECTION("QueueThreadPool") {
            auto pool = svs::threads::QueueThreadPoolWrapper(num_threads);
            start_time = std::chrono::steady_clock::now();
            svs::threads::parallel_for(pool, svs::threads::StaticPartition{v.size()}, f);
            stop_time = std::chrono::steady_clock::now();
            time_seconds = std::chrono::duration<float>(stop_time - start_time).count();
            std::cout << "QueueThreadPool: " << time_seconds << " seconds" << std::endl;

            start_time = std::chrono::steady_clock::now();
            svs::threads::parallel_for(pool, svs::threads::StaticPartition{v.size()}, f);
            stop_time = std::chrono::steady_clock::now();
            time_seconds = std::chrono::duration<float>(stop_time - start_time).count();
            std::cout << "QueueThreadPool: " << time_seconds << " seconds" << std::endl;

            CATCH_REQUIRE(std::all_of(v.begin(), v.end(), [](const uint64_t& v) {
                return v == 2;
            }));
        }

        /////
        ///// SwitchNativeThreadPool
        /////
        CATCH_SECTION("SwitchNativeThreadPool") {
            auto pool = svs::threads::SwitchNativeThreadPool(num_threads);
            start_time = std::chrono::steady_clock::now();
            svs::threads::parallel_for(pool, svs::threads::StaticPartition{v.size()}, f);
            stop_time = std::chrono::steady_clock::now();
            time_seconds = std::chrono::duration<float>(stop_time - start_time).count();
            std::cout << "SwitchNativeThreadPool: " << time_seconds << " seconds"
                      << std::endl;

            start_time = std::chrono::steady_clock::now();
            svs::threads::parallel_for(pool, svs::threads::StaticPartition{v.size()}, f);
            stop_time = std::chrono::steady_clock::now();
            time_seconds = std::chrono::duration<float>(stop_time - start_time).count();
            std::cout << "SwitchNativeThreadPool: " << time_seconds << " seconds"
                      << std::endl;

            CATCH_REQUIRE(std::all_of(v.begin(), v.end(), [](const uint64_t& v) {
                return v == 2;
            }));
        }
    }
    // CATCH_SECTION("SwitchNativeThreadPool and NativeThreadPool with Parallel Calls") {
    // constexpr size_t num_external_threads = 4;
    // constexpr size_t num_internal_threads = 2;
    // constexpr size_t num_elements = 50000000;

    // auto start_time = std::chrono::steady_clock::now();
    // auto stop_time = std::chrono::steady_clock::now();
    // float time_seconds, switch_time_seconds;

    //{
    // std::vector<std::vector<size_t>> v;
    // std::vector<size_t> sum(num_external_threads, 0);
    // v.resize(num_external_threads);
    // for (auto& vv : v) {
    // vv.resize(num_elements, 1);
    //}

    // std::vector<std::thread> external_threads;
    // auto pool = svs::threads::NativeThreadPool(num_internal_threads);
    // start_time = std::chrono::steady_clock::now();

    //// NativeThreadPool will block external parallelism due to internal lock.
    // for (size_t i = 0; i < num_external_threads; ++i) {
    // external_threads.emplace_back([&v, &pool, &sum, i]() {
    // svs::threads::parallel_for(
    // pool,
    // svs::threads::StaticPartition{1},
    //[i, &vv = v[i], &sum](const auto& [>unused*/, size_t /*unused<]) {
    // for (auto val : vv) {
    // sum[i] += val;
    //}
    //}
    //);
    //});
    //}

    // for (auto& t : external_threads) {
    // t.join();
    //}
    // stop_time = std::chrono::steady_clock::now();
    // time_seconds = std::chrono::duration<float>(stop_time - start_time).count();

    // for (auto s : sum) {
    // CATCH_REQUIRE(s == num_elements);
    //}
    //}

    //{
    // std::vector<std::vector<size_t>> v;
    // std::vector<size_t> sum(num_external_threads, 0);
    // v.resize(num_external_threads);
    // for (auto& vv : v) {
    // vv.resize(num_elements, 1);
    //}

    // std::vector<std::thread> external_threads;
    // auto switch_pool = svs::threads::SwitchNativeThreadPool(num_internal_threads);
    // start_time = std::chrono::steady_clock::now();

    // for (size_t i = 0; i < num_external_threads; ++i) {
    // external_threads.emplace_back([&v, &switch_pool, &sum, i]() {
    // svs::threads::parallel_for(
    // switch_pool,
    // svs::threads::StaticPartition{1},
    //[i, &vv = v[i], &sum](const auto& [>unused*/, size_t /*unused<]) {
    // for (auto val : vv) {
    // sum[i] += val;
    //}
    //}
    //);
    //});
    //}

    // for (auto& t : external_threads) {
    // t.join();
    //}
    // stop_time = std::chrono::steady_clock::now();
    // switch_time_seconds =
    // std::chrono::duration<float>(stop_time - start_time).count();

    // for (auto s : sum) {
    // CATCH_REQUIRE(s == num_elements);
    //}
    //}
    // CATCH_REQUIRE(switch_time_seconds < time_seconds);
    //}
}
