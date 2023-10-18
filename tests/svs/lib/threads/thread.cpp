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
#include <random>
#include <thread>
#include <tuple>

// local includes
#include "svs/lib/exception.h"
#include "svs/lib/threads/thread.h"

// catch macros
#include "catch2/catch_test_macros.hpp"

/////
///// Helper functions
/////

namespace {

std::array<svs::threads::ThreadState, 6> all_states() {
    using ThreadState = svs::threads::ThreadState;
    return {
        ThreadState::Working,
        ThreadState::Spinning,
        ThreadState::Sleeping,
        ThreadState::Exception,
        ThreadState::RequestShutdown,
        ThreadState::Shutdown,
    };
}

template <typename T> struct SpunUpThread {
    std::future<void> future;
    std::thread worker;
    std::unique_ptr<svs::threads::ThreadControlBlock<T>> block;
};

template <typename T = svs::threads::telemetry::NoTelemetry>
SpunUpThread<T> spin_up(size_t spin_count = 1'000, bool wait = true) {
    auto block = std::make_unique<svs::threads::ThreadControlBlock<T>>();

    // Spawn thread future and promise.
    std::promise<void> promise{};
    std::future<void> future = promise.get_future();

    auto worker_function = [block_local = block.get(),
                            spin_count](std::promise<void>&& exception_channel) {
        block_local->unsafe_run(std::move(exception_channel), spin_count);
    };
    auto worker = std::thread(std::move(worker_function), std::move(promise));
    if (wait) {
        block->wait_until_fully_asleep();
    }
    return SpunUpThread<T>{
        std::move(future),
        std::move(worker),
        std::move(block),
    };
}

} // namespace

CATCH_TEST_CASE("Basics", "[core][threads]") {
    CATCH_SECTION("Spin While") {
        using namespace std::chrono_literals;

        std::atomic<size_t> x{0};
        std::atomic<size_t> channel{0};
        auto thread = std::thread([&]() {
            svs::threads::detail::spin_while<size_t>(x, 0);
            channel.store(100);
            svs::threads::detail::spin_while<size_t>(x, 10);
            channel.store(1000);
        });

        // Sleep the main thread.
        // The worker thread should still be waiting.
        std::this_thread::sleep_for(1ms);
        CATCH_REQUIRE(channel.load() == 0);
        x.store(10);

        std::this_thread::sleep_for(1ms);
        CATCH_REQUIRE(channel.load() == 100);
        std::this_thread::sleep_for(1ms);
        CATCH_REQUIRE(channel.load() == 100);

        x.store(0);
        std::this_thread::sleep_for(1ms);
        CATCH_REQUIRE(channel.load() == 1000);
        thread.join();
    }

    CATCH_SECTION("ThreadFunction") {
        std::vector<int> x{};
        auto fn = [&x](uint64_t val) { x.push_back(val); };
        auto ref = svs::threads::FunctionRef(fn);

        svs::threads::ThreadFunctionRef fn_ref{ref, 10};
        fn_ref();
        CATCH_REQUIRE(x.size() == 1);
        CATCH_REQUIRE(x.at(0) == 10);

        fn_ref = {ref, 100};
        fn_ref();
        CATCH_REQUIRE(x.size() == 2);
        CATCH_REQUIRE(x.at(0) == 10);
        CATCH_REQUIRE(x.at(1) == 100);
    }
}

CATCH_TEST_CASE("Control Block", "[core][threads][thread_control_block]") {
    CATCH_SECTION("Basic Accessors") {
        auto block = svs::threads::ThreadControlBlock();
        // Get and set state.
        for (const auto& state : all_states()) {
            block.set_state(state);
            CATCH_REQUIRE(block.get_state() == state);
        }

        // Get and set work.
        std::vector<int> x{};
        auto fn = [&x](uint64_t val) { x.push_back(val); };
        svs::threads::ThreadFunctionRef fn_ref = {svs::threads::FunctionRef(fn), 10};
        block.unsafe_set_work(fn_ref);
        block.get_work()();
        CATCH_REQUIRE(x.size() == 1);
        CATCH_REQUIRE(x.at(0) == 10);
    }
    CATCH_SECTION("Work Assignment") {
        using namespace std::chrono_literals;
        auto block =
            svs::threads::ThreadControlBlock<svs::threads::telemetry::ActionTelemetry>();
        std::vector<int> vector;
        auto lambda = [&vector](uint64_t val) { vector.push_back(val); };
        auto fn = svs::threads::FunctionRef(lambda);

        CATCH_SECTION("Working to Spinning") {
            auto f = [&block]() {
                std::this_thread::sleep_for(1ms);
                block.set_state(svs::threads::ThreadState::Spinning);
            };

            // When the initial value is `Working`, then we should wait until it changes
            // value.
            svs::threads::ThreadFunctionRef fn{{}, 123};
            block.set_state(svs::threads::ThreadState::Working);
            {
                auto thread = std::thread(f);
                block.assign(fn);
                auto fn_retrieved = block.get_work();
                CATCH_REQUIRE(fn_retrieved.fn.fn == nullptr);
                CATCH_REQUIRE(fn_retrieved.fn.arg == nullptr);
                CATCH_REQUIRE(fn_retrieved.thread_id == 123);
                thread.join();

                CATCH_REQUIRE(
                    block.get_state(std::memory_order_relaxed) ==
                    svs::threads::ThreadState::Working
                );
            }
        }

        // There are several cases we need to test when dealing with the possible orderings
        // of spinning to sleeping and how that can interact with the assignment of new
        // work.
        //
        // (1) The thread could already be asleep.
        // (2) The thread could be transitioning to sleep, after acquiring the condition
        //     variable mutex but before exchanging the threadstate variable.
        //
        // (3) The thread could be transitioning to sleep after transitioning the
        //     threadstate variable to sleep but before actually going to sleep.
        CATCH_SECTION("Spinning to Sleeping") {
            CATCH_SECTION("Case 1") {
                bool slept = false;
                block.set_state(svs::threads::ThreadState::Spinning);
                auto thread =
                    std::thread([&block, &slept]() { slept = block.try_sleep(); });

                // Give the worker thread enough time to actually go to sleep.
                block.wait_until_fully_asleep();

                // The threadstate variable should be set to `Sleeping`.
                CATCH_REQUIRE(block.get_state() == svs::threads::ThreadState::Sleeping);
                block.assign({fn, 10});

                thread.join();
                CATCH_REQUIRE(block.get_state() == svs::threads::ThreadState::Working);
                CATCH_REQUIRE(slept == true);
                auto work = block.get_work();
                CATCH_REQUIRE(work.fn == fn);
                CATCH_REQUIRE(work.thread_id == 10);

                auto telemetry = block.get_telemetry();
                CATCH_REQUIRE(telemetry.sleep_attempts_ == 1);
                // Two predicate checks:
                // - Once when going to sleep
                // - Once when waking from sleep.
                CATCH_REQUIRE(telemetry.sleep_predicate_checks_ == 2);
                CATCH_REQUIRE(telemetry.sleep_success_ == 1);
                CATCH_REQUIRE(telemetry.sleep_fail_ == 0);
            }

            CATCH_SECTION("Case 2") {
                bool slept = false;
                block.set_state(svs::threads::ThreadState::Spinning);

                std::atomic<size_t> wait{0};
                std::atomic<bool> in_pre_op{false};
                auto thread = std::thread([&block, &slept, &wait, &in_pre_op] {
                    auto pre_op = [&wait, &in_pre_op]() {
                        in_pre_op.store(true);
                        svs::threads::detail::spin_while<size_t>(wait, 0);
                    };
                    slept = block.try_sleep(pre_op);
                });

                // Sleep to let everything settle into a steady state.
                // Because we injectd a pre-op wait, the thread state variable should not
                // have been set yet.
                svs::threads::detail::spin_until<bool>(in_pre_op, true);
                CATCH_REQUIRE(block.get_state() == svs::threads::ThreadState::Spinning);

                // Assign work and then let the spawned thread continue.
                block.assign({fn, 10});
                wait.store(1);
                thread.join();
                CATCH_REQUIRE(block.get_state() == svs::threads::ThreadState::Working);
                // State should have been updated before the worker thread had a chance to
                // sleep.
                CATCH_REQUIRE(slept == false);
                auto work = block.get_work();
                CATCH_REQUIRE(work.fn == fn);
                CATCH_REQUIRE(work.thread_id == 10);

                auto telemetry = block.get_telemetry();
                CATCH_REQUIRE(telemetry.sleep_attempts_ == 1);
                CATCH_REQUIRE(telemetry.sleep_success_ == 0);
                CATCH_REQUIRE(telemetry.sleep_fail_ == 1);
                CATCH_REQUIRE(telemetry.sleep_predicate_checks_ == 0);
            }

            CATCH_SECTION("Case 3") {
                bool slept = false;
                block.set_state(svs::threads::ThreadState::Spinning);

                std::atomic<bool> in_post_op{false};

                // For post-op injection, we greatly exaggerate the time between when
                // the mutex is locked before going to sleep and when the thread actually
                // sleeps.
                //
                // Since `assign_work` is blocking on the thread going to sleep, we need
                // to hope that this window is long enough to induce the pathological case
                // we are interested in.
                auto thread = std::thread([&block, &slept, &in_post_op] {
                    auto post_op = [&in_post_op]() {
                        in_post_op.store(true);
                        std::this_thread::sleep_for(100ms);
                    };
                    slept = block.try_sleep(svs::lib::donothing{}, post_op);
                });

                // Sleep to let everything settle into a steady state.
                // Because we injectd a post_op wait, the thread state variable should not
                svs::threads::detail::spin_until<bool>(in_post_op, true);
                CATCH_REQUIRE(block.get_state() == svs::threads::ThreadState::Sleeping);

                // Assign work and then let the spawned thread continue.
                block.assign({fn, 10});
                thread.join();
                CATCH_REQUIRE(block.get_state() == svs::threads::ThreadState::Working);

                // When the worker reaches the post-op, it should have made it all the way
                // to sleep.
                CATCH_REQUIRE(slept == true);
                auto work = block.get_work();
                CATCH_REQUIRE(work.fn == fn);
                CATCH_REQUIRE(work.thread_id == 10);

                auto telemetry = block.get_telemetry();
                CATCH_REQUIRE(telemetry.sleep_attempts_ == 1);
                CATCH_REQUIRE(telemetry.sleep_success_ == 1);
                CATCH_REQUIRE(telemetry.sleep_fail_ == 0);
                CATCH_REQUIRE(telemetry.sleep_predicate_checks_ == 2);
            }
        }

        CATCH_SECTION("Error Handling") {
            // The `Exception` is handled differently from the normal default.
            block.set_state(svs::threads::ThreadState::Exception);
            for (const auto& state : all_states()) {
                CATCH_REQUIRE_THROWS_AS(
                    block.notify_thread(state), svs::threads::ThreadCrashedError
                );
            }

            // General error states.
            auto error_states = std::vector<svs::threads::ThreadState>{
                svs::threads::ThreadState::Working,
                svs::threads::ThreadState::RequestShutdown,
                svs::threads::ThreadState::Shutdown,
            };

            for (const auto& error_state : error_states) {
                block.set_state(error_state);
                for (const auto& state : all_states()) {
                    CATCH_REQUIRE_THROWS_AS(block.notify_thread(state), svs::ANNException);
                }
            }
        }
    }

    CATCH_SECTION("Shutdown") {
        using namespace std::chrono_literals;
        auto block =
            svs::threads::ThreadControlBlock<svs::threads::telemetry::ActionTelemetry>();

        CATCH_SECTION("Spinning") {
            block.set_state(svs::threads::ThreadState::Working);
            auto thread = std::thread([&block] {
                std::this_thread::sleep_for(1ms);
                block.set_state(svs::threads::ThreadState::Spinning);
            });

            block.shutdown(false);
            thread.join();
            CATCH_REQUIRE(block.get_state() == svs::threads::ThreadState::RequestShutdown);
        }

        CATCH_SECTION("Sleeping 1", "Thread is fully asleep") {
            block.set_state(svs::threads::ThreadState::Spinning);
            bool slept = false;
            auto thread = std::thread([&block, &slept]() { slept = block.try_sleep(); });
            block.wait_until_fully_asleep();

            CATCH_REQUIRE(block.get_state() == svs::threads::ThreadState::Sleeping);
            block.shutdown(false);
            thread.join();
            CATCH_REQUIRE(block.get_state() == svs::threads::ThreadState::RequestShutdown);

            const auto& telemetry = block.get_telemetry();
            CATCH_REQUIRE(telemetry.sleep_attempts_ == 1);
            CATCH_REQUIRE(telemetry.sleep_predicate_checks_ == 2);
            CATCH_REQUIRE(telemetry.sleep_success_ == 1);
            CATCH_REQUIRE(telemetry.sleep_fail_ == 0);
        }

        CATCH_SECTION("Sleeping 2", "Thread has acquired mutex but not updated state.") {
            bool slept = false;
            block.set_state(svs::threads::ThreadState::Spinning);

            std::atomic<size_t> wait{0};
            std::atomic<bool> in_pre_op{false};
            auto thread = std::thread([&block, &slept, &wait, &in_pre_op] {
                auto pre_op = [&wait, &in_pre_op]() {
                    in_pre_op.store(true);
                    svs::threads::detail::spin_while<size_t>(wait, 0);
                };
                slept = block.try_sleep(pre_op);
            });

            // Sleep to let everything settle into a steady state.
            // Because we injectd a pre-op wait, the thread state variable should not
            // have been set yet.
            svs::threads::detail::spin_until<bool>(in_pre_op, true);
            CATCH_REQUIRE(block.get_state() == svs::threads::ThreadState::Spinning);

            // Assign work and then let the spawned thread continue.
            block.shutdown(false);
            wait.store(1);
            thread.join();
            CATCH_REQUIRE(block.get_state() == svs::threads::ThreadState::RequestShutdown);
            // State should have been updated before the worker thread had a chance to
            // sleep.
            CATCH_REQUIRE(slept == false);

            auto telemetry = block.get_telemetry();
            CATCH_REQUIRE(telemetry.sleep_attempts_ == 1);
            CATCH_REQUIRE(telemetry.sleep_success_ == 0);
            CATCH_REQUIRE(telemetry.sleep_fail_ == 1);
            CATCH_REQUIRE(telemetry.sleep_predicate_checks_ == 0);
        }

        CATCH_SECTION("Sleeping 3", "Thread has updated state but not yet slept.") {
            bool slept = false;
            block.set_state(svs::threads::ThreadState::Spinning);
            std::atomic<bool> in_post_op{false};

            // For post-op injection, we greatly exaggerate the time between when
            // the mutex is locked before going to sleep and when the thread actually
            // sleeps.
            //
            // Since `assign_work` is blocking on the thread going to sleep, we need
            // to hope that this window is long enough to induce the pathological case
            // we are interested in.
            auto thread = std::thread([&block, &slept, &in_post_op] {
                auto post_op = [&in_post_op]() {
                    in_post_op.store(true);
                    std::this_thread::sleep_for(100ms);
                };
                slept = block.try_sleep(svs::lib::donothing{}, post_op);
            });

            // Sleep to let everything settle into a steady state.
            // Because we injectd a post_op wait, the thread state variable should not
            svs::threads::detail::spin_until<bool>(in_post_op, true);
            CATCH_REQUIRE(block.get_state() == svs::threads::ThreadState::Sleeping);

            // Assign work and then let the spawned thread continue.
            block.shutdown(false);
            thread.join();
            CATCH_REQUIRE(block.get_state() == svs::threads::ThreadState::RequestShutdown);

            // When the worker reaches the post-op, it should have made it all the way
            // to sleep.
            CATCH_REQUIRE(slept == true);
            auto telemetry = block.get_telemetry();
            CATCH_REQUIRE(telemetry.sleep_attempts_ == 1);
            CATCH_REQUIRE(telemetry.sleep_success_ == 1);
            CATCH_REQUIRE(telemetry.sleep_fail_ == 0);
            CATCH_REQUIRE(telemetry.sleep_predicate_checks_ == 2);
        }

        CATCH_SECTION("Failure modes") {
            bool would_terminate = false;
            auto on_terminate = [&would_terminate] { would_terminate = true; };

            auto exception_states = std::vector<svs::threads::ThreadState>{
                svs::threads::ThreadState::RequestShutdown,
            };

            for (auto state : exception_states) {
                would_terminate = false;
                block.set_state(state);
                block.shutdown(false, on_terminate);
                CATCH_REQUIRE(would_terminate == true);
            }
        }

        CATCH_SECTION("Shutdown or Exception") {
            auto graceful_states = std::vector<svs::threads::ThreadState>{
                svs::threads::ThreadState::Shutdown, svs::threads::ThreadState::Exception};

            for (auto state : graceful_states) {
                block.set_state(state);
                block.shutdown(true);
                // Make sure after the call to `shutdown`, that the threadstate is
                // left unmodified.
                CATCH_REQUIRE(block.get_state() == state);
            }
        }
    }
}

CATCH_TEST_CASE("Simple Threading", "[core][threads]") {
    using namespace std::chrono_literals;
    auto block = svs::threads::ThreadControlBlock();

    // Spawn thread future and promise.
    std::promise<void> promise{};
    std::future<void> future = promise.get_future();

    auto worker_function = [&block](std::promise<void> exception_channel) {
        block.unsafe_run(std::move(exception_channel));
    };
    auto worker = std::thread(std::move(worker_function), std::move(promise));
    block.wait_until_fully_asleep();
    // std::this_thread::sleep_for(1ms);

    // Now that the worker is running, try assigning some jobs to it.
    {
        std::vector<size_t> test_vector{};
        auto lambda = [&test_vector](size_t i) {
            test_vector.push_back(i);
        };
        auto f = svs::threads::FunctionRef(lambda);

        // Assign some jobs and wait until finished.
        block.assign({f, 10});
        block.assign({f, 20});
        block.assign({f, 30});
        block.wait_while_busy();

        CATCH_REQUIRE(test_vector.size() == 3);
        CATCH_REQUIRE(test_vector.at(0) == 10);
        CATCH_REQUIRE(test_vector.at(1) == 20);
        CATCH_REQUIRE(test_vector.at(2) == 30);
    }

    // Try again, this time assigning different types of functions.
    {
        std::vector<size_t> test_vector_f{};
        std::vector<float> test_vector_g{};
        auto f_lambda = [&test_vector_f](size_t i) {
            test_vector_f.push_back(i);
        };
        auto f = svs::threads::FunctionRef(f_lambda);

        auto g_lambda = [&test_vector_g](size_t i) {
            test_vector_g.push_back(i);
        };
        auto g = svs::threads::FunctionRef(g_lambda);

        block.assign({f, 10});
        block.assign({g, 20});
        block.assign({f, 30});
        block.assign({g, 40});
        block.wait_while_busy();

        CATCH_REQUIRE(test_vector_f.size() == 2);
        CATCH_REQUIRE(test_vector_f.at(0) == 10);
        CATCH_REQUIRE(test_vector_f.at(1) == 30);

        CATCH_REQUIRE(test_vector_g.size() == 2);
        CATCH_REQUIRE(test_vector_g.at(0) == 20);
        CATCH_REQUIRE(test_vector_g.at(1) == 40);
    }

    // Request a shut down and join the worker thread.
    block.shutdown();
    worker.join();
    CATCH_REQUIRE(block.is_shutdown());
}

CATCH_TEST_CASE("Extented Test", "[core][threads]") {
    using namespace std::chrono_literals;
    using ActionTelemetry = svs::threads::telemetry::ActionTelemetry;

    auto block = svs::threads::ThreadControlBlock<ActionTelemetry>();

    // Spawn thread future and promise.
    std::promise<void> promise{};
    std::future<void> future = promise.get_future();

    auto worker_function = [&block](std::promise<void> exception_channel) {
        block.unsafe_run(std::move(exception_channel), 1'000);
    };
    auto worker = std::thread(std::move(worker_function), std::move(promise));
    std::this_thread::sleep_for(1ms);

    auto vector = std::vector<size_t>{};
    auto lambda = [&vector](size_t i) { vector.push_back(i); };
    auto f = svs::threads::FunctionRef(lambda);

    // Get a random number generator
    std::mt19937_64 eng{std::random_device{}()};
    std::uniform_int_distribution<> dist{1, 2};
    constexpr size_t trip_count = 200000;
    for (size_t i = 0; i < trip_count; ++i) {
        if ((i % 4) == 0) {
            block.assign({f, i});
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds{dist(eng)});
            block.assign({f, i});
        }
    }
    block.wait_while_busy();

    CATCH_REQUIRE(vector.size() == trip_count);
    for (size_t i = 0; i < trip_count; ++i) {
        CATCH_REQUIRE(vector.at(i) == i);
    }

    // Request a shut down and join the worker thread.
    block.shutdown();
    worker.join();
    future.get();
    CATCH_REQUIRE(block.is_shutdown());

    // Inspect the number of actions taken.
    const auto& telemetry = block.get_telemetry();
    std::cout << "Sleep Attempts: " << telemetry.sleep_attempts_ << std::endl;
    std::cout << "Sleep Success: " << telemetry.sleep_success_ << std::endl;
    std::cout << "Sleep Fail: " << telemetry.sleep_fail_ << std::endl;

    std::cout << "Successful Spins: " << telemetry.spin_success_ << std::endl;
    std::cout << "Failed Spins: " << telemetry.spin_fail_ << std::endl;

    ///// Make sure we've covered corner cases.
    CATCH_REQUIRE(telemetry.sleep_attempts_ == telemetry.spin_fail_);
    // Result should be `trip_count + 1` for the final shutdown request.
    CATCH_REQUIRE(telemetry.spin_fail_ + telemetry.spin_success_ == trip_count + 1);
    CATCH_REQUIRE(
        telemetry.sleep_attempts_ == telemetry.sleep_fail_ + telemetry.sleep_success_
    );

    ///
    /// TODO: The below test is too flaky for CI in DEBUG mode
    ///

    // Make sure we've handled some spurious sleep failures.
    // CATCH_REQUIRE(telemetry.sleep_fail_ > 0);
}

/////
///// Exceptions
/////

CATCH_TEST_CASE("Exception Handling", "[core][threads]") {
    using namespace std::chrono_literals;
    auto thread = spin_up();
    auto& [future, worker, block] = thread;

    // First, make sure we can submit successful jobs to the worker thread.
    size_t x = 0;
    auto fn_good_lambda = [&x](uint64_t new_val) { x = new_val; };
    auto fn_good = svs::threads::FunctionRef(fn_good_lambda);
    block->assign({fn_good, 10});
    block->wait_while_busy();
    CATCH_REQUIRE(x == 10);

    // Now, assign a job that throws an exception.
    auto fn_bad_lambda = [&x](uint64_t /*new_val*/) {
        auto message = std::to_string(x);
        throw std::runtime_error("Something went wrong: " + message);
    };
    auto fn_bad = svs::threads::FunctionRef(fn_bad_lambda);

    block->assign({fn_bad, 10});
    block->wait_while_busy();

    // An error should now be visible.
    CATCH_REQUIRE(!block->is_okay());
    CATCH_REQUIRE(block->get_state() == svs::threads::ThreadState::Exception);
    CATCH_REQUIRE(future.valid() == true);
    bool threw_exception = false;
    try {
        // Try to get the future.
        // This should throw an exception.
        future.get();
        block->shutdown();
    } catch (const std::exception& err) {
        threw_exception = true;
        std::string what = err.what();
        CATCH_REQUIRE(what == "Something went wrong: 10");
    }
    CATCH_REQUIRE(threw_exception);
    worker.join();
}

/////
///// Thread
/////

CATCH_TEST_CASE("Testing Thread", "[core][threads][high_level]") {
    using namespace std::chrono_literals;
    CATCH_SECTION("Construction and Destruction") {
        auto thread = svs::threads::Thread();
        CATCH_REQUIRE(thread.is_okay());
    }

    CATCH_SECTION("Move Constructor") {
        auto thread = svs::threads::Thread();
        auto other = std::move(thread);
        int x = 0;
        auto lambda = [&x](uint64_t i) { x = i; };
        auto f = svs::threads::FunctionRef(lambda);
        other.assign({f, 10});
        other.wait();
        CATCH_REQUIRE(x == 10);
    }

    CATCH_SECTION("Move Assignment") {
        auto thread = svs::threads::Thread();
        auto other = svs::threads::Thread();
        // Shutdown the first thread.
        thread.shutdown();
        // Make sure it's save to call shutdown multiple times.
        thread.shutdown();
        CATCH_REQUIRE(thread.is_shutdown());
        thread = std::move(other);
        int x = 0;
        auto lambda = [&x](uint64_t i) { x = i; };
        auto f = svs::threads::FunctionRef(lambda);
        thread.assign({f, 10});
        thread.wait();
        CATCH_REQUIRE(x == 10);
    }

    CATCH_SECTION("Simple Tests") {
        std::array<std::string, 3> words{"Cat", "Dog", "Ferret"};
        std::vector<std::string> words_dest{};
        std::vector<size_t> ints_dest{};

        auto thread = svs::threads::Thread{};
        CATCH_REQUIRE(thread.is_okay());
        CATCH_REQUIRE(!thread.is_shutdown());
        CATCH_REQUIRE(thread.is_running());

        // N.B.: The function `f` can throw if it receives an out-of-bounds
        // Index.
        auto f_lambda = [&words, &words_dest](uint64_t i) {
            words_dest.push_back(words.at(i));
        };
        auto f = svs::threads::FunctionRef(f_lambda);

        auto g_lambda = [&ints_dest](uint64_t i) { ints_dest.push_back(i); };
        auto g = svs::threads::FunctionRef(g_lambda);

        // Assign jobs to the thread and wait for all to complete.
        thread.assign({f, 2});
        thread.assign({g, 1});
        thread.assign({g, 2});
        thread.assign({f, 1});
        thread.assign({f, 0});
        thread.assign({g, 10});
        thread.assign({g, 4});
        thread.wait();

        CATCH_REQUIRE(thread.is_running());
        CATCH_REQUIRE(words_dest.size() == 3);

        std::vector<std::string> expected_words = {"Ferret", "Dog", "Cat"};
        CATCH_REQUIRE(std::equal(
            words_dest.begin(),
            words_dest.end(),
            expected_words.begin(),
            expected_words.end()
        ));

        std::vector<size_t> expected_ints = {1, 2, 10, 4};
        CATCH_REQUIRE(std::equal(
            ints_dest.begin(), ints_dest.end(), expected_ints.begin(), expected_ints.end()
        ));
    }

    CATCH_SECTION("Exception Handling") {
        auto thread = svs::threads::Thread{};
        auto lambda = [](uint64_t i) {
            throw std::runtime_error("Hello world " + std::to_string(i));
        };
        auto fn = svs::threads::FunctionRef(lambda);
        try {
            thread.assign({fn, 0});
            thread.assign({fn, 1});
        } catch (svs::threads::ThreadError& error) {
            CATCH_REQUIRE(
                error.what() == svs::threads::ThreadError::make_message("Hello world 0")
            );
        }

        // Reinitialize thread
        CATCH_REQUIRE(!thread.is_okay());
        thread.shutdown();
        thread = svs::threads::Thread{};
        try {
            thread.assign({fn, 10});
            // Wait until the thread is asleep.
            std::this_thread::sleep_for(5ms);
            thread.assign({fn, 20});
        } catch (svs::threads::ThreadError& error) {
            CATCH_REQUIRE(
                error.what() == svs::threads::ThreadError::make_message("Hello world 10")
            );
        }
    }
}
