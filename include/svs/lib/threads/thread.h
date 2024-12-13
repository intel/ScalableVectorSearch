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

#pragma once

#include "svs/lib/exception.h"
#include "svs/lib/misc.h"
#include "svs/lib/numa.h"
#include "svs/lib/spinlock.h"
#include "svs/lib/threads/thunks.h"
#include "svs/third-party/fmt.h"

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>

namespace svs {
namespace threads {

class ThreadCrashedError : public std::runtime_error {
  public:
    explicit ThreadCrashedError(const std::string& message)
        : std::runtime_error{message} {}
};

///
/// Thread pool consisting of SVS threads.
///
constexpr size_t default_spintime() { return 1'000'000; }
constexpr size_t short_spintime() { return 1'000; }

// ThreadState is used to communicate information between the worker thread and the
// controlling context.
//
// ASSUMPTIONS:
// - There is only one worker for the Control Block.
// - Access to the control block from the controller side API *is not threadsafe*.
//   If multiple threads are going to be working on the control side, write access must
//   be arbitrated externally.
//
// THREAD STATES:
// * Working (controller set): The worker thread is actively executing an assigned
//   function. In this state, the worker has ownership of the ThreadState atomic
//   variable and a controlling thread may only read.
//
// * Spinning: (worker set) Worker thread has completed its previously assigned tasks and
//   is actively polling the `ThreadState` atomic variable waiting for new work.
//
//   To assign new work, the controller must wait until it observes a non-"Working" state.
//   At which point, it may assign new work and update the thread control atomic variable.
//
//   This update *must* be performed using compare-and-swap because the worker may try
//   to transition to sleeping.
//
// * Sleeping: (worker set) The worker thread timed out on its busy spin loop and has
//   instead gone to sleep. New work should be assigned and the thread should be woken up
//   using an associated condition variable.
//
// * Exception: (worker set) An exception was thrown and the thread has shut downs.
//
// * RequestShutdown: (controller set) Request that the worker thread terminate.
//
// * Shutdown: (worker set) Worker thread has shut down gracefully.
enum class ThreadState : uint64_t {
    Working,
    Spinning,
    Sleeping,
    Exception,
    RequestShutdown,
    Shutdown
};

#define SVS_THREADSTATE_STRING(state) \
    case ThreadState::state:          \
        return "ThreadState::" #state;

inline std::string name(ThreadState state) {
    switch (state) {
        SVS_THREADSTATE_STRING(Working)
        SVS_THREADSTATE_STRING(Spinning)
        SVS_THREADSTATE_STRING(Sleeping)
        SVS_THREADSTATE_STRING(Exception)
        SVS_THREADSTATE_STRING(RequestShutdown)
        SVS_THREADSTATE_STRING(Shutdown)
    };
    // Make GCC happy.
    return "";
}

#undef SVS_THREADSTATE_PRINT

inline std::ostream& operator<<(std::ostream& stream, ThreadState state) {
    return stream << name(state);
}

constexpr ThreadState boot_state() { return ThreadState::Shutdown; }
namespace detail {
constexpr size_t CACHELINE_SIZE_BYTES = 64;

// clang-format off
template <typename F>
concept SpinAbort = requires(F f) {
    { f() } -> std::convertible_to<bool>;
};
// clang-format on

///
/// Efficiently spin on the atomic variable `var` as long as comparison to `test` is `true`.
/// Return the value of `var` with "acquire" memory ordering (default case).
///
/// This function also accepts an optional fourth element `f` that will be called every
/// time the value of `var` is sampled *during the busy wait loop* (i.e., it is not
/// guarenteed that `f` will ever be called).
///
/// The callable `f` provides a methods for aborting the spin loop and must satisfy
/// ```
/// f() -> std::convertible_to<bool>;
/// ```
/// If `f()` returns `true`, then the function will return the value of `test`.
///
/// @param var Atomic variable to wait on.
/// @param test Wait while logically `var.load() == test`.
/// @param compare Spin while logically `compare(var.load(), test)` evaluates to `true`.
/// @param f Optional exit hook (see description above).
///
template <typename T, typename Cmp = std::equal_to<T>, SpinAbort F = lib::Returns<bool>>
T spin_while_true(
    const std::atomic<T>& var,
    T test,
    Cmp&& compare = std::equal_to<T>{},
    F&& f = lib::Returns{false}
) {
    for (;;) {
        T value = var.load(std::memory_order_acquire);
        if (!compare(value, test)) {
            return value;
        }

        // Wait until the atomic variable changes.
        // Use `pause` to reduce contention between threads.
        while (compare(value, test)) {
            if (f()) {
                return value;
            }
            svs::detail::pause();
            value = var.load(std::memory_order_relaxed);
        }
    }
}

template <typename T, SpinAbort F = lib::Returns<bool>>
T spin_while(const std::atomic<T>& var, T test, F&& f = lib::Returns{false}) {
    return spin_while_true(var, test, std::equal_to<T>{}, std::forward<F>(f));
}

template <typename T, SpinAbort F = lib::Returns<bool>>
T spin_until(const std::atomic<T>& var, T test, F&& f = lib::Returns{false}) {
    return spin_while_true(var, test, std::not_equal_to<T>{}, std::forward<F>(f));
}
} // namespace detail

// Telemetry types to help with debugging and extracting performance information.
namespace telemetry {
class NoTelemetry {
  public:
    // Sleeping
    void sleep_attempt() {}
    void sleep_predicate_check() {}
    void sleep_success() {}
    void sleep_fail() {}

    // Spinning
    void enter_spinloop() {}
    void exit_spinloop_success() {}
    void exit_spinloop_fail() {}
};

// Make a struct to explicitly be able to inspect the fields.
struct ActionTelemetry {
    ActionTelemetry() = default;

    // Sleeping
    void sleep_attempt() { sleep_attempts_++; }
    void sleep_predicate_check() { sleep_predicate_checks_++; }
    void sleep_success() { sleep_success_++; }
    void sleep_fail() { sleep_fail_++; }

    // Spinning
    void enter_spinloop() {}
    void exit_spinloop_success() { spin_success_++; }
    void exit_spinloop_fail() { spin_fail_++; }

    ///// Members
    size_t sleep_attempts_{0};
    size_t sleep_predicate_checks_{0};
    size_t sleep_success_{0};
    size_t sleep_fail_{0};

    size_t spin_success_{0};
    size_t spin_fail_{0};
};
} // namespace telemetry

///
/// The default startup thunk used by threads.
/// A "startup thunk" is a bit of code that may return a handle to a resouce to live
/// through-out the duration of a Thread's lifetime.
///
/// Model as a struct instead of a free function to help out static analysis tools.
///
struct DefaultStartup {
    constexpr DefaultStartup() = default;
    constexpr bool operator()() const { return true; }
};

///
/// @brief Functor to terminate the program.
///
/// Same as with ``DefaultStartup``, model the call to ``std::terminate`` to help the
/// coverity static analyzer understand some of the use cases.
///
struct Terminate {
    constexpr Terminate() = default;
    void operator()() const { std::terminate(); }
};

// Key:
//
// Definitions:
// - Worker: The thread that is executing the work assigned to a control block.
// - Control: The thread that is assigning work to the control block.
//
// Pre-conditions:
// - Pr-X: Pre-condition X.
//
// Transition ownership:
// - W-only: Worker only
// - C-only: Control only
//                                      W-only
//                                       Pr-E
//                      +-----------------------------------+
//                      |                                   |
//                 ***********                              V
//         +------>| Working |<------+                *************
//         |       ***********       |                | Exception |
//         |            |            | C-only         *************
//         |            | W-only     | Pr-C
//         |            | Pr-A       |
//         |            V            |
// C-only  |      ************-------+                              W-only
//   Pr-D  |      | Spinning |                *******************   (Pr-H)  ************
//         |      ************--------------->| RequestShutdown |---------->| Shutdown |
//         |            |      C-only (Pr-F)  *******************           ************
//         |            | W-only                       ^
//         |            | Pr-B                         |
//         |            V                              |
//         |      ************     C-only (Pr-G)       |
//         +------| Sleeping |-------------------------+
//                ************
//
// (A) Working -> Spinning
// Pre-conditions:
// - Worker has successfully finished the previous job.
// Action:
// - Atomic Store (release)
// Post-conditions:
// - None
//
// (B) Spinning -> Sleeping
// Pre-conditions:
// - Condition variable mutex is acquired.
// Action:
// - Compare-and-swap (strong) of `threadstate` from Spinning to Sleeping.
// Failure:
// - CAS may fail if controller is transitioning to `Working` or `RequestShutdown`.
// - If that is the case, the pre-conditions for those transitions have been fulfilled
//   and work can continue without sleeping.
// Post-conditions:
// - Wait on the condition variable for `Working` or `RequestShutdown`.
//
// (C) Spinning -> Working
// Pre-conditions:
// - New work has been set in the control block.
// Action:
// - Compare-and-swap (strong) on `threadstate` from Spinning to Working.
// Failure:
// - Thread is transitioning to sleep.
// - Try transition (D)
// Post-conditions:
// - None
//
// (D) Sleeping -> working
// Pre-conditions:
// - New work has been set in the control block.
// - Acquire condition variable mutex to ensure thread is completely asleep.
// Action:
// - Compare-and-swap (strong) on `threadstate` from Sleeping to Working.
// Failure:
// - Must not fail. Failure is an error.
// Post-conditions:
// - Notify the condition variable.
//
// (E) Working -> Exception
// Pre-conditions:
// - Worker experiences an exception while executing.
// Action:
// - Atomic store.
// Post-conditions:
// - Set error to exception-channel promise.
// - Shutdown thread.
//
// (F) Spinning -> RequestShutdown
// Pre-conditions:
// - None
// Action:
// - Compare-and-swap (strong) `threadstate` from Spinning to Working.
// Failure:
// - Thread is transitioning to sleep.
// - Try transition G.
// Post-conditions:
// - None.
//
// (G) Sleeping -> RequestShutdown
// Pre-conditions:
// - Acquire condition variable mutex to ensure thread is completely asleep.
// Action:
// - Compare-and-swap (strong) on `threadstate` from Sleeping to RequestShutdown.
// Failure:
// - Must not fail.
// Post-conditions:
// - Notify the condition variable.
template <typename Telemetry = telemetry::NoTelemetry> class ThreadControlBlock {
  public:
    ThreadControlBlock() = default;

    /////
    ///// Shared API
    /////

    ///
    /// Atomically get the state of the thread control variable.
    /// @param order The memory ordering to be used for the atomic load.
    ///
    ThreadState get_state(std::memory_order order = std::memory_order_seq_cst) const {
        return threadstate_.load(order);
    }

    ///
    /// Atomically set the state of the thread control variable.
    /// @param new_state The new state to assign to the thread control variable.
    /// @param order The memory ordering to be used for the atomic store.
    ///
    void
    set_state(ThreadState new_state, std::memory_order order = std::memory_order_seq_cst) {
        threadstate_.store(new_state, order);
    }

    ///
    /// Perform a strong atomic compare-and-swap on the thread control variable.
    /// Returns whether or not the operation was successful.
    /// The read value of the thread state is assigned to `expected`.
    /// @param expected The expected thread state.
    /// @param new_state The state to transition to.
    /// @param order The memory ordering to use for the atomic operation.
    ///
    bool cas_state(
        ThreadState& expected,
        ThreadState new_state,
        std::memory_order order = std::memory_order_seq_cst
    ) {
        return threadstate_.compare_exchange_strong(expected, new_state, order);
    }

    ///
    /// Get the work function.
    /// NOTE: Non-synchronizing.
    ///
    ThreadFunctionRef get_work() const { return fn_; }

    ///
    /// Set the work function.
    /// NOTE: Non-synchronizing.
    ///
    void unsafe_set_work(ThreadFunctionRef fn) { fn_ = fn; }

    ///
    /// Wait while the threadstate is `test`. Return the new state.
    /// @param test Spin while the thread state is equal to `test`.
    ///
    ThreadState spin_while(ThreadState test) const {
        return detail::spin_while(threadstate_, test);
    }

    ///
    /// Time-out version of `spin_while`. If spinning times out, return `test`. Otherwise,
    /// return the new `threadstate`.
    /// @param test The state to wait on.
    /// @param f Time out function see `detail::spin_while_true`.
    ///
    /// @sa `svs::threads::detail::spin_while_true`.
    ///
    template <detail::SpinAbort F> ThreadState spin_while(ThreadState test, F&& f) const {
        return detail::spin_while(threadstate_, test, std::forward<F>(f));
    }

    ///
    /// Spin until the threadstate becomes `test`. Return `test`.
    /// @param test The state to wait for.
    ///
    ThreadState spin_until(ThreadState test) const {
        return detail::spin_until(threadstate_, test);
    }

    ///
    /// Block until the worker thread stops working.
    /// Return the new `threadstate`.
    ///
    ThreadState wait_while_busy() const { return spin_while(ThreadState::Working); }

    ///
    /// After a thread has been created, thus function can be called to block until it is
    /// safe to assign work to the worker thread.
    ///
    void wait_until_started() const { spin_while(boot_state()); }

    ///
    /// Return `true` if the worker thread is fully asleep. Otherwise, return `false`.
    ///
    bool is_fully_asleep() {
        if (get_state(std::memory_order_acquire) == ThreadState::Sleeping) {
            std::unique_lock lock{cv_mutex_};
            return true;
        }
        return false;
    }

    ///
    /// Block until it is guarenteed that the worker thread is fully gone to sleep on
    /// the condition variable.
    ///
    void wait_until_fully_asleep() {
        while (!is_fully_asleep()) {
            svs::detail::pause();
        }
    }

    ///
    /// Get control block telemetry. Only callable if attach telemetry is not an instance
    /// `svs::threads::telemetry::NoTelemetry`.
    ///
    const Telemetry& get_telemetry() const
        requires(!std::is_same_v<Telemetry, telemetry::NoTelemetry>)
    {
        return telemetry_;
    }

    /////
    ///// Control Side API
    /////

    bool is_okay(std::memory_order order = std::memory_order_acquire) const {
        return get_state(order) != ThreadState::Exception;
    }

    bool is_shutdown(std::memory_order order = std::memory_order_acquire) const {
        return get_state(order) == ThreadState::Shutdown;
    }

    // Preconditions:
    // - Worker thread must actually be sleeping on the condition variable.
    void unsafe_wake_thread() { cv_.notify_one(); }

    // Transition from `Spinning` or `Sleeping` to `Working`.
    // If the threadstate is in not one of these two states, then:
    // - Throw `ThreadCrashedError` is `get_state() == ThreadState::Exception`.
    // - Throw `ANNException` otherwise.
    //
    // Preconditions:
    // - Worker thread must not be in the `Working` state.
    void notify_thread(ThreadState current, ThreadState next = ThreadState::Working) {
        for (;;) {
            bool success = cas_state(current, next);

            // Rollback successful CAS in case something went wrong.
            auto rollback = [&, success, current]() {
                if (success) {
                    set_state(current);
                }
            };

            switch (current) {
                // If the thread was in the `Spinning` state and we successfully
                // performed a CAS, then the thread will now be in the working state.
                //
                // The worker thread must observe the change to working and will run
                // the new job.
                case ThreadState::Spinning: {
                    if (success) {
                        return;
                    }
                    break;
                }

                // N.B.: We already ensured the thread is asleep by acquiring the
                // condition variable mutex above.
                case ThreadState::Sleeping: {
                    if (success) {
                        { std::lock_guard lock{cv_mutex_}; }
                        unsafe_wake_thread();
                        return;
                    }
                    break;
                }
                case ThreadState::Exception: {
                    rollback();
                    throw ThreadCrashedError("Thread Crashed!");
                }
                case ThreadState::Shutdown: {
                    rollback();
                    throw ANNEXCEPTION("Trying to assign work to a shutdown thread.");
                }
                default: {
                    rollback();
                    throw ANNEXCEPTION("Concurrency Violation!");
                }
            }
        }
    }

    ///
    /// Safely assign a new job to the worker thread and notify the thread that work is
    /// available. Blocks while the worker thread is actively executing a previous job.
    ///
    /// @param fn New function to run on the worker thread.
    ///
    void assign(ThreadFunctionRef fn) {
        ThreadState current = wait_while_busy();
        unsafe_set_work(fn);
        notify_thread(current);
    }

    template <typename F = Terminate>
    void shutdown(bool wait = true, F on_error = Terminate()) {
        ThreadState current = wait_while_busy();
        bool shutdown_requested = false;
        for (;;) {
            bool exit_loop = false;
            bool success = cas_state(current, ThreadState::RequestShutdown);
            auto rollback = [&, success, current]() {
                if (success) {
                    set_state(current);
                }
            };

            switch (current) {
                case ThreadState::Spinning: {
                    if (success) {
                        shutdown_requested = true;
                        exit_loop = true;
                    }
                    break;
                }

                case ThreadState::Sleeping: {
                    if (success) {
                        { std::lock_guard lock{cv_mutex_}; }
                        unsafe_wake_thread();
                        shutdown_requested = true;
                        exit_loop = true;
                    }
                    break;
                }

                // If the thread is in an exception state or shutdown, that is okay.
                // Put everything back as we found it and exit the loop.
                case ThreadState::Exception:
                case ThreadState::Shutdown: {
                    rollback();
                    exit_loop = true;
                    break;
                }

                // Default case:
                // - If the thread was in any other state, it's the result of a concurrency
                //   violation on the controller side or otherwise a bug in the threading
                //   code.
                default: {
                    if (success) {
                        on_error();
                        return;
                    }
                }
            }

            if (exit_loop) {
                break;
            }
        }

        if (shutdown_requested && wait) {
            spin_while(ThreadState::RequestShutdown);
        }
    }

    /////
    ///// Thead-side API
    /////

    ThreadState spin_wait(size_t spin_count) {
        telemetry_.enter_spinloop();
        // Spin on the state variable for `spin_count` number of times.
        // If the variable hasn't changed, then it's time to go to sleep.
        ThreadState request =
            spin_while(ThreadState::Spinning, [local_spin_count = spin_count]() mutable {
                local_spin_count--;
                return (local_spin_count == 0);
            });

        // If the state changed and we're waiting for work, then spin until work
        // is available.
        bool timeout = (request == ThreadState::Spinning);
        if (!timeout) {
            telemetry_.exit_spinloop_success();
        } else {
            telemetry_.exit_spinloop_fail();
        }
        return request;
    }

    // Sleep the worker thread on the condition variable.
    //
    // Pre- and post-op functions allow injecting arbitrary delays in order to test
    // interleaving logic.
    //
    // When not explicitly testing the code, these template parameters should be left
    // alone.
    template <typename T = lib::donothing, typename U = lib::donothing>
    bool try_sleep(T pre_op = T{}, U post_op = U{}) {
        telemetry_.sleep_attempt();
        std::unique_lock lock{cv_mutex_};

        // Optional delay injection.
        pre_op();

        // Threadstate may change after acquiring the lock.
        // Ensure we either transition from `Spinning` to `Sleeping` or don't try sleeping.
        auto expected = ThreadState::Spinning;
        bool cansleep =
            cas_state(expected, ThreadState::Sleeping, std::memory_order_seq_cst);

        // At this point, it's okay if the controlling thread changes the state as that
        // will be observed in the in the condition variable predicate.
        if (cansleep) {
            cv_.wait(lock, [&] {
                ThreadState state = get_state(std::memory_order_acquire);
                telemetry_.sleep_predicate_check();
                post_op();
                return state == ThreadState::Working ||
                       state == ThreadState::RequestShutdown;
            });
            telemetry_.sleep_success();
            return true;
        }
        telemetry_.sleep_fail();
        return false;
    }

    ///
    /// Startup a thread.
    /// **Preconditions:**
    ///
    /// - Control block must be set to `boot_state`.
    ///
    /// @param promise The promise where thread shutdown and exceptions will live.
    /// @param spin_count The number of times to spin before attempting to sleep. A higher
    ///     count will increase the probability of new work being assigned while the thread
    ///     is spinning, at the cost of higher CPU utilization burned in idle cycles.
    /// @param startup Optional thunk to set up global state for the thread that will be
    ///     held until the thread is shutdown.
    ///
    template <typename F = DefaultStartup>
    void unsafe_run(
        std::promise<void> promise,
        size_t spin_count = default_spintime(),
        F&& startup = DefaultStartup()
    ) {
        // This variable is meant for RAII purposes and just supposed to hand-on to
        // something until the thread terminates.
        [[maybe_unused]] auto resource = startup();
        set_state(ThreadState::Spinning);
        for (;;) {
            // Spin until new work is available or we time out.
            // If we time-out, try to sleep.
            ThreadState request = spin_wait(spin_count);
            if (request == ThreadState::Spinning) {
                try_sleep();
                request = get_state(std::memory_order_acquire);
            }

            // The threadstate must be one of:
            // - RequestShutdown.
            // - Working.
            if (request == ThreadState::RequestShutdown) {
                // Failure to successfully assign the promise is a program bug.
                try {
                    promise.set_value_at_thread_exit();
                } catch (...) { std::terminate(); }
                set_state(ThreadState::Shutdown);
                return;
            }
            assert(request == ThreadState::Working);

            try {
                // Run the job and transition to `Spinning`.
                get_work()();
                set_state(ThreadState::Spinning, std::memory_order_release);
            } catch (...) {
                set_state(ThreadState::Exception);

                // `set_exception_at_thread_exit` can throw an exception.
                // However, those exceptions are either because:
                // 1. There is no shared state.
                // 2. A result or exception has already been put in the promise.
                //
                // Both of those should be considered errors so we take down the whole
                // program.
                try {
                    promise.set_exception_at_thread_exit(std::current_exception());
                } catch (...) { std::terminate(); }
                return;
            }
        }
    }

  private:
    std::atomic<ThreadState> threadstate_{boot_state()};
    ThreadFunctionRef fn_{};

    // Condition variable and associated mutex for waking up the thread if it is sleeping.
    std::condition_variable cv_{};
    std::mutex cv_mutex_{};
    [[no_unique_address]] Telemetry telemetry_{};
};

/////
///// Higher level Thread
/////

class ThreadError : public std::runtime_error {
  public:
    explicit ThreadError(const std::string& message)
        : std::runtime_error{message} {}
    explicit ThreadError(const std::exception& inner)
        : ThreadError{make_message(inner.what())} {}

    static std::string make_message(const char* message) {
        return "Spawned thread crashed with message: " + std::string(message);
    }
};

// Working with the control block, the actual `std::thread` object is external to the
// block.
//
// The `Thread` struct here packages everything together into a more useable interface.
template <typename T = telemetry::NoTelemetry> class ThreadImpl {
  public:
    template <typename F = DefaultStartup>
    explicit ThreadImpl(
        size_t spin_count = default_spintime(), F startup = DefaultStartup()
    )
        : control_{std::make_unique<ThreadControlBlock<T>>()} {
        std::promise<void> promise{};
        result_ = promise.get_future();

        // Capture the control block by reference to ensure that the controller (user of
        // the `ThreadImpl`) and the thread are sharing the same underlying data.
        //
        // Class invariant should keep the worker thead and control block alive for
        // eachother's duration.
        control_->set_state(boot_state());
        auto f = [&control_local = *control_,
                  spin_count,
                  inner_startup = std::move(startup)](std::promise<void>&& channel) {
            control_local.unsafe_run(std::move(channel), spin_count, inner_startup);
        };

        // Launch the thread and wait until it reaches far enough in its execution that
        // it is spinning. At this point, it should be save to assign work to the thread.
        worker_ = std::thread(std::move(f), std::move(promise));
        control_->spin_while(boot_state());
    }

    ///// Member Functions

    // General Queries

    ///
    /// Return `true` is the thread hasn't crashed. Note that a gracefully shutdown thread
    /// will return `true`.
    ///
    bool is_okay() { return control_->is_okay(); }

    ///
    /// Return `true` is the thread has gracefully shutdown.
    ///
    bool is_shutdown() { return control_->is_shutdown(); }

    ///
    /// Return `true` is the thread is either currently executing work or ready to accept
    /// new work.
    ///
    bool is_running() { return is_okay() && !is_shutdown(); }

    ///
    /// Block execution while the thread is executing a previously-assigned job.
    ///
    void wait() { control_->wait_while_busy(); }

    ///
    /// Get a thrown exception from a crashed thread.
    /// **Preconditions:**
    ///
    /// - The thread must have crashed and be in the `Exception` state.
    /// - No access to the `future`'s shard state must have been made before this call.
    ///   In particular, this function is not safe to call repeatedly.
    ///
    void unsafe_get_exception() {
        assert(control_->get_state() == ThreadState::Exception);
        // If we're calling this, than the thread *must* be either already shut down or
        // will be shutdown soon.
        //
        // Wait for the future to become available and rethrow the exception.
        wait_for_result();
        get_result();
        throw ANNEXCEPTION("Expected to get an exception from a crashed thread but no "
                           "exception was thrown!");
    }

    // Assign Work
    void unsafe_assign(ThreadFunctionRef fn) { control_->assign(fn); }
    void unsafe_assign_blocking(ThreadFunctionRef fn) {
        control_->assign(fn);
        wait();
    }

    void assign(ThreadFunctionRef fn) {
        // The thread we're trying to assign work to may have crashed and be in an
        // exception state.
        //
        // Also, it is possible that while we are trying to assign new work to the thread,
        // it throws an exception and enters the exception state.
        //
        // This latter case is very difficult to detect and correct from.
        //
        // What we do is:
        // * Try assigning the new work.
        // * If the thread-control assignment block sees that the thread is in the
        //   `ThreadState::Exception` state, it will throw an instance of
        //   `ThreadCrashedError`.
        //
        // * Catch the `ThreadCrashedError`.
        // * Access the `std::future` for the thread to get the actual exception that was
        //   thrown. Accessing the future will throw the exception.
        //
        // * Catch this exception and wrap its message inside a `ThreadError`.
        try {
            unsafe_assign(fn);
        } catch (const ThreadCrashedError& err) {
            try {
                unsafe_get_exception();
            } catch (const std::exception& inner_error) { throw ThreadError{inner_error}; }
        }
    }

    // Thread shutdown
    void request_shutdown() { control_->shutdown(false); }
    void wait_for_shutdown() {
        if (worker_.joinable()) {
            worker_.join();
        }
        if (result_.valid()) {
            get_result();
        }
    }

    ///
    /// Attempt to gracefully shutdown the Thread.
    /// This function is safe to call multiple times.
    ///
    void shutdown() {
        if (is_initialized()) {
            request_shutdown();
        } else {
            assert(!result_.valid());
            assert(!worker_.joinable());
        }
        wait_for_shutdown();
    }

    ///// Special Member Functions

    // Delete the copy operations.
    // `std::unique_ptr`, `std::future` and `std::thread` are not copyable anyways.
    ThreadImpl(const ThreadImpl& /*unused*/) = delete;
    ThreadImpl& operator=(const ThreadImpl& /*unused*/) = delete;

    // Movement is only safe if it is safe to move the underlying thread.
    // This, in turn, is only safe if thread is not joinable.
    //
    // In that case, optimistically just try to default move.
    // If this operation isn't okay, than the underlying `std::thread` is going kill
    // our program anyways.
    ThreadImpl(ThreadImpl&& other) noexcept = default;
    ThreadImpl& operator=(ThreadImpl&& other) noexcept = default;

    ~ThreadImpl() { shutdown(); }

  private:
    ///// Members
    std::unique_ptr<ThreadControlBlock<T>> control_;
    std::future<void> result_;
    std::thread worker_;

    ///// Private methods

    ///
    /// Wait until shared state is available in the future.
    ///
    void wait_for_result() { result_.wait(); }

    ///
    /// Get the result from the future. If the thread exited gracefully, then nothing will
    /// happen. If the thread crashed due to a job reporting an exception, then that
    /// exception will be thrown.
    ///
    /// Thus function is not safe to call if there is no shared state.
    /// In other words, there must be shared state to get.
    ///
    void get_result() { result_.get(); }

    ///
    /// Return `true` if the control block for this thread has been initialized. Return
    /// `false` otherwise. Examples of uninitialized state would be a default-constructed
    /// Thread or a moved-from Thread.
    ///
    bool is_initialized() const { return control_ != nullptr; }
};

using Thread = ThreadImpl<telemetry::NoTelemetry>;

} // namespace threads
} // namespace svs
