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
#include "svs/third-party/fmt.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace svs::lib {

using default_clock = std::chrono::steady_clock;
using default_time_point = std::chrono::time_point<default_clock>;

///
/// Return a time-stamp for the current time.
///
inline default_time_point now() noexcept { return default_clock::now(); }

///
/// Return the raw difference between two time points.
///
/// @tparam T The desired return type.
///
template <typename T = double>
T time_difference(default_time_point second, default_time_point first) {
    return std::chrono::duration<T>(second - first).count();
}

template <typename T> double as_seconds(T x) {
    return std::chrono::duration<double>(x).count();
}

///
/// Return the raw difference between the current time and a previous time point.
///
template <typename T = double> T time_difference(default_time_point first) {
    return time_difference<T>(now(), first);
}

inline std::pair<double, std::string_view> pretty_number(std::chrono::nanoseconds ns) {
    using rettype = std::pair<double, std::string_view>;
    auto count = ns.count();
    if (count < 1'000) {
        return rettype(count, "ns");
    }
    auto count_f64 = static_cast<double>(count);
    if (count < 1'000'000) {
        return rettype(count_f64 / 1'000., "us");
    }

    if (count < 1'000'000'000) {
        return rettype(count_f64 / 1'000'000., "ms");
    }

    return rettype(count_f64 / 1'000'000'000, "s");
}

inline std::pair<double, std::string_view> pretty_number(double seconds) {
    auto approx_ns =
        std::chrono::nanoseconds(static_cast<uint64_t>(std::round(seconds * 1'000'000'000.))
        );
    return pretty_number(approx_ns);
}

struct TimeData {
    // Type aliases
    using ns = std::chrono::nanoseconds;

    // Members
  public:
    uint64_t num_calls = 0;
    ns total_time{0};
    ns min_time = ns::max();
    ns max_time = ns::min();

    // Methods
  public:
    constexpr TimeData() = default;
    explicit constexpr TimeData(uint64_t num_calls, ns time)
        : num_calls{num_calls}
        , total_time{time}
        , min_time{time}
        , max_time{time} {}

    constexpr TimeData& operator+=(TimeData other) {
        num_calls += other.num_calls;
        total_time += other.total_time;
        min_time = std::min(min_time, other.min_time);
        max_time = std::max(max_time, other.max_time);
        return *this;
    }
};

// Forward declaration.
class Timer;

class AutoTime {
  public:
    explicit AutoTime(Timer* top_timer, Timer* timer)
        : top_timer_{top_timer}
        , timer_{timer}
        , start_{now()} {}

    std::chrono::nanoseconds elapsed() const { return now() - start_; }

    // Implementation deferred because it relies on `Timer` being defined.
    std::chrono::nanoseconds finish();

    // Make non-moveable or copyable.
    AutoTime(const AutoTime&) = delete;
    AutoTime& operator=(const AutoTime&) = delete;
    AutoTime(AutoTime&&) = delete;
    AutoTime& operator=(AutoTime&&) = delete;

    // On destruction, add the accumulated data to the timer.
    ~AutoTime() { finish(); }

  private:
    Timer* top_timer_;
    Timer* timer_;
    std::chrono::time_point<default_clock> start_;
    bool added_ = false;
};

///
/// @brief A timer class supporting nested, dynamicly scoped timing and pretty printing.
///
/// !!! NOTE
///
///     This data structure is inspired by TimerOutputs.jl by Kristoffer Carlsson.
///     The MIT licensed implementation can be found at
///     https://github.com/KristofferC/TimerOutputs.jl.
///     Please give that repository a star. :)
///
class Timer {
  private:
    static constexpr size_t indent_size = 2;

  public:
    /// @brief Construct a new empty timer.
    Timer()
        : start_time_{now()} {}

    ///
    /// @brief Push a new timer to the stack with the given label.
    ///
    /// @param label The label to assign to the timer.
    ///
    /// @returns An ``AutoTime`` which will automatically accumulate time when destroyed.
    ///
    AutoTime push_back(const std::string& label) {
        // Get the current timer in the stack.
        Timer* current_timer = this;
        if (!stack_.empty()) {
            current_timer = stack_.back();
        }

        // Check if the current timer already has this label. If so, return that timer.
        auto [it, created] = current_timer->timers_.try_emplace(label);
        auto& timer_unique_ptr = (*it).second;
        if (created) {
            timer_unique_ptr = std::make_unique<Timer>();
        }
        auto* timer = timer_unique_ptr.get();
        assert(timer != nullptr);
        stack_.push_back(timer);
        return AutoTime(this, timer);
    }

    ///
    /// @brief Get the sub timer corresponding to the label.
    ///
    /// @param label The name of the subtimer to return.
    ///
    /// @throws Exception if label does not exist.
    ///
    const Timer& get_timer(const std::string& label) const {
        return *timers_.at(label).get();
    }

    ///
    /// @brief Return the elapsed time since the timer was initialized.
    ///
    std::chrono::nanoseconds elapsed() const { return now() - start_time_; }

    ///
    /// @brief Pop a timer off the back of the stack.
    ///
    /// @param expected The expected pointer to remove (should match the timer that was
    ///     originally pushed).
    ///
    /// @throws ANNException if the timer that is removed is not equal to expected.
    ///
    void pop_back(Timer* expected) {
        assert(!stack_.empty());
        auto* back = stack_.back();
        if (back != expected) {
            throw ANNEXCEPTION("Timer corruption!");
        }
        stack_.pop_back();
    }

    /// Return the aggregate accumulated time from immediate subtimers.
    std::chrono::nanoseconds total_sub_time() const {
        auto time = std::chrono::nanoseconds(0);
        for (const auto& it : timers_) {
            time += it.second->get_time();
        }
        return time;
    }

    /// Return the accumulated time from all calls.
    std::chrono::nanoseconds get_time() const { return accumulated_data_.total_time; }

    /// Return the number of times this timer was called.
    uint64_t get_num_calls() const { return accumulated_data_.num_calls; }

    /// @brief Accumulate with the contents of ``data``.
    void accumulate(TimeData data) { accumulated_data_ += data; }

    /// @brief Return the maximum length of the sub-names of this timer.
    size_t longest_name(size_t indent = 0, size_t max_so_far = 0) const {
        for (const auto& kv : timers_) {
            max_so_far = std::max(max_so_far, kv.first.size() + indent);
            max_so_far = std::max(
                max_so_far, kv.second->longest_name(indent + indent_size, max_so_far)
            );
        }
        return max_so_far;
    }

    std::vector<std::pair<std::string, const Timer*>> subtimers() const {
        auto sorted = std::vector<std::pair<std::string, const Timer*>>{};
        for (const auto& it : timers_) {
            sorted.push_back(std::make_pair(it.first, (it.second).get()));
        }
        std::sort(sorted.begin(), sorted.end(), [](const auto& x, const auto& y) {
            return x.second->get_time() > y.second->get_time();
        });
        return sorted;
    }

    void print() const { print(std::cout); }

    void print(std::ostream& stream) const { stream << fmt::to_string(format()); }

    std::string format() const {
        auto buffer = fmt::memory_buffer();
        {
            auto out = std::back_inserter(buffer);
            format_into(out);
        }
        return fmt::to_string(std::move(buffer));
    }

    void format_into(auto& buffer) const {
        auto measured_time = as_seconds(total_sub_time());
        auto e = as_seconds(elapsed());
        // Print the header - figure out how long the section names are going to be.
        auto section_length = std::max(longest_name(), size_t(7)) + indent_size;
        auto padding = std::string(section_length - 7, ' ');
        auto str = fmt::format(
            "Section{}{:>10}{:>13}{:>12}{:>13}{:>13}{:>13}",
            padding,
            "num calls",
            "time",
            "%",
            "average",
            "min",
            "max"
        );

        auto hyphens = std::string(str.size(), '-');
        auto [measured_formatted, measured_units] = pretty_number(measured_time);
        fmt::format_to(
            buffer,
            "{}\n"
            "Total / % Measured: {:.4} {} / {:.4}\n"
            "{}\n"
            "{}\n",
            hyphens,
            measured_formatted,
            measured_units,
            measured_time / e,
            str,
            hyphens
        );
        // Sort sub-timers by execution time.
        for (const auto& it : subtimers()) {
            (it.second)->format_into_inner(
                buffer, section_length, measured_time, 0, it.first
            );
        }
        fmt::format_to(buffer, "{}", hyphens);
    }

    void format_into_inner(
        auto& buffer,
        size_t section_length,
        double measured_time,
        size_t this_indent,
        std::string_view label
    ) const {
        auto prefix = std::string(this_indent, ' ');
        auto padding = std::string(section_length - this_indent - label.size(), ' ');
        auto [num_calls, total_time, min_time, max_time] = accumulated_data_;
        auto total_time_seconds = as_seconds(total_time);

        auto [time_format, time_units] = pretty_number(total_time);
        auto [avg_format, avg_units] = pretty_number(total_time_seconds / num_calls);
        auto [min_time_format, min_time_units] = pretty_number(min_time);
        auto [max_time_format, max_time_units] = pretty_number(max_time);

        fmt::format_to(
            buffer,
            "{}{}{}{:10}{:10.4}{:>3}{:12.4}{:10.4}{:>3}{:10.4}{:>3}{:10.4}{:>3}\n",
            prefix,
            label,
            padding,
            num_calls,
            time_format,
            time_units,
            total_time_seconds / measured_time,
            avg_format,
            avg_units,
            min_time_format,
            min_time_units,
            max_time_format,
            max_time_units
        );
        for (const auto& it : subtimers()) {
            (it.second)->format_into_inner(
                buffer, section_length, measured_time, this_indent + indent_size, it.first
            );
        }
    }

    /// @brief Reset the timer.
    void clear() {
        timers_.clear();
        stack_.clear();
        accumulated_data_ = {};
        start_time_ = now();
    }

  private:
    /// Start time from when the class was constructed (or cleared).
    std::chrono::time_point<default_clock> start_time_;
    /// Accumulated runtime data.
    TimeData accumulated_data_{};
    /// Sub-timers one level of hierarchy below this timer.
    std::unordered_map<std::string, std::unique_ptr<Timer>> timers_{};
    /// Timer stack.
    std::vector<Timer*> stack_{};
};

inline std::chrono::nanoseconds AutoTime::finish() {
    auto e = elapsed();
    if (!added_) {
        added_ = true;
        timer_->accumulate(TimeData(1, e));
        top_timer_->pop_back(timer_);
    }
    return e;
}
} // namespace svs::lib

template <> struct fmt::formatter<svs::lib::Timer> : svs::format_empty {
    auto format(const svs::lib::Timer& timer, auto& ctx) const {
        // TODO: Figure out the right way to propage the context into the inner formatting
        // to save on some memory.
        return fmt::format_to(ctx.out(), "{}", timer.format());
    }
};
