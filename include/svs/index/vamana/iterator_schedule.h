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

#pragma once

// svs
#include "svs/index/vamana/search_params.h"
#include "svs/lib/preprocessor.h"

// stl
#include <cstddef>
#include <type_traits>
#include <utility>

namespace svs::index::vamana {

// clang-format off

/// @brief Protocol for ``BatchIterator`` compatible growth schedules.
///
/// Abstract schedules must take an iteration number and yield the search parameters to
/// use for that iteration.
///
/// Generally speaking, the schedule must (at least) increase the search buffer capacity
/// in order to ensure new valid items are returned on the next invocation of the iterator.
///
/// It is up to the implementation of the schedule to ensure that this holds.
/// @code{.cpp}
/// template<typename T>
/// concept IteratorSchedule = requires(const T& const_schedule, size_t iteration) {
///    // Return the schedule to use for the next iteration.
///    // The underlying schedule must be const invocable/pure for exception safety
///    // reasons.
///    { const_schedule.for_iteration(iteration) } ->
///        std::convertible_to<vamana::VamanaSearchParameters>;
///
///    // The maximum number of valid elements to yield from the search buffer.
///    // This is necessary for datasets that use reranking, since those datasets may need
///    // to super-sample in order to guarantee the quality of the top-K neighbors.
///    //
///    // This method takes the iteration number to provide a mechanism though with the
///    // number of neighbors can be modified.
///    { const_schedule.max_candidates(iteration) } -> std::same_as<size_t>;
/// };
/// @endcode
template <typename T>
concept IteratorSchedule =
std::is_nothrow_swappable_v<T> &&
requires(const T& const_schedule, size_t iteration) {
    { const_schedule.for_iteration(iteration) } ->
        std::convertible_to<vamana::VamanaSearchParameters>;
    { const_schedule.max_candidates(iteration) } -> std::same_as<size_t>;
};
// clang-format on

/////
///// Concrete Schedules
/////

/// @brief A simple schedule that accepts a batchsize and base parameters.
///
/// On each iteration, the search window size and buffer capacity are both increased by
/// the batchsize.
///
/// All other aspects of the base parameters are preserved (such as prefetch parameters
/// and visited filter).
class DefaultSchedule {
  public:
    /// @brief Construct a new default schedule.
    DefaultSchedule(const vamana::VamanaSearchParameters& base, size_t batch_size)
        : base_parameters_{base}
        , batch_size_{batch_size} {}

    /// @brief Return parameters for batch ``i``.
    ///
    /// Increments the search window size and capacity of the base search parameters by
    /// @code{.cpp}
    /// i * batch_size
    /// @endcode
    /// where ``batch_size`` is the value passed to the class constructor.
    vamana::VamanaSearchParameters for_iteration(size_t i) const {
        // Copy the base parameters, then scale the search window size and capacity by
        // `batch_size_` on each iteration.
        auto p = base_parameters_;
        p.buffer_config_.increment(i * batch_size_);
        return p;
    }

    /// @brief Return maximum number of candidates to return for iteration ``i``.
    ///
    /// The return value is constant and equal to the ``batch_size`` argument provided to
    /// the class constructor.
    size_t max_candidates(size_t SVS_UNUSED(i)) const { return batch_size_; }

    ///// Members
  private:
    // The starting search parameters to use for the first iteration of search.
    vamana::VamanaSearchParameters base_parameters_;
    // The new number of elements to return on each iteration.
    size_t batch_size_;
};

static_assert(
    IteratorSchedule<DefaultSchedule>,
    "Default Schedule must satisfy the IteratorSchedule concept."
);

/// An iterator schedule with separate scaling parameters for the Vamana buffer
/// configuration and batchsize.
class LinearSchedule {
  private:
    ///// Members
    vamana::VamanaSearchParameters base_parameters_;
    uint16_t scale_search_window_;
    uint16_t scale_buffer_capacity_;
    int16_t enable_filter_after_;
    uint16_t batch_size_start_;
    uint16_t scale_batch_size_;

    // Invariant Checks.
    void check_invariants() {
        // If the capacity scaling is slower than the window scaling, they will eventually
        // collide.
        if (scale_buffer_capacity_ < scale_search_window_) {
            throw ANNEXCEPTION("Capacity scaling must be at least as big as window scaling!"
            );
        }
        // Clamp filter enablement.
        enable_filter_after_ = std::max(enable_filter_after_, int16_t{-1});
        // Batch size should be at least one.
        // Otherwise, why are we creating an iterator?
        if (batch_size_start_ == 0) {
            throw ANNEXCEPTION(
                "Batch size start must be at least 1. Instead, got {}.", batch_size_start_
            );
        }
    }

  public:
    LinearSchedule(
        const vamana::VamanaSearchParameters& base_parameters,
        uint16_t scale_search_window,
        uint16_t scale_buffer_capacity,
        int16_t enable_filter_after,
        uint16_t batch_size_start,
        uint16_t scale_batch_size
    )
        : base_parameters_{base_parameters}
        , scale_search_window_{scale_search_window}
        , scale_buffer_capacity_{scale_buffer_capacity}
        , enable_filter_after_{enable_filter_after}
        , batch_size_start_{batch_size_start}
        , scale_batch_size_{scale_batch_size} {
        // Check invariants.
        check_invariants();
    }

    LinearSchedule(
        const vamana::VamanaSearchParameters& base_parameters,
        size_t batchsize,
        int16_t enable_filter_after = -1
    )
        : LinearSchedule{
              base_parameters,
              lib::narrow<uint16_t>(batchsize),
              lib::narrow_cast<uint16_t>(batchsize),
              enable_filter_after,
              lib::narrow_cast<uint16_t>(batchsize),
              uint16_t{0}} {}

    /// @brief Update the search buffer scaling parameters.
    ///
    /// @param config Config to apply for the scaling parameters.
    ///
    /// This method accepts a ``vamana::SearchBufferConfig`` because the invariant that
    /// the scaling for buffer capacity must be at least as large as the scaling for the
    /// search window size.
    ///
    /// This invariant is automatically guarenteed by the ``vamana::SearchBufferConfig``.
    ///
    /// @code{cpp}
    /// auto schedule = svs::index::vamana::LinearSchedule{...};
    /// // Increase the search window size by 10 and the buffer capacity by 20 on each
    /// // iteration.
    /// schedule.buffer_scaling({10, 20});
    /// @endcode
    LinearSchedule& buffer_scaling(vamana::SearchBufferConfig config) {
        scale_search_window_ = config.get_search_window_size();
        scale_buffer_capacity_ = config.get_total_capacity();
        return *this;
    }

    /// @brief Enable the visited filter on and after the specified iteration.
    ///
    /// If the visited filter is not needed, providing a negative value or using
    /// ``disable_filter()`` is sufficient.
    LinearSchedule& enable_filter_after(int16_t iteration) {
        enable_filter_after_ = std::max(iteration, int16_t{-1});
        return *this;
    }

    /// @brief Disable the visited filter from ever being used.
    LinearSchedule& disable_filter() { return enable_filter_after(-1); }

    /// @brief Configure the starting batch size.
    ///
    /// The starting batch size must be at least 1.
    ///
    /// @throws ANNException if the batchsize is no at least 1.
    LinearSchedule& starting_batch_size(uint16_t batch_size) {
        if (batch_size == 0) {
            throw ANNEXCEPTION("Starting batch size must be nonzero.");
        }
        batch_size_start_ = batch_size;
        return *this;
    }

    /// @brief Configure the batch size scaling.
    ///
    /// Batch size scaling provides a way of progressively yielding more neighbors on each
    /// iteration using the formula:
    /// @code{cpp}
    /// batch_size + scaling * iterations;
    /// @endcode
    /// To yield the same number of neighbors on each iteration, set the scaling to 0;
    ///
    /// Note that the number of yielded neighbors is not guarenteed to be the requested
    /// batchsize depending on the scaling of the buffer configuration.
    LinearSchedule& batch_size_scaling(uint16_t scaling) {
        scale_batch_size_ = scaling;
        return *this;
    }

    /// @brief Disable batch size scaling.
    ///
    /// This means the batch iterator will attempt to return the same number of neighbors
    /// on each iteration.
    ///
    /// Note that the number of yielded neighbors is not guarenteed to be the requested
    /// batchsize depending on the scaling of the buffer configuration.
    LinearSchedule& disable_batch_size_scaling() { return batch_size_scaling(0); }

    /// @brief Return search parameters for iteration ``i``.
    ///
    /// @param i The iteration number.
    ///
    /// The yielded ``vamana::VamanaSearchParameters`` will have its search window size
    /// and capacity scaled from their baseline.
    ///
    /// The visited filter will also be enabled on and after its specified iteration,
    /// if applicable.
    vamana::VamanaSearchParameters for_iteration(size_t i) const {
        // Copy the base parameters and scale the fields according to the internal
        // configuration.
        auto p = base_parameters_;
        p.buffer_config_.increment({scale_search_window_ * i, scale_buffer_capacity_ * i});

        // `narrow_cast` guarenteed to succeed since we've already ruled out negative
        // values for `enable_filter_after_`.
        if (enable_filter_after_ > -1 &&
            i >= lib::narrow_cast<size_t>(enable_filter_after_)) {
            p.search_buffer_visited_set_ = true;
        }
        return p;
    }

    /// @brief Return the maximum number of candidates to yield this iteration.
    ///
    /// @param i The iteration number.
    ///
    /// Linearly scales the starting batchsize with by scaling parameter.
    size_t max_candidates(size_t i) const {
        return batch_size_start_ + scale_batch_size_ * i;
    }
};

static_assert(
    IteratorSchedule<LinearSchedule>,
    "LinearSchedule must satisfy the IteratorSchedule concept."
);

/////
///// Type Erasure
/////

// A type-erased implementation of an IteratorSchedule.
class AbstractIteratorSchedule {
  private:
    struct Interface {
        virtual vamana::VamanaSearchParameters for_iteration(size_t iteration) const = 0;
        virtual size_t max_candidates(size_t iteration) const = 0;
        virtual std::unique_ptr<Interface> clone() const = 0;

        virtual ~Interface() = default;
    };

    template <vamana::IteratorSchedule Impl> struct Implementation : public Interface {
        ///// Constructors
        Implementation(Impl&& impl)
            : impl_{std::move(impl)} {}
        Implementation(const Impl& impl)
            : impl_{impl} {}
        template <typename... Args>
        Implementation(std::in_place_t SVS_UNUSED(tag), Args&&... args)
            : impl_{SVS_FWD(args)...} {}

        ///// Interface implementation
        virtual vamana::VamanaSearchParameters for_iteration(size_t iteration
        ) const override {
            return impl_.for_iteration(iteration);
        }

        virtual size_t max_candidates(size_t iteration) const override {
            return impl_.max_candidates(iteration);
        }

        virtual std::unique_ptr<Interface> clone() const override {
            // Clone by copy-construction.
            return std::make_unique<Implementation>(impl_);
        }

        // The concrete implementation.
        Impl impl_;
    };

    ///// Members
    std::unique_ptr<Interface> iface_;

  public:
    // Don't allow uninitialized schedules.
    // We have `std::optional` for that.
    AbstractIteratorSchedule() = delete;

    /// @brief Construct a new abstract schedule around the concrete implementation.
    template <
        IteratorSchedule Schedule,
        typename = std::enable_if_t<
            !std::is_same_v<std::remove_cvref_t<Schedule>, AbstractIteratorSchedule>>>
    explicit AbstractIteratorSchedule(Schedule schedule)
        : iface_{std::make_unique<Implementation<Schedule>>(std::move(schedule))} {}

    /// @brief Construct a new abstract schedule.
    ///
    /// The requested implementation will be constructed in-place by forwarding the
    /// arguments to the class's constructor.
    template <IteratorSchedule Schedule, typename... Args>
    AbstractIteratorSchedule(std::in_place_type_t<Schedule> SVS_UNUSED(tag), Args&&... args)
        : iface_{std::make_unique<Implementation<Schedule>>(
              std::in_place, SVS_FWD(args)...
          )} {}

    /// @brief Replace the wrapped schedule with a new schedule.
    template <IteratorSchedule Schedule> void reset(Schedule schedule) {
        iface_ = std::make_unique<Implementation<Schedule>>(std::move(schedule));
    }

    /// @brief Return the search parameters from the wrapped schedule.
    vamana::VamanaSearchParameters for_iteration(size_t iteration) const {
        return iface_->for_iteration(iteration);
    }

    /// @brief Return the maximum candidates from the wrapped schedule.
    size_t max_candidates(size_t iteration) const {
        return iface_->max_candidates(iteration);
    }

    // Special member functions.
    // By default, `std::unique_ptr` is not cloneable.
    // However, we've explicitly provided a `.clone()` method, so we can define copy
    // constructors and assignment operators.
    AbstractIteratorSchedule(const AbstractIteratorSchedule& other) noexcept
        : iface_{other.iface_->clone()} {}

    AbstractIteratorSchedule& operator=(const AbstractIteratorSchedule& other) {
        if (this != &other) {
            iface_ = other.iface_->clone();
        }
        return *this;
    }

    // Default the move operations and the destructor.
    AbstractIteratorSchedule(AbstractIteratorSchedule&& other) = default;
    AbstractIteratorSchedule& operator=(AbstractIteratorSchedule&& other) = default;
    ~AbstractIteratorSchedule() = default;
};

static_assert(
    IteratorSchedule<AbstractIteratorSchedule>,
    "AbstractIteratorSchedule must satisfy the IteratorSchedule concept."
);

} // namespace svs::index::vamana
