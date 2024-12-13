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

// Header Under Test
#include "svs/core/loading.h"

// Additional headers.
#include "svs/lib/misc.h"
#include "svs/lib/threads.h"

// Catch2
#include "catch2/catch_test_macros.hpp"

namespace {

template <bool Threaded, bool NoArg> struct Loadable {
    static constexpr size_t threaded_call = 1;
    static constexpr size_t noarg_call = 2;
    static_assert(threaded_call != noarg_call);

    template <svs::threads::ThreadPool Pool>
        requires Threaded
    size_t load(Pool& SVS_UNUSED(threadpool)) const {
        return threaded_call;
    }

    size_t load() const
        requires NoArg
    {
        return noarg_call;
    }
};

struct MoveOnly {
  public:
    MoveOnly(size_t value)
        : value_{value} {}

    MoveOnly(const MoveOnly&) = delete;
    MoveOnly& operator=(const MoveOnly&) = delete;

    MoveOnly(MoveOnly&&) = default;
    MoveOnly& operator=(MoveOnly&&) = default;

  public:
    size_t value_;
};

} // namespace

CATCH_TEST_CASE("Dispatch Load", "[core][dispatch_load]") {
    CATCH_SECTION("Threaded") {
        auto pool = svs::threads::SequentialThreadPool();
        CATCH_SECTION("Threaded Load") {
            // Instantiate with both a threaded and non-threaded call - make sure the
            // threaded version is the one that is called.
            auto x = Loadable<true, true>();
            auto y = svs::detail::dispatch_load(x, pool);
            CATCH_REQUIRE(y == Loadable<true, true>::threaded_call);
        }

        CATCH_SECTION("Non-Threaded Load") {
            // Instantiate with both a threaded and non-threaded call - make sure the
            // threaded version is the one that is called.
            auto x = Loadable<false, true>();
            auto y = svs::detail::dispatch_load(x, pool);
            CATCH_REQUIRE(y == Loadable<false, true>::noarg_call);
        }

        CATCH_SECTION("Threaded Lazy") {
            // Construct an overloaded `Lazy` object to ensure the threaded alternative
            // has precedence.
            auto lazy_with_both = svs::lib::Lazy(
                [](svs::threads::ThreadPool auto&) { return 0; }, []() { return 1; }
            );

            CATCH_REQUIRE(svs::detail::dispatch_load(lazy_with_both, pool) == 0);
            CATCH_REQUIRE(lazy_with_both() == 1);

            // Make sure that if we only supply a threaded alternative, nothing untoward
            // occurs.
            auto lazy_only_threaded =
                svs::lib::Lazy([](svs::threads::ThreadPool auto&) { return 1; });
            CATCH_REQUIRE(svs::detail::dispatch_load(lazy_only_threaded, pool) == 1);
        }

        CATCH_SECTION("Non-Threaded Lazy") {
            auto lazy = svs::lib::Lazy([]() { return 1; });
            CATCH_REQUIRE(svs::detail::dispatch_load(lazy, pool) == 1);
        }

        CATCH_SECTION("Pass Through") {
            // Passing in move-only objects should work.
            auto moveonly = MoveOnly(10);
            auto result = svs::detail::dispatch_load(std::move(moveonly), pool);
            CATCH_REQUIRE(result.value_ == 10);
        }
    }

    CATCH_SECTION("Threaded") {
        CATCH_SECTION("Threaded Load") {
            // Instantiate with both a threaded and non-threaded call.
            // Since we're calling "dispatch_load" without a threadpool, the no-argument
            // method should be called.
            auto x = Loadable<true, true>();
            auto y = svs::detail::dispatch_load(x);
            CATCH_REQUIRE(y == Loadable<true, true>::noarg_call);
        }

        CATCH_SECTION("Non-Threaded Load") {
            // Instantiate with both a threaded and non-threaded call - make sure the
            // threaded version is the one that is called.
            auto x = Loadable<false, true>();
            auto y = svs::detail::dispatch_load(x);
            CATCH_REQUIRE(y == Loadable<false, true>::noarg_call);
        }

        CATCH_SECTION("Both Lazy") {
            auto lazy_with_both = svs::lib::Lazy(
                [](svs::threads::ThreadPool auto&) { return 0; }, []() { return 1; }
            );

            CATCH_REQUIRE(svs::detail::dispatch_load(lazy_with_both) == 1);
        }

        CATCH_SECTION("Non-Threaded Lazy") {
            auto lazy = svs::lib::Lazy([]() { return 1; });
            CATCH_REQUIRE(svs::detail::dispatch_load(lazy) == 1);
        }

        CATCH_SECTION("Pass Through") {
            auto moveonly = MoveOnly(10);
            auto result = svs::detail::dispatch_load(std::move(moveonly));
            CATCH_REQUIRE(result.value_ == 10);
        }
    }
}
