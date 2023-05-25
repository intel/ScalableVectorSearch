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
#include <iostream>
#include <sstream>
#include <thread>

#include "svs/lib/numa.h"

#include "catch2/catch_test_macros.hpp"

CATCH_TEST_CASE("Testing NUMA utilities", "[core][numa]") {
    CATCH_SECTION("Bitmask") {
        CATCH_REQUIRE(svs::numa::detail::max_count<svs::numa::CPUMask>() > 0);
        CATCH_SECTION("CPUMask") {
            auto bitmask = svs::numa::CPUBitMask{};
            CATCH_REQUIRE(bitmask.count() == 0);
            CATCH_REQUIRE(bitmask.get_nth(0) == bitmask.capacity());
            bitmask.set(2, true);
            bitmask.set(3, true);
            bitmask.set(4, true);
            CATCH_REQUIRE(bitmask.count() == 3);
            CATCH_REQUIRE(bitmask.get_nth(0) == 2);
            CATCH_REQUIRE(bitmask.get_nth(1) == 3);
            CATCH_REQUIRE(bitmask.get_nth(2) == 4);
            CATCH_REQUIRE(bitmask.get_nth(3) == bitmask.capacity());

            bitmask.set(2, false);
            CATCH_REQUIRE(bitmask.count() == 2);
            CATCH_REQUIRE(bitmask.get_nth(0) == 3);
            CATCH_REQUIRE(bitmask.get_nth(1) == 4);
            CATCH_REQUIRE(bitmask.get_nth(2) == bitmask.capacity());
        }

        CATCH_SECTION("Nodemask") {
            auto bitmask = svs::numa::NodeBitMask{};
            CATCH_REQUIRE(bitmask.count() == 0);
            CATCH_REQUIRE(bitmask.get_nth(0) == bitmask.capacity());
            bitmask.set(0, true);
            CATCH_REQUIRE(bitmask.count() == 1);
            CATCH_REQUIRE(bitmask.get_nth(0) == 0);
            CATCH_REQUIRE(bitmask.get_nth(1) == bitmask.capacity());

            bitmask.set(0, false);
            CATCH_REQUIRE(bitmask.count() == 0);
        }

        CATCH_SECTION("Copy Operations") {
            auto bitmask = svs::numa::CPUBitMask{};
            bitmask.set(0, true);
            bitmask.set(2, true);
            //// Copy constructor
            auto other = bitmask;
            CATCH_REQUIRE(other.count() == 2);
            CATCH_REQUIRE(other.get(0) == true);
            CATCH_REQUIRE(other.get(1) == false);
            CATCH_REQUIRE(other.get(2) == true);
            CATCH_REQUIRE(other.get(3) == false);

            // Make sure the two are independent.
            bitmask.set(1, true);
            CATCH_REQUIRE(bitmask.get(1) == true);
            CATCH_REQUIRE(other.get(1) == false);

            //// Copy Assignment.
            bitmask = other;
            CATCH_REQUIRE(bitmask.get(1) == false);
            CATCH_REQUIRE(other.get(1) == false);
            bitmask.set(1, true);
            CATCH_REQUIRE(bitmask.get(1) == true);
            CATCH_REQUIRE(other.get(1) == false);

            // Self assignment.
            bitmask = bitmask;
            CATCH_REQUIRE(bitmask.get(1) == true);
        }

        CATCH_SECTION("Move Operations") {
            auto bitmask = svs::numa::CPUBitMask{};
            bitmask.set(0, true);
            bitmask.set(2, true);
            //// Move constructor
            {
                auto other = std::move(bitmask);
                CATCH_REQUIRE(other.get(0) == true);
                CATCH_REQUIRE(other.get(1) == false);
                CATCH_REQUIRE(other.get(2) == true);
                CATCH_REQUIRE(other.get(3) == false);
            }

            //// Move assignment operator.
            auto other = svs::numa::CPUBitMask{};
            other.set(1, true);
            other.set(3, true);
            bitmask = std::move(other);
            CATCH_REQUIRE(bitmask.get(0) == false);
            CATCH_REQUIRE(bitmask.get(1) == true);
            CATCH_REQUIRE(bitmask.get(2) == false);
            CATCH_REQUIRE(bitmask.get(3) == true);
        }

        // Printing
        std::ostringstream stream{};
        CATCH_SECTION("CPU Bitmask Display") {
            auto cpumask = svs::numa::CPUBitMask{};
            cpumask.set(0, true);
            cpumask.set(2, true);
            stream << cpumask;
            CATCH_REQUIRE(stream.str() == "CPUMask[0 2]");
        }

        CATCH_SECTION("Node Bitmask Display") {
            auto cpumask = svs::numa::NodeBitMask{};
            cpumask.set(0, true);
            stream << cpumask;
            CATCH_REQUIRE(stream.str() == "NodeMask[0]");
        }
    }

    CATCH_SECTION("NUMA Local") {
        // Use `std::unique_ptr` because it is a moveable but non-copyable type.
        auto x = svs::numa::NumaLocal<std::unique_ptr<size_t>>(3, [](auto& v) {
            v[0] = std::make_unique<size_t>(0);
            v[1] = std::make_unique<size_t>(1);
            v[2] = std::make_unique<size_t>(2);
        });

        CATCH_REQUIRE(*x.get_direct(0) == 0);
        CATCH_REQUIRE(*x.get_direct(1) == 1);
        CATCH_REQUIRE(*x.get_direct(2) == 2);

        // Make sure we throw an `ANNException` if we don't fill all entries.
        CATCH_REQUIRE_THROWS_AS(
            svs::numa::NumaLocal<size_t>(3, [](auto& v) { v[1] = 0; }), svs::ANNException
        );

        ///
        /// Constant Iterator
        ///
        auto const_iter = x.cbegin();
        // The dreaded DOUBLE dereference! :D
        CATCH_REQUIRE(*(*const_iter) == 0);
        ++const_iter;
        CATCH_REQUIRE(*(*const_iter) == 1);
        ++const_iter;
        CATCH_REQUIRE(*(*const_iter) == 2);
        ++const_iter;
        CATCH_REQUIRE(const_iter == x.cend());

        ///
        /// Normal Iterator
        ///
        auto iter = x.begin();
        *(*iter) = 3;
        iter++;
        *(*iter) = 4;
        iter++;
        *(*iter) = 5;
        iter++;
        CATCH_REQUIRE(iter == x.end());

        const_iter = x.cbegin();
        CATCH_REQUIRE(*(*const_iter) == 3);
        ++const_iter;
        CATCH_REQUIRE(*(*const_iter) == 4);
        ++const_iter;
        CATCH_REQUIRE(*(*const_iter) == 5);
        ++const_iter;
        CATCH_REQUIRE(const_iter == x.cend());
    }

    CATCH_SECTION("Thread Local NUMA node") {
        // Clear out any interaction from other runs.
        svs::numa::tls::assigned_node = std::numeric_limits<size_t>::max();
        std::thread thread1{[] {
            CATCH_REQUIRE(svs::numa::tls::is_assigned() == false);
            svs::numa::tls::assigned_node = 0;
            CATCH_REQUIRE(svs::numa::tls::is_assigned() == true);
        }};
        thread1.join();

        // Setting the value in another thread should not affect this thread.
        CATCH_REQUIRE(svs::numa::tls::is_assigned() == false);
    }

    CATCH_SECTION("NUMA Local") {
        // Use `std::unique_ptr` because it is a moveable but non-copyable type.
        auto x = svs::numa::NumaLocal<std::unique_ptr<size_t>>(3, [](auto& v) {
            v[0] = std::make_unique<size_t>(0);
            v[1] = std::make_unique<size_t>(1);
            v[2] = std::make_unique<size_t>(2);
        });

        CATCH_REQUIRE(*x.get_direct(0) == 0);
        CATCH_REQUIRE(*x.get_direct(1) == 1);
        CATCH_REQUIRE(*x.get_direct(2) == 2);

        // Make sure we throw an `ANNException` if we don't fill all entries.
        CATCH_REQUIRE_THROWS_AS(
            svs::numa::NumaLocal<size_t>(3, [](auto& v) { v[1] = 0; }), svs::ANNException
        );
    }

    CATCH_SECTION("Thread Local NUMA node") {
        // Clear out any interaction from other runs.
        svs::numa::tls::assigned_node = std::numeric_limits<size_t>::max();
        std::thread thread1{[] {
            CATCH_REQUIRE(svs::numa::tls::is_assigned() == false);
            svs::numa::tls::assigned_node = 0;
            CATCH_REQUIRE(svs::numa::tls::is_assigned() == true);
        }};
        thread1.join();

        // Setting the value in another thread should not affect this thread.
        CATCH_REQUIRE(svs::numa::tls::is_assigned() == false);
    }
}
