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

// header under test
#include "svs/lib/concurrency/upgradable_mutex.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

namespace {
using svs::lib::UpgradableMutex;
using svs::lib::UpgradeLock;
using svs::lib::UpgradeToUniqueLock;
} // namespace

CATCH_TEST_CASE("Upgradable Mutex - Basic Ownership", "[core][utils][upgradable_mutex]") {
    CATCH_SECTION("Exclusive try_lock is exclusive") {
        UpgradableMutex mutex{};
        CATCH_REQUIRE(mutex.try_lock());
        // No other ownership can be acquired while exclusively held.
        CATCH_REQUIRE_FALSE(mutex.try_lock());
        CATCH_REQUIRE_FALSE(mutex.try_lock_shared());
        CATCH_REQUIRE_FALSE(mutex.try_lock_upgrade());
        mutex.unlock();
        // After release, ownership is available again.
        CATCH_REQUIRE(mutex.try_lock());
        mutex.unlock();
    }

    CATCH_SECTION("Multiple shared owners are allowed") {
        UpgradableMutex mutex{};
        CATCH_REQUIRE(mutex.try_lock_shared());
        CATCH_REQUIRE(mutex.try_lock_shared());
        // Exclusive ownership is blocked while shared owners exist.
        CATCH_REQUIRE_FALSE(mutex.try_lock());
        mutex.unlock_shared();
        mutex.unlock_shared();
        CATCH_REQUIRE(mutex.try_lock());
        mutex.unlock();
    }

    CATCH_SECTION("Upgrade coexists with shared but not with upgrade or exclusive") {
        UpgradableMutex mutex{};
        CATCH_REQUIRE(mutex.try_lock_upgrade());
        // Shared readers are still allowed alongside an upgrade owner.
        CATCH_REQUIRE(mutex.try_lock_shared());
        // A second upgrade owner is not allowed.
        CATCH_REQUIRE_FALSE(mutex.try_lock_upgrade());
        // Exclusive ownership is blocked.
        CATCH_REQUIRE_FALSE(mutex.try_lock());
        mutex.unlock_shared();
        mutex.unlock_upgrade();
        CATCH_REQUIRE(mutex.try_lock());
        mutex.unlock();
    }
}

CATCH_TEST_CASE("Upgradable Mutex - Transitions", "[core][utils][upgradable_mutex]") {
    CATCH_SECTION("try_unlock_upgrade_and_lock succeeds when sole owner") {
        UpgradableMutex mutex{};
        CATCH_REQUIRE(mutex.try_lock_upgrade());
        CATCH_REQUIRE(mutex.try_unlock_upgrade_and_lock());
        // Now exclusively held.
        CATCH_REQUIRE_FALSE(mutex.try_lock_shared());
        mutex.unlock();
    }

    CATCH_SECTION("try_unlock_upgrade_and_lock fails with concurrent shared owner") {
        UpgradableMutex mutex{};
        CATCH_REQUIRE(mutex.try_lock_upgrade());
        CATCH_REQUIRE(mutex.try_lock_shared());
        CATCH_REQUIRE_FALSE(mutex.try_unlock_upgrade_and_lock());
        mutex.unlock_shared();
        // With the shared owner gone, the upgrade can now be promoted.
        CATCH_REQUIRE(mutex.try_unlock_upgrade_and_lock());
        mutex.unlock();
    }

    CATCH_SECTION("unlock_upgrade_and_lock blocks until shared owners drain") {
        UpgradableMutex mutex{};
        CATCH_REQUIRE(mutex.try_lock_upgrade());
        CATCH_REQUIRE(mutex.try_lock_shared());

        std::atomic<bool> promoted{false};
        std::thread promoter{[&]() {
            mutex.unlock_upgrade_and_lock();
            promoted.store(true);
        }};

        // The promoter must wait for the shared owner to release.
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        CATCH_REQUIRE_FALSE(promoted.load());

        mutex.unlock_shared();
        promoter.join();
        CATCH_REQUIRE(promoted.load());
        mutex.unlock();
    }

    CATCH_SECTION("unlock_and_lock_upgrade downgrades exclusive to upgrade") {
        UpgradableMutex mutex{};
        CATCH_REQUIRE(mutex.try_lock());
        mutex.unlock_and_lock_upgrade();
        // Shared readers are now admitted again.
        CATCH_REQUIRE(mutex.try_lock_shared());
        mutex.unlock_shared();
        mutex.unlock_upgrade();
    }

    CATCH_SECTION("unlock_and_lock_shared downgrades exclusive to shared") {
        UpgradableMutex mutex{};
        CATCH_REQUIRE(mutex.try_lock());
        mutex.unlock_and_lock_shared();
        CATCH_REQUIRE(mutex.try_lock_shared());
        mutex.unlock_shared();
        mutex.unlock_shared();
    }

    CATCH_SECTION("unlock_upgrade_and_lock_shared downgrades upgrade to shared") {
        UpgradableMutex mutex{};
        CATCH_REQUIRE(mutex.try_lock_upgrade());
        mutex.unlock_upgrade_and_lock_shared();
        // Another upgrade owner may now enter.
        CATCH_REQUIRE(mutex.try_lock_upgrade());
        mutex.unlock_upgrade();
        mutex.unlock_shared();
    }
}

CATCH_TEST_CASE(
    "Upgradable Mutex - Standard Lock Helpers", "[core][utils][upgradable_mutex]"
) {
    CATCH_SECTION("std::shared_lock and std::unique_lock interoperate") {
        UpgradableMutex mutex{};
        {
            std::shared_lock shared{mutex};
            CATCH_REQUIRE(shared.owns_lock());
            CATCH_REQUIRE_FALSE(mutex.try_lock());
        }
        {
            std::unique_lock unique{mutex};
            CATCH_REQUIRE(unique.owns_lock());
            CATCH_REQUIRE_FALSE(mutex.try_lock_shared());
        }
        CATCH_REQUIRE(mutex.try_lock());
        mutex.unlock();
    }
}

CATCH_TEST_CASE("UpgradeLock helper", "[core][utils][upgradable_mutex]") {
    CATCH_SECTION("Scoped acquisition and release") {
        UpgradableMutex mutex{};
        {
            UpgradeLock lock{mutex};
            CATCH_REQUIRE(lock.owns_lock());
            CATCH_REQUIRE(static_cast<bool>(lock));
            CATCH_REQUIRE(lock.mutex() == &mutex);
            CATCH_REQUIRE_FALSE(mutex.try_lock_upgrade());
        }
        CATCH_REQUIRE(mutex.try_lock_upgrade());
        mutex.unlock_upgrade();
    }

    CATCH_SECTION("Deferred and try construction") {
        UpgradableMutex mutex{};
        UpgradeLock lock{mutex, std::defer_lock};
        CATCH_REQUIRE_FALSE(lock.owns_lock());
        CATCH_REQUIRE(lock.try_lock());
        CATCH_REQUIRE(lock.owns_lock());
        lock.unlock();
        CATCH_REQUIRE_FALSE(lock.owns_lock());
        lock.lock();
        CATCH_REQUIRE(lock.owns_lock());
    }

    CATCH_SECTION("Move semantics transfer ownership") {
        UpgradableMutex mutex{};
        UpgradeLock lock{mutex};
        UpgradeLock moved{std::move(lock)};
        CATCH_REQUIRE(moved.owns_lock());
        CATCH_REQUIRE_FALSE(lock.owns_lock());
        CATCH_REQUIRE(lock.mutex() == nullptr);
    }

    CATCH_SECTION("Unlocking an unowned lock throws") {
        UpgradableMutex mutex{};
        UpgradeLock lock{mutex, std::defer_lock};
        CATCH_REQUIRE_THROWS(lock.unlock());
    }
}

CATCH_TEST_CASE("UpgradeToUniqueLock helper", "[core][utils][upgradable_mutex]") {
    CATCH_SECTION("Promotes and restores upgrade ownership") {
        UpgradableMutex mutex{};
        UpgradeLock lock{mutex};
        {
            UpgradeToUniqueLock unique{lock};
            // While promoted, no shared readers may enter.
            CATCH_REQUIRE_FALSE(mutex.try_lock_shared());
        }
        // After the scope, upgrade ownership is restored and shared readers are allowed.
        CATCH_REQUIRE(mutex.try_lock_shared());
        mutex.unlock_shared();
    }

    CATCH_SECTION("Throws when the source lock owns nothing") {
        UpgradableMutex mutex{};
        UpgradeLock lock{mutex, std::defer_lock};
        CATCH_REQUIRE_THROWS(UpgradeToUniqueLock{lock});
    }
}

CATCH_TEST_CASE("Upgradable Mutex - Concurrent Stress", "[core][utils][upgradable_mutex]") {
    UpgradableMutex mutex{};
    int shared_value = 0;
    constexpr int num_writers = 4;
    constexpr int increments_per_writer = 1000;

    std::vector<std::thread> writers{};
    for (int i = 0; i < num_writers; ++i) {
        writers.emplace_back([&]() {
            for (int j = 0; j < increments_per_writer; ++j) {
                UpgradeLock lock{mutex};
                // Read phase under upgrade ownership (shared readers may coexist).
                int observed = shared_value;
                // Decide to write; promote to exclusive to mutate safely.
                UpgradeToUniqueLock unique{lock};
                shared_value = observed + 1;
            }
        });
    }
    for (auto& writer : writers) {
        writer.join();
    }
    CATCH_REQUIRE(shared_value == num_writers * increments_per_writer);
}
