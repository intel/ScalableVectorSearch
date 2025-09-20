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

// header under test.
#include "svs/core/data.h"

// test utils
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"

namespace {
template <typename T, typename F = svs::lib::identity>
void set_sequential(T& x, F&& f = svs::lib::identity()) {
    size_t count = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        for (auto& j : x.get_datum(i)) {
            j = f(count);
            ++count;
        }
    }
}

template <typename T> bool is_sequential(const T& x) {
    size_t count = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        for (auto& j : x.get_datum(i)) {
            if (j != count) {
                return false;
            }
            ++count;
        }
    }
    return true;
}
} // namespace

CATCH_TEST_CASE("Data Loading/Saving", "[core][data]") {
    svs_test::prepare_temp_directory();
    auto temp_directory = svs_test::temp_directory();

    CATCH_SECTION("Data") {
        auto x = svs::data::SimpleData<float, svs::Dynamic>(10, 10);

        // Populate the contents of `x`.
        set_sequential(x);
        CATCH_REQUIRE(is_sequential(x));
        CATCH_REQUIRE(x == x);

        // Save to the temporary directory.
        svs::lib::save_to_disk(x, temp_directory);
        auto loader = svs::VectorDataLoader<float, 10>(temp_directory);
        auto y = loader.load();
        CATCH_REQUIRE(x == y);

        // Make sure we get an error if we try to load with the wrong element type.
        auto otherloader = svs::VectorDataLoader<uint8_t, svs::Dynamic>(temp_directory);
        CATCH_REQUIRE_THROWS_AS(otherloader.load(), svs::ANNException);

        // If we get the dimensionality wrong, we should also throw an error.
        auto wrongloader = svs::VectorDataLoader<float, 100>(temp_directory);
        CATCH_REQUIRE_THROWS_AS(wrongloader.load(), svs::ANNException);

        ///// Make sure the loading escape hatch works.
        auto z = svs::data::SimpleData<float, svs::Dynamic>(10, 10);
        set_sequential(z, [](auto x) { return x + 100; });
        CATCH_REQUIRE(z != x);

        // Use the underlying direct save to avoid any TOML file generation.
        auto bypass_file = temp_directory / "file.svs";
        svs::io::save(z, svs::io::NativeFile(bypass_file));
        loader = svs::VectorDataLoader<float, 10>(bypass_file);
        auto w = loader.load();
        CATCH_REQUIRE(w == z);
    }
}

CATCH_TEST_CASE("Element Size", "[core][data]") {
    CATCH_SECTION("Check element_size()") {
        // Test with float, dynamic dimensions
        auto float_data = svs::data::SimpleData<float, svs::Dynamic>(5, 10);
        CATCH_REQUIRE(float_data.element_size() == sizeof(float) * 10);

        // Test with double, dynamic dimensions
        auto double_data = svs::data::SimpleData<double, svs::Dynamic>(3, 16);
        CATCH_REQUIRE(double_data.element_size() == sizeof(double) * 16);

        // Test with int8_t, fixed dimensions
        auto int8_data = svs::data::SimpleData<int8_t, 32>(10, 32);
        CATCH_REQUIRE(int8_data.element_size() == sizeof(int8_t) * 32);

        // Test with int16_t, dynamic dimensions
        auto int16_data = svs::data::SimpleData<int16_t, svs::Dynamic>(8, 64);
        CATCH_REQUIRE(int16_data.element_size() == sizeof(int16_t) * 64);

        // Test with int32_t, fixed dimensions
        auto int32_data = svs::data::SimpleData<int32_t, 128>(5, 128);
        CATCH_REQUIRE(int32_data.element_size() == sizeof(int32_t) * 128);

        // Test with uint8_t, dynamic dimensions
        auto uint8_data = svs::data::SimpleData<uint8_t, svs::Dynamic>(12, 256);
        CATCH_REQUIRE(uint8_data.element_size() == sizeof(uint8_t) * 256);

        // Test with uint16_t, fixed dimensions
        auto uint16_data = svs::data::SimpleData<uint16_t, 48>(7, 48);
        CATCH_REQUIRE(uint16_data.element_size() == sizeof(uint16_t) * 48);

        // Test with uint32_t, dynamic dimensions
        auto uint32_data = svs::data::SimpleData<uint32_t, svs::Dynamic>(6, 96);
        CATCH_REQUIRE(uint32_data.element_size() == sizeof(uint32_t) * 96);

        // Test fixed dimensions with blocked storage
        auto blocked_fixed = svs::data::BlockedData<int32_t, 64>(25, 64);
        CATCH_REQUIRE(blocked_fixed.element_size() == sizeof(int32_t) * 64);

        // Test element_size consistency across different instances
        auto data1 = svs::data::SimpleData<float, svs::Dynamic>(10, 20);
        // Different size, same dims
        auto data2 = svs::data::SimpleData<float, svs::Dynamic>(50, 20);
        CATCH_REQUIRE(data1.element_size() == data2.element_size());

        // Test consistency across different data types with same dimensions
        auto float_128 = svs::data::SimpleData<float, svs::Dynamic>(5, 128);
        auto double_128 = svs::data::SimpleData<double, svs::Dynamic>(5, 128);
        CATCH_REQUIRE(float_128.element_size() == sizeof(float) * 128);
        CATCH_REQUIRE(double_128.element_size() == sizeof(double) * 128);
        // double is 2x float
        CATCH_REQUIRE(double_128.element_size() == 2 * float_128.element_size());
    }
}
