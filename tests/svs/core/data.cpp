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
