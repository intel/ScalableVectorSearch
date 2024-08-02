/**
 *    Copyright (C) 2023, Intel Corporation
 *
 *    You can redistribute and/or modify this software under the terms of the
 *    GNU Affero General Public License version 3.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    version 3 along with this software. If not, see
 *    <https://www.gnu.org/licenses/agpl-3.0.en.html>.
 */

// svs
#include "svs/core/data.h"
#include "svs/core/data/simple.h"
#include "svs/lib/saveload.h"
#include "svs/lib/static.h"

// tests
#include "tests/svs/core/data/data.h"
#include "tests/utils/utils.h"

// stdlib
#include <span>
#include <type_traits>

// catch2
#include "catch2/catch_test_macros.hpp"

namespace {

// An invalid schema - should not be loadable.
constexpr std::string_view invalid_schema = R"(
__version__ = 'v0.0.2'

[object]
__schema__ = 'uncompressed_data_nomatch'
__version__ = 'v0.0.0'
binary_file = 'data_0.svs'
dims = 5
eltype = 'uint8'
name = 'uncompressed'
num_vectors = 5
uuid = 'bc2a95c1-f882-49c9-928f-437083800700'
)";

template <typename T, size_t N>
void generate_serialized_file(
    const std::filesystem::path& dir, size_t size, svs::lib::MaybeStatic<N> dims = {}
) {
    auto data = svs::data::SimpleData<T, N>(size, dims);
    svs_test::data::set_sequential(data);
    svs::lib::save_to_disk(data, dir);
}

// Test routine.
template <typename T, size_t N>
void test_serialization(
    const std::filesystem::path& dir, size_t size, svs::lib::MaybeStatic<N> dims = {}
) {
    generate_serialized_file<T>(dir, size, dims);
    size_t dynamic_dims = dims;

    auto e = svs::lib::try_load_from_disk<svs::data::Matcher>(dir);
    CATCH_REQUIRE(e);
    auto checker = e.value();
    CATCH_REQUIRE(checker.eltype == svs::datatype_v<T>);
    CATCH_REQUIRE(checker.dims == dynamic_dims);

    // Load Resolvers
    auto loader = svs::UnspecializedVectorDataLoader(dir);
    CATCH_REQUIRE(loader.type_ == svs::datatype_v<T>);
    CATCH_REQUIRE(loader.dims_ == dynamic_dims);

    // Generate an invalid schema - we should then fail to load a checker without throwing
    // an exception.
    auto file = dir / "svs_config.toml";
    {
        auto table = toml::parse(invalid_schema);
        auto io = svs::lib::open_write(file);
        io << table << '\n';
    }

    e = svs::lib::try_load_from_disk<svs::data::Matcher>(dir);
    CATCH_REQUIRE(!e);
    CATCH_REQUIRE(e.error() == svs::lib::TryLoadFailureReason::InvalidSchema);
}

} // namespace

CATCH_TEST_CASE("Testing Serialization", "[core][data][checker]") {
    auto temp_directory = svs_test::prepare_temp_directory_v2();
    test_serialization<float, svs::Dynamic>(temp_directory, 5, svs::lib::MaybeStatic(20));
    test_serialization<double, svs::Dynamic>(temp_directory, 5, svs::lib::MaybeStatic(20));
    test_serialization<int32_t, svs::Dynamic>(temp_directory, 5, svs::lib::MaybeStatic(20));
    test_serialization<uint8_t, svs::Dynamic>(temp_directory, 5, svs::lib::MaybeStatic(5));
}
