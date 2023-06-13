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

// header under test
#include "svs/lib/saveload.h"

// svs
#include "svs/lib/file.h"

// test helpers
#include "tests/utils/utils.h"

// catch2
#include "catch2/catch_test_macros.hpp"

// stl
#include <filesystem>

// Test local objects.
namespace {

// Test object saving.
struct NotSaveable {};

struct Saveable {
    // Static members
    static constexpr svs::lib::Version save_version = svs::lib::Version(0, 0, 1);

    // Members
    int64_t key;
    std::string file;

    // Methods
    svs::lib::SaveType save(const svs::lib::SaveContext& ctx) const {
        auto my_path = ctx.get_directory() / file;
        auto stream = svs::lib::open_write(my_path);
        svs::lib::write_binary(stream, key);
        return svs::lib::SaveType(
            toml::table({{"key", key}, {"file", file}}), save_version
        );
    }

    static Saveable load(
        const toml::table& table,
        const svs::lib::LoadContext& ctx,
        const svs::lib::Version& version
    ) {
        CATCH_REQUIRE(version == save_version);
        auto key = svs::get<int64_t>(table, "key");
        auto file = svs::get(table, "file").value();

        // Make sure any side-effects were saved correctly.
        auto path = ctx.get_directory() / file;
        auto stream = svs::lib::open_read(path);
        auto val = svs::lib::read_binary<int64_t>(stream);
        CATCH_REQUIRE(val == key);
        return Saveable{key, file};
    }

    friend bool operator==(const Saveable&, const Saveable&);
};

bool operator==(const Saveable& a, const Saveable& b) {
    return a.key == b.key && a.file == b.file;
}
} // namespace

CATCH_TEST_CASE("Save/Load", "[lib][save_load]") {
    // Setup the temporary directory.
    svs_test::prepare_temp_directory();
    auto temp_directory = svs_test::temp_directory();

    CATCH_SECTION("Saving") {
        CATCH_REQUIRE(svs::lib::is_saveable<Saveable>);
        CATCH_REQUIRE(!svs::lib::is_saveable<NotSaveable>);

        auto config_path = temp_directory / svs::lib::config_file_name;

        auto x = Saveable{10, "my_file.bin"};
        CATCH_REQUIRE(!std::filesystem::exists(config_path));
        svs::lib::save(x, temp_directory);
        CATCH_REQUIRE(std::filesystem::exists(config_path));

        // Read the generated config file.
        auto table = toml::parse_file(config_path.c_str());
        auto subtable = svs::subtable(table, "object");
        CATCH_REQUIRE((svs::get(subtable, "key", int64_t(0)) == 10));
        auto expected = std::string("my_file.bin");
        CATCH_REQUIRE((svs::get(subtable, "file", "none") == expected));

        // Loading
        auto y = svs::lib::load<Saveable>(temp_directory);
        CATCH_REQUIRE(y == x);

        // Make sure we can load from the full config file path as well.
        auto z = svs::lib::load<Saveable>(config_path);
        CATCH_REQUIRE(z == x);

        // Do this process again, but now with the automatic saving path.
        CATCH_REQUIRE(svs::lib::test_self_save_load(x, temp_directory / "another_level"));
    }
}
